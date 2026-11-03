"""Opt-in shared sampling-clock trajectories, separate from retained observations.

Every tick has a Gaussian state. Only positive-responsibility observations have
measurement statistics. The legacy emission histories remain observation-indexed
views, so observation policies, alignment and component ownership keep their
existing contracts. This is a discrete-time model, not continuous-time ODE fitting.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class SamplingTrajectory:
    prior_mean: torch.Tensor
    prior_covariance: torch.Tensor
    filtered_means: list = field(default_factory=list)
    filtered_covariances: list = field(default_factory=list)
    means: list = field(default_factory=list)
    covariances: list = field(default_factory=list)
    predicted_means: list = field(default_factory=list)
    predicted_covariances: list = field(default_factory=list)
    matrices: list = field(default_factory=list)
    noises: list = field(default_factory=list)
    observations: dict = field(default_factory=dict)
    parameter_tick: int = 0
    shared_parameter_tick: int = 0


def enabled(emission):
    if getattr(emission, 'dynamics_clock', 'cluster-clock') != 'physical-clock':
        return False
    independent=getattr(emission,'_independent_factors',None)
    return not (independent is not None and emission.A and independent())


def trajectory(emission):
    state = getattr(emission, '_sampling_trajectory', None)
    if state is None:
        device=emission.A[-1].device
        state = SamplingTrajectory(torch.as_tensor(emission.f_star[0],device=device,dtype=torch.float64).detach().clone().reshape(-1, 1),
                                   torch.as_tensor(emission.cov_f[0],device=device,dtype=torch.float64).detach().clone())
        emission._sampling_trajectory = state
    return state


def _symmetric(value):
    return .5*(value+value.T)


def _views(emission, state):
    indices = sorted(state.observations)
    emission.indexes = indices
    emission.N = len(indices)
    emission.x_train = [state.observations[t][0] for t in indices]
    emission.y_train = [state.observations[t][1] for t in indices]
    emission.f_star = [state.prior_mean]+[state.filtered_means[t] for t in indices]
    emission.cov_f = [state.prior_covariance]+[state.filtered_covariances[t] for t in indices]
    emission.f_star_sm = [state.prior_mean]+[state.means[t] for t in indices]
    emission.cov_f_sm = [state.prior_covariance]+[state.covariances[t] for t in indices]


def advance(emission, index):
    """Commit prediction-only ticks; no N, observation mass or parameter update."""
    state = trajectory(emission)
    while len(state.means) <= index:
        t = len(state.means)
        A, Q = emission.A[-1], emission.Gamma[-1]
        if t:
            mean = A @ state.filtered_means[-1]
            covariance = _symmetric(A @ state.filtered_covariances[-1] @ A.T+Q)
        else:
            mean, covariance = state.prior_mean, state.prior_covariance
        state.predicted_means.append(mean.detach().clone())
        state.predicted_covariances.append(covariance.detach().clone())
        state.filtered_means.append(mean.detach().clone())
        state.filtered_covariances.append(covariance.detach().clone())
        state.means.append(mean.detach().clone())
        state.covariances.append(covariance.detach().clone())
        state.matrices.append(A.detach().clone())
        state.noises.append(Q.detach().clone())
    return state


def state_at(emission, index=None, *, smoothed=True):
    """Read-only propagation, including queries beyond the last committed tick."""
    state = trajectory(emission)
    if index is None:
        index = getattr(emission, '_clock_target_index', len(state.means)-1)
    if index < 0:
        return state.prior_mean, state.prior_covariance
    if index < len(state.means):
        return ((state.means[index],state.covariances[index]) if smoothed else
                (state.filtered_means[index],state.filtered_covariances[index]))
    if state.means:
        mean, covariance = state.filtered_means[-1], state.filtered_covariances[-1]
        start = len(state.means)
    else:
        mean, covariance = state.prior_mean,state.prior_covariance
        start = 1  # the initial prior is the state at sampling tick zero
    for _ in range(start,index+1):
        A,Q=emission.A[-1],emission.Gamma[-1]
        mean=A @ mean; covariance=_symmetric(A @ covariance @ A.T+Q)
    return mean,covariance


def _flow_condition(emission, mean, covariance, grids, observation, weight):
    from ..alignment._gp_flow import _column, _kernel_torch
    device=mean.device
    grids=torch.as_tensor(grids,dtype=mean.dtype,device=device)
    if grids.ndim==2:grids=grids.unsqueeze(-1)
    basis=_column(emission.x_basis,device=device)
    y=_column(observation,device=device)
    C,Sigma=emission.C[-1],emission.Sigma[-1]
    eye=torch.eye(len(basis),device=device,dtype=mean.dtype)
    precision=torch.cholesky_inverse(emission._chol_spd(covariance))
    natural=precision @ mean
    kernel=emission.gp.kernel
    K=_kernel_torch(kernel,basis,basis)
    chol=torch.linalg.cholesky(_symmetric(K)+1e-4*eye)
    eye_y=torch.eye(len(y),device=device,dtype=mean.dtype)
    constant=bool(torch.all(torch.isclose(torch.diag(Sigma),torch.diag(Sigma).mean())))
    for grid in grids:
        cross=_kernel_torch(kernel,basis,grid)
        projection=torch.cholesky_solve(cross,chol).T
        H=projection @ C
        R=(torch.diag(Sigma).mean()*eye_y if constant else
           _kernel_torch(kernel,grid,grid)-cross.T @ projection.T+projection @ Sigma @ projection.T)
        L=torch.linalg.cholesky(_symmetric(R)+1e-6*eye_y)
        precision=precision+weight/len(grids)*H.T @ torch.cholesky_solve(H,L)
        natural=natural+weight/len(grids)*H.T @ torch.cholesky_solve(y,L)
    covariance=torch.cholesky_inverse(emission._chol_spd(precision))
    return covariance @ natural,covariance


def include(emission,index,grid,warped,observation,weight,*,snr=None,grids=None):
    observation=emission.cond_to_cuda(emission.cond_to_torch(observation)).reshape(-1,1)
    grid=emission.cond_to_cuda(emission.cond_to_torch(grid))
    if weight>0 and emission.N==0 and not emission.fitted:
        valid=torch.allclose(torch.from_numpy(emission.gp.kernel.theta),
                             torch.from_numpy(emission.ini_kernel_theta))
        emission.fit_kernel_params(grid,observation,emission.Sigma[-1],emission.Gamma[-1],valid=valid)
        # Fitting the original kernel changes the prior before any data were retained.
        emission._sampling_trajectory=None
    state=advance(emission,int(index))
    if weight>0 and (snr is None or snr>.5):
        mean,covariance=state.predicted_means[index],state.predicted_covariances[index]
        if grids is None:
            identity=torch.eye(len(mean),device=mean.device,dtype=mean.dtype)
            mean,covariance=emission.gp.posterior(mean,covariance,observation,identity,
                torch.zeros_like(identity),emission.C[-1],emission.Sigma[-1],x_train=grid,
                x_warped=warped,h=float(weight))
        else:
            mean,covariance=_flow_condition(emission,mean,covariance,grids,observation,float(weight))
            grid=torch.as_tensor(grids,dtype=mean.dtype,device=mean.device).mean(0)
        state.filtered_means[index]=mean.detach()
        state.filtered_covariances[index]=covariance.detach()
        state.means[index]=mean.detach().clone()
        state.covariances[index]=covariance.detach().clone()
        state.observations[int(index)]=(grid.detach(),observation.detach(),float(weight))
    _views(emission,state)
    emission._clock_target_index=int(index)
    return emission.x_basis


def smooth(emission):
    state=trajectory(emission)
    state.means=[mean.clone() for mean in state.filtered_means]
    state.covariances=[covariance.clone() for covariance in state.filtered_covariances]
    for t in range(len(state.means)-2,-1,-1):
        A,Q=state.matrices[t+1],state.noises[t+1]
        P=state.filtered_covariances[t]
        predicted=state.predicted_covariances[t+1]
        gain=torch.cholesky_solve(A @ P.T,emission._chol_spd(predicted)).T
        state.means[t]=state.filtered_means[t]+gain @ (state.means[t+1]-state.predicted_means[t+1])
        eye=torch.eye(len(P),device=P.device,dtype=P.dtype)
        projection=eye-gain @ A
        reverse=projection @ P @ projection.T+gain @ Q @ gain.T
        state.covariances[t]=_symmetric(reverse+gain @ state.covariances[t+1] @ gain.T)
    _views(emission,state)


def transition_statistics(emission,*,start=1,end=None):
    """Unit-mass physical transitions; missing measurements have no emission mass."""
    state=trajectory(emission)
    end=(max(state.observations,default=-1)+1) if end is None else min(end,len(state.means))
    y,x=[],[]
    zero=torch.zeros_like(state.prior_covariance)
    yy,xx,yx=zero.clone(),zero.clone(),zero.clone()
    for t in range(max(1,start),end):
        current,previous=state.means[t],state.means[t-1]
        P=state.filtered_covariances[t-1]
        gain=torch.cholesky_solve(state.matrices[t] @ P.T,
                                 emission._chol_spd(state.predicted_covariances[t])).T
        y.append(current);x.append(previous)
        yy+=state.covariances[t];xx+=state.covariances[t-1]
        yx+=state.covariances[t] @ gain.T
    if not y:return None
    return len(y),torch.cat(y,1),torch.cat(x,1),yy,xx,yx


def observation_statistics(emission,indices):
    state=trajectory(emission)
    y,x,roots=[],[],[]
    dim=state.prior_mean.numel()
    out=emission.C[-1].shape[0]
    device,dtype=state.prior_mean.device,state.prior_mean.dtype
    cov_y=torch.zeros((out,out),device=device,dtype=dtype)
    cov_x=torch.zeros((dim,dim),device=device,dtype=dtype)
    cross=torch.zeros((out,dim),device=device,dtype=dtype)
    mass=0.
    for t in indices:
        grid,observation,weight=state.observations[t]
        if getattr(emission,'direct_observation_offset',None) is not None:
            observation=observation-emission.direct_observation_offset
        else:
            observation=emission.project_observation_to_basis(observation,grid,allow_warped=True)
        root=weight**.5
        y.append(root*observation);x.append(root*state.means[t]);roots.append(root)
        cov_x+=weight*state.covariances[t];mass+=weight
    if not y:return None
    return (mass,torch.cat(y,1),torch.cat(x,1),cov_y,cov_x,cross),roots


def _posterior(emission,base,statistics,*,batch,roots=None):
    from ._distributions import inv_wishart
    from . import _trajectory
    mass,y,x,yy,xx,yx=statistics
    if isinstance(base,inv_wishart):
        C=base.get_C();error=y-C @ x
        scatter=error @ error.T+C @ xx @ C.T
        degrees=base.n0+mass
        return inv_wishart(degrees,_symmetric(((base.n0-2)*base.scale+scatter)/(degrees-2)),C)
    extra={'intercept_weights':roots} if getattr(base,'affine_intercept',False) else {}
    method=(base.posterior if batch else base.posterior_natural_gradient
            if emission.lds_update=='natural_gradient' else base.posterior_legacy
            if emission.lds_update=='legacy' and hasattr(base,'posterior_legacy') else base.posterior)
    if not batch and emission.lds_update=='natural_gradient':
        extra['step_size']=emission.lds_step_size
    result=method(*statistics,**extra)
    _trajectory._require_spd_posterior(result,'physical-clock')
    return result


def _positive(value):
    value=_symmetric(value)
    vals,vectors=torch.linalg.eigh(value)
    floor=max(float(vals.abs().max())*1e-8,1e-10)
    return (vectors*vals.clamp_min(floor)) @ vectors.T


def update_parameters(emission,weight=1.,*,batch=False,full_data=False,first_observation=False,
                      force=False,reestimate=True,observation_window=None,check_var=False):
    from . import _trajectory
    state=trajectory(emission)
    if not state.observations or weight<=0:return
    if not (batch or full_data) and getattr(emission,'_clock_target_index',-1) not in state.observations:
        return
    count=len(state.observations)
    permitted=(batch or full_data or force or count<emission.estimation_limit)
    if not permitted or not reestimate or (count==1 and not (batch or first_observation or full_data)):
        if not batch:_trajectory.hold_lds_params(emission)
        return
    indices=sorted(state.observations)
    if np.isfinite(emission.estimation_limit):indices=indices[:int(emission.estimation_limit)]
    if observation_window is not None:
        indices=indices[-int(observation_window):] if observation_window>0 else []
    if not indices:
        if not batch:_trajectory.hold_lds_params(emission)
        return
    last=indices[-1]
    start=1 if batch or full_data else max(1,state.parameter_tick+1)
    if full_data and observation_window is not None:start=max(1,indices[0]+1)
    trans=transition_statistics(emission,start=start,end=last+1)
    if not bool(torch.count_nonzero(emission.Gamma_def)):trans=None
    observed=indices if batch or full_data else [last]
    obs,roots=observation_statistics(emission,observed)
    A,Q,C,R=emission.A[-1],emission.Gamma[-1],emission.C[-1],emission.Sigma[-1]
    new_internal=getattr(emission,'internal_params',None)
    new_observation=getattr(emission,'observation_params',None)
    try:
        if emission.bayesian:
            if trans is not None and getattr(emission,'transition_policy',None) is None:
                new_internal=_posterior(emission,emission.internal_params,trans,batch=batch)
                A=new_internal.get_mean()
                Q=_trajectory._lds_scale(emission,new_internal,final=batch or full_data)
            if getattr(emission,'observation_policy',None) is None:
                new_observation=_posterior(emission,emission.observation_params,obs,batch=batch,roots=roots)
                C=new_observation.get_mean()
                R=_trajectory._lds_scale(emission,new_observation,final=batch or full_data)
        else:
            def fit(statistics):
                mass,y,x,yy,xx,yx=statistics
                syy=y @ y.T+yy;sxx=x @ x.T+xx;syx=y @ x.T+yx
                matrix=torch.linalg.solve(_positive(sxx),syx.T).T
                noise=(syy-matrix @ syx.T-syx @ matrix.T+matrix @ sxx @ matrix.T)/mass
                return matrix,_positive(noise)
            if trans is not None:A,Q=fit(trans)
            C,R=fit(obs)
    except torch.linalg.LinAlgError as error:
        emission._lds_update_failures=getattr(emission,'_lds_update_failures',0)+1
        if emission.verbose:
            print(f'Physical-clock LDS update skipped: {error}',flush=True)
        if not batch:_trajectory.hold_lds_params(emission)
        return
    emission.internal_params=new_internal
    emission.observation_params=new_observation
    if emission.annealing:
        if trans is not None:Q=Q+emission.Gamma_def/max(1,count)**2
        if getattr(emission,'direct_observation_offset',None) is None:
            R=R+emission.Sigma_def/max(1,count)**2
    policies=getattr(emission,'lds_pair_policies',None)
    if policies is not None:
        if policies['transition'][1].mode=='fixed':Q=emission.Gamma_def
        if policies['observation'][1].mode=='fixed':R=emission.Sigma_def
    if not emission.bayesian and check_var:R=emission.check_bound_sigma(R)
    if batch:
        emission.A[-1],emission.Gamma[-1]=A,Q
        emission.C[-1],emission.Sigma[-1]=C,R
    else:
        for name,value in (('A',A),('Gamma',Q),('C',C),('Sigma',R)):
            getattr(emission,name).append(value)
    var=torch.diag(Q).clamp_min(0).sqrt().reshape(-1,1)
    yvar=torch.diag(R).clamp_min(0).sqrt().reshape(-1,1)
    if batch:emission.var[-1],emission.y_var[-1]=var,yvar
    else:emission.var.append(var);emission.y_var.append(yvar)
    if getattr(emission,'factor_analysis',False):
        from ..emissions._factor_analysis import refresh_factor_observation
        refresh_factor_observation(emission)
    state.parameter_tick=last
    _trajectory.record_lds_parameter_score(emission,replace_last=batch)


def full_pass(emission,grids,values,weights,*,samples=None,deterministic=False):
    from . import _trajectory
    active=set(emission._active_weighted_indices(weights).tolist())
    if not active:
        return emission.compute_sq_err_all(grids,values),emission.compute_q_lat_all(grids)
    batch=emission.lds_update=='conjugate_batch'
    emission.deterministic_input_warp=deterministic
    for index in range(len(values)):
        weight=float(weights[index]) if index in active else 0.
        include(emission,index,grids[index],grids[index],values[index],weight,
                grids=None if samples is None else samples[index])
        if weight>0:
            if batch:_trajectory.hold_lds_params(emission)
            elif bool(torch.count_nonzero(emission.Gamma[-1])):
                smooth(emission)
                update_parameters(emission,weight,first_observation=(
                    getattr(emission,'factor_analysis',False) and emission.N==1))
    smooth(emission)
    if batch:update_parameters(emission,batch=True)
    mean_grids=grids if samples is None else torch.stack([sample.mean(0) for sample in samples])
    return emission.compute_sq_err_all(mean_grids,values),emission.compute_q_lat_all(mean_grids)


def latent_scores(emission,count=None):
    state=trajectory(emission)
    count=len(state.means) if count is None else min(count,len(state.means))
    scores=state.prior_covariance.new_zeros(count)
    if not bool(torch.count_nonzero(emission.Gamma[-1])):return scores
    for t in range(count):
        current,P=state.means[t],state.covariances[t]
        if t==0:
            residual=current-state.prior_mean;variance=P
            noise=state.prior_covariance
            logdet_cond=2*emission._chol_spd(P).diag().log().sum()
        else:
            A_q,Q_q=state.matrices[t],state.noises[t]
            A,Q=(emission.A[-1],emission.Gamma[-1]) if emission.lds_scoring=='final' else (A_q,Q_q)
            filtered=state.filtered_covariances[t-1]
            gain=torch.cholesky_solve(A_q @ filtered.T,
                                     emission._chol_spd(state.predicted_covariances[t])).T
            eye=torch.eye(len(P),device=P.device,dtype=P.dtype)
            reverse=(eye-gain @ A_q) @ filtered @ (eye-gain @ A_q).T+gain @ Q_q @ gain.T
            previous=reverse+gain @ P @ gain.T
            residual=current-A @ state.means[t-1]
            variance=(eye-A @ gain) @ P @ (eye-A @ gain).T+A @ reverse @ A.T
            noise=Q
            logdet_cond=2*(emission._chol_spd(P).diag().log().sum()
                +emission._chol_spd(reverse).diag().log().sum()
                -emission._chol_spd(previous).diag().log().sum())
        chol=emission._chol_spd(noise)
        scores[t]=.5*(len(P)+logdet_cond-2*chol.diag().log().sum()
            -(residual*torch.cholesky_solve(residual,chol)).sum()
            -torch.trace(torch.cholesky_solve(variance,chol)))
    return scores


def observation_score(emission,grid,value,index=None,*,mean=None,covariance=None,C=None,Sigma=None,first=False):
    if mean is None:mean,covariance=state_at(emission,index)
    if emission.lds_scoring=='historical' and C is None and Sigma is None:
        from bisect import bisect_right
        index=getattr(emission,'_clock_target_index',len(trajectory(emission).means)-1) if index is None else index
        _,_,C,Sigma=emission.get_params(bisect_right(emission.indexes,index))
    C=emission.C[-1] if C is None else C
    Sigma=emission.Sigma[-1] if Sigma is None else Sigma
    value=emission.cond_to_cuda(emission.cond_to_torch(value)).reshape(-1,1)
    model=getattr(emission,'fixed_observation_model',None)
    if model is not None:
        if not torch.equal(emission.cond_to_torch(grid).reshape(-1,1),emission.direct_observation_grid):
            raise ValueError('Fixed observation requires its configured waveform grid')
        return model.expected_log_likelihood(value,mean,covariance,noise=Sigma)
    projected,noise=emission._observation_distribution(grid,emission.x_basis,C @ mean,Sigma)
    latent=emission._observation_latent_cov(grid,covariance,C)
    return emission._gaussian_score_shared_cov(value.reshape(1,-1,1),projected,noise,latent)[0]


def convert(emission,target):
    """Move/copy the extra clock state with the existing emission conversions."""
    state=getattr(emission,'_sampling_trajectory',None)
    if state is None:return
    def tensor(value):
        if isinstance(value,(torch.Tensor,np.ndarray)):
            if target=='numpy':return torch.as_tensor(value).detach().cpu().numpy().copy()
            result=torch.as_tensor(value,dtype=torch.float64)
            return result.to(target) if target in ('cpu','cuda') else result
        if isinstance(value,list):return [tensor(item) for item in value]
        if isinstance(value,tuple):return tuple(tensor(item) for item in value)
        if isinstance(value,dict):return {key:tensor(item) for key,item in value.items()}
        return value
    for name in state.__dataclass_fields__:
        setattr(state,name,tensor(getattr(state,name)))
