"""Synthetic Bayesian/finite-MFA comparisons using train-only fitting/selection.

Run through examples/synthetic_mfa_comparison.py. Outputs use ExperimentRun's
content-derived folders and logs; no external data or downloads are required.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..config import HDPMFAConfig, HDPConfig, NumericalConfig
from ..model import HDPMFA
from ._mfa_reference import FiniteMFAReference
from ._runs import ExperimentRun, write_json


CASES = ('separated', 'overlapping', 'high_dimensional', 'markov')


@dataclass(frozen=True)
class SyntheticMFAData:
    train: np.ndarray
    test: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    means: np.ndarray
    loadings: np.ndarray
    noise: np.ndarray
    train_lengths: tuple[int, ...]
    test_lengths: tuple[int, ...]
    transition: np.ndarray


def synthetic_mfa(case='separated', *, seed=42, train_size=360, test_size=180):
    """Independent train/test draws from one known three-component MFA.

    Markov sequences restart independently in each split. Features themselves
    have no temporal order. Ground-truth labels are used only after fitting,
    except for the explicitly named oracle-K baseline's known component count.
    """
    if case not in CASES or train_size < 12 or test_size < 6:
        raise ValueError('Unknown case or insufficient train/test observations')
    rng = np.random.default_rng(seed)
    d, rank = (48, 3) if case == 'high_dimensional' else (10, 2)
    separation = {'separated': 5., 'overlapping': 1.3, 'high_dimensional': 3., 'markov': 1.3}[case]
    means = np.zeros((3, d))
    means[:, :2] = np.array([[-1.,0.], [1.,0.], [0.,1.]]) * separation
    loadings = rng.normal(0, .45, (3,d,rank))
    noise = np.linspace(.12, .24, d)
    transition = .96 * np.eye(3) + .02 * (1 - np.eye(3))

    def sample(count):
        lengths = (count,)
        if case == 'markov':
            lengths = (count//3, count//3, count-2*(count//3))
            labels = []
            for length in lengths:
                current = int(rng.integers(3))
                for _ in range(length):
                    labels.append(current)
                    current = int(rng.choice(3, p=transition[current]))
            labels = np.asarray(labels)
        else:
            labels = rng.choice(3, count, p=[.35,.4,.25])
        z = rng.normal(size=(count,rank))
        x = means[labels] + np.einsum('ndr,nr->nd',loadings[labels],z)
        x += rng.normal(size=x.shape) * np.sqrt(noise)
        return x, labels, lengths

    train, train_labels, train_lengths = sample(train_size)
    test, test_labels, test_lengths = sample(test_size)
    return SyntheticMFAData(train,test,train_labels,test_labels,means,loadings,noise,
                            train_lengths,test_lengths,transition)


def _covariance_error(w, noise, reference_w, reference_noise):
    """Relative Frobenius error via low-rank Gram matrices, without dense d²."""
    diagonal = noise - reference_noise
    difference = (np.square(w.T@w).sum() + np.square(reference_w.T@reference_w).sum()
        - 2*np.square(w.T@reference_w).sum() + np.square(diagonal).sum()
        + 2*np.dot(diagonal,np.square(w).sum(1)-np.square(reference_w).sum(1)))
    norm = (np.square(reference_w.T@reference_w).sum() + np.square(reference_noise).sum()
            + 2*np.dot(reference_noise,np.square(reference_w).sum(1)))
    return float(np.sqrt(max(0.,difference)/norm))


def _metrics(data, labels, test_labels, means, loadings, noise):
    ids = np.unique(labels)
    contingency = np.array([[np.sum((labels == k)&(data.train_labels == j)) for j in range(3)] for k in ids])
    rows, columns = linear_sum_assignment(-contingency)
    covariance_errors, mean_errors = [], []
    for row, reference in zip(rows,columns):
        k = ids[row]
        covariance_errors.append(_covariance_error(loadings[k],noise,data.loadings[reference],data.noise))
        mean_errors.append(float(np.linalg.norm(means[k]-data.means[reference])/np.sqrt(len(noise))))
    return dict(train_ari=float(adjusted_rand_score(data.train_labels,labels)),
        train_nmi=float(normalized_mutual_info_score(data.train_labels,labels)),
        test_ari=float(adjusted_rand_score(data.test_labels,test_labels)),
        test_nmi=float(normalized_mutual_info_score(data.test_labels,test_labels)),
        map_clusters=int(len(ids)), matched_covariance_error=float(np.mean(covariance_errors)),
        matched_mean_rmse=float(np.mean(mean_errors)), matched_regimes=len(rows),
        covariance_matching='Hungarian training contingency; unmatched regimes reported separately',
        unmatched_reference_regimes=3-len(rows), unmatched_occupied_components=len(ids)-len(rows))


def _iid_plugin_score(model, values):
    """Common iid emission-mixture diagnostic, including for the HMM ablation."""
    from scipy.special import logsumexp
    terms = np.column_stack([component.marginal_log_prob(values).cpu().numpy()
                             for component in model.components_])
    return logsumexp(terms+np.log(model.weights_),axis=1)


def run_comparison(data, *, seed=42, support=5, max_iter=200, n_init=3):
    """Fit every model on training data; test labels never select parameters/K."""
    if support < 3 or support > len(data.train):
        raise ValueError('Comparison support must be between 3 and training N')
    models, records = {}, []
    rank = data.loadings.shape[2]
    variants = [('dp_mfa_fixed',False,'fixed'), ('dp_mfa_adaptive',False,'birth_prune')]
    if len(data.train_lengths) > 1:
        variants.append(('hdp_hmm_mfa',True,'fixed'))
    for name, markov, strategy in variants:
        config = HDPMFAConfig(n_factors=rank,
            hdp=HDPConfig(M=1 if strategy=='birth_prune' else support, max_models=support,
                hmm_switch=markov,hdp_hyp='more_sticky' if markov else 'more'),
            numerics=NumericalConfig(seed=seed,num_threads=1),loading_prior_precision=.1,
            noise_initial=.4, noise_floor=1e-6, max_iter=max_iter,tol=1e-6,
            component_strategy=strategy,n_explore_steps=2,proposal_refine_steps=3)
        start = time.perf_counter()
        model = HDPMFA(config).fit(data.train,data.train_lengths)
        seconds = time.perf_counter()-start
        test_labels = model.predict(data.test,data.test_lengths)
        record = _metrics(data,model.labels_,test_labels,model.means_,model.loadings_,model.noise_)
        record.update(method=name,represented_components=model.n_components_,fit_seconds=seconds,
            iterations=model.n_iter_,converged=model.converged_,objective_kind='variational_lower_bound',
            training_objective=model.elbo_,minimum_objective_step=float(np.diff(model.elbo_history_).min()),
            test_iid_plugin_log_density=float(_iid_plugin_score(model,data.test).mean()),
            test_sequence_plugin_log_density=model.score(data.test,data.test_lengths),
            accepted_births=sum(m['accepted'] and m['move']=='birth' for m in model.structure_history_),
            accepted_prunes=sum(m['accepted'] and m['move']=='prune' for m in model.structure_history_),
            test_sequence_count=len(data.test_lengths),configuration=config.to_dict())
        models[name], records = model, records+[record]

    references = {}
    for k in range(1,support+1):
        start = time.perf_counter()
        reference = FiniteMFAReference(k,rank,seed=seed,n_init=n_init,max_iter=max_iter,
            noise_initial=.4,noise_floor=1e-6,tol=1e-6).fit(data.train)
        references[k] = (reference,time.perf_counter()-start)
    bic_k = min(references,key=lambda k: references[k][0].bic(data.train))
    for name,k in (('finite_mfa_oracle_k',3),('finite_mfa_same_support',support),('finite_mfa_bic',bic_k)):
        model, seconds = references[k]
        record = _metrics(data,model.labels_,model.predict_proba(data.test).argmax(1),
                          model.means_,model.loadings_,model.noise_)
        test_score = float(model.score_samples(data.test).mean())
        record.update(method=name,represented_components=k,fit_seconds=seconds,iterations=model.n_iter_,
            converged=model.converged_,objective_kind='observed_training_log_likelihood',
            training_objective=model.log_likelihood_,
            minimum_objective_step=float(np.diff(model.log_likelihood_history_).min()),
            test_iid_plugin_log_density=test_score,test_sequence_plugin_log_density=test_score,
            bic=float(model.bic(data.train)),n_init=n_init,selected_initialization=model.initialization_,
            failed_initializations=model.failed_initializations_,
            k_selection='known generative K (oracle)' if name.endswith('oracle_k') else
                        ('training BIC over 1..support' if name.endswith('bic') else 'same fixed support as DP-MFA'))
        if name.endswith('bic'):
            record['selected_fit_seconds'] = seconds
            record['selection_fit_seconds'] = sum(seconds for _,seconds in references.values())
            record['fit_seconds'] = record['selection_fit_seconds']
            record['bic_candidates'] = {str(k):float(ref.bic(data.train)) for k,(ref,_) in references.items()}
        models[name], records = model, records+[record]
    return models, records


def _plots(path, data, models, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    projection = PCA(2,svd_solver='full').fit_transform(data.train)
    fig,axes = plt.subplots(2,3,figsize=(16,9),layout='constrained')
    names = ['Truth','dp_mfa_fixed','finite_mfa_oracle_k']
    for axis,name in zip(axes[0],names):
        labels = data.train_labels if name=='Truth' else models[name].labels_
        axis.scatter(projection[:,0],projection[:,1],c=labels,s=9,cmap='tab10',alpha=.7)
        axis.set_title(name); axis.set_xlabel('Common PCA display coordinate 1')
    axes[1,0].bar(np.arange(len(records)),[r['test_iid_plugin_log_density'] for r in records])
    axes[1,0].set_xticks(np.arange(len(records)),[r['method'] for r in records],rotation=45,ha='right',fontsize=8)
    axes[1,0].set_title('Held-out iid plug-in density (higher is better)')
    for name,model in models.items():
        if isinstance(model,HDPMFA): axes[1,1].plot(model.elbo_history_,label=name)
        else: axes[1,2].plot(model.log_likelihood_history_,label=name)
    axes[1,1].set_title('Bayesian training bounds'); axes[1,2].set_title('Finite MFA training log likelihoods')
    for axis in axes[1,1:]:
        axis.set_xlabel('Iteration'); axis.legend(fontsize=8)
    fig.suptitle('MFA comparison | display PCA is not the inference representation')
    fig.savefig(path/'comparison.png',dpi=130); plt.close(fig)
    chosen = ['dp_mfa_fixed','dp_mfa_adaptive','finite_mfa_oracle_k']
    if 'hdp_hmm_mfa' in models: chosen.append('hdp_hmm_mfa')
    fig,axes = plt.subplots(3,len(chosen)+1,figsize=(4*(len(chosen)+1),10),layout='constrained')
    true_cov = data.loadings@data.loadings.transpose(0,2,1)+np.diag(data.noise)
    for j in range(3):
        # Dense matrices are created solely for the diagnostic heatmaps.
        im=axes[j,0].imshow(true_cov[j],cmap='RdBu_r')
        axes[j,0].set_title(f'True component {j}'); fig.colorbar(im,ax=axes[j,0],fraction=.045)
        for column,name in enumerate(chosen,1):
            model=models[name]; occupied=np.unique(model.labels_)
            k=int(occupied[np.argmin(np.square(model.means_[occupied]-data.means[j]).sum(1))])
            cov=model.loadings_[k]@model.loadings_[k].T+np.diag(model.noise_)
            im=axes[j,column].imshow(cov,cmap='RdBu_r',vmin=im.get_clim()[0],vmax=im.get_clim()[1])
            axes[j,column].set_title(f'{name}\nnearest centroid: {k}',fontsize=9)
            fig.colorbar(im,ax=axes[j,column],fraction=.045)
    fig.suptitle('Reference and fitted plug-in covariance | centroid matching for display')
    fig.savefig(path/'covariances.png',dpi=110); plt.close(fig)
    if len(data.train_lengths)>1:
        fig,axes=plt.subplots(1+len(chosen),1,figsize=(14,2*(1+len(chosen))),layout='constrained')
        for axis,name in zip(axes,['Truth']+chosen):
            labels=data.train_labels if name=='Truth' else models[name].labels_
            axis.scatter(np.arange(len(labels)),labels,c=labels,cmap='tab10',s=8)
            for boundary in np.cumsum(data.train_lengths)[:-1]: axis.axvline(boundary-.5,c='k',ls='--',lw=.7)
            axis.set_title(name); axis.set_yticks(np.unique(labels))
        fig.savefig(path/'sequences.png',dpi=130); plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,19,42])
    parser.add_argument('--train-size',type=int,default=360)
    parser.add_argument('--test-size',type=int,default=180)
    parser.add_argument('--support',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=200)
    parser.add_argument('--reference-starts',type=int,default=3)
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=Path('agents/experiments/mfa/comparison'))
    args=parser.parse_args(argv)
    rows=[]
    for case in args.cases:
        for seed in args.seeds:
            data=synthetic_mfa(case,seed=seed,train_size=args.train_size,test_size=args.test_size)
            fingerprint=hashlib.sha256(data.train.tobytes()+data.test.tobytes()).hexdigest()
            settings={key:value for key,value in vars(args).items() if key not in ('output_dir','cases','seeds')}
            settings.update(case=case,seed=seed)
            run=ExperimentRun(args.output_dir,f'{case}_seed_{seed}','synthetic_mfa_comparison',settings,fingerprint)
            with run:
                models,records=run_comparison(data,seed=seed,support=args.support,
                    max_iter=args.iterations,n_init=args.reference_starts)
                arrays=dict(train=data.train,test=data.test,train_labels=data.train_labels,test_labels=data.test_labels,
                    true_means=data.means,true_loadings=data.loadings,true_noise=data.noise,
                    train_lengths=data.train_lengths,test_lengths=data.test_lengths)
                for name,model in models.items():
                    arrays[name+'_labels']=model.labels_
                    arrays[name+'_responsibilities']=model.responsibilities_
                    arrays[name+'_means']=model.means_; arrays[name+'_loadings']=model.loadings_
                    arrays[name+'_noise']=model.noise_
                    arrays[name+'_weights']=model.weights_
                    arrays[name+'_objective_history']=(model.elbo_history_ if isinstance(model,HDPMFA)
                                                       else model.log_likelihood_history_)
                    if isinstance(model,HDPMFA):
                        model.save(run.path/(name+'.json'))
                        arrays[name+'_coordinates']=model.transform(data.train,sequence_lengths=data.train_lengths)
                        clone=HDPMFA.load(run.path/(name+'.json'))
                        error=float(np.max(np.abs(model.predict_proba(data.test,data.test_lengths)-
                                                  clone.predict_proba(data.test,data.test_lengths))))
                        next(r for r in records if r['method']==name)['roundtrip_assignment_error']=error
                np.savez_compressed(run.path/'data_and_models.npz',**arrays)
                if not args.no_plots: _plots(run.path,data,models,records)
                report=dict(status='ok',run_id=run.run_id,case=case,seed=seed,records=records,
                    note='ELBOs and ML training objectives are not interchangeable; compare held-out densities.',
                    artifacts=sorted(p.name for p in run.path.iterdir() if p.name not in ('results.json','clustering.log')))
                write_json(run.path/'results.json',report)
                for record in records:
                    print(f'{case} seed={seed}: {record["method"]}, K={record["map_clusters"]}, '
                          f'test ARI={record["test_ari"]:.3f}, iid logp={record["test_iid_plugin_log_density"]:.4f}')
                    rows.append(dict(case=case,seed=seed,run_id=run.run_id,**record))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/'comparison.json',rows)
    fields=['case','seed','method','represented_components','map_clusters','train_ari','test_ari','test_nmi',
            'test_iid_plugin_log_density','test_sequence_plugin_log_density','matched_covariance_error',
            'matched_regimes','unmatched_reference_regimes','unmatched_occupied_components',
            'fit_seconds','iterations','converged','run_id']
    with (args.output_dir/'comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    aggregates=[]
    for case in args.cases:
        for method in sorted({row['method'] for row in rows if row['case']==case}):
            group=[row for row in rows if row['case']==case and row['method']==method]
            entry=dict(case=case,method=method,seeds=len(group))
            for metric in ('test_ari','test_nmi','test_iid_plugin_log_density','test_sequence_plugin_log_density',
                           'matched_covariance_error','map_clusters'):
                values=[row[metric] for row in group]
                entry[metric+'_mean']=float(np.mean(values)); entry[metric+'_std']=float(np.std(values))
            aggregates.append(entry)
    write_json(args.output_dir/'aggregate.json',aggregates)
    return rows


if __name__=='__main__':
    main()
