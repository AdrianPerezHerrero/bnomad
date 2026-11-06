"""Fitted latent assignments and projected one-step motion at encoded samples."""
from numbers import Integral

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from ._embedding_plots import _embedding_projection
from ._lds_plots import model_lds_arrays


def model_embedding_dynamics(model, codes, *, output=0):
    """Copy a common latent PCA view and P((A_k-I)z+b_k) for each MAP sample.

    Codes must use the selected output's native LDS coordinates. No decoder,
    clustering or dynamics are fitted here. Each full-dimensional sample is
    transformed before projecting its change; omitted coordinates are retained
    in the calculation. These are conditional mean changes under final fitted
    parameters, not inferred next observations or observed velocities.
    """
    codes = np.asarray(codes, dtype=np.float64)
    assignments = np.asarray(model.labels_, dtype=int).reshape(-1)
    if (codes.ndim != 2 or not codes.shape[1] or len(codes) < 2
            or len(assignments) != len(codes) or not np.isfinite(codes).all()):
        raise ValueError('Codes must be finite native latent rows, one per fitted assignment')
    matrices = model_lds_arrays(model, output=output)
    change = np.empty_like(codes)
    for cluster in np.unique(assignments):
        parameters = matrices[int(cluster)]
        A = parameters['A']
        b = parameters.get('b', np.zeros((codes.shape[1], 1)))
        if (A.shape != (codes.shape[1], codes.shape[1]) or b.shape != (codes.shape[1], 1)
                or not np.isfinite(A).all() or not np.isfinite(b).all()):
            raise ValueError('Codes and fitted transition matrices must share the native latent dimension')
        chosen = assignments == cluster
        change[chosen] = codes[chosen] @ (A-np.eye(len(A))).T+b.reshape(1,-1)
    coordinates, components, center, variance = _embedding_projection(codes)
    return dict(coordinates=coordinates, changes=change @ components.T,
                projection_components=components.copy(), projection_center=center.copy(),
                explained_variance_ratio=np.asarray(variance).copy(), assignments=assignments.copy())


def _operator_fields(data, coverage):
    """Least-squares affine motion and empirical support from every member.

    Each field minimizes sum ||d_i - J(u_i-center) - mean(d)||^2. With
    omitted latent coordinates this is a display summary, not a replacement
    LDS. Rank-deficient groups identify only motion along their sampled span;
    a singleton contributes its mean motion and no identifiable Jacobian.
    """
    if isinstance(coverage,bool) or not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError('coverage must be in (0, 1]')
    xy, changes, assignments = data['coordinates'],data['changes'],data['assignments']
    minimum_variance = (.01*max(float(np.linalg.norm(np.ptp(xy,axis=0))),1.))**2
    fields = {}
    for cluster in np.unique(assignments):
        member = assignments == cluster
        points, motion = xy[member],changes[member]
        center, mean_motion = points.mean(0),motion.mean(0)
        centered = points-center
        coefficients,_,rank,_ = np.linalg.lstsq(centered,motion-mean_motion,rcond=None)
        jacobian = coefficients.T
        residual = motion-(centered @ jacobian.T+mean_motion)
        covariance = centered.T @ centered/len(points)
        eigenvalues,eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues,minimum_variance)
        precision = (eigenvectors/eigenvalues) @ eigenvectors.T
        distances = np.einsum('ni,ij,nj->n',centered,precision,centered)
        fields[int(cluster)] = dict(center=center,mean_motion=mean_motion,jacobian=jacobian,
            operator=np.eye(2)+jacobian,intercept=mean_motion-jacobian @ center,
            members=len(points),rank=int(rank),rms_error=float(np.sqrt(np.mean(residual**2))),
            region_precision=precision,region_logdet=float(np.log(eigenvalues).sum()),
            region_radius_squared=max(float(np.quantile(distances,coverage,method='higher')),1.))
    return fields


def _region_assignment(points, fields):
    """Resolve overlapping empirical core regions by projected Gaussian density."""
    assigned = np.full(len(points),-1,dtype=int)
    best = np.full(len(points),-np.inf)
    for index,field in enumerate(fields.values()):
        centered = points-field['center']
        distance = np.einsum('ni,ij,nj->n',centered,field['region_precision'],centered)
        score = -.5*(distance+field['region_logdet'])
        better = ((distance <= field['region_radius_squared']*(1+1e-12)) & (score > best))
        assigned[better],best[better] = index,score[better]
    return assigned


def _field_arrows(fields, low, high, maximum):
    """Evaluate each fitted field on a regular grid, not a subset of samples."""
    budget = max(maximum,len(fields))
    side = max(1,int(np.sqrt(budget-len(fields))))
    gx,gy = np.meshgrid(np.linspace(low[0],high[0],side+2)[1:-1],
                        np.linspace(low[1],high[1],side+2)[1:-1])
    points = np.column_stack((gx.ravel(),gy.ravel()))
    owned = _region_assignment(points,fields)
    points,owned = points[owned>=0],owned[owned>=0]
    # Small cores may fall between grid nodes. Evaluate their operator at its
    # center so those clusters remain represented, without selecting samples.
    for index,field in enumerate(fields.values()):
        if not np.any(owned == index):
            points = np.vstack((points,field['center']))
            owned = np.append(owned,index)
    changes = np.empty_like(points)
    for index,field in enumerate(fields.values()):
        member = owned == index
        changes[member] = ((points[member]-field['center']) @ field['jacobian'].T
                           +field['mean_motion'])
    return points,changes


def plot_embedding_dynamics(codes, model, labels=None, *, title=None,
                            projection_name='Autoencoder', output=0, max_arrows=180,
                            region_resolution=200, coverage=.9):
    """Draw one affine operator region per cluster, plus reference labels.

    Every assigned full-code increment contributes to a least-squares projected
    field. Arrows evaluate these fields on a regular grid within empirical
    ellipses covering at least coverage of their members; density resolves
    overlaps. These are display summaries, not temporal decision boundaries.
    All arrows share a multiplier. max_arrows controls grid density, not how
    many observations contribute. Reference labels do not affect the fields.
    """
    if (isinstance(max_arrows,bool) or not isinstance(max_arrows,Integral) or max_arrows < 1
            or isinstance(region_resolution,bool) or not isinstance(region_resolution,Integral)
            or region_resolution < 2):
        raise ValueError('max_arrows must be positive and region_resolution must be at least two')
    data = model_embedding_dynamics(model,codes,output=output)
    xy, assignments = data['coordinates'],data['assignments']
    fields = _operator_fields(data,coverage)
    labels = None if labels is None else np.asarray(labels).reshape(-1)
    if labels is not None and len(labels) != len(xy):
        raise ValueError('Reference labels must align with encoded samples')
    clusters, cluster_index = np.unique(assignments,return_inverse=True)
    palette = plt.get_cmap('tab10' if len(clusters)<=10 else 'tab20' if len(clusters)<=20 else 'turbo')
    positions = (np.arange(len(clusters)) if len(clusters)<=10 else
                 np.r_[np.arange(0,20,2),np.arange(1,20,2)][:len(clusters)] if len(clusters)<=20
                 else np.mod(np.arange(len(clusters))*.6180339887498949,1.))
    colors = [palette(position) for position in positions]
    columns = min(6,len(clusters))
    legend_height = .45+.28*int(np.ceil(len(clusters)/columns))
    figure = plt.figure(figsize=(14 if labels is not None else 8,6.2+legend_height),constrained_layout=True)
    layout = figure.add_gridspec(3,2 if labels is not None else 1,
                                height_ratios=(5,legend_height,.7))
    axis = figure.add_subplot(layout[0,0])
    low, high = xy.min(0),xy.max(0)
    span = np.maximum(high-low,max(float(np.ptp(xy,axis=0).max()),1.)*.05)
    low, high = low-.08*span,high+.08*span
    # Expand a collapsed coordinate as well (constant/one-dimensional codes).
    middle = (low+high)/2
    span = np.maximum(high-low,span)
    low,high = middle-span/2,middle+span/2
    xx,yy = np.meshgrid(np.linspace(low[0],high[0],region_resolution),
                        np.linspace(low[1],high[1],region_resolution))
    regions = _region_assignment(np.column_stack((xx.ravel(),yy.ravel())),fields)
    axis.imshow(np.ma.masked_less(regions.reshape(xx.shape),0),origin='lower',
                extent=(low[0],high[0],low[1],high[1]),cmap=ListedColormap(colors),
                vmin=-.5,vmax=len(clusters)-.5,alpha=.18,interpolation='nearest',aspect='auto')
    for index,cluster in enumerate(clusters):
        member = cluster_index == index
        axis.scatter(xy[member,0],xy[member,1],s=10,alpha=.75,color=colors[index],
                     linewidths=.15,edgecolors='white',rasterized=True,
                     label=f'{cluster} (n={member.sum()})')
    arrow_points,arrow_changes = _field_arrows(fields,low,high,max_arrows)
    lengths = np.linalg.norm(arrow_changes,axis=1)
    moving = lengths > np.finfo(float).eps*max(1.,float(np.linalg.norm(span)))
    multiplier = (float(.07*np.linalg.norm(span)/np.quantile(lengths[moving],.9))
                  if moving.any() else 1.)
    if moving.any():
        axis.quiver(arrow_points[moving,0],arrow_points[moving,1],
                    arrow_changes[moving,0],arrow_changes[moving,1],
                    angles='xy',scale_units='xy',scale=1/multiplier,color='#222222',
                    width=.003,headwidth=3.5,alpha=.8,zorder=4)
    axis.set_title(f'{projection_name} projection · fitted operator regions')
    shown = [axis]
    if labels is not None:
        reference = figure.add_subplot(layout[0,1],sharex=axis,sharey=axis)
        classes,indices = np.unique(labels.astype(str),return_inverse=True)
        scatter = reference.scatter(xy[:,0],xy[:,1],c=indices,
            cmap=plt.get_cmap('tab20' if len(classes)<=20 else 'turbo',max(len(classes),2)),
            vmin=-.5,vmax=len(classes)-.5,s=10,alpha=.65,rasterized=True)
        reference.set_title(f'{projection_name} projection · true labels ({len(classes)})')
        if len(classes)<=20:
            colorbar = figure.colorbar(scatter,ax=reference,ticks=np.arange(len(classes)),fraction=.045,pad=.02)
            colorbar.ax.set_yticklabels(classes)
        shown.append(reference)
    variance = data['explained_variance_ratio']
    name = 'Latent' if projection_name.lower()=='autoencoder' else projection_name
    for panel in shown:
        panel.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),
                  xlabel=f'{name} PC1 ({variance[0]:.1%} variance)',
                  ylabel=f'{name} PC2 ({variance[1]:.1%} variance)')
        panel.set_aspect('equal',adjustable='box')
        panel.grid(alpha=.12)
    legend = figure.add_subplot(layout[1,:]);legend.axis('off')
    legend.legend(*axis.get_legend_handles_labels(),title='Fitted clusters (sample counts)',
                  ncol=columns,loc='center',fontsize=8,frameon=False,markerscale=2)
    caption = figure.add_subplot(layout[2,:]);caption.axis('off')
    caption.text(.5,.5,f'Shading: empirical {coverage:.0%} cluster regions (not temporal decision boundaries).\n'
        'One projected affine field per cluster, fitted to all (A − I)z + b vectors.\n'
        f'Regular-grid arrows; common display multiplier ×{multiplier:.3g}.',
        ha='center',va='center',fontsize=9,transform=caption.transAxes)
    dynamics = getattr(getattr(model,'config',None),'dynamics',None)
    clock = getattr(dynamics,'clock','')
    figure.suptitle(f'{title or "Latent clustering"} · one fitted LDS step'+(f' · {clock}' if clock else ''))
    return figure
