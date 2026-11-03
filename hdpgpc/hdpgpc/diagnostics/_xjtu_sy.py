"""Acquisition-clock bearing diagnostics; full-burst statistics are display only."""
import matplotlib.pyplot as plt
import numpy as np

from ._affine_dynamics import affine_dynamics_diagnostics


def representation_label(representation):
    return {'pca':'PCA', 'autoencoder':'AE', 'tied_linear_orthonormal_autoencoder':'AE',
            'fourier':'Fourier', 'wavelet':'Haar wavelet'}[representation]


def bearing_colors(labels):
    palette = plt.get_cmap('tab20')
    order = list(range(0, 20, 2))+list(range(1, 20, 2))
    return {int(cluster): palette(order[i % 20]) for i, cluster in enumerate(np.unique(labels))}


def emission_reconstruction(values, decoder):
    """Read-only decoder reconstruction and centered-variance score on supplied rows."""
    codes = (values-decoder.offset) @ np.linalg.pinv(decoder.C).T
    reconstruction = decoder.offset+codes @ decoder.C.T
    variance = float(np.sum((values-values.mean(0))**2))
    explained = 1-float(np.sum((values-reconstruction)**2))/variance if variance else None
    return codes, reconstruction, explained


def xjtu_diagnostics(model, data, values, decoder, center, scale, paths, *, samples, seed, representation='autoencoder'):
    codes, reconstruction, explained = emission_reconstruction(values, decoder)
    errors = np.mean((values-reconstruction)**2, axis=1)
    full = reconstruction.reshape(-1, data.window_samples, 2)*scale+center
    original = data.values.reshape(-1, data.window_samples, 2)
    result = dict(reference_scope='Terminal fault inspection only; no acquisition-level health labels',
        reconstruction_scope=f'In-sample retrospective {representation}, not held-out reconstruction or causal prediction',
        reconstruction_explained_variance=explained,
        reconstruction_scaled_mse=float(errors.mean()),
        raw_window_channel_rmse=np.sqrt(np.mean((original-full)**2, axis=(0, 1))),
        first_quarter_scaled_mse=float(errors[:max(1,len(errors)//4)].mean()),
        last_quarter_scaled_mse=float(errors[-max(1,len(errors)//4):].mean()),
        clusters=affine_dynamics_diagnostics(model, paths, data.times, samples=samples, seed=seed,
            interpretation='Successive retained acquisitions; unobserved vibration gaps and random phase; rho is not a mechanical flutter or failure criterion'))
    result[f'{representation}_explained_variance']=result['reconstruction_explained_variance']
    result[f'{representation}_scaled_mse']=result['reconstruction_scaled_mse']
    return result, codes


def plot_xjtu_summary(data, labels, diagnostics, *, title, representation='autoencoder'):
    colors = bearing_colors(labels)
    figure, axes = plt.subplots(5, 1, figsize=(14, 11), constrained_layout=True,
        gridspec_kw={'height_ratios': [1.6, 1.6, .7, .8, 1.8]})
    minutes = data.times/60
    for channel, name in enumerate(('Horizontal', 'Vertical')):
        axis = axes[channel]
        axis.plot(minutes, data.acquisition_peak[:, channel], color='.75', lw=.8, label='Full-burst absolute peak')
        axis.plot(minutes, data.acquisition_rms[:, channel], color='.35', lw=.8, label='Full-burst RMS')
        for cluster, color in colors.items():
            chosen = labels == cluster
            axis.scatter(minutes[chosen], data.acquisition_rms[chosen, channel], s=12, color=color)
        axis.set(ylabel=f'{name} vibration\n(source acceleration units)', xlabel='Time from first acquisition (min)')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[2].set_facecolor('#eeeeee')
    axes[2].text(.5,.5,f'Run-to-failure; terminal inspection: {data.fault_element}. No health-state boundaries supplied.',
        transform=axes[2].transAxes, ha='center', va='center')
    axes[2].set(xticks=[],yticks=[],title='Published bearing-level reference (evaluation only)')
    for cluster, color in colors.items():
        chosen = labels == cluster
        axes[3].scatter(minutes[chosen], np.zeros(chosen.sum()), marker='|', s=300,
            color=color, label=f'{cluster} (n={chosen.sum()})')
    axes[3].set(yticks=[],xlabel='Raw-window acquisition time (min)',title='Inferred states; IDs local to this bearing')
    axes[3].legend(fontsize=8,ncol=min(len(colors),8),loc='upper left')
    for row in diagnostics['clusters']:
        cluster = row['cluster']; color = colors[cluster]
        axes[4].scatter(cluster,row['latent_spectral_radius'],marker='x',s=45,color=color)
        if 'rho_quantiles' in row:
            lo,mid,hi=row['rho_quantiles']
            axes[4].errorbar(cluster+.08,mid,yerr=[[mid-lo],[hi-mid]],fmt='o',capsize=3,color=color)
    axes[4].axhline(1,color='.4',linestyle='--')
    axes[4].set(xticks=list(colors),xlabel='Local state ID',ylabel='Acquisition-clock spectral radius',
        title='× mean A; ● variational median and 95% interval; not fast mechanical stability')
    label='linear AE' if representation=='autoencoder' else representation_label(representation)
    figure.suptitle(f'XJTU-SY {title}: {data.rpm} rpm, {data.radial_load_kn:g} kN\n'
        f'Raw window per burst → shared {label} → separate bearing fit; RMS/peak are display only')
    return figure


def plot_xjtu_embedding(codes, labels, times, *, title, representation='autoencoder'):
    xy=np.column_stack((codes[:,0],codes[:,1] if codes.shape[1]>1 else np.zeros(len(codes))))
    figure,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for cluster,color in bearing_colors(labels).items():
        chosen=labels==cluster
        axes[0].scatter(*xy[chosen].T,s=12,color=color,label=str(cluster))
    axes[0].legend(title='Local states',fontsize=8)
    points=axes[1].scatter(*xy.T,s=12,c=times/60,cmap='viridis')
    figure.colorbar(points,ax=axes[1],label='Acquisition time (min)')
    label=representation_label(representation)
    for axis in axes:
        axis.set(xlabel=f'{label} coordinate 1',ylabel=f'{label} coordinate 2 (0 if r=1)')
        axis.grid(alpha=.2)
    figure.suptitle(title+f'\nFirst two raw-window {label} coordinates; other latent dimensions omitted')
    return figure


def plot_xjtu_overview(reports):
    import csv
    from pathlib import Path
    figure, axes = plt.subplots(len(reports), 2, figsize=(14, 2.5*len(reports)),
                               squeeze=False, constrained_layout=True)
    for pair, report in zip(axes, reports):
        path=Path(report['output_directory'])
        with np.load(path/'prepared_data.npz') as saved:
            minutes=saved['window_times']/60
            rms=saved['acquisition_rms']
        with (path/'assignments.csv').open(newline='') as stream:
            labels=np.array([int(row['cluster']) for row in csv.DictReader(stream)])
        pair[0].plot(minutes,rms[:,0],color='.6',lw=.8)
        for cluster,color in bearing_colors(labels).items():
            chosen=labels==cluster
            pair[0].scatter(minutes[chosen],rms[chosen,0],s=10,color=color)
            pair[1].scatter(minutes[chosen],labels[chosen],s=10,color=color)
        info=report['metadata']
        pair[0].set(title=f"{report['dataset']}: {info['configuration']}; terminal {info['fault_element']}",
            xlabel='Acquisition time (min)',ylabel='Horizontal burst RMS\n(display only)')
        pair[1].set(title=f"Inferred K={report['clusters']}; local IDs",xlabel='Acquisition time (min)',ylabel='State')
    kinds={report['metadata']['representation'] for report in reports}
    label=representation_label(next(iter(kinds))) if len(kinds)==1 else 'mixed representation'
    figure.suptitle(f'XJTU-SY raw-window {label} comparison; separate bearing sequences\n'
        'Full-burst RMS is display metadata, not a training input; no intermediate health labels')
    return figure


def plot_xjtu_basis_calibration(batch, estimate, *, representation, sample_rate):
    """Calibration energy and one raw-window reconstruction, separately per sensor."""
    values=np.asarray(batch)
    decoder=estimate.model
    reconstruction=decoder.offset+(values-decoder.offset) @ decoder.C @ decoder.C.T
    figure,axes=plt.subplots(2,2,figsize=(12,6),constrained_layout=True)
    time=np.arange(values.shape[1]//2)/sample_rate*1000
    for channel,name in enumerate(('Horizontal','Vertical')):
        energy=estimate.candidate_energy[channel]
        chosen=estimate.selected_columns[channel]
        axes[channel,0].plot(np.arange(len(energy)),energy,color='.6',lw=.8)
        axes[channel,0].scatter(chosen,energy[chosen],s=16,color='tab:orange',label='Retained')
        axes[channel,0].set(xlabel='Native basis column',ylabel=f'{name} calibration energy',
                            title=f'{len(chosen)} / {len(energy)} coefficients')
        axes[channel,0].legend(fontsize=8)
        axes[channel,1].plot(time,values[-1,channel::2],lw=.8,label='Raw window')
        axes[channel,1].plot(time,reconstruction[-1,channel::2],lw=.8,label='Basis reconstruction')
        axes[channel,1].set(xlabel='Within-burst time (ms)',ylabel=f'{name} scaled acceleration',
                            title='Last calibration window')
        axes[channel,1].legend(fontsize=8)
        for axis in axes[channel]:axis.grid(alpha=.2)
    figure.suptitle(f'{representation_label(representation)}: {decoder.latent_dim} total coefficients; '
                   f'{estimate.explained_variance_ratio:.1%} calibration variance explained\n'
                   'Shared channel-wise basis; raw waveform likelihood; no PCA preprocessing')
    return figure


def plot_xjtu_pca_calibration(batch, estimate):
    """Calibration variance and examples; no refit or additional display PCA."""
    indices=np.asarray(estimate.beat_indices,dtype=int)
    data=batch[indices]
    scores=(data-estimate.mean) @ estimate.components
    reconstructed=estimate.mean+scores @ estimate.components.T
    total=float(np.sum(np.var(data,axis=0,ddof=1)))
    ratios=np.var(scores,axis=0,ddof=1)/total if total>0 else np.zeros(scores.shape[1])
    figure,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    axes[0].plot(np.arange(1,len(ratios)+1),np.cumsum(ratios),marker='o')
    axes[0].set(xlabel='Retained principal components',ylabel='Cumulative variance explained',
        ylim=(0,1.02),title=f'Calibration variance explained: {estimate.explained_variance_ratio:.1%}')
    for j,index in enumerate(np.unique(np.linspace(0,len(data)-1,3,dtype=int))):
        color=plt.get_cmap('tab10')(j)
        axes[1].plot(data[index],color=color,alpha=.6,label=f'Calibration row {index}')
        axes[1].plot(reconstructed[index],color=color,linestyle='--')
    axes[1].set(xlabel='Flattened raw acquisition-window coordinates',ylabel='Scaled acceleration',
        title='Observed / dashed PCA reconstruction')
    axes[1].legend(fontsize=8)
    for axis in axes:axis.grid(alpha=.2)
    figure.suptitle('Shared fixed PCA calibration; in-sample raw windows')
    return figure
