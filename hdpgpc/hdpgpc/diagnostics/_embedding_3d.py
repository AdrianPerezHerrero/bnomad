"""Interactive views of a common latent coordinate system and its dynamics."""
from numbers import Integral

from matplotlib.colors import to_hex
import numpy as np
import plotly.graph_objects as go

from ._embedding_dynamics import _cluster_colors, _field_arrows, _operator_fields
from ._image_trajectories import _times


def _region_surface(field):
    azimuth, polar = np.meshgrid(np.linspace(0, 2*np.pi, 19), np.linspace(0, np.pi, 11))
    sphere = np.stack((np.cos(azimuth)*np.sin(polar), np.sin(azimuth)*np.sin(polar),
                       np.cos(polar)), axis=-1)
    precision, basis = np.linalg.eigh(field['region_precision'])
    transform = basis*np.sqrt(field['region_radius_squared']/precision)
    return sphere @ transform.T+field['center']


def _arrow_segments(origins, changes):
    """Build shafts and arrowheads without independently scaling their lengths."""
    direction = changes/np.linalg.norm(changes, axis=1, keepdims=True)
    perpendicular = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction), axis=1)])
    perpendicular /= np.linalg.norm(perpendicular, axis=1, keepdims=True)
    tips = origins+changes
    base = tips-.26*changes
    wings = .11*np.linalg.norm(changes, axis=1, keepdims=True)*perpendicular
    gap = np.full_like(origins, np.nan)
    return np.stack((origins, tips, gap, base+wings, tips, base-wings, gap), axis=1).reshape(-1, 3)


def plot_embedding_dynamics_3d(data, *, trajectories=None, times=None, labels=None,
                               title=None, clock=None, max_arrows=230, coverage=.9):
    """Return a Plotly figure from a three-component ``model_embedding_dynamics`` view.

    Each empirical ellipsoid and operator uses all assigned observations. The
    paths use the saved smoothed means on each cluster's own clock, projected
    with the same center and PCA basis as the encoded observation markers.
    """
    points = data['coordinates']
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('A display projection with exactly three coordinates is required')
    if isinstance(max_arrows, bool) or not isinstance(max_arrows, Integral) or max_arrows < 1:
        raise ValueError('max_arrows must be a positive integer')
    clock_label = 'Sample' if times is None else 'Observation time / index'
    times = _times(times, len(points))
    labels = None if labels is None else np.asarray(labels).reshape(-1)
    if labels is not None and len(labels) != len(points):
        raise ValueError('Reference labels must align with encoded samples')
    fields = _operator_fields(data, coverage)
    assignments = data['assignments']
    low, high = points.min(0), points.max(0)
    span = np.maximum(high-low, max(float(np.ptp(points, axis=0).max()), 1.)*.05)
    middle = (low+high)/2
    origins, motion, owners = _field_arrows(fields, middle-.58*span, middle+.58*span,
                                            max_arrows, return_owners=True)
    lengths = np.linalg.norm(motion, axis=1)
    span_norm = float(np.linalg.norm(span))
    moving = lengths > np.finfo(float).eps*max(1., span_norm)
    multiplier = float(.07*span_norm/np.quantile(lengths[moving], .9)) if moving.any() else 1.

    figure = go.Figure()
    layers = []
    for index, (cluster, color) in enumerate(zip(fields, _cluster_colors(len(fields)))):
        color = to_hex(color)
        field = fields[cluster]
        group = f'Cluster {cluster}'
        member = assignments == cluster
        x, y, z = points[member].T
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', legendgroup=group,
            name=f'{group} ({field["members"]})', marker=dict(color=color, size=3.5, opacity=.78),
            customdata=times[member], text=None if labels is None else labels[member].astype(str),
            hovertemplate=(f'{group}<br>{clock_label} %{{customdata:g}}'
                           + ('<br>Reference %{text}' if labels is not None else '')+'<extra></extra>')))
        layers.append('observations')

        x, y, z = np.moveaxis(_region_surface(field), -1, 0)
        figure.add_trace(go.Surface(x=x, y=y, z=z, legendgroup=group, name=f'{group} region',
            opacity=.13, colorscale=[(0, color), (1, color)], showscale=False,
            showlegend=False, hoverinfo='skip'))
        layers.append('regions')

        path = None if trajectories is None else trajectories.get(cluster)
        if path is not None:
            rows = np.asarray(path['indices'], dtype=int)
            means = np.asarray(path['means'])
            if (means.shape != (len(rows), len(data['projection_center']))
                    or (rows < 0).any() or (rows >= len(times)).any()):
                raise ValueError(f'Cluster {cluster} trajectory must align with the native latent space and clock')
            x, y, z = ((means-data['projection_center']) @ data['projection_components'].T).T
            status = np.where(path.get('observed', np.ones(len(rows), dtype=bool)),
                              'Retained observation', 'Unobserved tick')
            figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers',
                legendgroup=group, name=f'{group} smoothed path', showlegend=False,
                marker=dict(color=color, size=2), line=dict(color=color, width=5),
                customdata=times[rows], text=status,
                hovertemplate=(f'{group} smoothed mean<br>{clock_label} %{{customdata:g}}'
                               '<br>%{text}<extra></extra>')))
            layers.append('trajectories')

        arrow = (owners == index) & moving
        if arrow.any():
            x, y, z = _arrow_segments(origins[arrow], multiplier*motion[arrow]).T
            figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', legendgroup=group,
                name=f'{group} one-step field', line=dict(color=color, width=4),
                showlegend=False, hoverinfo='skip'))
            layers.append('arrows')

    selections = [('All layers', set(layers)), ('Observations', {'observations'}),
                  ('Trajectories', {'observations', 'trajectories'}),
                  ('Operator regions and arrows', {'observations', 'regions', 'arrows'})]
    buttons = [dict(label=name, method='restyle',
                    args=[{'visible': [layer in visible for layer in layers]}])
               for name, visible in selections if name != 'Trajectories' or 'trajectories' in layers]
    axes = {f'{axis}axis_title': f'Latent PC{index} ({variance:.1%} variance)'
            for index, (axis, variance) in enumerate(zip('xyz', data['explained_variance_ratio']), 1)}
    figure.update_layout(template='plotly_white',
        title=dict(text=(f'{title or "Latent clustering"} · latent trajectories and dynamics'
                         + (f' · {clock}' if clock else '')), x=.02, y=.97),
        scene=dict(aspectmode='data', **axes), legend=dict(groupclick='togglegroup'),
        margin=dict(l=8, r=8, t=110, b=52), height=780,
        updatemenus=[dict(type='dropdown', x=0, xanchor='left', y=1.06, buttons=buttons)],
        annotations=[dict(x=0, xref='paper', y=-.03, yref='paper', showarrow=False,
            xanchor='left', yanchor='top', align='left',
            text=(f'Ellipsoids show empirical {coverage:.0%} cluster regions. '
                  f'Arrows show projected (A − I)z + b per model step, scaled ×{multiplier:.3g}.'))])
    return figure
