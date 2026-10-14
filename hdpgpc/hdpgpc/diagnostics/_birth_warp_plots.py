"""Visualize the warped curves and score decomposition of a birth proposal."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_birth_warp_diagnostic(record, *, title=None):
    """Plot an optional offline birth snapshot captured by ``HDPGPC.fit``.

    Each row shows one observation: the birth seed or a series with a large
    negative emission-score change. ``aligned`` is the raw series resampled
    onto the common coordinate grid through the recorded input warp. The
    inference score itself uses the raw series on that warped input grid.
    """
    rows = record.get('rows', ())
    if not rows:
        raise ValueError(record.get('diagnostic_error', 'No birth warp curves were captured'))
    figure, axes = plt.subplots(
        len(rows), 3, figsize=(17, max(3.5 * len(rows), 4)),
        squeeze=False, constrained_layout=True,
        gridspec_kw={'width_ratios': (1.35, 1, 1)},
    )
    colors = {'baseline': '#27844c', 'parent': '#2563a6',
              'newborn': '#d17a1e'}
    for row_number, row in enumerate(rows):
        waveform, warp_axis, score_axis = axes[row_number]
        grid = np.asarray(row['grid'])
        raw = np.asarray(row['raw'])
        waveform.plot(grid, raw, color='black', lw=1.6, label='Raw observation')
        warp_axis.plot(grid, grid, color='0.4', lw=1, ls=':', label='Identity map')
        score_values = []
        for label in ('baseline', 'parent', 'newborn'):
            if label not in row:
                continue
            item = row[label]
            color = colors[label]
            waveform.plot(grid, item['aligned'], color=color, lw=1.4,
                          label=f'{label.capitalize()} aligned')
            waveform.plot(grid, item['scoring_mean'], color=color, lw=1.2,
                          ls='--', label=f'{label.capitalize()} scoring mean')
            if (label == 'newborn' and not np.allclose(
                    item['warp_reference_mean'], item['scoring_mean'])):
                waveform.plot(grid, item['warp_reference_mean'], color='0.55',
                              lw=1, ls='-.',
                              label=f'Newborn warp reference ({item.get("warp_reference_label", "cluster " + str(item["warp_reference_cluster"]))})')
            warp_axis.plot(grid, item['mapped_grid'], color=color, lw=1.5,
                           label=f'{label.capitalize()} map')
            score_values.append((label, item))
        waveform.set_title(f'Series {row["index"] + 1}' + (
            ' (birth seed)' if row['index'] == record['seed'] else ' (large emission loss)'))
        waveform.set_xlabel('Common time grid')
        waveform.set_ylabel('Normalized value')
        waveform.grid(alpha=0.15)
        warp_axis.set_title('Input warp: observed time to model time')
        warp_axis.set_xlabel('Observed time')
        warp_axis.set_ylabel('Mapped model time')
        warp_axis.grid(alpha=0.15)
        score_axis.axis('off')
        cells = [
            [f'{item[key]:,.1f}' for _, item in score_values]
            for key in ('data_score', 'warp_penalty', 'total_score')
        ]
        table = score_axis.table(
            cellText=cells,
            rowLabels=('GP data log density', 'Warp penalty', 'Net score'),
            colLabels=[name.capitalize() for name, _ in score_values],
            cellLoc='right', loc='center', bbox=(0.31, 0.28, 0.67, 0.55),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for column, (name, _) in enumerate(score_values):
            table[(0, column)].set_facecolor(colors[name])
            table[(0, column)].get_text().set_color('white')
        score_axis.set_title('Emission score decomposition (log units)')
        if row_number == 0:
            waveform.legend(fontsize=7, loc='best')
            warp_axis.legend(fontsize=8, loc='best')
    def change(first, second):
        if first not in record or second not in record:
            return 'n/a'
        return f'{record[second] - record[first]:+,.1f}'

    decision = 'accepted' if record.get('accepted') else 'rejected'
    suffix = (f'  |  assigned newborn={record["newborn_occupancy"]:g}'
              f'  |  decision={decision}'
              f'  |  Δ birth score={change("decision_baseline_score", "decision_proposal_score")}'
              f'  |  Δ emission={change("baseline_emission", "proposal_emission")}')
    if 'baseline_warp_penalty' in record:
        suffix += (f'\nΔ GP data={change("baseline_gp_data", "proposal_gp_data")}'
                   f'  |  Δ warp penalty={change("baseline_warp_penalty", "proposal_warp_penalty")}')
    figure.suptitle(
        f'{title or "Birth warp diagnostic"}  |  iteration {record["iteration"]}, '
        f'seed {record["seed"] + 1}, parent {record["parent_cluster"]}'
        f'\n{record["phase"]}{suffix}', fontsize=11)
    return figure
