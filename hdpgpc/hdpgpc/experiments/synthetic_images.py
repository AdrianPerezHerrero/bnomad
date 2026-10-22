"""Generate image data and ground-truth artifacts, without fitting a model."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..datasets import generate_switching_lds_images


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('agents/synthetic_lds_images'))
    parser.add_argument('--image-size', type=int, nargs=2, default=(32, 32), metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--regime-order', type=int, nargs='+', default=(0, 1, 2, 0, 2, 1))
    parser.add_argument('--segment-length', type=int, default=120)
    parser.add_argument('--scenario', choices=('physical_fields', 'appearance_dynamics', 'dynamics_only'), default='physical_fields')
    parser.add_argument('--state-mode', choices=('switching', 'per_regime'), default='per_regime')
    parser.add_argument('--rho', type=float, default=.995)
    parser.add_argument('--contrast', type=float, help='Oscillator pixel SD scale: default .06 for fields, .05 for tissue, .12 for dynamics_only.')
    parser.add_argument('--appearance-strength', type=float, default=1.)
    parser.add_argument('--appearance-drift', type=float, help='Latent appearance random-walk SD: .04 for fields, .002 for tissue.')
    parser.add_argument('--observation-std', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=15, help='Animation playback only; does not change LDS time steps')
    parser.add_argument('--gif', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--frames', action=argparse.BooleanOptionalAction, default=True)
    return parser


def run_synthetic_images(args):
    """Save one deterministic run in a separate mode/seed directory."""
    if not 1 <= args.fps <= 100:
        raise ValueError('fps must be between 1 and 100')
    data = generate_switching_lds_images(
        image_size=tuple(args.image_size), regime_order=tuple(args.regime_order),
        segment_lengths=args.segment_length, rho=args.rho, contrast=args.contrast,
        observation_std=args.observation_std, state_mode=args.state_mode, seed=args.seed,
        scenario=args.scenario, appearance_strength=args.appearance_strength,
        appearance_drift=args.appearance_drift)
    # Include every data/presentation option, so different settings never share
    # a folder. Refuse overwrites, including partially generated outputs.
    import hashlib
    settings = {key: value for key, value in vars(args).items() if key != 'output_dir'}
    run_id = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output_dir / f'{data.scenario}_{args.state_mode}_seed_{args.seed}_{run_id}'
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / 'dataset.npz', images=data.images, clean_images=data.clean_images,
        values=data.values, latent_states=data.states, labels=data.labels,
        segment_ids=data.segment_ids, previous_indices=data.previous_indices,
        A=data.A, Gamma=data.Gamma, C=data.C, Sigma_diagonal=data.Sigma_diagonal,
        offset=data.offset, initial_covariance=data.initial_covariance, grid=data.grid,
        regime_names=np.asarray(data.regime_names), state_mode=data.state_mode, seed=data.seed,
        scenario=data.scenario, initial_state_means=data.initial_state_means,
        initial_state_covariances=data.initial_state_covariances,
        regime_mean_images=data.regime_mean_images)
    with (output / 'labels.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('frame', 'segment', 'regime', 'regime_name', 'previous_state_frame'))
        writer.writerows((t, int(data.segment_ids[t]), int(k), data.regime_names[k],
                         int(data.previous_indices[t])) for t, k in enumerate(data.labels))
    low, high = min(0., float(data.images.min())), max(1., float(data.images.max()))
    metadata = dict(
        settings=settings, frames=len(data.images), image_shape=list(data.images.shape[1:]),
        latent_dimension=data.C.shape[1], regime_names=data.regime_names, scenario=data.scenario,
        stationary_covariance=('I for all regimes' if data.scenario == 'dynamics_only' else
            'Appearance coordinates are random walks; oscillators have regime-dependent activity covariance.'),
        process_equation=('z[t] = A[s[t]] z[t-1] + e[t]' if data.state_mode == 'switching' else
                          'z[t] = A[s[t]] z[previous occurrence of s[t]] + e[t]; first occurrence N(initial_state_means[k], initial_state_covariances[k])'),
        observation_equation='y[t] = offset + C z[t] + eta[t]',
        process_covariance_diagonals=np.diagonal(data.Gamma, axis1=1, axis2=2).tolist(),
        initial_state_means=data.initial_state_means.tolist(),
        model_initialization='Common pooled covariance; true regime-specific initial distributions are stored separately.',
        observation_variance=float(data.Sigma_diagonal[0]),
        labels='Scheduled regime IDs; no stochastic HMM label generation',
        preview_intensity_range=[low, high],
        preview_note='PNG/GIF are globally rescaled uint8 previews. Fit dataset.npz values, which are unclipped float64.',
        fitting_note='No HDP-GPC fit performed. Standard SLDS switches differ from per-cluster resumed state histories.')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..diagnostics import plot_switching_lds_images
    for name, figure in plot_switching_lds_images(data).items():
        figure.savefig(output / f'{name}.png', dpi=140)
        plt.close(figure)
    if args.frames or args.gif:
        from PIL import Image, ImageDraw
        previews = np.rint((data.images - low) * (255 / (high-low))).astype(np.uint8)
        if args.frames:
            frame_dir = output / 'frames'
            frame_dir.mkdir()
            for t, frame in enumerate(previews):
                Image.fromarray(frame).save(frame_dir / f'frame_{t:05d}.png')
        if args.gif:
            animation = []
            for t, frame in enumerate(previews):
                image = Image.fromarray(frame).convert('RGB').resize((192, 192), Image.Resampling.NEAREST)
                canvas = Image.new('RGB', (192, 220), 'white')
                canvas.paste(image, (0, 28))
                ImageDraw.Draw(canvas).text((5, 8), f'{t:04d} | {data.regime_names[data.labels[t]]}', fill='black')
                animation.append(canvas)
            animation[0].save(output / 'sequence.gif', save_all=True, append_images=animation[1:],
                              duration=round(1000/args.fps), loop=0)
    print(f'Generated {len(data.images)} ordered {data.images.shape[1]}x{data.images.shape[2]} images; '
          f'{data.C.shape[1]} latent dimensions; scenario={data.scenario}; state mode={data.state_mode}')
    print(f'Gamma diagonals={np.diagonal(data.Gamma, axis1=1, axis2=2).tolist()}; '
          f'Sigma diagonal={data.Sigma_diagonal[0]:.6g}')
    print(f'Output: {output.resolve()}')
    return output


def main():
    run_synthetic_images(build_parser().parse_args())
