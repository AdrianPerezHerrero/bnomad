"""Cluster complete bundled MIT-BIH records and compare clusters with annotations.

Run from the repository root after installing the package::

    python examples/full_record_clustering.py
    python examples/full_record_clustering.py --records 101 102
    python examples/full_record_clustering.py --records 100 --save-plots
    python examples/full_record_clustering.py --records 100 --no-scale-waveforms --unit-grid
    python examples/full_record_clustering.py --records 102 --calibrate-lds-noise-priors
    python examples/full_record_clustering.py --records 100 --no-fit-gp

Edit the experiment controls below to choose records, channels, inference schedule,
alignment policy, warping, and model hyperparameters. If there is one extra trailing annotation,
the script crops the annotation vector to the number of fitted observations.
Console output is also saved to one log file per record in the plot directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tutorial_utils import plot_record_results
from hdpgpc import HDPGPC
from hdpgpc.dynamics import estimate_lds_noise_priors, estimate_pca_observation
from hdpgpc.diagnostics import (
    Tee, cluster_label_statistics, format_confusion_matrix, format_lds_priors,
    model_lds_arrays, model_warp_arrays, plot_lds_diagnostics, plot_warp_diagnostics,
)
from hdpgpc.preprocessing import prepare_sequences
from hdpgpc.config import (
    AlignmentConfig,
    DynamicsConfig,
    GPEmissionConfig,
    ObservationConfig,
    HDPConfig,
    HDPGPCConfig,
    InferenceConfig,
    MultiOutputConfig,
    NumericalConfig,
    MatrixPolicyConfig,
    NoisePolicyConfig,
    with_lds_preset,
    with_pca_emission,
)


# -----------------------------------------------------------------------------
# Experiment controls: edit these values for the desired whole-record run.
# -----------------------------------------------------------------------------
# Full run: use this tuple; for a single record, choose the alternative below.
RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
)
# RECORDS = ("102",)            # Current record selection.
CHANNELS = (0,)                 # Choose channel 0, 1, or both: (0, 1).
SCALE_WAVEFORMS = False          # First-beat mean/std scaling; False keeps stored amplitudes.
UNIT_GRID = False               # True: coordinates 0..1; False: point indices.
INFERENCE_MODE = "offline"      # "offline", "online", or "online_fast".
LDS_INDUCING_POINTS = None        # Reduced LDS basis size; 0 or None uses every waveform point.
# Set a component count for a fixed PCA emission calibrated before fitting.
# It overrides LDS_PRESET and all manual LDS matrix modes. Online calibration
# pretrains on a record prefix, then replays that prefix during clustering;
# use a separate calibration record for a strictly causal evaluation.
PCA_COMPONENTS = None
PCA_CALIBRATION_MAX_BEATS = 2000
PCA_RESIDUAL_NOISE_FLOOR = 1.0e-3
# PCA keeps C fixed. Sigma uses the existing noise modes: fixed, cluster,
# cluster_diagonal, shared, or shared_diagonal. The calibrated residual
# covariance is the initial value/prior mean in every mode.
PCA_SIGMA_MODE = "cluster"
HARD_ASSIGNMENTS = True         # False uses soft responsibility weights during inference.
ENABLE_WARP = True              # Whether to request alignment during fitting.
ALIGNMENT_POLICY = "dtw_gp"    # "gp", (gp-based alignment), "dtw_gp" (cheap point warp), or "gp_flow".
VERBOSE = "progress"           # "silent", "progress", or "debug"; False/True remain aliases.
SHARE_GP_FOR_PROPOSALS = True   # Proposals/new clusters may reuse the fitted or fixed GP kernel.
SAVE_PLOTS = True               # Also save the final figures when True.
PLOT_WARPS = True               # Plot fitted warp maps, record heatmap, and cluster summaries.
PLOT_LDS = True                 # Plot fitted LDS matrices, spectra, and stability measures.
PLOT_DIRECTORY = Path(__file__).resolve().parents[1] / "agents" / "plots_presets"
PLOT_DPI = 160

# Typed configuration values; these are experiment settings, not package defaults.
# Lengthscale is measured in the selected grid units; noise settings depend on waveform scale.
MAX_CLUSTERS = 30
HDP_PRESET = "more"            # "less", "balanced" or "more" (see HDPConfig preset descriptions).
MODEL_TYPE = "dynamic"         # "static" or "dynamic" (see DynamicsConfig model_type descriptions).
LDS_DEGREES_OF_FREEDOM = 10  # None uses the package default: 5 offline, 20 online.
LDS_UPDATE = "conjugate_batch"  # "conjugate_sequential", "conjugate_batch", "natural_gradient", or "legacy".
LDS_SCORING = "final"          # "final" uses last LDS matrices; "historical" uses per-beat matrices.
# Set None to use the four manual modes below. Named bundles are documented in
# docs/model_choices.md; "only_plastic" retains this example's prior behavior.
LDS_PRESET = "stable"
# Independent LDS policies; all four choices are explicit in this script.
# Matrix: cluster, shrinkage, shared, fixed. Noise: cluster, cluster_diagonal,
# shared, shared_diagonal, fixed. Matched learned scopes are required within
# each pair (A/Gamma and C/Sigma); fixed may accompany either learned scope.
LDS_A_MODE = "cluster"
LDS_GAMMA_MODE = "cluster"
LDS_C_MODE = "fixed"
LDS_SIGMA_MODE = "fixed"
LDS_SHRINK_STRENGTH = 10.0
LDS_A_SHRINKAGE = 10.0       # Used only if LDS_A_MODE == 'shrinkage'.
LDS_C_SHRINKAGE = 10.0       # Used only if LDS_C_MODE == 'shrinkage'.
CUSTOM_A = None              # Square LDS-grid matrix for a fixed/custom prior map.
CUSTOM_GAMMA = None          # Positive-definite LDS-grid covariance.
CUSTOM_C = None              # Square LDS-grid matrix; None selects identity.
CUSTOM_SIGMA = None          # Positive-definite LDS-grid covariance; None uses the initial noise.
# Model-owned shared/fixed observation parameters use final scoring.
LDS_STEP_SIZE = 0.05            # Fraction of new LDS evidence in natural_gradient mode.
BIRTH_LDS_KL_PER_SQRT_POINT = True  # Historical flag name; LDS KL / LDS dimension in both schedules.
INITIAL_LENGTHSCALE = 1.0
LENGTHSCALE_BOUNDS = (0.1, 10.0)
FIT_GP = True                  # False keeps the supplied GP kernel fixed for every cluster.
GP_OUTPUTSCALE_INITIAL = 1.0
GP_OUTPUTSCALE_BOUNDS = (0.001, 1.0e+4)
GP_WHITE_NOISE_INITIAL = 1.0
GP_WHITE_NOISE_BOUNDS = (0.0001, 1.0e+8)  # Lower and upper bounds for first-beat GP fitting.
CUSTOM_GP_KERNEL = None         # Set to a sklearn kernel; with FIT_GP=False it is never optimized.
INITIAL_OBSERVATION_NOISE = 5.0e+1
OBSERVATION_NOISE_BOUNDS = (0.001, 1.0e+8)  # LDS Sigma, separate from GP white noise.
INITIAL_PROCESS_NOISE = 1.0e+1
PROCESS_NOISE_BOUNDS = (0.001, 1.0e+8)
# Optional, independent identity-LDS pilot calibration before model.fit.
# It uses random contiguous blocks from this record, so enable it only offline.
CALIBRATE_LDS_NOISE_PRIORS = False
LDS_CALIBRATION_MAX_BEATS = 20
LDS_CALIBRATION_SEGMENT_LENGTH = 20

# Inference hyperparameters
OFFLINE_ITERATIONS = 50
RANDOM_SEED = 42

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "hdpgpc" / "data" / "mitbih"


def _configuration(
    grid: np.ndarray, *, lds_degrees_of_freedom: int | None = LDS_DEGREES_OF_FREEDOM,
    lds_update: str = LDS_UPDATE, lds_scoring: str = LDS_SCORING,
    lds_step_size: float = LDS_STEP_SIZE,
    lds_inducing_points: int | None = LDS_INDUCING_POINTS,
    fit_gp: bool = FIT_GP,
    birth_lds_kl_per_sqrt_point: bool = BIRTH_LDS_KL_PER_SQRT_POINT,
    initial_observation_noise: float = INITIAL_OBSERVATION_NOISE,
    initial_process_noise: float = INITIAL_PROCESS_NOISE,
    pca_estimate=None,
) -> HDPGPCConfig:
    """Build the full experiment recipe in one visible place."""
    if ENABLE_WARP and INFERENCE_MODE != "offline" and ALIGNMENT_POLICY not in ("gp_flow", "dtw_gp"):
        raise ValueError("Online warping in this example requires 'gp_flow' or 'dtw_gp'.")
    if len(CHANNELS) > 1 and INFERENCE_MODE in ("online", "online_fast"):
        raise ValueError("The characterized online multi-output path is unsupported; select one channel.")
    pca_selected = pca_estimate is not None
    if pca_selected or lds_inducing_points is None or lds_inducing_points == 0:
        basis = grid
        sparse_lds = False
    else:
        if (isinstance(lds_inducing_points, bool)
                or not isinstance(lds_inducing_points, int)
                or not 1 <= lds_inducing_points <= len(grid)):
            raise ValueError("lds_inducing_points must be between 1 and the waveform point count, or 0/None")
        sparse_lds = lds_inducing_points < len(grid)
        basis = (grid[np.linspace(0, len(grid) - 1, lds_inducing_points, dtype=int)].copy()
                 if sparse_lds else grid)

    gp_kernel = (CUSTOM_GP_KERNEL if CUSTOM_GP_KERNEL is not None else
                 ConstantKernel(GP_OUTPUTSCALE_INITIAL, GP_OUTPUTSCALE_BOUNDS)
                 * RBF(INITIAL_LENGTHSCALE, LENGTHSCALE_BOUNDS)
                 + WhiteKernel(GP_WHITE_NOISE_INITIAL, GP_WHITE_NOISE_BOUNDS))
    if (LDS_C_MODE == "shared"
            or LDS_SIGMA_MODE in ("shared", "shared_diagonal")
            or LDS_A_MODE == "shared"
            or LDS_GAMMA_MODE in ("shared", "shared_diagonal")):
        lds_scoring = "final"

    if pca_selected and (lds_update == 'legacy' or
                         (lds_update == 'conjugate_batch' and INFERENCE_MODE != 'offline')):
        lds_update = 'conjugate_sequential'
    manual_model_type = "dynamic" if (LDS_PRESET is not None or pca_selected) else MODEL_TYPE
    config = HDPGPCConfig(
        x_basis=basis,
        verbose=VERBOSE,
        hdp=HDPConfig(hdp_hyp=HDP_PRESET, max_models=MAX_CLUSTERS),
        emission=GPEmissionConfig(
            kernels=[gp_kernel.clone_with_theta(gp_kernel.theta)],
            fit_gp=fit_gp,
            ini_lengthscale=INITIAL_LENGTHSCALE,
            bound_lengthscale=LENGTHSCALE_BOUNDS,
            ini_sigma=initial_observation_noise,
            bound_sigma=OBSERVATION_NOISE_BOUNDS,
            ini_outputscale=GP_OUTPUTSCALE_INITIAL,
            inducing_points=sparse_lds,
        ),
        observation=ObservationConfig(
            C_policy=None if pca_selected else MatrixPolicyConfig(
                mode=LDS_C_MODE, value=CUSTOM_C,
                strength=LDS_C_SHRINKAGE if LDS_C_MODE == 'shrinkage' else 1.0),
            Sigma_policy=None if pca_selected else NoisePolicyConfig(
                mode=LDS_SIGMA_MODE, value=CUSTOM_SIGMA)),
        dynamics=DynamicsConfig(
            model_type=manual_model_type,
            ini_gamma=initial_process_noise,
            A_policy=(None if manual_model_type == "static" or pca_selected else MatrixPolicyConfig(
                mode=LDS_A_MODE, value=CUSTOM_A,
                strength=LDS_A_SHRINKAGE if LDS_A_MODE == 'shrinkage' else 1.0)),
            Gamma_policy=(None if manual_model_type == "static" or pca_selected else NoisePolicyConfig(
                mode=LDS_GAMMA_MODE, value=CUSTOM_GAMMA)),
            bound_gamma=PROCESS_NOISE_BOUNDS,
            annealing=False,
            bayesian_params=True,
            free_deg_MNIV=lds_degrees_of_freedom,
            lds_update=lds_update,
            lds_scoring=lds_scoring,
            lds_step_size=lds_step_size,
        ),
        alignment=AlignmentConfig(
            policy=ALIGNMENT_POLICY if ENABLE_WARP else "none",
            x_basis_warp=grid if ENABLE_WARP else None,
            mode_warp="rough",
        ),
        inference=InferenceConfig(
            schedule=INFERENCE_MODE,
            hard_assignments=HARD_ASSIGNMENTS,
            birth_lds_kl_per_sqrt_point=birth_lds_kl_per_sqrt_point,
            warp=ENABLE_WARP,
            it_limit=OFFLINE_ITERATIONS if INFERENCE_MODE == "offline" else None,
            n_explore_steps=10,
        ),
        multi_output=MultiOutputConfig(
            n_outputs=len(CHANNELS),
            share_gp=SHARE_GP_FOR_PROPOSALS,
            use_snr=False,
        ),
        numerics=NumericalConfig(
            seed=RANDOM_SEED,
            num_threads=1,
            dtype="float64",
            deterministic=True,
        ),
    )
    if pca_estimate is not None:
        return with_pca_emission(
            config, pca_estimate,
            sigma_policy=NoisePolicyConfig(mode=PCA_SIGMA_MODE))
    return (with_lds_preset(config, LDS_PRESET,
                            shrink_strength=LDS_SHRINK_STRENGTH)
            if LDS_PRESET is not None else config)


def _load_record(record: str) -> tuple[np.ndarray, np.ndarray]:
    signal_path = DATA_DIRECTORY / f"{record}.npy"
    label_path = DATA_DIRECTORY / f"{record}_labels.npy"
    if not signal_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(
            f"Record {record} needs both {signal_path.name} and {label_path.name} in {DATA_DIRECTORY}"
        )
    signal = np.load(signal_path, allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False).reshape(-1)
    if signal.ndim != 3 or signal.shape[-1] < 1:
        raise ValueError(f"Expected {signal_path.name} shaped (beats, points, channels), got {signal.shape}")
    if min(CHANNELS) < 0 or max(CHANNELS) >= signal.shape[-1]:
        raise ValueError(f"Selected channels {CHANNELS} are invalid for {signal_path.name} {signal.shape}")
    return signal, labels



def run_record(record: str, *, save_plots: bool = SAVE_PLOTS,
               plot_warps: bool = PLOT_WARPS,
               plot_lds: bool = PLOT_LDS,
               fit_gp: bool = FIT_GP,
               lds_degrees_of_freedom: int | None = LDS_DEGREES_OF_FREEDOM,
               lds_update: str = LDS_UPDATE,
               lds_scoring: str = LDS_SCORING,
               lds_step_size: float = LDS_STEP_SIZE,
               lds_inducing_points: int | None = LDS_INDUCING_POINTS,
               birth_lds_kl_per_sqrt_point: bool = BIRTH_LDS_KL_PER_SQRT_POINT,
               plot_directory: Path = PLOT_DIRECTORY,
               scale_waveforms: bool = SCALE_WAVEFORMS,
               unit_grid: bool = UNIT_GRID,
               calibrate_lds_noise_priors: bool = CALIBRATE_LDS_NOISE_PRIORS) -> None:
    from hdpgpc.config._verbosity import normalize_verbosity
    verbosity = normalize_verbosity(VERBOSE)

    def detail_print(*args, **kwargs):
        if verbosity == 'debug':
            print(*args, **kwargs)

    signal, annotations = _load_record(record)
    values, grid = prepare_sequences(
        signal, channels=CHANNELS, scale_from_first=scale_waveforms,
        unit_grid=unit_grid,
    )
    initial_observation_noise = INITIAL_OBSERVATION_NOISE
    initial_process_noise = INITIAL_PROCESS_NOISE
    pca_estimate = None
    if PCA_COMPONENTS is not None:
        if calibrate_lds_noise_priors:
            raise ValueError('PCA uses its calibrated residual noise; disable LDS noise calibration')
        if ENABLE_WARP or len(CHANNELS) != 1:
            raise ValueError('PCA currently requires warp=False and one selected channel')
        pca_estimate = estimate_pca_observation(
            values, grid, components=PCA_COMPONENTS,
            max_beats=PCA_CALIBRATION_MAX_BEATS, random_state=RANDOM_SEED,
            sampling='uniform' if INFERENCE_MODE == 'offline' else 'prefix',
            noise_floor=PCA_RESIDUAL_NOISE_FLOOR,
        )
        detail_print(f'PCA calibration: {len(pca_estimate.beat_indices)} beats, '
              f'{PCA_COMPONENTS} components, explained variance '
              f'{pca_estimate.explained_variance_ratio:.1%}, '
              f'mean residual variance={np.diag(pca_estimate.residual_covariance).mean():.6g}')
    if calibrate_lds_noise_priors:
        effective_model_type = "static" if LDS_PRESET == "static_mixture" else (
            "dynamic" if LDS_PRESET is not None else MODEL_TYPE)
        if INFERENCE_MODE != "offline" or effective_model_type != "dynamic":
            raise ValueError(
                "Record-based LDS noise calibration requires offline dynamic inference; "
                "for online inference calibrate on a separate record before fitting"
            )
        estimate = estimate_lds_noise_priors(
            values, max_beats=LDS_CALIBRATION_MAX_BEATS,
            segment_length=LDS_CALIBRATION_SEGMENT_LENGTH,
            random_state=RANDOM_SEED,
            require_positive_components=False,
        )
        shared_noise_prior = 0.5 * (estimate.observation + estimate.process)
        if not np.isfinite(shared_noise_prior) or shared_noise_prior <= 0:
            raise ValueError(
                "Pooled LDS noise prior must be positive and finite; "
                f"estimated Sigma={estimate.observation:.6g}, "
                f"Gamma={estimate.process:.6g}"
            )
        initial_observation_noise = shared_noise_prior
        initial_process_noise = shared_noise_prior
        detail_print(
            "Identity-LDS pilot noise scales: "
            f"estimated Sigma={estimate.observation:.6g}, "
            f"estimated Gamma={estimate.process:.6g}, "
            f"shared Sigma/Gamma prior={shared_noise_prior:.6g}; "
            f"{len(estimate.segment_starts)} blocks of {estimate.segment_length} beats, "
            f"negative lag fraction={estimate.negative_lag_fraction:.2f}"
        )
    config = _configuration(
        grid, lds_degrees_of_freedom=lds_degrees_of_freedom,
        lds_update=lds_update, lds_scoring=lds_scoring,
        lds_step_size=lds_step_size,
        lds_inducing_points=lds_inducing_points,
        fit_gp=fit_gp,
        birth_lds_kl_per_sqrt_point=birth_lds_kl_per_sqrt_point,
        initial_observation_noise=initial_observation_noise,
        initial_process_noise=initial_process_noise,
        pca_estimate=pca_estimate,
    )

    if verbosity != 'silent':
        print(f"Record {record}: {len(values)} samples; schedule={INFERENCE_MODE}")
    detail_print(f"Waveform shape={signal.shape}, annotations={len(annotations)}")
    detail_print(f"Input shape={values.shape} (beats, points, outputs), channels={CHANNELS}")
    detail_print(f"LDS basis points={len(config.x_basis)} of {len(grid)}, "
          f"inducing mode={config.emission.inducing_points}")
    detail_print(f"Waveform scaling={'first-beat mean/std' if scale_waveforms else 'off'}, "
          f"grid range=0..{grid[-1, 0]:g}")
    detail_print(f"Schedule={INFERENCE_MODE}, hard assignments={HARD_ASSIGNMENTS}, warp={ENABLE_WARP}, "
          f"alignment={ALIGNMENT_POLICY if ENABLE_WARP else 'none'}, verbose={VERBOSE}")
    detail_print(f"Model={config.dynamics.model_type}, Markov={config.hdp.hmm_switch}, "
          f"LDS preset={'PCA' if pca_estimate is not None else (LDS_PRESET or 'manual')}, "
          f"HDP preset={HDP_PRESET}, "
          f"cluster cap={MAX_CLUSTERS}")
    from hdpgpc.config._models import resolved_lds_policies
    selected = resolved_lds_policies(config)
    detail_print("LDS policies=" + ", ".join(
        f"{name}:{selected[name].mode}" for name in ('A', 'Gamma', 'C', 'Sigma')))
    detail_print("LDS prior strengths=" + ", ".join(
        f"{name}:{selected[name].strength:g}"
        for name in ('A', 'Gamma', 'C', 'Sigma')))
    detail_print(f"Custom C={selected['C'].value is not None}, "
          f"custom Sigma={selected['Sigma'].value is not None}")
    detail_print(f"GP fitting={'on' if config.emission.fit_gp else 'off'}, "
          f"kernel={config.emission.kernels[0]}")
    gp_white_bounds = config.emission.kernels[0].get_params().get("k2__noise_level_bounds")
    if gp_white_bounds is not None:
        detail_print(f"GP white-noise bounds={gp_white_bounds}")
    effective_dof = (lds_degrees_of_freedom if lds_degrees_of_freedom is not None
                     else (5 if INFERENCE_MODE == "offline" else 20))
    detail_print(f"LDS MNIW degrees of freedom={effective_dof}"
          + (" (schedule default)" if lds_degrees_of_freedom is None else ""))
    detail_print(f"LDS update={config.dynamics.lds_update}"
          + (f", step size={lds_step_size:g}"
             if config.dynamics.lds_update == "natural_gradient" else ""))
    detail_print(f"LDS scoring={config.dynamics.lds_scoring}")
    birth_lds_divisor = len(config.x_basis) if birth_lds_kl_per_sqrt_point else 1
    detail_print("Experimental birth LDS KL / LDS dimension="
          f"{birth_lds_kl_per_sqrt_point} "
          f"(divisor={birth_lds_divisor:.6g})")
    detail_print(format_lds_priors(
        dimension=len(config.x_basis), model_type=config.dynamics.model_type,
        observation_noise=initial_observation_noise,
        process_noise=initial_process_noise,
        degrees_of_freedom=effective_dof,
        source="identity-LDS pilot" if calibrate_lds_noise_priors else "configured values",
        policies=selected,
    ))

    model = HDPGPC(config)
    start = time.perf_counter()
    model.fit(values, grid=grid)
    elapsed = time.perf_counter() - start

    assignments = model.labels_
    print(f"Elapsed: {elapsed:.1f} s; clusters found: {model.n_clusters_}")
    print("Cluster sizes:", Counter(assignments.tolist()))

    if len(annotations) == len(assignments) + 1:
        print("Dropping the trailing annotation without a stored waveform observation.")
        annotations = annotations[:len(assignments)]
    print("Annotation counts:", Counter(annotations.tolist()))
    scores = None
    if len(annotations) != len(assignments):
        print(
            "Classification statistics skipped: expected equal lengths or one extra trailing annotation; "
            f"got {len(assignments)} observations and {len(annotations)} annotations."
        )
    else:
        scores, classes, confusion, cluster_annotations = cluster_label_statistics(
            assignments, annotations
        )
        print("Label-based clustering statistics:")
        print(f"  Purity:                         {scores['purity']:.4f}")
        print(f"  Adjusted Rand index:            {scores['adjusted_rand_index']:.4f}")
        print(f"  Normalized mutual information:  {scores['normalized_mutual_information']:.4f}")
        print(f"  Majority-label accuracy*:       {scores['majority_label_accuracy_in_sample']:.4f}")
        print("  * Cluster-to-label mapping is fitted and scored on this same record.")
        print("Cluster annotation assignments (majority vote): "
              + ", ".join(f"C{cluster}={label}" for cluster, label in cluster_annotations.items()))
        print(format_confusion_matrix(
            classes, confusion, true_name='real annotation',
            assigned_name='cluster-assigned annotation'))

    figures = plot_record_results(
        model,
        values,
        annotations if len(annotations) == len(assignments) else None,
        grid=grid,
        record=record,
        metrics=scores,
        channels=CHANNELS,
        warped=ENABLE_WARP,
    )
    if not scale_waveforms:
        for figure in figures:
            for axis in figure.axes:
                if axis.get_ylabel() == "Scaled amplitude":
                    axis.set_ylabel("Stored amplitude")
    warp_figures = []
    if ENABLE_WARP and plot_warps:
        # Both candidate input-grid policies use one warp for all channels.
        # Legacy GP alignment may produce a different warp for each output.
        outputs = (0,) if ALIGNMENT_POLICY in ("gp_flow", "dtw_gp") else range(len(CHANNELS))
        for output in outputs:
            offsets, posterior_std = model_warp_arrays(model, grid, output=output)
            figure = plot_warp_diagnostics(
                grid, offsets, labels=assignments, posterior_std=posterior_std,
                title=f"Record {record}, channel {CHANNELS[output]}: fitted input warps",
            )
            warp_figures.append((CHANNELS[output], figure))
    lds_figures = []
    if plot_lds:
        for output, channel in enumerate(CHANNELS):
            matrices = model_lds_arrays(model, output=output)
            for page, figure in enumerate(plot_lds_diagnostics(
                matrices, title=f'Record {record}, channel {channel}: fitted LDS parameters',
            ), start=1):
                if save_plots:
                    plot_directory.mkdir(parents=True, exist_ok=True)
                    path = plot_directory / f"record_{record}_channel_{channel}_lds_{page:02d}.png"
                    figure.savefig(path, dpi=PLOT_DPI)
                    print(f"Saved LDS plot: {path}")
                    plt.close(figure)
                else:
                    lds_figures.append(figure)
    if save_plots:
        plot_directory.mkdir(parents=True, exist_ok=True)
        for channel, figure in zip(CHANNELS, figures):
            path = plot_directory / f"record_{record}_channel_{channel}_clustering.png"
            figure.savefig(path, dpi=PLOT_DPI)
            print(f"Saved plot: {path}")
            plt.close(figure)
        for channel, figure in warp_figures:
            path = plot_directory / f"record_{record}_channel_{channel}_warps.png"
            figure.savefig(path, dpi=PLOT_DPI)
            print(f"Saved warp plot: {path}")
            plt.close(figure)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        nargs="+",
        default=RECORDS,
        help="MIT-BIH record stems to run; defaults to the RECORDS tuple in this script.",
    )
    parser.add_argument(
        "--scale-waveforms", action=argparse.BooleanOptionalAction, default=SCALE_WAVEFORMS,
        help="Scale each channel with the first beat's mean/std (default: SCALE_WAVEFORMS).",
    )
    parser.add_argument(
        "--unit-grid", action=argparse.BooleanOptionalAction, default=UNIT_GRID,
        help="Use grid coordinates 0..1 instead of the existing 0..4 (default: UNIT_GRID).",
    )
    parser.add_argument(
        "--save-plots", action=argparse.BooleanOptionalAction, default=SAVE_PLOTS,
        help="Save figures without displaying them (default: SAVE_PLOTS).",
    )
    parser.add_argument(
        "--plot-warps", action=argparse.BooleanOptionalAction, default=PLOT_WARPS,
        help="Include fitted warp diagnostics when alignment is enabled (default: PLOT_WARPS).",
    )
    parser.add_argument(
        "--plot-lds", action=argparse.BooleanOptionalAction, default=PLOT_LDS,
        help="Include fitted LDS matrix and stability diagnostics (default: PLOT_LDS).",
    )
    parser.add_argument(
        "--fit-gp", action=argparse.BooleanOptionalAction, default=FIT_GP,
        help="Fit the supplied GP kernel on the first beat; --no-fit-gp keeps it fixed.",
    )
    parser.add_argument(
        "--lds-degrees-of-freedom", type=int, default=LDS_DEGREES_OF_FREEDOM,
        help="MNIW prior degrees of freedom for LDS parameters; omit for the schedule default.",
    )
    parser.add_argument(
        "--lds-update", choices=("conjugate_sequential", "conjugate_batch", "natural_gradient", "legacy"),
        default=LDS_UPDATE, help="LDS Bayesian parameter update (default: LDS_UPDATE).",
    )
    parser.add_argument(
        "--lds-scoring", choices=("final", "historical"), default=LDS_SCORING,
        help="LDS matrices used in observation and latent ELBO scores (default: LDS_SCORING).",
    )
    parser.add_argument(
        "--lds-step-size", type=float, default=LDS_STEP_SIZE,
        help="Fraction of new evidence used by the natural-gradient LDS update.",
    )
    parser.add_argument(
        "--lds-inducing-points", type=int, default=LDS_INDUCING_POINTS,
        help="Number of evenly spaced LDS basis points; 0 uses the full waveform grid.",
    )
    parser.add_argument(
        "--birth-lds-kl-per-sqrt-point", action=argparse.BooleanOptionalAction,
        default=BIRTH_LDS_KL_PER_SQRT_POINT,
        help="Scale proposal LDS KL by LDS state dimension in either inference schedule.",
    )
    parser.add_argument(
        "--calibrate-lds-noise-priors", action=argparse.BooleanOptionalAction,
        default=CALIBRATE_LDS_NOISE_PRIORS,
        help="Estimate initial Sigma/Gamma scales from contiguous record blocks before offline fitting.",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=PLOT_DIRECTORY,
        help="Directory for saved PNGs and record logs (default: PLOT_DIRECTORY).",
    )
    args = parser.parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    for record in args.records:
        record = str(record)
        log_path = args.plot_dir / f"record_{record}_clustering.log"
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
                try:
                    run_record(
                        record, save_plots=args.save_plots, plot_warps=args.plot_warps,
                        plot_lds=args.plot_lds, fit_gp=args.fit_gp,
                        lds_degrees_of_freedom=args.lds_degrees_of_freedom,
                        lds_update=args.lds_update, lds_scoring=args.lds_scoring,
                        lds_step_size=args.lds_step_size,
                        lds_inducing_points=args.lds_inducing_points,
                        birth_lds_kl_per_sqrt_point=args.birth_lds_kl_per_sqrt_point,
                        plot_directory=args.plot_dir,
                        scale_waveforms=args.scale_waveforms, unit_grid=args.unit_grid,
                        calibrate_lds_noise_priors=args.calibrate_lds_noise_priors,
                    )
                    print(f"Log file: {log_path}")
                except BaseException:
                    traceback.print_exc(file=log_file)
                    raise


if __name__ == "__main__":
    main()
