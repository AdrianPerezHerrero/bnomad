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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tutorial_utils import plot_record_results
from hdpgpc import HDPGPC
from hdpgpc.dynamics import estimate_lds_noise_priors
from hdpgpc.diagnostics import (
    model_lds_arrays, model_warp_arrays, plot_lds_diagnostics, plot_warp_diagnostics,
)
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
)


# -----------------------------------------------------------------------------
# Experiment controls: edit these values for the desired whole-record run.
# -----------------------------------------------------------------------------
# Full run: uncomment this tuple and comment out the single-record choice below.
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
UNIT_GRID = False               # True: coordinates 0..1; False: existing 0..4 grid.
INFERENCE_MODE = "offline"      # "offline", "online", or "online_fast".
LDS_INDUCING_POINTS = None        # Reduced LDS basis size; 0 or None uses every waveform point.
HARD_ASSIGNMENTS = True         # False uses soft responsibility weights during inference.
ENABLE_WARP = False              # Whether to request alignment during fitting.
ALIGNMENT_POLICY = "dtw_gp"    # "gp", (gp-based alignment), "dtw_gp" (cheap point warp), or "gp_flow".
VERBOSE = True                 # False: progress/summary; True: per-sample/iteration detail.
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
LDS_DEGREES_OF_FREEDOM = 5  # None uses the package default: 5 offline, 20 online.
LDS_UPDATE = "conjugate_batch"  # "conjugate_sequential", "conjugate_batch", "natural_gradient", or "legacy".
LDS_SCORING = "final"          # "final" uses last LDS matrices; "historical" uses per-beat matrices.
# Set None to use the four manual modes below. Named bundles are documented in
# agents/lds_presets.md; "only_plastic" retains this example's prior behavior.
LDS_PRESET = "only_stable"
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
INITIAL_PROCESS_NOISE = 5.0e+1
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
) -> HDPGPCConfig:
    """Build the full experiment recipe in one visible place."""
    if ENABLE_WARP and INFERENCE_MODE != "offline" and ALIGNMENT_POLICY not in ("gp_flow", "dtw_gp"):
        raise ValueError("Online warping in this example requires 'gp_flow' or 'dtw_gp'.")
    if len(CHANNELS) > 1 and INFERENCE_MODE in ("online", "online_fast"):
        raise ValueError("The characterized online multi-output path is unsupported; select one channel.")
    if lds_inducing_points is None or lds_inducing_points == 0:
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

    manual_model_type = "dynamic" if LDS_PRESET is not None else MODEL_TYPE
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
            C_policy=MatrixPolicyConfig(
                mode=LDS_C_MODE, value=CUSTOM_C,
                strength=LDS_C_SHRINKAGE if LDS_C_MODE == 'shrinkage' else 1.0),
            Sigma_policy=NoisePolicyConfig(
                mode=LDS_SIGMA_MODE, value=CUSTOM_SIGMA)),
        dynamics=DynamicsConfig(
            model_type=manual_model_type,
            ini_gamma=initial_process_noise,
            A_policy=(None if manual_model_type == "static" else MatrixPolicyConfig(
                mode=LDS_A_MODE, value=CUSTOM_A,
                strength=LDS_A_SHRINKAGE if LDS_A_MODE == 'shrinkage' else 1.0)),
            Gamma_policy=(None if manual_model_type == "static" else NoisePolicyConfig(
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
            n_explore_steps=5,
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


def _prepare_record(
    signal: np.ndarray, *, scale_waveforms: bool = SCALE_WAVEFORMS,
    unit_grid: bool = UNIT_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep every point and apply the selected waveform/grid scaling."""
    raw = np.asarray(signal[:, :, CHANNELS], dtype=np.float64)
    if scale_waveforms:
        center = raw[0].mean(axis=0)
        scale = raw[0].std(axis=0)
        if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(np.float64).eps):
            raise ValueError("The first beat has a zero/invalid scale in a selected channel.")
        values = (raw - center[None, None, :]) / scale[None, None, :]
    else:
        values = raw
    grid_end = 1.0 if unit_grid else raw.shape[1] - 1.0
    grid = np.linspace(0.0, grid_end, raw.shape[1], dtype=np.float64)[:, None]
    return values, grid


def _classification_statistics(
    assignments: np.ndarray, labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[int, str]]:
    """Score clusters after assigning each its majority ECG annotation."""
    assignments = np.asarray(assignments).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if len(assignments) != len(labels):
        raise ValueError("Assignments and annotations must have a verified one-to-one alignment.")

    clusters = np.unique(assignments)
    classes = np.unique(labels)
    contingency = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for row, cluster in enumerate(clusters):
        members = labels[assignments == cluster]
        counts = Counter(members.tolist())
        for column, label in enumerate(classes):
            contingency[row, column] = counts[label]

    majority_labels = classes[np.argmax(contingency, axis=1)]
    cluster_annotations = dict(zip(clusters.tolist(), majority_labels.tolist()))
    assigned_labels = majority_labels[np.searchsorted(clusters, assignments)]
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (np.searchsorted(classes, labels),
                          np.searchsorted(classes, assigned_labels)), 1)
    majority_correct = int(np.trace(confusion))
    scores = {
        "purity": majority_correct / len(labels),
        "majority_label_accuracy_in_sample": majority_correct / len(labels),
        "adjusted_rand_index": float(adjusted_rand_score(labels, assignments)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(labels, assignments, average_method="arithmetic")
        ),
    }
    return scores, classes, confusion, cluster_annotations


def _print_confusion_matrix(classes: np.ndarray, matrix: np.ndarray) -> None:
    """Print true annotations against cluster-assigned annotations."""
    row_width = max(13, *(len(str(label)) for label in classes))
    column_names = [str(label) for label in classes]
    cell_width = max(5, len(str(matrix.sum())), *(len(name) for name in column_names))
    print("Confusion matrix (rows=real annotation, columns=cluster-assigned annotation; counts):")
    print(f"  {'Real/assigned':>{row_width}} | "
          + " ".join(f"{name:>{cell_width}}" for name in column_names)
          + f" | {'Total':>{cell_width}}")
    for label, row in zip(classes, matrix):
        print(f"  {str(label):>{row_width}} | "
              + " ".join(f"{int(count):>{cell_width}}" for count in row)
              + f" | {int(row.sum()):>{cell_width}}")
    print(f"  {'Total':>{row_width}} | "
          + " ".join(f"{int(total):>{cell_width}}" for total in matrix.sum(axis=0))
          + f" | {int(matrix.sum()):>{cell_width}}")


def _print_lds_priors(*, dimension: int, model_type: str,
                      observation_noise: float, process_noise: float,
                      degrees_of_freedom: int, source: str, policies: dict) -> None:
    """Log the LDS prior centers and concentrations selected by this recipe."""
    def matrix_center(name):
        value = policies[name].value
        return f"I_{dimension}" if value is None else f"custom {np.asarray(value).shape} matrix"

    def noise_center(name, scalar):
        value = policies[name].value
        return (f"{scalar:.6g} * I_{dimension}" if value is None
                else f"custom {np.asarray(value).shape} covariance")

    print("------ Initial LDS parameter priors ------")
    print(f"Source={source}; waveform dimension={dimension}")
    if model_type == "dynamic":
        for matrix, noise, scale in (('A', 'Gamma', process_noise),
                                     ('C', 'Sigma', observation_noise)):
            matrix_policy, noise_policy = policies[matrix], policies[noise]
            n0 = degrees_of_freedom * noise_policy.strength
            print(f"{matrix}: {matrix_policy.mode}, center={matrix_center(matrix)}, "
                  f"column prior precision={matrix_policy.strength:g} * I_{dimension}"
                  if matrix_policy.mode != 'fixed' else
                  f"{matrix}: fixed at {matrix_center(matrix)}")
            print(f"{noise}: {noise_policy.mode}, center={noise_center(noise, scale)}, "
                  f"excess degrees n0={n0:g}" if noise_policy.mode != 'fixed' else
                  f"{noise}: fixed at {noise_center(noise, scale)}")
    else:
        print("A: absent (static dynamics)")
        print(f"C: {policies['C'].mode} at {matrix_center('C')}")
        print(f"Gamma: 0_{dimension} (static model; no A/Gamma posterior KL)")
        print(f"Sigma: {policies['Sigma'].mode}, "
              f"center={noise_center('Sigma', observation_noise)}")
    print("------------------------------------------")


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
    signal, annotations = _load_record(record)
    values, grid = _prepare_record(
        signal, scale_waveforms=scale_waveforms, unit_grid=unit_grid,
    )
    initial_observation_noise = INITIAL_OBSERVATION_NOISE
    initial_process_noise = INITIAL_PROCESS_NOISE
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
        print(
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
    )

    print(f"\nRecord {record}: waveform shape={signal.shape}, annotations={len(annotations)}")
    print(f"Input shape={values.shape} (beats, points, outputs), channels={CHANNELS}")
    print(f"LDS basis points={len(config.x_basis)} of {len(grid)}, "
          f"inducing mode={config.emission.inducing_points}")
    print(f"Waveform scaling={'first-beat mean/std' if scale_waveforms else 'off'}, "
          f"grid range=0..{grid[-1, 0]:g}")
    print(f"Schedule={INFERENCE_MODE}, hard assignments={HARD_ASSIGNMENTS}, warp={ENABLE_WARP}, "
          f"alignment={ALIGNMENT_POLICY if ENABLE_WARP else 'none'}, verbose={VERBOSE}")
    print(f"Model={config.dynamics.model_type}, Markov={config.hdp.hmm_switch}, "
          f"LDS preset={LDS_PRESET or 'manual'}, HDP preset={HDP_PRESET}, "
          f"cluster cap={MAX_CLUSTERS}")
    from hdpgpc.config._models import resolved_lds_policies
    selected = resolved_lds_policies(config)
    print("LDS policies=" + ", ".join(
        f"{name}:{selected[name].mode}" for name in ('A', 'Gamma', 'C', 'Sigma')))
    print("LDS prior strengths=" + ", ".join(
        f"{name}:{selected[name].strength:g}"
        for name in ('A', 'Gamma', 'C', 'Sigma')))
    print(f"Custom C={selected['C'].value is not None}, "
          f"custom Sigma={selected['Sigma'].value is not None}")
    print(f"GP fitting={'on' if config.emission.fit_gp else 'off'}, "
          f"kernel={config.emission.kernels[0]}")
    gp_white_bounds = config.emission.kernels[0].get_params().get("k2__noise_level_bounds")
    if gp_white_bounds is not None:
        print(f"GP white-noise bounds={gp_white_bounds}")
    effective_dof = (lds_degrees_of_freedom if lds_degrees_of_freedom is not None
                     else (5 if INFERENCE_MODE == "offline" else 20))
    print(f"LDS MNIW degrees of freedom={effective_dof}"
          + (" (schedule default)" if lds_degrees_of_freedom is None else ""))
    print(f"LDS update={lds_update}"
          + (f", step size={lds_step_size:g}" if lds_update == "natural_gradient" else ""))
    print(f"LDS scoring={config.dynamics.lds_scoring}")
    birth_lds_divisor = len(config.x_basis) if birth_lds_kl_per_sqrt_point else 1
    print("Experimental birth LDS KL / LDS dimension="
          f"{birth_lds_kl_per_sqrt_point} "
          f"(divisor={birth_lds_divisor:.6g})")
    _print_lds_priors(
        dimension=len(config.x_basis), model_type=config.dynamics.model_type,
        observation_noise=initial_observation_noise,
        process_noise=initial_process_noise,
        degrees_of_freedom=effective_dof,
        source="identity-LDS pilot" if calibrate_lds_noise_priors else "configured values",
        policies=selected,
    )

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
        scores, classes, confusion, cluster_annotations = _classification_statistics(
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
        _print_confusion_matrix(classes, confusion)

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


class _Tee:
    """Write Python console output to the terminal and a record log."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


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
            with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
                print(f"Log file: {log_path}")
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
                except BaseException:
                    traceback.print_exc(file=log_file)
                    raise


if __name__ == "__main__":
    main()
