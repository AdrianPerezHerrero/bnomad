"""Cluster complete bundled MIT-BIH records and compare clusters with annotations.

Run from the repository root after installing the package::

    python examples/full_record_clustering.py
    python examples/full_record_clustering.py --records 101 102
    python examples/full_record_clustering.py --records 100 --save-plots

Edit the experiment controls below to choose records, channels, inference schedule,
warping, and model hyperparameters. If there is one extra trailing annotation,
the script crops the annotation vector to the number of fitted observations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tutorial_utils import plot_record_results
from hdpgpc import HDPGPC
from hdpgpc.config import (
    AlignmentConfig,
    DynamicsConfig,
    GPEmissionConfig,
    HDPConfig,
    HDPGPCConfig,
    InferenceConfig,
    MultiOutputConfig,
    NumericalConfig,
)


# -----------------------------------------------------------------------------
# Experiment controls: edit these values for the desired whole-record run.
# -----------------------------------------------------------------------------
# Full run: uncomment this tuple and comment out the single-record choice below.
# RECORDS = (
#     "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
#     "111", "112", "113", "114", "115", "116", "117", "118", "119",
#     "121", "122", "123", "124",
#     "200", "201", "202", "203", "205", "207", "208", "209", "210",
#     "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
#     "228", "230", "231", "232", "233", "234",
# )
RECORDS = ("100",)            # Current record selection.
CHANNELS = (0,)                 # Choose channel 0, 1, or both: (0, 1).
INFERENCE_MODE = "offline"      # "offline", "online", or "online_fast".
ENABLE_WARP = True              # The supported warping tutorial path is offline.
VERBOSE = False                 # False: progress/summary; True: per-sample/iteration detail.
SHARE_GP_FOR_PROPOSALS = True   # Offline proposals/new clusters start from a fitted GP copy.
SAVE_PLOTS = True               # Also save the final figures when True.
PLOT_DIRECTORY = Path(__file__).resolve().parents[1] / "agents" / "plots"
PLOT_DPI = 160

# Typed configuration values; these are experiment settings, not package defaults.
MAX_CLUSTERS = 12
HDP_PRESET = "more"            # "less", "balanced" or "more" (see HDPConfig preset descriptions).
MODEL_TYPE = "dynamic"         # "static" or "dynamic" (see DynamicsConfig model_type descriptions).
INITIAL_LENGTHSCALE = 1.5
LENGTHSCALE_BOUNDS = (0.1, 4.0)
INITIAL_OBSERVATION_NOISE = 0.1
OBSERVATION_NOISE_BOUNDS = (0.001, 0.5)
INITIAL_PROCESS_NOISE = 0.1
PROCESS_NOISE_BOUNDS = (0.001, 0.5)
OFFLINE_ITERATIONS = 20
RANDOM_SEED = 42

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "hdpgpc" / "data" / "mitbih"


def _configuration(grid: np.ndarray) -> HDPGPCConfig:
    """Build the full experiment recipe in one visible place."""
    if ENABLE_WARP and INFERENCE_MODE != "offline":
        raise ValueError("This example enables warping only with INFERENCE_MODE='offline'.")
    if len(CHANNELS) > 1 and INFERENCE_MODE in ("online", "online_fast"):
        raise ValueError("The characterized online multi-output path is unsupported; select one channel.")

    return HDPGPCConfig(
        x_basis=grid,
        verbose=VERBOSE,
        hdp=HDPConfig(hdp_hyp=HDP_PRESET, max_models=MAX_CLUSTERS),
        emission=GPEmissionConfig(
            ini_lengthscale=INITIAL_LENGTHSCALE,
            bound_lengthscale=LENGTHSCALE_BOUNDS,
            ini_sigma=INITIAL_OBSERVATION_NOISE,
            bound_sigma=OBSERVATION_NOISE_BOUNDS,
            ini_outputscale=1.0,
        ),
        dynamics=DynamicsConfig(
            model_type=MODEL_TYPE,
            ini_gamma=INITIAL_PROCESS_NOISE,
            bound_gamma=PROCESS_NOISE_BOUNDS,
            annealing=False,
            bayesian_params=True,
        ),
        alignment=AlignmentConfig(
            policy="gp" if ENABLE_WARP else "none",
            x_basis_warp=grid if ENABLE_WARP else None,
            mode_warp="rough",
        ),
        inference=InferenceConfig(
            schedule=INFERENCE_MODE,
            with_warp=ENABLE_WARP,
            warp=ENABLE_WARP,
            it_limit=OFFLINE_ITERATIONS if INFERENCE_MODE == "offline" else None,
            n_explore_steps=2,
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


def _prepare_record(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep every point and scale each selected channel from the first beat only."""
    raw = np.asarray(signal[:, :, CHANNELS], dtype=np.float64)
    center = raw[0].mean(axis=0)
    scale = raw[0].std(axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("The first beat has a zero/invalid scale in a selected channel.")
    values = (raw - center[None, None, :]) / scale[None, None, :]
    grid = np.linspace(0.0, 4.0, raw.shape[1], dtype=np.float64)[:, None]
    return values, grid


def _classification_statistics(
    assignments: np.ndarray, labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Return scores and annotation-by-cluster counts with their row/column labels."""
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

    majority_correct = int(contingency.max(axis=1).sum())
    scores = {
        "purity": majority_correct / len(labels),
        "majority_label_accuracy_in_sample": majority_correct / len(labels),
        "adjusted_rand_index": float(adjusted_rand_score(labels, assignments)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(labels, assignments, average_method="arithmetic")
        ),
    }
    return scores, classes, clusters, contingency.T


def _print_confusion_matrix(classes: np.ndarray, clusters: np.ndarray,
                            matrix: np.ndarray) -> None:
    """Print annotation counts by assigned cluster, including row/column totals."""
    row_width = max(10, *(len(str(label)) for label in classes))
    column_names = [f"C{cluster}" for cluster in clusters]
    cell_width = max(5, len(str(matrix.sum())), *(len(name) for name in column_names))
    print("Confusion matrix (rows=annotations, columns=assigned clusters; counts):")
    print(f"  {'Annotation':>{row_width}} | "
          + " ".join(f"{name:>{cell_width}}" for name in column_names)
          + f" | {'Total':>{cell_width}}")
    for label, row in zip(classes, matrix):
        print(f"  {str(label):>{row_width}} | "
              + " ".join(f"{int(count):>{cell_width}}" for count in row)
              + f" | {int(row.sum()):>{cell_width}}")
    print(f"  {'Total':>{row_width}} | "
          + " ".join(f"{int(total):>{cell_width}}" for total in matrix.sum(axis=0))
          + f" | {int(matrix.sum()):>{cell_width}}")


def run_record(record: str, *, save_plots: bool = SAVE_PLOTS,
               plot_directory: Path = PLOT_DIRECTORY) -> None:
    signal, annotations = _load_record(record)
    values, grid = _prepare_record(signal)
    config = _configuration(grid)

    print(f"\nRecord {record}: waveform shape={signal.shape}, annotations={len(annotations)}")
    print(f"Input shape={values.shape} (beats, points, outputs), channels={CHANNELS}")
    print(f"Schedule={INFERENCE_MODE}, warp={ENABLE_WARP}, verbose={VERBOSE}")
    print(f"Model={MODEL_TYPE}, HDP preset={HDP_PRESET}, cluster cap={MAX_CLUSTERS}")

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
        scores, classes, clusters, confusion = _classification_statistics(assignments, annotations)
        print("Label-based clustering statistics:")
        print(f"  Purity:                         {scores['purity']:.4f}")
        print(f"  Adjusted Rand index:            {scores['adjusted_rand_index']:.4f}")
        print(f"  Normalized mutual information:  {scores['normalized_mutual_information']:.4f}")
        print(f"  Majority-label accuracy*:       {scores['majority_label_accuracy_in_sample']:.4f}")
        print("  * Cluster-to-label mapping is fitted and scored on this same record.")
        _print_confusion_matrix(classes, clusters, confusion)

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
    if save_plots:
        plot_directory.mkdir(parents=True, exist_ok=True)
        for channel, figure in zip(CHANNELS, figures):
            path = plot_directory / f"record_{record}_channel_{channel}_clustering.png"
            figure.savefig(path, dpi=PLOT_DPI)
            print(f"Saved plot: {path}")
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
        "--save-plots", action=argparse.BooleanOptionalAction, default=SAVE_PLOTS,
        help="Save one PNG for each selected channel (default: SAVE_PLOTS).",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=PLOT_DIRECTORY,
        help="Directory for saved PNGs (default: PLOT_DIRECTORY).",
    )
    args = parser.parse_args()
    for record in args.records:
        run_record(str(record), save_plots=args.save_plots, plot_directory=args.plot_dir)


if __name__ == "__main__":
    main()
