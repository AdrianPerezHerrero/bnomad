"""Cluster complete bundled MIT-BIH records and compare clusters with annotations.

Run from the repository root after installing the package::

    python examples/full_record_clustering.py
    python examples/full_record_clustering.py --records 101 102
    python examples/full_record_clustering.py --records 100 --save-plots
    python examples/full_record_clustering.py --records 100 --no-scale-waveforms --unit-grid

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from tutorial_utils import plot_record_results
from hdpgpc import HDPGPC
from hdpgpc.diagnostics import model_warp_arrays, plot_warp_diagnostics
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
RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
)
#RECORDS = ("100",)            # Current record selection.
CHANNELS = (0,)                 # Choose channel 0, 1, or both: (0, 1).
SCALE_WAVEFORMS = False          # First-beat mean/std scaling; False keeps stored amplitudes.
UNIT_GRID = False               # True: coordinates 0..1; False: existing 0..4 grid.
INFERENCE_MODE = "offline"      # "offline", "online", or "online_fast".
ENABLE_WARP = True              # Whether to request alignment during fitting.
ALIGNMENT_POLICY = "gp_flow"    # "gp" (legacy) or "gp_flow" (posterior input warp).
VERBOSE = False                 # False: progress/summary; True: per-sample/iteration detail.
SHARE_GP_FOR_PROPOSALS = True   # Offline proposals/new clusters start from a fitted GP copy.
SAVE_PLOTS = True               # Also save the final figures when True.
PLOT_WARPS = True               # Plot fitted warp maps, record heatmap, and cluster summaries.
PLOT_DIRECTORY = Path(__file__).resolve().parents[1] / "agents" / "plots_mo"
PLOT_DPI = 160

# Typed configuration values; these are experiment settings, not package defaults.
# Lengthscale is measured in the selected grid units; noise settings depend on waveform scale.
MAX_CLUSTERS = 30
HDP_PRESET = "more"            # "less", "balanced" or "more" (see HDPConfig preset descriptions).
MODEL_TYPE = "dynamic"         # "static" or "dynamic" (see DynamicsConfig model_type descriptions).
INITIAL_LENGTHSCALE = 1.0
LENGTHSCALE_BOUNDS = (0.1, 4.0)
INITIAL_OBSERVATION_NOISE = 0.1
OBSERVATION_NOISE_BOUNDS = (0.001, 0.5)
INITIAL_PROCESS_NOISE = 0.1
PROCESS_NOISE_BOUNDS = (0.001, 0.5)
OFFLINE_ITERATIONS = 50
RANDOM_SEED = 42


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "hdpgpc" / "data" / "mitbih"


def _configuration(grid: np.ndarray) -> HDPGPCConfig:
    """Build the full experiment recipe in one visible place."""
    if ENABLE_WARP and INFERENCE_MODE != "offline" and ALIGNMENT_POLICY != "gp_flow":
        raise ValueError("Online warping in this example requires ALIGNMENT_POLICY='gp_flow'.")
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
            policy=ALIGNMENT_POLICY if ENABLE_WARP else "none",
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


def run_record(record: str, *, save_plots: bool = SAVE_PLOTS,
               plot_warps: bool = PLOT_WARPS,
               plot_directory: Path = PLOT_DIRECTORY,
               scale_waveforms: bool = SCALE_WAVEFORMS,
               unit_grid: bool = UNIT_GRID) -> None:
    signal, annotations = _load_record(record)
    values, grid = _prepare_record(
        signal, scale_waveforms=scale_waveforms, unit_grid=unit_grid,
    )
    config = _configuration(grid)

    print(f"\nRecord {record}: waveform shape={signal.shape}, annotations={len(annotations)}")
    print(f"Input shape={values.shape} (beats, points, outputs), channels={CHANNELS}")
    print(f"Waveform scaling={'first-beat mean/std' if scale_waveforms else 'off'}, "
          f"grid range=0..{grid[-1, 0]:g}")
    print(f"Schedule={INFERENCE_MODE}, warp={ENABLE_WARP}, "
          f"alignment={ALIGNMENT_POLICY if ENABLE_WARP else 'none'}, verbose={VERBOSE}")
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
        # GP-flow uses one warp for all channels; legacy GP alignment may
        # produce a different warp for each output.
        outputs = (0,) if ALIGNMENT_POLICY == "gp_flow" else range(len(CHANNELS))
        for output in outputs:
            offsets, posterior_std = model_warp_arrays(model, grid, output=output)
            figure = plot_warp_diagnostics(
                grid, offsets, labels=assignments, posterior_std=posterior_std,
                title=f"Record {record}, channel {CHANNELS[output]}: fitted input warps",
            )
            warp_figures.append((CHANNELS[output], figure))
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
                        plot_directory=args.plot_dir,
                        scale_waveforms=args.scale_waveforms, unit_grid=args.unit_grid,
                    )
                except BaseException:
                    traceback.print_exc(file=log_file)
                    raise


if __name__ == "__main__":
    main()
