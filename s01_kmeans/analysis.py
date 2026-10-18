"""Reads results/s01/raw_results.csv, computes Speedup/Efficiency from the
raw times (never typed in by hand), writes results/s01/summary.csv, and
saves the required plots (+ this generating script) under
results/s01/plots/.

K is encoded in the shared `Workload` column as "kmeans_k<K>" since the
shared CSV schema (common/csv_schema.py) has no dedicated K column.

Usage: python3 -m s01_kmeans.analysis [--quick]
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


_WORKLOAD_K_RE = re.compile(r"kmeans_k(\d+)")


def _extract_k(workload):
    m = _WORKLOAD_K_RE.match(workload)
    if not m:
        raise ValueError(f"Unrecognized Workload value: {workload!r}")
    return int(m.group(1))


def load_raw(raw_csv):
    df = pd.read_csv(raw_csv)
    df["K"] = df["Workload"].apply(_extract_k)
    return df


def build_summary(df):
    group_cols = ["Dataset", "Problem_Size", "K", "Algorithm", "Resource_Count"]
    agg = df.groupby(group_cols).agg(
        Median_Execution_Time=("Execution_Time", "median"),
        Std_Execution_Time=("Execution_Time", "std"),
        Median_Communication_Time=("Communication_Time", "median"),
        Median_Synchronization_Time=("Synchronization_Time", "median"),
        Median_Scheduling_Time=("Scheduling_Time", "median"),
        Median_Memory_Use=("Memory_Use", "median"),
        N_Runs=("Run_ID", "count"),
    ).reset_index()

    baseline = (
        agg[agg["Algorithm"] == "Lloyd-Sequential"]
        .set_index(["Dataset", "Problem_Size", "K"])["Median_Execution_Time"]
    )

    def _speedup(row):
        key = (row["Dataset"], row["Problem_Size"], row["K"])
        if key not in baseline.index:
            return np.nan
        return baseline.loc[key] / row["Median_Execution_Time"]

    agg["Speedup"] = agg.apply(_speedup, axis=1)
    agg["Efficiency"] = agg["Speedup"] / agg["Resource_Count"]

    agg = agg.sort_values(group_cols).reset_index(drop=True)
    return agg


def _save(fig, plots_dir, name):
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_vs_samples(summary, plots_dir):
    blobs = summary[summary["Dataset"] == "synthetic_blobs"]
    for k, sub_k in blobs.groupby("K"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for (algo, workers), sub in sub_k.groupby(["Algorithm", "Resource_Count"]):
            sub = sub.sort_values("Problem_Size")
            label = "Sequential" if algo == "Lloyd-Sequential" else f"Parallel ({workers} workers)"
            ax.plot(sub["Problem_Size"], sub["Median_Execution_Time"], marker="o", label=label)
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Median execution time (s)")
        ax.set_title(f"Time vs. samples (K={k}, synthetic blobs)")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, plots_dir, f"time_vs_samples_k{k}.png")


def plot_speedup_vs_workers(summary, plots_dir):
    parallel = summary[summary["Algorithm"] == "Lloyd-Parallel-Pool"]
    for (dataset, size), sub_ds in parallel.groupby(["Dataset", "Problem_Size"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        for k, sub in sub_ds.groupby("K"):
            sub = sub.sort_values("Resource_Count")
            ax.plot(sub["Resource_Count"], sub["Speedup"], marker="o", label=f"K={k}")
        ax.plot(
            parallel["Resource_Count"].drop_duplicates().sort_values(),
            parallel["Resource_Count"].drop_duplicates().sort_values(),
            linestyle="--",
            color="gray",
            label="Ideal (linear)",
        )
        ax.set_xlabel("Workers")
        ax.set_ylabel("Speedup vs. sequential baseline")
        ax.set_title(f"Speedup vs. workers ({dataset}, n={size})")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, plots_dir, f"speedup_vs_workers_{dataset}_n{size}.png")


def plot_iterations_vs_configuration(iterations_df, plots_dir):
    agg = iterations_df.groupby(
        ["Dataset", "Problem_Size", "K", "Algorithm", "Resource_Count"]
    )["Iterations"].median().reset_index()
    agg["config_label"] = (
        agg["Dataset"].astype(str)
        + "_n"
        + agg["Problem_Size"].astype(str)
        + "_k"
        + agg["K"].astype(str)
        + "_"
        + agg["Algorithm"]
        + "_w"
        + agg["Resource_Count"].astype(str)
    )
    agg = agg.sort_values("config_label")

    fig, ax = plt.subplots(figsize=(max(8, len(agg) * 0.3), 5))
    ax.bar(range(len(agg)), agg["Iterations"])
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["config_label"], rotation=90, fontsize=6)
    ax.set_ylabel("Iterations to convergence")
    ax.set_title("Iterations vs. configuration")
    _save(fig, plots_dir, "iterations_vs_configuration.png")


def plot_time_per_iteration(summary, iterations_df, plots_dir):
    """Median time / median iterations. Most runs stop at the iteration cap
    rather than converging, so raw time mixes workload size with iteration count."""
    keys = ["Dataset", "Problem_Size", "K", "Algorithm", "Resource_Count"]
    iters = iterations_df.groupby(keys)["Iterations"].median().reset_index()
    merged = summary.merge(iters, on=keys)
    merged["Time_Per_Iteration"] = merged["Median_Execution_Time"] / merged["Iterations"]

    blobs = merged[merged["Dataset"] == "synthetic_blobs"]
    for k, sub_k in blobs.groupby("K"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for (algo, workers), sub in sub_k.groupby(["Algorithm", "Resource_Count"]):
            sub = sub.sort_values("Problem_Size")
            label = "Sequential" if algo == "Lloyd-Sequential" else f"Parallel ({workers} workers)"
            ax.plot(sub["Problem_Size"], sub["Time_Per_Iteration"] * 1000, marker="o", label=label)
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Median time per iteration (ms)")
        ax.set_title(f"Time per iteration vs. samples (K={k}, synthetic blobs)")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, plots_dir, f"time_per_iteration_vs_samples_k{k}.png")


def plot_time_breakdown(summary, plots_dir):
    parallel = summary[summary["Algorithm"] == "Lloyd-Parallel-Pool"].copy()
    parallel["Compute_Time"] = (
        parallel["Median_Execution_Time"]
        - parallel["Median_Communication_Time"]
        - parallel["Median_Synchronization_Time"]
        - parallel["Median_Scheduling_Time"]
    ).clip(lower=0)

    for (dataset, size, k), sub in parallel.groupby(["Dataset", "Problem_Size", "K"]):
        sub = sub.sort_values("Resource_Count")
        fig, ax = plt.subplots(figsize=(7, 5))
        x = sub["Resource_Count"].astype(str)
        bottom = np.zeros(len(sub))
        for col, label in [
            ("Compute_Time", "Compute"),
            ("Median_Communication_Time", "Communication (IPC)"),
            ("Median_Synchronization_Time", "Synchronization (reduction)"),
            ("Median_Scheduling_Time", "Scheduling (pool+dispatch)"),
        ]:
            ax.bar(x, sub[col], bottom=bottom, label=label)
            bottom += sub[col].to_numpy()
        ax.set_xlabel("Workers")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Time breakdown ({dataset}, n={size}, K={k})")
        ax.legend()
        _save(fig, plots_dir, f"time_breakdown_{dataset}_n{size}_k{k}.png")


def main():
    parser = argparse.ArgumentParser(description="Analyse S01 results.")
    parser.add_argument("--quick", action="store_true", help="Analyse results/s01_quick instead of results/s01.")
    paths = config.results_paths(quick=parser.parse_args().quick)

    df = load_raw(paths.raw_csv)
    summary = build_summary(df)
    summary.to_csv(paths.summary_csv, index=False)
    print(f"Summary written to {paths.summary_csv} ({len(summary)} rows)")

    plots_dir = paths.plots_dir
    plot_time_vs_samples(summary, plots_dir)
    plot_speedup_vs_workers(summary, plots_dir)
    plot_time_breakdown(summary, plots_dir)

    if paths.iterations_csv.exists():
        iterations_df = pd.read_csv(paths.iterations_csv)
        plot_iterations_vs_configuration(iterations_df, plots_dir)
        plot_time_per_iteration(summary, iterations_df, plots_dir)
    else:
        print(f"Skipping iterations plot: {paths.iterations_csv} not found.")

    print(f"Plots written to {plots_dir}")


if __name__ == "__main__":
    main()
