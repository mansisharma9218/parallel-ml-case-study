"""Reads results/s03/{raw_results,stages,run_metrics}.csv, computes Speedup,
Efficiency and throughput from the raw times (never typed in by hand),
writes summary.csv and stage_summary.csv, and saves the plots (PNG + SVG)
under results/s03/plots/.

Usage: python -m s03_preprocessing.analysis [--quick]
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .experiment_runner import QUICK_RESULTS_DIR, RESULTS_DIR
from .stages import STAGES

CONFIG_KEY = ["Dataset", "Problem_Size", "Algorithm", "Resource_Count"]
PARALLEL_ALGS = [config.ALG_STAGE, config.ALG_PIPE]
LABELS = {
    config.ALG_SEQ: "Sequential",
    config.ALG_STAGE: "Stage-level parallel",
    config.ALG_PIPE: "Pipeline parallel",
}


def build_summary(raw, metrics):
    df = raw.merge(metrics, on=["Dataset", "Problem_Size", "Workload", "Algorithm", "Resource_Count", "Run_ID"])
    agg = df.groupby(CONFIG_KEY).agg(
        Median_Execution_Time=("Execution_Time", "median"),
        Std_Execution_Time=("Execution_Time", "std"),
        Median_Communication_Time=("Communication_Time", "median"),
        Median_Synchronization_Time=("Synchronization_Time", "median"),
        Median_Scheduling_Time=("Scheduling_Time", "median"),
        Median_Data_Movement_Time=("Data_Movement_Time", "median"),
        Median_Memory_Use=("Memory_Use", "median"),
        Median_CPU_Utilization_Pct=("CPU_Utilization_Pct", "median"),
        N_Runs=("Run_ID", "count"),
    ).reset_index()

    baseline = (agg[agg["Algorithm"] == config.ALG_SEQ]
                .set_index(["Dataset", "Problem_Size"])["Median_Execution_Time"]
                .rename("_seq"))
    agg = agg.join(baseline, on=["Dataset", "Problem_Size"])
    agg["Speedup"] = agg["_seq"] / agg["Median_Execution_Time"]
    agg["Efficiency"] = agg["Speedup"] / agg["Resource_Count"]
    agg["Throughput_Images_Per_Sec"] = agg["Problem_Size"] / agg["Median_Execution_Time"]
    return agg.drop(columns="_seq").sort_values(CONFIG_KEY).reset_index(drop=True)


def build_stage_summary(stages):
    agg = stages.groupby(CONFIG_KEY + ["Stage"]).agg(
        Median_Wall_Time=("Wall_Time", "median"),
        Median_Busy_Time=("Busy_Time", "median"),
    ).reset_index()

    seq = (agg[agg["Algorithm"] == config.ALG_SEQ]
           .set_index(["Dataset", "Problem_Size", "Stage"])["Median_Wall_Time"]
           .rename("_seq"))
    agg = agg.join(seq, on=["Dataset", "Problem_Size", "Stage"])
    # Wall_Time is empty for the pipeline strategy (its stages are fused),
    # so its stage speedup / share stay NaN
    agg["Stage_Speedup"] = agg["_seq"] / agg["Median_Wall_Time"]
    totals = agg.groupby(CONFIG_KEY)["Median_Wall_Time"].transform("sum", min_count=1)
    agg["Share_Of_Stage_Time"] = agg["Median_Wall_Time"] / totals
    agg["Stage"] = pd.Categorical(agg["Stage"], categories=STAGES, ordered=True)
    return agg.drop(columns="_seq").sort_values(CONFIG_KEY + ["Stage"]).reset_index(drop=True)


def _save(fig, plots_dir, name):
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(plots_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_contribution(summary, stage_summary, plots_dir):
    """Stacked per-stage wall time: sequential and stage-level parallel at each
    worker count, plus the remainder of Execution_Time (pool start-up,
    shared-memory set-up, copy-out) as 'overhead'."""
    for (dataset, size), sub in stage_summary.groupby(["Dataset", "Problem_Size"]):
        stage_workers = sorted(sub.loc[sub["Algorithm"] == config.ALG_STAGE, "Resource_Count"].unique())
        configs = [(config.ALG_SEQ, 1)] + [(config.ALG_STAGE, w) for w in stage_workers]
        labels, bottoms = [], np.zeros(len(configs))
        fig, ax = plt.subplots(figsize=(8, 5))
        per_stage = {stage: [] for stage in STAGES}
        overhead = []
        for algo, workers in configs:
            labels.append("Seq" if algo == config.ALG_SEQ else f"{workers}w")
            rows = sub[(sub["Algorithm"] == algo) & (sub["Resource_Count"] == workers)].set_index("Stage")
            for stage in STAGES:
                per_stage[stage].append(rows.loc[stage, "Median_Wall_Time"])
            total = summary[(summary["Dataset"] == dataset) & (summary["Problem_Size"] == size)
                            & (summary["Algorithm"] == algo) & (summary["Resource_Count"] == workers)]
            overhead.append(max(total["Median_Execution_Time"].iloc[0] - rows["Median_Wall_Time"].sum(), 0.0))
        for stage in STAGES:
            ax.bar(labels, per_stage[stage], bottom=bottoms, label=stage)
            bottoms += np.array(per_stage[stage])
        ax.bar(labels, overhead, bottom=bottoms, label="overhead (pool/shm/copy-out)", color="lightgray")
        ax.set_xlabel("Configuration (Seq = sequential; Nw = stage-level parallel, N workers)")
        ax.set_ylabel("Median time (s)")
        ax.set_title(f"Stage contribution ({dataset}, n={size})")
        ax.legend()
        _save(fig, plots_dir, f"stage_contribution_{dataset}_n{size}")


def _ideal(ax, workers):
    ax.plot(workers, workers, linestyle="--", color="gray", label="Ideal (linear)")


def plot_speedup_vs_workers(summary, plots_dir):
    for (dataset, size), sub in summary[summary["Algorithm"].isin(PARALLEL_ALGS)].groupby(["Dataset", "Problem_Size"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in PARALLEL_ALGS:
            s = sub[sub["Algorithm"] == algo].sort_values("Resource_Count")
            ax.plot(s["Resource_Count"], s["Speedup"], marker="o", label=LABELS[algo])
        _ideal(ax, sorted(sub["Resource_Count"].unique()))
        ax.set_xlabel("Workers")
        ax.set_ylabel("Speedup vs. sequential baseline")
        ax.set_title(f"Speedup vs. workers ({dataset}, n={size})")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, plots_dir, f"speedup_vs_workers_{dataset}_n{size}")


def plot_stage_speedup(stage_summary, plots_dir):
    """Which stages scale: per-stage speedup of the stage-level strategy."""
    stage_level = stage_summary[stage_summary["Algorithm"] == config.ALG_STAGE]
    for (dataset, size), sub in stage_level.groupby(["Dataset", "Problem_Size"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        for stage in STAGES:
            s = sub[sub["Stage"] == stage].sort_values("Resource_Count")
            ax.plot(s["Resource_Count"], s["Stage_Speedup"], marker="o", label=stage)
        _ideal(ax, sorted(sub["Resource_Count"].unique()))
        ax.set_xlabel("Workers")
        ax.set_ylabel("Stage speedup vs. sequential stage")
        ax.set_title(f"Per-stage speedup, stage-level parallel ({dataset}, n={size})")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, plots_dir, f"stage_speedup_vs_workers_{dataset}_n{size}")


def plot_throughput(summary, plots_dir):
    for dataset, sub in summary.groupby("Dataset"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for (algo, size), s in sub[sub["Algorithm"].isin(PARALLEL_ALGS)].groupby(["Algorithm", "Problem_Size"]):
            s = s.sort_values("Resource_Count")
            style = "-" if algo == config.ALG_STAGE else "--"
            ax.plot(s["Resource_Count"], s["Throughput_Images_Per_Sec"], style, marker="o",
                    label=f"{LABELS[algo]}, n={size}")
        seq = sub[sub["Algorithm"] == config.ALG_SEQ]
        ax.scatter(seq["Resource_Count"], seq["Throughput_Images_Per_Sec"], marker="x", color="black",
                   zorder=3, label="Sequential (each n)")
        ax.set_xlabel("Workers")
        ax.set_ylabel("Pipeline throughput (images/s)")
        ax.set_title(f"Pipeline throughput vs. workers ({dataset})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        _save(fig, plots_dir, f"throughput_vs_workers_{dataset}")


def plot_time_breakdown(summary, plots_dir):
    parallel = summary[summary["Algorithm"].isin(PARALLEL_ALGS)].copy()
    overhead_cols = [
        ("Median_Communication_Time", "Communication (IPC + imbalance)"),
        ("Median_Synchronization_Time", "Synchronization (reductions)"),
        ("Median_Scheduling_Time", "Scheduling (pool start/stop)"),
        ("Median_Data_Movement_Time", "Data movement (shm + copy-out)"),
    ]
    parallel["Compute_Time"] = (parallel["Median_Execution_Time"]
                                - parallel[[c for c, _ in overhead_cols]].sum(axis=1)).clip(lower=0)
    for (dataset, size), sub in parallel.groupby(["Dataset", "Problem_Size"]):
        sub = sub.sort_values(["Algorithm", "Resource_Count"], ascending=[False, True])
        x = [f"{'Stage' if a == config.ALG_STAGE else 'Pipe'}\n{w}w"
             for a, w in zip(sub["Algorithm"], sub["Resource_Count"])]
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(sub))
        for col, label in [("Compute_Time", "Compute")] + overhead_cols:
            ax.bar(x, sub[col], bottom=bottom, label=label)
            bottom += sub[col].to_numpy()
        ax.set_xlabel("Strategy / workers")
        ax.set_ylabel("Median time (s)")
        ax.set_title(f"Time breakdown ({dataset}, n={size})")
        ax.legend(fontsize=8)
        _save(fig, plots_dir, f"time_breakdown_{dataset}_n{size}")


def plot_cpu_utilization(summary, plots_dir):
    for dataset, sub in summary.groupby("Dataset"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for (algo, size), s in sub[sub["Algorithm"].isin(PARALLEL_ALGS)].groupby(["Algorithm", "Problem_Size"]):
            s = s.sort_values("Resource_Count")
            style = "-" if algo == config.ALG_STAGE else "--"
            ax.plot(s["Resource_Count"], s["Median_CPU_Utilization_Pct"], style, marker="o",
                    label=f"{LABELS[algo]}, n={size}")
        ax.set_xlabel("Workers")
        ax.set_ylabel("System-wide CPU utilization (%)")
        ax.set_title(f"CPU utilization vs. workers ({dataset})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        _save(fig, plots_dir, f"cpu_utilization_vs_workers_{dataset}")


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot S03 results.")
    parser.add_argument("--quick", action="store_true", help="Analyze results/s03_quick instead.")
    args = parser.parse_args()
    results_dir = QUICK_RESULTS_DIR if args.quick else RESULTS_DIR
    plots_dir = results_dir / "plots"

    raw = pd.read_csv(results_dir / "raw_results.csv")
    stages = pd.read_csv(results_dir / "stages.csv")
    metrics = pd.read_csv(results_dir / "run_metrics.csv")

    summary = build_summary(raw, metrics)
    summary.to_csv(results_dir / "summary.csv", index=False)
    stage_summary = build_stage_summary(stages)
    stage_summary.to_csv(results_dir / "stage_summary.csv", index=False)
    print(f"Summaries written to {results_dir} ({len(summary)} configs, {len(stage_summary)} stage rows)")

    plot_stage_contribution(summary, stage_summary, plots_dir)
    plot_speedup_vs_workers(summary, plots_dir)
    plot_stage_speedup(stage_summary, plots_dir)
    plot_throughput(summary, plots_dir)
    plot_time_breakdown(summary, plots_dir)
    plot_cpu_utilization(summary, plots_dir)
    print(f"Plots written to {plots_dir}")


if __name__ == "__main__":
    main()
