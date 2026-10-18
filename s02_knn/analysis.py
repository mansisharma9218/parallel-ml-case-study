"""Computes Speedup/Efficiency/throughput from raw times (never hand-typed),
writes summary.csv and the plots.

    python3 -m s02_knn.analysis [--quick]
"""
from . import _pin_blas  # noqa: F401
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C

KEY = ["Dataset", "Problem_Size", "Workload", "Algorithm", "Resource_Count"]
PAR = [C.ALG_QUERY, C.ALG_DATA]


def build_summary(raw, acc=None):
    raw = raw.copy()
    raw["Num_Queries"] = raw["Workload"].str.extract(r"_q(\d+)$")[0].astype(int)
    s = raw.groupby(KEY + ["Num_Queries"], as_index=False).agg(
        Median_Execution_Time=("Execution_Time", "median"),
        Std_Execution_Time=("Execution_Time", "std"),
        Median_Communication_Time=("Communication_Time", "median"),
        Median_Synchronization_Time=("Synchronization_Time", "median"),
        Median_Scheduling_Time=("Scheduling_Time", "median"),
        Median_Memory_Use=("Memory_Use", "median"),
        Num_Runs=("Run_ID", "count"))
    base = (s[s.Algorithm == C.ALG_SEQ][["Dataset", "Problem_Size", "Workload", "Median_Execution_Time"]]
            .rename(columns={"Median_Execution_Time": "_seq"}))
    s = s.merge(base, on=["Dataset", "Problem_Size", "Workload"], how="left")
    s["Speedup"] = s["_seq"] / s["Median_Execution_Time"]
    s["Efficiency"] = s["Speedup"] / s["Resource_Count"]
    s["Queries_Per_Sec"] = s["Num_Queries"] / s["Median_Execution_Time"]
    if acc is not None:
        a = acc.groupby(KEY, as_index=False).agg(Mean_Accuracy=("Accuracy", "mean"))
        s = s.merge(a, on=KEY, how="left")
    return s.drop(columns="_seq")


def _save(fig, path):
    fig.tight_layout()
    for ext in ("png", "svg"):  # PNG for viewing, SVG as the editable form
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _series(d, algo, wmax):
    return d[(d.Algorithm == algo) & (d.Resource_Count == (1 if algo == C.ALG_SEQ else wmax))]


def make_plots(s, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ds, d in s.groupby("Dataset"):
        qmax, nmax, wmax = d.Num_Queries.max(), d.Problem_Size.max(), d.Resource_Count.max()

        # 1. time vs dataset size
        sub = d[d.Num_Queries == qmax]
        fig, ax = plt.subplots()
        for algo in [C.ALG_SEQ] + PAR:
            x = _series(sub, algo, wmax).sort_values("Problem_Size")
            lab = algo if algo == C.ALG_SEQ else f"{algo} ({wmax} workers)"
            ax.errorbar(x.Problem_Size, x.Median_Execution_Time, yerr=x.Std_Execution_Time.fillna(0),
                        marker="o", capsize=3, label=lab)
        ax.set(xlabel="Training set size", ylabel="Median search time (s)",
               title=f"Time vs dataset size ({ds}, {qmax} queries)")
        ax.legend(); ax.grid(alpha=.3)
        _save(fig, plots_dir / f"time_vs_dataset_size_{ds}_q{qmax}.png")

        # 2/3. throughput and speedup vs workers (largest training set, most queries)
        sub = d[(d.Problem_Size == nmax) & (d.Num_Queries == qmax)]
        for col, ylab, fname in (("Queries_Per_Sec", "Queries / sec", "throughput_vs_workers"),
                                 ("Speedup", "Speedup (T_seq / T_p)", "speedup_vs_workers")):
            fig, ax = plt.subplots()
            for algo in PAR:
                x = sub[sub.Algorithm == algo].sort_values("Resource_Count")
                ax.plot(x.Resource_Count, x[col], marker="o", label=algo)
            if col == "Speedup":
                ax.plot([1, wmax], [1, wmax], "k--", label="ideal")
            ax.set(xlabel="Workers", ylabel=ylab, title=f"{ylab} vs workers ({ds}, n={nmax}, {qmax} queries)")
            ax.legend(); ax.grid(alpha=.3)
            _save(fig, plots_dir / f"{fname}_{ds}_n{nmax}_q{qmax}.png")

        # 3b. same metrics, one line per training size (answers the research question)
        sub = d[d.Num_Queries == qmax]
        for col, ylab, fname in (("Speedup", "Speedup (T_seq / T_p)", "speedup_vs_workers_by_size"),
                                 ("Queries_Per_Sec", "Queries / sec", "throughput_vs_workers_by_size")):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
            for ax, algo in zip(axes, PAR):
                for n in sorted(sub.Problem_Size.unique()):
                    x = sub[(sub.Algorithm == algo) & (sub.Problem_Size == n)].sort_values("Resource_Count")
                    ax.plot(x.Resource_Count, x[col], marker="o", label=f"n={n}")
                if col == "Speedup":
                    ax.plot([1, wmax], [1, wmax], "k--", label="ideal")
                ax.set(xlabel="Workers", title=algo)
                ax.grid(alpha=.3)
            axes[0].set_ylabel(ylab)
            axes[0].legend(fontsize=8)
            fig.suptitle(f"{ylab} vs workers by training size ({ds}, {qmax} queries)")
            _save(fig, plots_dir / f"{fname}_{ds}_q{qmax}.png")

        # 4. speedup vs dataset size (the research question)
        sub = d[d.Num_Queries == qmax]
        fig, ax = plt.subplots()
        for algo in PAR:
            x = _series(sub, algo, wmax).sort_values("Problem_Size")
            ax.plot(x.Problem_Size, x.Speedup, marker="o", label=algo)
        ax.set(xlabel="Training set size", ylabel=f"Speedup at {wmax} workers",
               title=f"Speedup vs dataset size ({ds}, {qmax} queries)")
        ax.legend(); ax.grid(alpha=.3)
        _save(fig, plots_dir / f"speedup_vs_dataset_size_{ds}_q{qmax}.png")

        # 5. time vs number of queries
        sub = d[d.Problem_Size == nmax]
        fig, ax = plt.subplots()
        for algo in [C.ALG_SEQ] + PAR:
            x = _series(sub, algo, wmax).sort_values("Num_Queries")
            lab = algo if algo == C.ALG_SEQ else f"{algo} ({wmax} workers)"
            ax.plot(x.Num_Queries, x.Median_Execution_Time, marker="o", label=lab)
        ax.set(xlabel="Number of queries", ylabel="Median search time (s)",
               title=f"Time vs query count ({ds}, n={nmax})")
        ax.legend(); ax.grid(alpha=.3)
        _save(fig, plots_dir / f"time_vs_queries_{ds}_n{nmax}.png")

        # 6. stacked time breakdown
        sub = d[(d.Problem_Size == nmax) & (d.Num_Queries == qmax) & (d.Algorithm.isin(PAR))]
        sub = sub.sort_values(["Algorithm", "Resource_Count"])
        comm, sync = sub.Median_Communication_Time, sub.Median_Synchronization_Time
        comp = (sub.Median_Execution_Time - comm - sync).clip(lower=0)
        labels = [f"{a.replace('kNN-Parallel-', '')[:1]}{p}" for a, p in zip(sub.Algorithm, sub.Resource_Count)]
        fig, ax = plt.subplots()
        ax.bar(labels, comp, label="compute")
        ax.bar(labels, comm, bottom=comp, label="communication (IPC/dispatch)")
        ax.bar(labels, sync, bottom=comp + comm, label="synchronization (merge)")
        ax.bar(labels, sub.Median_Scheduling_Time, bottom=comp + comm + sync, hatch="//",
               alpha=.5, label="scheduling (pool start-up, one-off)")
        ax.set(xlabel="Q=query-parallel, D=data-parallel; number = workers", ylabel="Median time (s)",
               title=f"Time breakdown ({ds}, n={nmax}, {qmax} queries)")
        ax.legend(fontsize=8)
        _save(fig, plots_dir / f"time_breakdown_{ds}_n{nmax}_q{qmax}.png")

        # 7. accuracy (kept separate from runtime)
        if "Mean_Accuracy" in d:
            sub = d[(d.Num_Queries == qmax) & (d.Algorithm == C.ALG_SEQ)].sort_values("Problem_Size")
            fig, ax = plt.subplots()
            ax.plot(sub.Problem_Size, sub.Mean_Accuracy, marker="o")
            ax.set(xlabel="Training set size", ylabel="Accuracy", title=f"k-NN accuracy ({ds}, k={C.K_NEIGHBORS})")
            ax.grid(alpha=.3)
            _save(fig, plots_dir / f"accuracy_vs_dataset_size_{ds}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    d = C.QUICK_RESULTS_DIR if a.quick else C.RESULTS_DIR
    raw = pd.read_csv(d / "raw_results.csv")
    acc = pd.read_csv(d / "accuracy.csv") if (d / "accuracy.csv").exists() else None
    s = build_summary(raw, acc)
    s.to_csv(d / "summary.csv", index=False)
    make_plots(s, d / "plots")
    print(f"wrote {d}/summary.csv and plots in {d}/plots")


if __name__ == "__main__":
    main()