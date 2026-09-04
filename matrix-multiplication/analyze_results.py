#!/usr/bin/env python3
"""
analyze_results.py — turn the CSVs produced by run_sweep.sh, profile_ncu.sh,
and param_sweep.sh into the plots the assignment asks for.

Usage:
    python3 analyze_results.py --sweep benchmark_results.csv \
                                --ncu ncu_results/ncu_merged.csv \
                                --params param_sweep_results.csv \
                                --outdir plots

Any of --sweep / --ncu / --params can be omitted if you haven't generated
that CSV yet — the script skips the plots that need it and tells you so,
rather than failing.

Requires: pandas, matplotlib  (pip install pandas matplotlib)
"""
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path, label):
    if not path or not os.path.exists(path):
        print(f"[skip] {label}: no file at {path!r}")
        return None
    df = pd.read_csv(path)
    print(f"[ok]   {label}: {len(df)} rows from {path}")
    return df


def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------
# 1. GFLOPS vs matrix size, one line per kernel, faceted by GPU
# ---------------------------------------------------------------------
def plot_scaling(df, outdir):
    df = clean_numeric(df.copy(), ["M", "N", "K", "avg_ms", "gflops"])
    df = df.dropna(subset=["gflops"])
    df["dim_label"] = df.apply(lambda r: f"{int(r.M)}x{int(r.N)}x{int(r.K)}", axis=1)
    df["is_square"] = df["M"] == df["N"]
    df["is_square"] = df["is_square"] & (df["N"] == df["K"])

    for gpu, gdf in df.groupby("gpu"):
        square = gdf[gdf["is_square"]].sort_values("M")
        if square.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for kernel, kdf in square.groupby("kernel"):
            ax.plot(kdf["M"], kdf["gflops"], marker="o", label=kernel)
        ax.set_xlabel("Matrix dimension N (square NxNxN)")
        ax.set_ylabel("GFLOPS")
        ax.set_xscale("log", base=2)
        ax.set_title(f"Kernel scaling on {gpu}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fname = os.path.join(outdir, f"scaling_{safe(gpu)}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")

        # non-square / edge-case dims as grouped bars
        nonsq = gdf[~gdf["is_square"]]
        if not nonsq.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot = nonsq.pivot_table(index="dim_label", columns="kernel",
                                       values="gflops", aggfunc="mean")
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("GFLOPS")
            ax.set_title(f"Non-square / non-power-of-two dims on {gpu}")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fname = os.path.join(outdir, f"nonsquare_{safe(gpu)}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


# ---------------------------------------------------------------------
# 2. Multi-architecture comparison at a fixed size
# ---------------------------------------------------------------------
def plot_multi_arch(df, outdir, target_dim=2048):
    df = clean_numeric(df.copy(), ["M", "N", "K", "gflops"])
    subset = df[(df.M == target_dim) & (df.N == target_dim) & (df.K == target_dim)]
    if subset["gpu"].nunique() < 2:
        print(f"[skip] multi-arch plot: only {subset['gpu'].nunique()} GPU(s) in data "
              f"— run run_sweep.sh on more than one architecture to populate this.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = subset.pivot_table(index="kernel", columns="gpu", values="gflops", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"Architecture comparison at {target_dim}x{target_dim}x{target_dim}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fname = os.path.join(outdir, "multi_arch_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


# ---------------------------------------------------------------------
# 3. Tile-parameter sensitivity heatmaps
# ---------------------------------------------------------------------
def plot_param_sensitivity(df, outdir):
    df = df[df["status"] == "ok"].copy()
    df = clean_numeric(df, ["BM", "BN", "BK", "TM", "TN", "gflops"])
    df = df.dropna(subset=["gflops"])
    if df.empty:
        print("[skip] param sensitivity: no successful configs in param_sweep CSV")
        return

    # Heatmap of BM x BN, averaged over BK/TM/TN, per kernel
    for kernel, kdf in df.groupby("kernel"):
        pivot = kdf.pivot_table(index="BM", columns="BN", values="gflops", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("BN")
        ax.set_ylabel("BM")
        ax.set_title(f"{kernel}: GFLOPS by block tile (avg over BK/TM/TN)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="GFLOPS")
        fig.tight_layout()
        fname = os.path.join(outdir, f"param_heatmap_{safe(kernel)}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")

    # TM x TN sensitivity (register pressure axis), averaged over block dims
    fig, ax = plt.subplots(figsize=(7, 5))
    for (tm, tn), g in df.groupby(["TM", "TN"]):
        ax.scatter([f"{int(tm)}x{int(tn)}"] * len(g), g["gflops"], alpha=0.5)
    ax.set_xlabel("TM x TN (per-thread work)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Per-thread tile size vs achieved GFLOPS (all BM/BN/BK)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(outdir, "param_tm_tn_scatter.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")

    # Failure-rate table (compile_failed / runtime_failed / skipped) — useful
    # for discussing "performance cliffs" that are really "invalid regions"
    print("\nParameter sweep status breakdown:")
    print(pd.read_csv(sys.argv[sys.argv.index("--params") + 1])["status"]
          .value_counts().to_string())


# ---------------------------------------------------------------------
# 4. Hardware metrics vs achieved GFLOPS (needs both --sweep and --ncu)
# ---------------------------------------------------------------------
def plot_hw_correlation(sweep_df, ncu_df, outdir):
    merge_cols = ["kernel", "M", "N", "K"]
    for c in merge_cols:
        sweep_df[c] = pd.to_numeric(sweep_df[c], errors="coerce") if c != "kernel" else sweep_df[c]
        ncu_df[c] = pd.to_numeric(ncu_df[c], errors="coerce") if c != "kernel" else ncu_df[c]
    merged = pd.merge(sweep_df, ncu_df, on=merge_cols, how="inner")
    if merged.empty:
        print("[skip] hw correlation: no matching (kernel,M,N,K) rows between "
              "sweep and ncu CSVs — make sure profile_ncu.sh used the same dims.")
        return

    candidate_metrics = [c for c in merged.columns if c not in
                          ["kernel", "M", "N", "K", "avg_ms", "gflops", "correct", "gpu"]]
    interesting = [c for c in candidate_metrics if any(
        k in c for k in ["hit_rate", "occupancy", "registers", "bank_conflicts",
                          "sectors_per_request", "throughput"])]

    for metric in interesting:
        merged[metric] = pd.to_numeric(merged[metric], errors="coerce")
        sub = merged.dropna(subset=[metric, "gflops"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for kernel, kdf in sub.groupby("kernel"):
            ax.scatter(kdf[metric], kdf["gflops"], label=kernel, alpha=0.7)
        ax.set_xlabel(metric)
        ax.set_ylabel("GFLOPS")
        ax.set_title(f"GFLOPS vs {metric}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = os.path.join(outdir, f"hw_corr_{safe(metric)}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")


def safe(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="benchmark_results.csv")
    ap.add_argument("--ncu", default="ncu_results/ncu_merged.csv")
    ap.add_argument("--params", default="param_sweep_results.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sweep_df = load_csv(args.sweep, "sweep")
    ncu_df = load_csv(args.ncu, "ncu")
    param_df = load_csv(args.params, "param sweep")

    if sweep_df is not None:
        print("\n-- scaling plots --")
        plot_scaling(sweep_df, args.outdir)
        print("\n-- multi-arch comparison --")
        plot_multi_arch(sweep_df, args.outdir)

    if param_df is not None:
        print("\n-- parameter sensitivity --")
        plot_param_sensitivity(param_df, args.outdir)

    if sweep_df is not None and ncu_df is not None:
        print("\n-- hardware-metric correlation --")
        plot_hw_correlation(sweep_df, ncu_df, args.outdir)

    print(f"\nAll plots written to: {args.outdir}/")


if __name__ == "__main__":
    main()