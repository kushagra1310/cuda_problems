#!/usr/bin/env python3
"""
analyze_results.py — turn the CSVs from run_sweep.sh, profile_ncu.sh, and
param_sweep.sh into report-ready plots.

Design principle for this version: every plot answers "is this good or bad?"
at a glance, via one of:
  - a %-of-cuBLAS axis (100% = matched cuBLAS; that's the reference line)
  - a fixed color per kernel, reused across every plot, so you can track
    one kernel without re-reading the legend each time
  - a leaderboard ranking (one plot, no per-size digging required)

Usage:
    python3 analyze_results.py --sweep benchmark_results.csv \
                                --ncu ncu_results/ncu_merged.csv \
                                --params param_sweep_results.csv \
                                --outdir plots

Any of --sweep / --ncu / --params can be omitted if you haven't generated
that CSV yet. Requires: pandas, matplotlib (pip install pandas matplotlib)
"""
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------
# Fixed kernel -> color mapping, used across EVERY plot in this script.
# cuBLAS is always black-dashed since it's the reference, not a competitor.
# ---------------------------------------------------------------------
KERNEL_ORDER = [
    "cublas_baseline", "k1_naive", "k2_coalesced", "k3_shared",
    "k4_1d_blocktile", "k5_2d_blocktile", "k6_vectorized",
    "k9_autotuned", "k10_warptiling",
]
# Explicit high-contrast palette (tab10 with its gray/olive slots swapped
# out) so no kernel accidentally lands on a "muted/disabled"-looking color.
_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
KERNEL_COLORS = {"cublas_baseline": "black"}
_color_i = 0
for k in KERNEL_ORDER:
    if k == "cublas_baseline":
        continue
    KERNEL_COLORS[k] = _palette[_color_i % len(_palette)]
    _color_i += 1


def color_for(kernel):
    if kernel in KERNEL_COLORS:
        return KERNEL_COLORS[kernel]
    # unseen kernel name (e.g. k7/k8 added later) - assign deterministically
    return _palette[hash(kernel) % len(_palette)]


def sorted_kernels(names):
    known = [k for k in KERNEL_ORDER if k in names]
    unknown = sorted(n for n in names if n not in KERNEL_ORDER)
    return known + unknown


def safe(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))[:60]


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
# 0. Add a %-of-cuBLAS column — THE key reference metric for "good/bad".
# ---------------------------------------------------------------------
def add_pct_of_cublas(df):
    df = clean_numeric(df.copy(), ["M", "N", "K", "gflops"])
    baseline = (df[df.kernel == "cublas_baseline"]
                .set_index(["M", "N", "K", "gpu"])["gflops"])
    def lookup(row):
        try:
            b = baseline.loc[(row.M, row.N, row.K, row.gpu)]
            return 100.0 * row.gflops / b if b and pd.notna(b) else float("nan")
        except KeyError:
            return float("nan")
    df["pct_of_cublas"] = df.apply(lookup, axis=1)
    return df


# ---------------------------------------------------------------------
# 1. Leaderboard — ONE plot, average %-of-cuBLAS per kernel across all
#    square sizes. This is the "tell me who's good" plot.
# ---------------------------------------------------------------------
def plot_leaderboard(df, outdir):
    df = df.copy()
    df["is_square"] = (df["M"] == df["N"]) & (df["N"] == df["K"])
    sq = df[df["is_square"] & (df["kernel"] != "cublas_baseline")]
    if sq.empty:
        print("[skip] leaderboard: no square-dim rows to rank")
        return
    for gpu, gdf in sq.groupby("gpu"):
        agg = (gdf.groupby("kernel")["pct_of_cublas"]
               .mean().dropna().sort_values())
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(agg) + 2))
        colors = [color_for(k) for k in agg.index]
        bars = ax.barh(agg.index, agg.values, color=colors)
        ax.axvline(100, color="black", linestyle="--", linewidth=1.5,
                   label="cuBLAS (100%)")
        for bar, val in zip(bars, agg.values):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", fontsize=9)
        ax.set_xlabel("% of cuBLAS performance (avg over square sizes)")
        ax.set_title(f"Kernel leaderboard — {gpu}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fname = os.path.join(outdir, f"leaderboard_{safe(gpu)}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")


# ---------------------------------------------------------------------
# 2. %-of-cuBLAS vs size — the scaling plot, but on the good/bad axis
#    instead of raw GFLOPS. 100% line is the visual anchor.
# ---------------------------------------------------------------------
def plot_pct_scaling(df, outdir):
    df = df.copy()
    df["is_square"] = (df["M"] == df["N"]) & (df["N"] == df["K"])
    sq = df[df["is_square"] & (df["kernel"] != "cublas_baseline")].sort_values("M")

    for gpu, gdf in sq.groupby("gpu"):
        if gdf.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.axhline(100, color="black", linestyle="--", linewidth=1.5, zorder=1,
                   label="cuBLAS (100%)")
        for kernel in sorted_kernels(gdf["kernel"].unique()):
            kdf = gdf[gdf.kernel == kernel].dropna(subset=["pct_of_cublas"])
            if kdf.empty:
                continue
            ax.plot(kdf["M"], kdf["pct_of_cublas"], marker="o",
                    color=color_for(kernel), label=kernel, linewidth=2)
            last = kdf.iloc[-1]
            ax.annotate(f"{last.pct_of_cublas:.0f}%",
                        (last.M, last.pct_of_cublas),
                        textcoords="offset points", xytext=(6, 0), fontsize=8,
                        color=color_for(kernel))
        ax.set_xlabel("Matrix dimension N (square NxNxN)")
        ax.set_ylabel("% of cuBLAS performance")
        ax.set_xscale("log", base=2)
        ax.set_title(f"Kernel performance relative to cuBLAS — {gpu}\n"
                     f"(above the dashed line = faster than cuBLAS, below = slower)")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = os.path.join(outdir, f"pct_of_cublas_{safe(gpu)}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")


# ---------------------------------------------------------------------
# 3. Raw GFLOPS scaling (kept, but now color-consistent + annotated) —
#    useful alongside the %-plot since absolute throughput still matters.
# ---------------------------------------------------------------------
def plot_scaling(df, outdir):
    df = clean_numeric(df.copy(), ["M", "N", "K", "avg_ms", "gflops"])
    df = df.dropna(subset=["gflops"])
    df["dim_label"] = df.apply(lambda r: f"{int(r.M)}x{int(r.N)}x{int(r.K)}", axis=1)
    df["is_square"] = (df["M"] == df["N"]) & (df["N"] == df["K"])

    for gpu, gdf in df.groupby("gpu"):
        square = gdf[gdf["is_square"]].sort_values("M")
        if not square.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            for kernel in sorted_kernels(square["kernel"].unique()):
                kdf = square[square.kernel == kernel]
                style = "--" if kernel == "cublas_baseline" else "-"
                width = 2.5 if kernel == "cublas_baseline" else 1.8
                ax.plot(kdf["M"], kdf["gflops"], style, marker="o",
                        color=color_for(kernel), label=kernel, linewidth=width)
                last = kdf.iloc[-1]
                ax.annotate(f"{last.gflops:.0f}", (last.M, last.gflops),
                            textcoords="offset points", xytext=(6, 0),
                            fontsize=7, color=color_for(kernel))
            ax.set_xlabel("Matrix dimension N (square NxNxN)")
            ax.set_ylabel("GFLOPS")
            ax.set_xscale("log", base=2)
            ax.set_title(f"Absolute throughput — {gpu}")
            ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fname = os.path.join(outdir, f"scaling_{safe(gpu)}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")

        nonsq = gdf[~gdf["is_square"]]
        if not nonsq.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            pivot = nonsq.pivot_table(index="dim_label", columns="kernel",
                                       values="gflops", aggfunc="mean")
            pivot = pivot[[k for k in sorted_kernels(pivot.columns) if k in pivot.columns]]
            colors = [color_for(k) for k in pivot.columns]
            pivot.plot(kind="bar", ax=ax, color=colors)
            ax.set_ylabel("GFLOPS")
            ax.set_title(f"Non-square / non-power-of-two dims — {gpu}")
            ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
            ax.grid(alpha=0.3, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            fig.tight_layout()
            fname = os.path.join(outdir, f"nonsquare_{safe(gpu)}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


# ---------------------------------------------------------------------
# 4. Multi-architecture comparison, normalized to each GPU's own cuBLAS
#    (fair comparison across GPUs with very different raw FLOPS ceilings)
# ---------------------------------------------------------------------
def plot_multi_arch(df, outdir, target_dim=2048):
    subset = df[(df.M == target_dim) & (df.N == target_dim) & (df.K == target_dim)
                & (df.kernel != "cublas_baseline")]
    if subset["gpu"].nunique() < 2:
        print(f"[skip] multi-arch plot: only {subset['gpu'].nunique()} GPU(s) in data")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot = subset.pivot_table(index="kernel", columns="gpu",
                                values="pct_of_cublas", aggfunc="mean")
    pivot = pivot.reindex([k for k in sorted_kernels(pivot.index) if k in pivot.index])
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(100, color="black", linestyle="--", linewidth=1.5)
    ax.set_ylabel("% of that GPU's own cuBLAS performance")
    ax.set_title(f"Architecture comparison at {target_dim}³ (normalized per-GPU)")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fname = os.path.join(outdir, "multi_arch_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


# ---------------------------------------------------------------------
# 5. Tile-parameter sensitivity heatmaps — best config starred, colorbar
#    fixed to [0, max] so color intensity is comparable across kernels.
# ---------------------------------------------------------------------
def plot_param_sensitivity(df, outdir):
    df = df[df["status"] == "ok"].copy()
    df = clean_numeric(df, ["BM", "BN", "BK", "TM", "TN", "gflops"])
    df = df.dropna(subset=["gflops"])
    if df.empty:
        print("[skip] param sensitivity: no successful configs")
        return

    for kernel, kdf in df.groupby("kernel"):
        pivot = kdf.pivot_table(index="BM", columns="BN", values="gflops", aggfunc="mean")
        valid_vals = pivot.values[~pd.isna(pivot.values)]
        # Scale the colormap to the ACTUAL spread in this data, not [0, max] —
        # tile-size differences are typically a much smaller % swing than the
        # absolute GFLOPS value, so anchoring at 0 washes out all contrast.
        vmin, vmax = valid_vals.min(), valid_vals.max()
        if vmin == vmax:
            vmin, vmax = vmin * 0.95, vmax * 1.05
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        ax.set_xlabel("BN"); ax.set_ylabel("BM")
        ax.set_title(f"{kernel}: GFLOPS by block tile\n(green=faster, red=slower; avg over BK/TM/TN)")
        best_i, best_j = None, -1
        best_val = -1
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="black", fontsize=9)
                    if v > best_val:
                        best_val, best_i, best_j = v, i, j
        if best_i is not None:
            ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                       fill=False, edgecolor="blue", linewidth=3))
            ax.text(0.02, -0.12, f"Best: BM={pivot.index[best_i]}, BN={pivot.columns[best_j]} "
                    f"-> {best_val:.0f} GFLOPS (blue box)",
                    transform=ax.transAxes, fontsize=8)
        fig.colorbar(im, ax=ax, label="GFLOPS")
        fig.tight_layout()
        fname = os.path.join(outdir, f"param_heatmap_{safe(kernel)}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")

    fig, ax = plt.subplots(figsize=(7, 5))
    tmtn_groups = sorted(df.groupby(["TM", "TN"]).groups.keys())
    positions = {g: i for i, g in enumerate(tmtn_groups)}
    for (tm, tn), g in df.groupby(["TM", "TN"]):
        x = [positions[(tm, tn)]] * len(g)
        ax.scatter(x, g["gflops"], alpha=0.6, color="steelblue")
    means = df.groupby(["TM", "TN"])["gflops"].mean()
    ax.plot(range(len(tmtn_groups)), [means[g] for g in tmtn_groups],
            "D-", color="darkred", label="mean", zorder=5)
    ax.set_xticks(range(len(tmtn_groups)))
    ax.set_xticklabels([f"{int(tm)}x{int(tn)}" for tm, tn in tmtn_groups])
    ax.set_xlabel("TM x TN (per-thread work)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Per-thread tile size vs achieved GFLOPS (all BM/BN/BK)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(outdir, "param_tm_tn_scatter.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


# ---------------------------------------------------------------------
# 6. Hardware metrics vs GFLOPS (needs both --sweep and --ncu)
# ---------------------------------------------------------------------
def plot_hw_correlation(sweep_df, ncu_df, outdir):
    merge_cols = ["kernel", "M", "N", "K"]
    sweep_df = sweep_df.copy(); ncu_df = ncu_df.copy()
    for c in ["M", "N", "K"]:
        sweep_df[c] = pd.to_numeric(sweep_df[c], errors="coerce")
        ncu_df[c] = pd.to_numeric(ncu_df[c], errors="coerce")
    merged = pd.merge(sweep_df, ncu_df, on=merge_cols, how="inner")
    if merged.empty:
        print("[skip] hw correlation: no matching (kernel,M,N,K) rows")
        return

    candidate_metrics = [c for c in merged.columns if c not in
                          ["kernel", "M", "N", "K", "avg_ms", "gflops", "correct",
                           "gpu", "pct_of_cublas", "is_square", "dim_label"]]
    interesting = [c for c in candidate_metrics if any(
        k in c for k in ["hit_rate", "occupancy", "registers", "bank_conflicts",
                          "sectors_per_request", "throughput"])]

    for metric in interesting:
        merged[metric] = pd.to_numeric(merged[metric], errors="coerce")
        sub = merged.dropna(subset=[metric, "gflops"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for kernel in sorted_kernels(sub["kernel"].unique()):
            kdf = sub[sub.kernel == kernel]
            ax.scatter(kdf[metric], kdf["gflops"], label=kernel,
                       color=color_for(kernel), alpha=0.8, s=60)
        ax.set_xlabel(metric)
        ax.set_ylabel("GFLOPS")
        ax.set_title(f"GFLOPS vs {metric}")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = os.path.join(outdir, f"hw_corr_{safe(metric)}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {fname}")


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
        sweep_df = add_pct_of_cublas(sweep_df)
        print("\n-- leaderboard (start here) --")
        plot_leaderboard(sweep_df, args.outdir)
        print("\n-- %-of-cuBLAS scaling --")
        plot_pct_scaling(sweep_df, args.outdir)
        print("\n-- absolute GFLOPS scaling --")
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
    print("Start with leaderboard_*.png, then pct_of_cublas_*.png for the "
          "'is this good or bad' view; scaling_*.png has the raw numbers.")


if __name__ == "__main__":
    main()