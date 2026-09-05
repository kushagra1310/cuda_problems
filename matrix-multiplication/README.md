# GEMM Profiling Pipeline — Assignment 1

Four scripts, run in this order, produce everything the assignment asks for.

```
gemm_profiling/
├── gpu_info.sh            # tags results with GPU name/arch
├── run_sweep.sh           # wall-clock sweep: square/non-square/odd dims, all kernels
├── profile_ncu.sh         # Nsight Compute: memory/occupancy/coalescing/bank-conflict metrics
├── param_sweep.sh         # tile-size (BM/BN/BK/TM/TN) sensitivity sweep on k9_autotuned
└── analyze_results.py     # turns all three CSVs into plots
```

Put these four scripts alongside your kernel `.cu` files and `gemm_utils.cuh`
(same directory `run_benchmarks.sh` was already in).

## Step 1 — Wall-clock sweep (on each GPU you have access to)

```bash
chmod +x gpu_info.sh run_sweep.sh profile_ncu.sh param_sweep.sh
./run_sweep.sh
```

This covers square power-of-two, square non-power-of-two (1000³), non-square
(3000×1500×2048), and two aspect-ratio extremes, for all 9 kernels. It
appends to `benchmark_results.csv` with a `gpu` column — so:

```bash
# on machine A (e.g. an A100):
./run_sweep.sh

# copy benchmark_results.csv to machine B (e.g. a 4090 or a T4), then:
./run_sweep.sh          # appends its rows to the SAME csv
# copy the merged csv back
```

That single CSV is now your multi-architecture dataset. If you only have one
GPU, that's fine — say so explicitly in the report and treat the
architecture-comparison ask as a discussion of *published* specs (SM count,
memory bandwidth, L2 size) vs your one measured curve, rather than fabricating
data for hardware you don't have.

Custom dims:
```bash
./run_sweep.sh 777,777,777 8192,8192,4096
```

## Step 2 — Nsight Compute profiling (the deep hardware metrics)

```bash
./profile_ncu.sh
```

Profiles every kernel at 3 representative sizes (edit `DEFAULT_DIMS` in the
script to match/extend what you used in Step 1 if you want the correlation
plots in Step 4 to have matching rows). This needs elevated perf-counter
permissions — if you see `ERR_NVGPUCTRPERM`, you need either root or:

```bash
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0   # then reboot
```

On a shared/cluster GPU you don't control, ask whoever admins it — this is a
common blocker and worth mentioning in the report if you hit it and had to
work around it with fewer metrics or a different machine.

This produces `ncu_results/ncu_merged.csv` with, per (kernel, size):
register count, occupancy limiters, L1/L2 hit rate, DRAM throughput,
global-load/store sectors-per-request (coalescing efficiency — closer to 1.0
is more efficient; 4.0 means every 32B thread request is pulling a full
128B/32B-aligned sector it didn't need, i.e. poor coalescing), and shared-memory
bank conflict counts for loads and stores separately.

## Step 3 — Parameter sensitivity sweep

```bash
./param_sweep.sh
```

Sweeps `BM,BN,BK,TM,TN` on `k9_autotuned` (already confirmed to use those
exact macro names) at a fixed 2048³ problem size, ~108 configs. Expect this
to take a while — each config is a full `nvcc` compile. Invalid/failing
configs (over shared-mem budget, illegal thread count) are recorded with a
`status` column rather than silently dropped — keep those in your analysis,
they're the "performance cliff" boundary the assignment wants explained.

If you also want to sweep `k10_warptiling`'s tile params, pass it explicitly:
```bash
./param_sweep.sh k10_warptiling warptiling.cu
```
(warptiling.cu would need its local `K10_BM` etc. consts turned into
`#ifndef`-guarded defines first — say the word if you want that patch.)

## Step 4 — Generate plots

```bash
pip install pandas matplotlib
python3 analyze_results.py \
  --sweep benchmark_results.csv \
  --ncu ncu_results/ncu_merged.csv \
  --params param_sweep_results.csv \
  --outdir plots
```

Produces, per GPU tag:
- `scaling_<gpu>.png` — GFLOPS vs N, log-x, one line per kernel (square dims)
- `nonsquare_<gpu>.png` — grouped bars for the non-square/odd dims
- `multi_arch_comparison.png` — if ≥2 GPUs are in the CSV
- `param_heatmap_<kernel>.png` — BM×BN heatmap of GFLOPS
- `param_tm_tn_scatter.png` — per-thread tile size sensitivity
- `hw_corr_<metric>.png` — GFLOPS vs each hardware metric (occupancy,
  hit rates, bank conflicts, coalescing ratio), colored by kernel

Any step you skip, the script just skips the plots that need it — you can
run Step 4 after Step 1 alone to sanity-check before doing the slower
profiling steps.

## Turning this into the report

The assignment explicitly penalizes a code walkthrough — structure the
report around *claims backed by a plot*, not kernel-by-kernel narration:

1. **Roofline framing** — for each kernel, is it compute-bound or
   memory-bound at each size? Use `dram__throughput.avg.pct_of_peak...` vs
   `sm__throughput.avg.pct_of_peak...` from the ncu data: if DRAM% is high
   and SM% is low, you're memory-bound; explain *why* using the specific
   optimization that kernel introduces (coalescing, SMEM caching, tiling).
2. **Where each optimization actually pays off** — pair the scaling plot
   with the coalescing-ratio and bank-conflict plots to explain *why* k2→k3
   or k5→k6 jumps happen, not just that they happen.
3. **The cliff in the param sweep** — pick 2-3 specific transitions (e.g.
   BK=8→16 at fixed BM/BN) and explain via occupancy/register pressure
   (`launch__occupancy_limit_registers`) why GFLOPS drops or jumps.
4. **Small vs large matrices** — explain the 128×128 vs 4096×4096 cuBLAS
   gap mentioned in Boehm's article in your own data: is it launch overhead,
   tile-size mismatch (your BM/BN don't divide small M/N well, wasting a
   whole block on padding), or something else? Your non-power-of-two and
   extreme-aspect-ratio rows from Step 1 are exactly for this.
5. **Multi-architecture section** — if you have ≥2 GPUs, explain
   differences via SM count, L2 size, and memory bandwidth deltas, not just
   "GPU X was faster."
6. **Honest negative results** — if a param combo or optimization made
   things worse (per Simon Boehm's own thread-swizzling anecdote in the
   source article), report it and explain why, exactly like the article does.