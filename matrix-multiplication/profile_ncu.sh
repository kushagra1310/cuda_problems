#!/bin/bash
# profile_ncu.sh — low-level Nsight Compute profiling of each kernel at a
# small set of REPRESENTATIVE sizes (full ncu profiling is slow — this is
# not meant to replace run_sweep.sh's wall-clock sweep, it's for the deep
# hardware-metric side of the assignment).
#
# Requires: ncu (NVIDIA Nsight Compute CLI) on PATH, and either root or
# `sudo sysctl -w kernel.perf_event_paranoid=0` / an admin-granted
# NVreg_RestrictProfilingToAdminUsers=0 driver setting — Nsight Compute
# needs elevated perf-counter access on most systems. If `ncu` reports
# "ERR_NVGPUCTRPERM", that's the permission issue, not a script bug.
#
# Usage:
#   ./profile_ncu.sh                       # default kernels + sizes below
#   ./profile_ncu.sh k5_2d_blocktile k10_warptiling
#
# Output: ncu_results/<kernel>_<M>x<N>x<K>.csv  (one per kernel/size)
#         ncu_results/ncu_merged.csv            (combined, for analyze_results.py)

set -e

NVCC_FLAGS="-O3 -std=c++17 -lcublas"
OUTDIR="ncu_results"
mkdir -p "$OUTDIR"

declare -A KERNELS=(
  ["cublas_baseline"]="cublas_baseline.cu"
  ["k1_naive"]="naive.cu"
  ["k2_coalesced"]="glob-memory-coalescing.cu"
  ["k3_shared"]="shared-memory.cu"
  ["k4_1d_blocktile"]="shared-memory-multiple.cu"
  ["k5_2d_blocktile"]="shared-memory-multiple-2.cu"
  ["k6_vectorized"]="vectorized-access.cu"
  ["k9_autotuned"]="autotuning.cu"
  ["k10_warptiling"]="warptiling.cu"
)

# Keep this list short — each entry multiplies total profiling time.
# Pick sizes that let you talk about small-vs-large and square-vs-non-square
# behavior without profiling the entire wall-clock sweep.
DEFAULT_DIMS=("1024,1024,1024" "3000,1500,2048" "4096,4096,4096")

if [ "$#" -eq 0 ]; then
  BINS=(k1_naive k2_coalesced k3_shared k4_1d_blocktile k5_2d_blocktile k6_vectorized k9_autotuned k10_warptiling)
else
  BINS=("$@")
fi

# Metric set: memory hierarchy (L1/L2/DRAM), register pressure & occupancy,
# global-load/store coalescing efficiency, and shared-memory bank conflicts.
# This is deliberately a --metrics list (fast) rather than --set full
# (thorough but ~10-50x slower per launch) — switch to --set full on one or
# two runs if you want the richest possible report source, see note below.
METRICS="launch__registers_per_thread,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum.per_second,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__sass_average_branch_targets_threads_uniform.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

echo "kernel,M,N,K,$(echo $METRICS | tr ',' ',')" > "$OUTDIR/ncu_merged.csv"

for bin in "${BINS[@]}"; do
  src="${KERNELS[$bin]}"
  [ -z "$src" ] && { echo "  [SKIP] unknown kernel $bin"; continue; }
  [ ! -f "$src" ] && { echo "  [SKIP] $bin — source $src not found"; continue; }

  for DIM in "${DEFAULT_DIMS[@]}"; do
    IFS=',' read -r M N K <<< "$DIM"
    echo "Profiling $bin @ ${M}x${N}x${K} ..."

    nvcc $NVCC_FLAGS -DM_OVERRIDE=$M -DN_OVERRIDE=$N -DK_OVERRIDE=$K \
      -o "${bin}_ncu" "$src" 2>/dev/null || {
      echo "  [SKIP] compile failed"; continue
    }

    RAW_CSV="$OUTDIR/${bin}_${M}x${N}x${K}.csv"

    # --launch-skip 5: our benchmark_ms() does 5 warmup launches before the
    #   timed loop — skip those so we profile a steady-state launch, not a
    #   cold one still paying for cache/TLB warmup.
    # --launch-count 1: profiling every launch is extremely slow and
    #   unnecessary — kernel behavior is deterministic across launches here.
    ncu --metrics "$METRICS" \
        --csv --page raw \
        --launch-skip 5 --launch-count 1 \
        --target-processes all \
        ./"${bin}_ncu" > "$RAW_CSV" 2>"$OUTDIR/${bin}_${M}x${N}x${K}.log" || {
      echo "  [WARN] ncu failed for $bin @ ${M}x${N}x${K} — see ${bin}_${M}x${N}x${K}.log"
      echo "         (often ERR_NVGPUCTRPERM — see permission note at top of this script)"
      rm -f "${bin}_ncu"
      continue
    }

    # ncu's raw CSV is metric-per-row; pivot to one row of metric:value pairs
    # tagged with kernel/M/N/K, appended to the merged file for pandas.
    python3 - "$RAW_CSV" "$bin" "$M" "$N" "$K" "$OUTDIR/ncu_merged.csv" <<'PYEOF'
import csv, sys
raw_path, kernel, M, N, K, merged_path = sys.argv[1:7]
metrics = {}
with open(raw_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get("Metric Name")
        val = row.get("Metric Value")
        if name:
            metrics[name] = val
header = ["kernel", "M", "N", "K"] + list(metrics.keys())
write_header = False
try:
    with open(merged_path) as f:
        existing_header = f.readline().strip().split(",")
    write_header = existing_header != header
except FileNotFoundError:
    write_header = True
mode = "a"
with open(merged_path, mode) as f:
    if write_header:
        f.write(",".join(header) + "\n")
    row = [kernel, M, N, K] + [str(metrics[k]) for k in metrics]
    f.write(",".join(row) + "\n")
PYEOF

    rm -f "${bin}_ncu"
  done
done

echo "Per-kernel raw CSVs + merged summary written to: $OUTDIR/"
echo "Feed $OUTDIR/ncu_merged.csv into analyze_results.py --ncu"