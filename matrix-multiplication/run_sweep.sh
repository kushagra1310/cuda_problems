#!/bin/bash
# run_sweep.sh — build all kernels and run a sweep across matrix DIMENSIONS
# (not just square sizes), tagging every row with the GPU it ran on.
#
# This supersedes run_benchmarks.sh: it accepts arbitrary M,N,K triples
# (so it covers square, non-square, and non-power-of-two cases in one run)
# and adds a `gpu` column so results from different machines/architectures
# can be concatenated and compared directly.
#
# Usage:
#   ./run_sweep.sh                                   # uses DEFAULT_DIMS below
#   ./run_sweep.sh 512,512,512 3000,1500,2048 4096,4096,4096
#
# Each argument is "M,N,K" (no spaces). Output: benchmark_results.csv
# Columns: kernel,M,N,K,avg_ms,gflops,correct,gpu

set -e

NVCC_FLAGS="-O3 -std=c++17 -lcublas"
LOG="benchmark_results.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Covers: square power-of-two, square non-power-of-two, tall/wide non-square,
# and a couple of small/large extremes to probe launch-overhead vs
# compute-bound regimes.
DEFAULT_DIMS=(
  "512,512,512"
  "1024,1024,1024"
  "2048,2048,2048"
  "4096,4096,4096"

  "4096,4096,512"     # K much smaller than M,N
  "512,4096,4096"     # wide
  "4096,512,4096"     # tall
)

if [ "$#" -eq 0 ]; then
  DIMS=("${DEFAULT_DIMS[@]}")
else
  DIMS=("$@")
fi

GPU_TAG=$("$SCRIPT_DIR/gpu_info.sh")
echo "Tagging this run as GPU: $GPU_TAG"

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

if [ ! -f "$LOG" ]; then
  echo "kernel,M,N,K,avg_ms,gflops,correct,gpu" > "$LOG"
fi

for DIM in "${DIMS[@]}"; do
  IFS=',' read -r M N K <<< "$DIM"
  echo "========================================"
  echo "Matrix dims: M=$M N=$N K=$K"
  echo "========================================"

  for bin in cublas_baseline k1_naive k2_coalesced k3_shared \
    k4_1d_blocktile k5_2d_blocktile k6_vectorized k9_autotuned k10_warptiling; do
    src="${KERNELS[$bin]}"
    [ -z "$src" ] && continue
    [ ! -f "$src" ] && { echo "  [SKIP] $bin — source $src not found"; continue; }

    nvcc $NVCC_FLAGS -DM_OVERRIDE=$M -DN_OVERRIDE=$N -DK_OVERRIDE=$K \
      -o "${bin}_sz" "$src" 2>/dev/null || {
      echo "  [SKIP] $bin — compile failed for M=$M N=$N K=$K"; continue
    }

    OUTPUT=$(./"${bin}_sz" 2>&1) || {
      echo "  [SKIP] $bin — runtime error at M=$M N=$N K=$K"
      rm -f "${bin}_sz"; continue
    }

    # "Avg kernel time : 0.1234 ms"   -> number is second-to-last field
    # "Performance     : 1234.56 GFLOPS" -> same
    AVG_MS=$(echo "$OUTPUT" | grep "Avg kernel time" | awk '{print $(NF-1)}')
    GFLOPS=$(echo "$OUTPUT" | grep "Performance"     | awk '{print $(NF-1)}')

    if echo "$OUTPUT" | grep -q "PASS"; then
      CORRECT=1
    elif echo "$OUTPUT" | grep -q "FAIL"; then
      CORRECT=0
    else
      CORRECT="N/A"   # cublas_baseline has no self-check
    fi

    printf "  %-22s  M=%-5s N=%-5s K=%-5s  %8s ms  %8s GFLOPS  correct=%s\n" \
      "$bin" "$M" "$N" "$K" "$AVG_MS" "$GFLOPS" "$CORRECT"
    echo "$bin,$M,$N,$K,$AVG_MS,$GFLOPS,$CORRECT,$GPU_TAG" >> "$LOG"

    rm -f "${bin}_sz"
  done
  echo ""
done

echo "Results appended to: $LOG"
echo "Run this same script on each GPU/architecture you have access to —"
echo "results accumulate in the same CSV, tagged by the 'gpu' column."