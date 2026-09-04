#!/bin/bash
# run_benchmarks.sh — build all kernels and run a sweep across matrix sizes.
#
# Usage:
#   ./run_benchmarks.sh              # uses default sizes
#   ./run_benchmarks.sh 512 1024 4096
#
# Output: benchmark_results.csv  (kernel, M, N, K, avg_ms, gflops, correct)

set -e

NVCC_FLAGS="-O3 -std=c++17 -lcublas"
LOG="benchmark_results.csv"

# Default sizes; override by passing args
if [ "$#" -eq 0 ]; then
SIZES="512 1024 2048 4096"
else
SIZES="$*"
fi

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

echo "kernel,M,N,K,avg_ms,gflops,correct" > "$LOG"

# ---- run over each size ---------------------------------------------------
# (Per-size builds are compiled inside the loop below via -DM_OVERRIDE etc.,
# so there's no need for a separate up-front compile pass at the default size.)

for SZ in $SIZES; do
echo "========================================"
echo "Matrix size: ${SZ} x ${SZ} x ${SZ}"
echo "========================================"

for bin in cublas_baseline k1_naive k2_coalesced k3_shared \
k4_1d_blocktile k5_2d_blocktile k6_vectorized k9_autotuned k10_warptiling; do
        src="${KERNELS[$bin]}"
        [ -z "$src" ] && continue
        [ ! -f "$src" ] && { echo "  [SKIP] $bin — source $src not found"; continue; }

# Pass size via -DM/N/K overrides and recompile per-size.
nvcc $NVCC_FLAGS -DM_OVERRIDE=$SZ -DN_OVERRIDE=$SZ -DK_OVERRIDE=$SZ \
-o "${bin}_sz" "$src" 2>/dev/null || {
echo "  [SKIP] $bin — compile failed for size $SZ"; continue
        }

OUTPUT=$(./"${bin}_sz" 2>&1) || { echo "  [SKIP] $bin — runtime error"; rm -f "${bin}_sz"; continue; }

# Field indices: "Avg kernel time : 0.1234 ms" and "Performance : 1234.56 GFLOPS"
# — the number is the second-to-last field, not the last (that's the unit).
AVG_MS=$(echo "$OUTPUT" | grep "Avg kernel time" | awk '{print $(NF-1)}')
GFLOPS=$(echo "$OUTPUT" | grep "Performance"     | awk '{print $(NF-1)}')

# grep -c always prints a count (even 0) and only fails on exit status,
# so `grep -c ... || echo N/A` doubles up output. Check match presence instead.
if echo "$OUTPUT" | grep -q "PASS"; then
    CORRECT=1
elif echo "$OUTPUT" | grep -q "FAIL"; then
    CORRECT=0
else
    CORRECT="N/A"
fi

printf "  %-22s  %8s ms  %8s GFLOPS  correct=%s\n" \
"$bin" "$AVG_MS" "$GFLOPS" "$CORRECT"
echo "$bin,$SZ,$SZ,$SZ,$AVG_MS,$GFLOPS,$CORRECT" >> "$LOG"

rm -f "${bin}_sz"
done
echo ""
done

echo "Results saved to: $LOG"