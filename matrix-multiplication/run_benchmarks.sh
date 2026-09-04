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
)

echo "kernel,M,N,K,avg_ms,gflops,correct" > "$LOG"

# ---- compile all kernels --------------------------------------------------
echo "Compiling all kernels..."
for bin in "${!KERNELS[@]}"; do
    src="${KERNELS[$bin]}"
    echo "  nvcc $src → $bin"
    nvcc $NVCC_FLAGS -o "$bin" "$src"
done
echo "Done compiling."
echo ""

# ---- run over each size ---------------------------------------------------
for SZ in $SIZES; do
    echo "========================================"
    echo "Matrix size: ${SZ} x ${SZ} x ${SZ}"
    echo "========================================"

    for bin in cublas_baseline k1_naive k2_coalesced k3_shared \
               k4_1d_blocktile k5_2d_blocktile k6_vectorized k9_autotuned; do
        [ ! -f "$bin" ] && continue

        # Pass size via environment variables (each main() reads M,N,K from env
        # if set, else defaults to 4096). The binaries we built hardcode 4096,
        # so for the sweep we recompile with -DM= -DN= -DK= overrides.
        # Recompile per-size:
        src="${KERNELS[$bin]}"
        nvcc $NVCC_FLAGS -DM_OVERRIDE=$SZ -DN_OVERRIDE=$SZ -DK_OVERRIDE=$SZ \
             -o "${bin}_sz" "$src" 2>/dev/null || {
            echo "  [SKIP] $bin — compile failed for size $SZ"; continue
        }

        OUTPUT=$(./"${bin}_sz" 2>&1) || { echo "  [SKIP] $bin — runtime error"; continue; }

        AVG_MS=$(echo "$OUTPUT" | grep "Avg kernel time" | awk '{print $NF}' | tr -d 'ms')
        GFLOPS=$(echo "$OUTPUT" | grep "Performance"     | awk '{print $NF}')
        CORRECT=$(echo "$OUTPUT" | grep -c "PASS" 2>/dev/null || echo "N/A")

        printf "  %-22s  %8s ms  %8s GFLOPS  correct=%s\n" \
               "$bin" "$AVG_MS" "$GFLOPS" "$CORRECT"
        echo "$bin,$SZ,$SZ,$SZ,$AVG_MS,$GFLOPS,$CORRECT" >> "$LOG"

        rm -f "${bin}_sz"
    done
    echo ""
done

echo "Results saved to: $LOG"
