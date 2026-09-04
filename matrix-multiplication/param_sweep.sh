#!/bin/bash
# param_sweep.sh
#
# Parameter sensitivity analysis for the warp-tiled GEMM kernel.
#
# Sweeps:
#   BM, BN, BK, TM, TN
#
# The script validates the structural constraints imposed by the
# warp-tiled kernel before compiling/running a configuration.
#
# Usage:
#   ./param_sweep.sh
#   ./param_sweep.sh k10_warptiling warptiling.cu

set -u

NVCC_FLAGS="-O3 -std=c++17 -lcublas"
LOG="param_sweep_results.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN_NAME="${1:-k10_warptiling}"
SRC="${2:-warptiling.cu}"

# ------------------------------------------------------------
# Fixed problem size
# ------------------------------------------------------------
M=2048
N=2048
K=2048

GPU_TAG=$("$SCRIPT_DIR/gpu_info.sh")

# ------------------------------------------------------------
# Parameter grid
# ------------------------------------------------------------
BM_VALS=(64 128 256)
BN_VALS=(64 128 256)
BK_VALS=(8 16 32)
TM_VALS=(4 8)
TN_VALS=(4 8)

echo "kernel,BM,BN,BK,TM,TN,M,N,K,threads,smem_bytes,avg_ms,gflops,correct,status,gpu" > "$LOG"


# ============================================================
# Validate a parameter configuration
# ============================================================
validate_config() {

    local BM=$1
    local BN=$2
    local BK=$3
    local TM=$4
    local TN=$5

    # --------------------------------------------------------
    # Basic divisibility requirements
    # --------------------------------------------------------

    if (( BM % TM != 0 )); then
        echo "BM_not_divisible_by_TM"
        return 1
    fi

    if (( BN % TN != 0 )); then
        echo "BN_not_divisible_by_TN"
        return 1
    fi

    # float4 access along K for A
    if (( BK % 4 != 0 )); then
        echo "BK_not_divisible_by_4"
        return 1
    fi

    # float4 access along N for B/C
    if (( BN % 4 != 0 )); then
        echo "BN_not_divisible_by_4"
        return 1
    fi

    # --------------------------------------------------------
    # Warp-tile constraints
    #
    # Current k10 uses:
    #   WM = 64
    #   WN = 64
    # --------------------------------------------------------

    local WM=64
    local WN=64

    if (( BM % WM != 0 )); then
        echo "BM_not_divisible_by_WM"
        return 1
    fi

    if (( BN % WN != 0 )); then
        echo "BN_not_divisible_by_WN"
        return 1
    fi

    if (( WM % TM != 0 )); then
        echo "WM_not_divisible_by_TM"
        return 1
    fi

    if (( WN % TN != 0 )); then
        echo "WN_not_divisible_by_TN"
        return 1
    fi

    # --------------------------------------------------------
    # Number of threads implied by block tiling
    #
    # threads = (BM/TM) * (BN/TN)
    # --------------------------------------------------------

    if (( BM % TM != 0 || BN % TN != 0 )); then
        echo "invalid_thread_mapping"
        return 1
    fi

    local THREADS=$(( (BM / TM) * (BN / TN) ))

    if (( THREADS < 32 )); then
        echo "too_few_threads"
        return 1
    fi

    if (( THREADS > 1024 )); then
        echo "too_many_threads"
        return 1
    fi

    if (( THREADS % 32 != 0 )); then
        echo "threads_not_multiple_of_warp"
        return 1
    fi

    # Current launcher is fixed at 128 threads.
    #
    # IMPORTANT:
    # k10_warptiling.cu currently hardcodes:
    #     K10_NUM_THREADS = 128
    #
    # Therefore configurations whose implied thread count differs
    # from 128 cannot simply be passed to the current kernel.
    if (( THREADS != 128 )); then
        echo "threads_mismatch_current_k10"
        return 1
    fi

    # --------------------------------------------------------
    # Shared memory
    #
    # As = BM * BK floats
    # Bs = BK * BN floats
    # --------------------------------------------------------

    local SMEM_BYTES=$(( BK * (BM + BN) * 4 ))

    # Use conservative 48 KiB limit for portability.
    if (( SMEM_BYTES > 49152 )); then
        echo "shared_memory_over_48KB"
        return 1
    fi

    # --------------------------------------------------------
    # Warp-tile decomposition must be integral
    #
    # WMITER =
    #   (WM*WN)/(32*TM*TN*WNITER)
    #
    # Current k10:
    #   WM=64
    #   WN=64
    #   WNITER=4
    # --------------------------------------------------------

    local WNITER=4
    local NUMERATOR=$(( WM * WN ))
    local DENOMINATOR=$(( 32 * TM * TN * WNITER ))

    if (( NUMERATOR % DENOMINATOR != 0 )); then
        echo "invalid_WMITER"
        return 1
    fi

    local WMITER=$(( NUMERATOR / DENOMINATOR ))

    if (( WMITER < 1 )); then
        echo "invalid_WMITER"
        return 1
    fi

    if (( WM % WMITER != 0 )); then
        echo "WM_not_divisible_by_WMITER"
        return 1
    fi

    # --------------------------------------------------------
    # Shared-memory warp subdivision
    # --------------------------------------------------------

    local WSUBM=$(( WM / WMITER ))
    local WSUBN=$(( WN / WNITER ))

    if (( WSUBM % TM != 0 )); then
        echo "WSUBM_not_divisible_by_TM"
        return 1
    fi

    if (( WSUBN % TN != 0 )); then
        echo "WSUBN_not_divisible_by_TN"
        return 1
    fi

    # --------------------------------------------------------
    # Vectorized loading coverage
    #
    # A:
    #   innerRowA = thread / (BK/4)
    #   rowStrideA = (THREADS*4)/BK
    #
    # B:
    #   innerRowB = thread / (BN/4)
    #   rowStrideB = THREADS/(BN/4)
    #
    # These need to produce integral mappings.
    # --------------------------------------------------------

    if (( THREADS * 4 % BK != 0 )); then
        echo "invalid_A_vector_mapping"
        return 1
    fi

    if (( THREADS % (BN / 4) != 0 )); then
        echo "invalid_B_vector_mapping"
        return 1
    fi

    echo "valid"
    return 0
}


# ============================================================
# Sweep
# ============================================================

for BM in "${BM_VALS[@]}"; do
for BN in "${BN_VALS[@]}"; do
for BK in "${BK_VALS[@]}"; do
for TM in "${TM_VALS[@]}"; do
for TN in "${TN_VALS[@]}"; do

    THREADS=$(( (BM / TM) * (BN / TN) ))
    SMEM_BYTES=$(( BK * (BM + BN) * 4 ))

    REASON=$(validate_config "$BM" "$BN" "$BK" "$TM" "$TN")

    if [[ "$REASON" != "valid" ]]; then

        echo "  BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN -> SKIPPED ($REASON)"

        echo "$BIN_NAME,$BM,$BN,$BK,$TM,$TN,$M,$N,$K,$THREADS,$SMEM_BYTES,,,N/A,skipped_$REASON,$GPU_TAG" >> "$LOG"

        continue
    fi

    TAG="${BIN_NAME}_${BM}_${BN}_${BK}_${TM}_${TN}"

    echo
    echo "============================================================"
    echo "Testing BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN"
    echo "threads=$THREADS smem=${SMEM_BYTES}B"
    echo "============================================================"

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    if ! nvcc $NVCC_FLAGS \
        -DM_OVERRIDE=$M \
        -DN_OVERRIDE=$N \
        -DK_OVERRIDE=$K \
        -DBM=$BM \
        -DBN=$BN \
        -DBK=$BK \
        -DTM=$TM \
        -DTN=$TN \
        -o "$TAG" \
        "$SRC" \
        2>"${TAG}.compile.log"; then

        echo "  -> COMPILE FAILED"

        echo "$BIN_NAME,$BM,$BN,$BK,$TM,$TN,$M,$N,$K,$THREADS,$SMEM_BYTES,,,N/A,compile_failed,$GPU_TAG" >> "$LOG"

        rm -f "$TAG"
        continue
    fi

    rm -f "${TAG}.compile.log"

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    if ! OUTPUT=$(./"$TAG" 2>&1); then

        echo "  -> RUNTIME FAILED"

        echo "$BIN_NAME,$BM,$BN,$BK,$TM,$TN,$M,$N,$K,$THREADS,$SMEM_BYTES,,,N/A,runtime_failed,$GPU_TAG" >> "$LOG"

        rm -f "$TAG"
        continue
    fi

    # --------------------------------------------------------
    # Extract benchmark values
    # --------------------------------------------------------

    AVG_MS=$(echo "$OUTPUT" |
        grep "Avg kernel time" |
        awk '{print $(NF-1)}')

    GFLOPS=$(echo "$OUTPUT" |
        grep "Performance" |
        awk '{print $(NF-1)}')

    if echo "$OUTPUT" | grep -q "PASSED"; then
        CORRECT=1
    elif echo "$OUTPUT" | grep -q "FAILED"; then
        CORRECT=0
    else
        CORRECT="N/A"
    fi

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    echo "  -> ${GFLOPS:-N/A} GFLOPS"
    echo "  -> ${AVG_MS:-N/A} ms"
    echo "  -> correct=$CORRECT"

    # --------------------------------------------------------
    # Record result
    # --------------------------------------------------------

    echo "$BIN_NAME,$BM,$BN,$BK,$TM,$TN,$M,$N,$K,$THREADS,$SMEM_BYTES,$AVG_MS,$GFLOPS,$CORRECT,ok,$GPU_TAG" >> "$LOG"

    rm -f "$TAG"

done
done
done
done
done

echo
echo "============================================================"
echo "Parameter sweep complete"
echo "============================================================"
echo "Results: $LOG"
echo
echo "Valid configurations were compiled/run."
echo "Structurally invalid configurations were recorded as skipped."
echo "Compile/runtime failures were recorded separately."