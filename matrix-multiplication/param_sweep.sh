#!/bin/bash
# param_sweep.sh
#
# Parameter sensitivity analysis for the warp-tiled GEMM kernel (k10).
#
# Sweeps: BM, BN, BK, TM, TN  (WM, WN, WNITER held fixed at the kernel's
# tuned defaults — see WM/WN/WNITER below, matching warptiling.cu)
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

# Fixed problem size
M=2048
N=2048
K=2048

GPU_TAG=$("$SCRIPT_DIR/gpu_info.sh")

# Parameter grid
BM_VALS=(64 128 256)
BN_VALS=(64 128 256)
BK_VALS=(8 16 32)
TM_VALS=(4 8)
TN_VALS=(4 8)

# Fixed warp-tile geometry — must match the WM/WN/WNITER defaults in
# warptiling.cu. If you want to sweep these too, add them to the grid
# above and thread them through validate_config the same way BM/BN/BK are.
WM=64
WN=64
WNITER=4

if [ ! -f "$LOG" ]; then
    echo "kernel,BM,BN,BK,TM,TN,M,N,K,threads,smem_bytes,avg_ms,gflops,correct,status,gpu" > "$LOG"
else
    echo "Appending to existing $LOG (delete it first if you want a clean sweep)"
fi


# ============================================================
# Validate a parameter configuration
# ============================================================
validate_config() {
    local BM=$1
    local BN=$2
    local BK=$3
    local TM=$4
    local TN=$5

    # -- Basic divisibility --------------------------------------------
    if (( BK % 4 != 0 )); then echo "BK_not_divisible_by_4"; return 1; fi
    if (( BN % 4 != 0 )); then echo "BN_not_divisible_by_4"; return 1; fi

    # -- Warp-tile fits inside block-tile -------------------------------
    if (( BM % WM != 0 )); then echo "BM_not_divisible_by_WM"; return 1; fi
    if (( BN % WN != 0 )); then echo "BN_not_divisible_by_WN"; return 1; fi

    # -- Thread count, DERIVED from warp geometry ------------------------
    # This is the corrected formula. Warptiling decouples per-thread tile
    # size (TM/TN) from block size (BM/BN) via the warp layer — thread
    # count is (# warps in the block) * 32, NOT (BM/TM)*(BN/TN) (that
    # formula belongs to the simpler 2D-blocktile kernel, k9, which has
    # no warp layer in between).
    local numWarpsM=$(( BM / WM ))
    local numWarpsN=$(( BN / WN ))
    local THREADS=$(( numWarpsM * numWarpsN * 32 ))

    if (( THREADS > 1024 )); then echo "too_many_threads"; return 1; fi

    # -- WMITER must be an integer ---------------------------------------
    # WMITER = (WM*WN) / (32*TM*TN*WNITER)
    local NUMERATOR=$(( WM * WN ))
    local DENOMINATOR=$(( 32 * TM * TN * WNITER ))
    if (( NUMERATOR % DENOMINATOR != 0 )); then echo "invalid_WMITER"; return 1; fi
    local WMITER=$(( NUMERATOR / DENOMINATOR ))
    if (( WMITER < 1 )); then echo "invalid_WMITER"; return 1; fi

    local WSUBM=$(( WM / WMITER ))
    local WSUBN=$(( WN / WNITER ))
    if (( WSUBM % TM != 0 )); then echo "WSUBM_not_divisible_by_TM"; return 1; fi
    if (( WSUBN % TN != 0 )); then echo "WSUBN_not_divisible_by_TN"; return 1; fi

    # -- Vectorized (float4) load mapping must be integral ----------------
    if (( THREADS * 4 % BK != 0 )); then echo "invalid_A_vector_mapping"; return 1; fi
    if (( THREADS % (BN / 4) != 0 )); then echo "invalid_B_vector_mapping"; return 1; fi

    # -- Load-loop COVERAGE — this is a correctness check, not just a
    # compile-validity check. If BM isn't a multiple of rowStrideA (or BK
    # of rowStrideB), the GMEM->SMEM load loop's `offset + stride <= BM`
    # condition stops early and never fills the remaining rows. That
    # doesn't crash — it silently leaves part of shared memory
    # uninitialized, which the kernel then reads as garbage. Skipping
    # these here avoids burning time on configs that would report
    # correct=0 for a reason that has nothing to do with tiling
    # performance.
    local rowStrideA=$(( (THREADS * 4) / BK ))
    local rowStrideB=$(( THREADS / (BN / 4) ))
    if (( BM % rowStrideA != 0 )); then echo "load_loop_incomplete_coverage_A"; return 1; fi
    if (( BK % rowStrideB != 0 )); then echo "load_loop_incomplete_coverage_B"; return 1; fi

    # -- Shared memory ----------------------------------------------------
    local SMEM_BYTES=$(( BK * (BM + BN) * 4 ))
    if (( SMEM_BYTES > 49152 )); then echo "shared_memory_over_48KB"; return 1; fi

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

    THREADS=0
    if (( BM % WM == 0 && BN % WN == 0 )); then
        THREADS=$(( (BM / WM) * (BN / WN) * 32 ))
    fi
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

    if ! nvcc $NVCC_FLAGS \
        -DM_OVERRIDE=$M -DN_OVERRIDE=$N -DK_OVERRIDE=$K \
        -DBM=$BM -DBN=$BN -DBK=$BK -DTM=$TM -DTN=$TN \
        -DWM=$WM -DWN=$WN -DWNITER=$WNITER \
        -o "$TAG" "$SRC" \
        2>"${TAG}.compile.log"; then
        echo "  -> COMPILE FAILED (see ${TAG}.compile.log)"
        echo "$BIN_NAME,$BM,$BN,$BK,$TM,$TN,$M,$N,$K,$THREADS,$SMEM_BYTES,,,N/A,compile_failed,$GPU_TAG" >> "$LOG"
        rm -f "$TAG"
        continue
    fi
    rm -f "${TAG}.compile.log"

    if ! OUTPUT=$(./"$TAG" 2>&1); then
        echo "  -> RUNTIME FAILED"
        echo "$BIN_NAME,$BM,$BN,$BK,$TM,$TN,$M,$N,$K,$THREADS,$SMEM_BYTES,,,N/A,runtime_failed,$GPU_TAG" >> "$LOG"
        rm -f "$TAG"
        continue
    fi

    AVG_MS=$(echo "$OUTPUT" | grep "Avg kernel time" | awk '{print $(NF-1)}')
    GFLOPS=$(echo "$OUTPUT" | grep "Performance"     | awk '{print $(NF-1)}')

    if echo "$OUTPUT" | grep -q "PASSED"; then CORRECT=1
    elif echo "$OUTPUT" | grep -q "FAILED"; then CORRECT=0
    else CORRECT="N/A"; fi

    echo "  -> ${GFLOPS:-N/A} GFLOPS"
    echo "  -> ${AVG_MS:-N/A} ms"
    echo "  -> correct=$CORRECT"

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