#!/bin/bash
# autotune.sh — sweep BM, BN, BK, TM, TN for kernel 9 (autotuning.cu)
#
# Constraints (enforced before compilation):
#   BM % TM == 0,  BN % TN == 0
#   BK % 4  == 0,  BN % 4  == 0
#   TM % 4  == 0,  TN % 4  == 0
#   (BM/TM) * (BN/TN) <= 1024  (max threads per block)
#   Shared memory (BK*(BM+BN)*4 bytes) <= 48 KB
#   GMEM->SMEM load coverage: rowStrideA <= BM and BM % rowStrideA == 0
#                              rowStrideB <= BK and BK % rowStrideB == 0

set -e

SOURCE="autotuning.cu"
OUT_BIN="gemm_test"
LOG="autotune_results.csv"

echo "kernel,BM,BN,BK,TM,TN,threads,smem_bytes,avg_ms,gflops,correct" > "$LOG"

best_gflops=0
best_config=""

for BM in 64 128; do
for BN in 64 128; do
for BK in 8 16; do
for TM in 4 8; do
for TN in 4 8; do

    # ---- compile-time constraint checks -----------------------------------
    [ $((BM % TM)) -ne 0 ] && continue
    [ $((BN % TN)) -ne 0 ] && continue
    [ $((BK % 4))  -ne 0 ] && continue
    [ $((BN % 4))  -ne 0 ] && continue
    [ $((TM % 4))  -ne 0 ] && continue
    [ $((TN % 4))  -ne 0 ] && continue

    NUM_THREADS=$(( (BM/TM) * (BN/TN) ))
    [ $NUM_THREADS -gt 1024 ] && continue
    [ $NUM_THREADS -lt 32   ] && continue

    SMEM=$(( BK * (BM + BN) * 4 ))
    [ $SMEM -gt 49152 ] && continue   # 48 KB limit

    # ---- NEW: GMEM->SMEM load-coverage constraints -------------------------
    ROW_STRIDE_A=$(( NUM_THREADS / (BK / 4) ))
    ROW_STRIDE_B=$(( NUM_THREADS / (BN / 4) ))

    [ $ROW_STRIDE_A -eq 0 ] && continue          # divide-by-zero guard
    [ $ROW_STRIDE_B -eq 0 ] && continue
    [ $ROW_STRIDE_A -gt $BM ] && continue        # A-tile would never fully load
    [ $ROW_STRIDE_B -gt $BK ] && continue        # B-tile would never fully load
    [ $((BM % ROW_STRIDE_A)) -ne 0 ] && continue # partial coverage -> wrong result
    [ $((BK % ROW_STRIDE_B)) -ne 0 ] && continue

    echo ""
    echo "Testing BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN  (threads=$NUM_THREADS smem=${SMEM}B)"

    # ---- compile ------------------------------------------------------------
    if ! nvcc -O3 -std=c++17 \
        -DBM=$BM -DBN=$BN -DBK=$BK -DTM=$TM -DTN=$TN \
        "$SOURCE" -o "$OUT_BIN" -lcublas 2>/dev/null; then
        echo "  [SKIP] Compilation failed"
        continue
    fi

    # ---- run and parse output ------------------------------------------------
    OUTPUT=$(./"$OUT_BIN" 2>&1) || { echo "  [SKIP] Runtime error"; continue; }

    RESULT_LINE=$(echo "$OUTPUT" | grep "^RESULT,")
    if [ -z "$RESULT_LINE" ]; then
        echo "  [SKIP] No RESULT line found in output"
        continue
    fi
    AVG_MS=$(echo "$RESULT_LINE" | cut -d',' -f2)
    GFLOPS=$(echo "$RESULT_LINE" | cut -d',' -f3)

    CORRECT=0
    echo "$OUTPUT" | grep -qi "PASS" && CORRECT=1

    echo "  → ${GFLOPS} GFLOPS  (${AVG_MS} ms)  correct=${CORRECT}"

    echo "k9,$BM,$BN,$BK,$TM,$TN,$NUM_THREADS,$SMEM,$AVG_MS,$GFLOPS,$CORRECT" >> "$LOG"

    # Track best (only consider correct configs)
    if [ "$CORRECT" -eq 1 ] && awk "BEGIN{exit !($GFLOPS > $best_gflops)}"; then
        best_gflops=$GFLOPS
        best_config="BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN"
    fi

done; done; done; done; done

echo ""
echo "===================================================="
echo "Autotuning complete.  Results saved to: $LOG"
echo "Best config: $best_config  →  ${best_gflops} GFLOPS"
echo "===================================================="