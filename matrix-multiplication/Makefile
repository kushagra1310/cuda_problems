NVCC       := nvcc
NVCC_FLAGS := -O3 -std=c++17 -lcublas

# All kernel binaries
KERNELS := \
    k1_naive \
    k2_coalesced \
    k3_shared \
    k4_1d_blocktile \
    k5_2d_blocktile \
    k6_vectorized \
    k9_autotuned \
    k10_warptiling \
    cublas_baseline

.PHONY: all clean benchmark

all: $(KERNELS)

k1_naive: naive.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k2_coalesced: glob-memory-coalescing.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k3_shared: shared-memory.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k4_1d_blocktile: shared-memory-multiple.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k5_2d_blocktile: shared-memory-multiple-2.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k6_vectorized: vectorized-access.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k9_autotuned: autotuning.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k10_warptiling: warptiling.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cublas_baseline: cublas_baseline.cu gemm_utils.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

benchmark: all
	@echo ""
	@echo "========== cuBLAS Baseline =========="
	./cublas_baseline
	@echo ""
	@echo "========== Kernel 1: Naive =========="
	./k1_naive
	@echo ""
	@echo "========== Kernel 2: Coalesced =========="
	./k2_coalesced
	@echo ""
	@echo "========== Kernel 3: Shared Memory =========="
	./k3_shared
	@echo ""
	@echo "========== Kernel 4: 1D Blocktile =========="
	./k4_1d_blocktile
	@echo ""
	@echo "========== Kernel 5: 2D Blocktile =========="
	./k5_2d_blocktile
	@echo ""
	@echo "========== Kernel 6: Vectorized =========="
	./k6_vectorized
	@echo ""
	@echo "========== Kernel 9: Autotuned =========="
	./k9_autotuned
	@echo ""
    @echo "========== Kernel 10: Warp Tiling =========="
    ./k10_warptiling

clean:
	rm -f $(KERNELS) gemm_test a.out *.o
