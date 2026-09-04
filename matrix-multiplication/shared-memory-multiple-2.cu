/*
 * Kernel 5: 2-D Blocktiling (TM × TN results per thread)
 *
 * Extends kernel 4 to compute a 2-D tile (TM rows × TN cols) per thread.
 * Each thread caches TM entries of A and TN entries of B in registers,
 * then performs an outer-product accumulation — maximising register reuse
 * and further raising arithmetic intensity.
 *
 * Tile dimensions: BM × BK (A tile) and BK × BN (B tile).
 * Threads per block: (BN/TN) × (BM/TM)
 *
 * Reference: https://siboehm.com/articles/22/CUDA-MMM  (Kernel 5)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

#define BM  64
#define BN  64
#define BK   8
#define TM   8
#define TN   8

static_assert(BM % TM == 0, "BM must be divisible by TM");
static_assert(BN % TN == 0, "BN must be divisible by TN");
// The tile-loading loops iterate multiple times if numThreads < tile size — that's fine.

// ---------------------------------------------------------------------------
__global__ void sgemm_2d_blocktile(int M, int N, int K,
                                    const float * __restrict__ A,
                                    const float * __restrict__ B,
                                    float       * __restrict__ C)
{
    __shared__ float As[BM * BK];  // stored row-major: As[row*BK + col]
    __shared__ float Bs[BK * BN];  // stored row-major: Bs[row*BN + col]

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Thread position within the block tile
    const int threadRow = threadIdx.y; // 0 .. BM/TM - 1
    const int threadCol = threadIdx.x; // 0 .. BN/TN - 1
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Advance base pointers to this block's starting position
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Per-thread result accumulator (TM × TN)
    float threadResults[TM * TN] = {0.f};
    float regM[TM] = {0.f};
    float regN[TN] = {0.f};

    const int numThreads = blockDim.x * blockDim.y;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ---- load A tile: BM×BK elements, numThreads threads ---------------
        for (int i = tid; i < BM * BK; i += numThreads) {
            int row = i / BK, col = i % BK;
            As[row * BK + col] =
                (cRow * BM + row < M && bkIdx + col < K)
                ? A[row * K + col] : 0.f;
        }

        // ---- load B tile: BK×BN elements -----------------------------------
        for (int i = tid; i < BK * BN; i += numThreads) {
            int row = i / BN, col = i % BN;
            Bs[row * BN + col] =
                (bkIdx + row < K && cCol * BN + col < N)
                ? B[row * N + col] : 0.f;
        }

        __syncthreads();
        A += BK;
        B += BK * N;

        // ---- compute outer-product accumulation ----------------------------
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // load this thread's slice of As and Bs into registers
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            // outer product
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
    }

    // ---- write back -------------------------------------------------------
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int row = cRow * BM + threadRow * TM + i;
            int col = cCol * BN + threadCol * TN + j;
            if (row < M && col < N)
                C[(threadRow * TM + i) * N + threadCol * TN + j] =
                    threadResults[i * TN + j];
        }
    }
}

// ---------------------------------------------------------------------------
int main()
{
    const int M = GEMM_M, N = GEMM_N, K = GEMM_K;

    float *h_A = (float *)malloc((size_t)M * K * sizeof(float));
    float *h_B = (float *)malloc((size_t)K * N * sizeof(float));
    float *h_C = (float *)malloc((size_t)M * N * sizeof(float));

    init_random(h_A, M * K, 42u);
    init_random(h_B, K * N, 43u);

    float *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_ref, (size_t)M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublas_sgemm_ref(M, N, K, d_A, d_B, d_C_ref);

    dim3 blockDim(BN / TN, BM / TM, 1);
    dim3 gridDim((N + BN - 1) / BN,
                 (M + BM - 1) / BM, 1);

    printf("Matrix: %d x %d x %d\n", M, N, K);
    printf("BM=%d BN=%d BK=%d TM=%d TN=%d\n", BM, BN, BK, TM, TN);
    printf("Block : %d x %d  |  Grid: %d x %d\n",
           blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    auto kernel = [&]() {
        sgemm_2d_blocktile<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    };

    float avg_ms = benchmark_ms(kernel);
    print_results("Kernel 5: 2D Blocktiling", M, N, K, avg_ms);

    kernel();
    CHECK_CUDA(cudaDeviceSynchronize());
    verify_correctness(M, N, d_C, d_C_ref);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));
    free(h_A); free(h_B); free(h_C);

    return 0;
}
