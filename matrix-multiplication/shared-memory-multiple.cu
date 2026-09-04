/*
 * Kernel 4: 1-D Blocktiling (Multiple results per thread, along M)
 *
 * Each thread computes TM output elements in a column of C instead of just
 * one.  This raises arithmetic intensity by reusing the loaded B tile across
 * TM rows, reducing SMEM pressure and hiding memory latency.
 *
 * Tile dimensions: BM × BK (A tile) and BK × BN (B tile).
 * Threads per block: (BN) × (BM/TM)  (threadIdx.x → N, threadIdx.y → M/TM)
 *
 * Reference: https://siboehm.com/articles/22/CUDA-MMM  (Kernel 4)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

#define BM  64
#define BN  64
#define BK   8
#define TM   8   // results per thread along M

// Threads: (BN) × (BM/TM) = 64 × 8 = 512
static_assert(BM % TM == 0, "BM must be divisible by TM");

// ---------------------------------------------------------------------------
__global__ void sgemm_1d_blocktile(int M, int N, int K,
                                    const float * __restrict__ A,
                                    const float * __restrict__ B,
                                    float       * __restrict__ C)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Thread coordinates
    const int threadRow = threadIdx.y;        // 0 .. BM/TM - 1
    const int threadCol = threadIdx.x;        // 0 .. BN - 1

    // Block-level offsets into A, B, C
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Move base pointers to this block's tile
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Each thread accumulates TM results
    float threadResults[TM] = {0.f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ---- load A tile (BM × BK) into shared memory --------------------
        // Total elements = BM*BK, total threads = BN*(BM/TM)
        // Each thread loads BM*BK / (BN*BM/TM) = BK*TM/BN elements
        // For our params: 8*8/64 = 1  → one element per thread
        int tid = threadIdx.y * blockDim.x + threadIdx.x; // flat thread index
        {
            int row = tid / BK;
            int col = tid % BK;
            if (cRow * BM + row < M && bkIdx + col < K)
                As[row][col] = A[row * K + col];
            else
                As[row][col] = 0.f;
        }

        // ---- load B tile (BK × BN) into shared memory --------------------
        {
            int row = tid / BN;
            int col = tid % BN;
            if (bkIdx + row < K && cCol * BN + col < N)
                Bs[row][col] = B[row * N + col];
            else
                Bs[row][col] = 0.f;
        }

        __syncthreads();
        A += BK;    // advance A tile right
        B += BK * N; // advance B tile down

        // ---- compute partial dot-products --------------------------------
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float bTmp = Bs[dotIdx][threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[threadRow * TM + resIdx][dotIdx] * bTmp;
            }
        }

        __syncthreads();
    }

    // ---- write results ---------------------------------------------------
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        int row = cRow * BM + threadRow * TM + resIdx;
        int col = cCol * BN + threadCol;
        if (row < M && col < N)
            C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
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

    // Threads: BN columns × (BM/TM) rows
    dim3 blockDim(BN, BM / TM, 1);
    dim3 gridDim((N + BN - 1) / BN,
                 (M + BM - 1) / BM, 1);

    printf("Matrix: %d x %d x %d\n", M, N, K);
    printf("BM=%d BN=%d BK=%d TM=%d\n", BM, BN, BK, TM);
    printf("Block : %d x %d  |  Grid: %d x %d\n",
           blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    auto kernel = [&]() {
        sgemm_1d_blocktile<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    };

    float avg_ms = benchmark_ms(kernel);
    print_results("Kernel 4: 1D Blocktiling", M, N, K, avg_ms);

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
