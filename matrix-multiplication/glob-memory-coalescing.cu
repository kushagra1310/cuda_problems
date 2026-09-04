/*
 * Kernel 2: Global Memory Coalescing
 *
 * Threads in a warp access consecutive columns of C (and consecutive
 * elements of B's row), enabling the hardware to coalesce 32 × 4-byte loads
 * into a single 128-byte transaction.  The 1-D block layout maps threadIdx.x
 * to the N dimension (column) so adjacent threads → adjacent memory.
 *
 * Reference: https://siboehm.com/articles/22/CUDA-MMM  (Kernel 2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

#define BLOCKSIZE 32

// ---------------------------------------------------------------------------
__global__ void sgemm_coalesced(int M, int N, int K,
                                 const float * __restrict__ A,
                                 const float * __restrict__ B,
                                 float       * __restrict__ C)
{
    // 1-D block: remap threadIdx.x so that warp-adjacent threads own
    // adjacent columns (coalesced B and C access).
    const int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE; // M
    const int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE; // N

    if (row < M && col < N) {
        float tmp = 0.f;
        for (int k = 0; k < K; ++k)
            tmp += A[row * K + k] * B[k * N + col];
        C[row * N + col] = tmp;
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

    // 1-D block of BLOCKSIZE*BLOCKSIZE threads
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE, 1, 1);
    dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE,
                 (N + BLOCKSIZE - 1) / BLOCKSIZE, 1);

    printf("Matrix: %d x %d x %d\n", M, N, K);
    printf("Block : %d  |  Grid: %d x %d\n",
           blockDim.x, gridDim.x, gridDim.y);

    auto kernel = [&]() {
        sgemm_coalesced<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    };

    float avg_ms = benchmark_ms(kernel);
    print_results("Kernel 2: Global Memory Coalescing", M, N, K, avg_ms);

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
