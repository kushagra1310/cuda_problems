#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

#define TILESIZE 32

__global__ void sgemm_shared(int M, int N, int K,
                              const float * __restrict__ A,
                              const float * __restrict__ B,
                              float       * __restrict__ C)
{
    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];

    const int row = blockIdx.y * TILESIZE + threadIdx.y;
    const int col = blockIdx.x * TILESIZE + threadIdx.x;

    float tmp = 0.f;
    for (int t = 0; t < K; t += TILESIZE) {
        As[threadIdx.y][threadIdx.x] =
            (row < M && t + threadIdx.x < K)
            ? A[row * K + t + threadIdx.x] : 0.f;

        Bs[threadIdx.y][threadIdx.x] =
            (t + threadIdx.y < K && col < N)
            ? B[(t + threadIdx.y) * N + col] : 0.f;

        __syncthreads();

        for (int k = 0; k < TILESIZE; ++k)
            tmp += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = tmp;
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

    dim3 blockDim(TILESIZE, TILESIZE, 1);
    dim3 gridDim((N + TILESIZE - 1) / TILESIZE,
                 (M + TILESIZE - 1) / TILESIZE, 1);

    printf("Matrix: %d x %d x %d\n", M, N, K);
    printf("Block : %d x %d  |  Grid: %d x %d\n",
           blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    auto kernel = [&]() {
        sgemm_shared<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    };

    float avg_ms = benchmark_ms(kernel);
    print_results("Kernel 3: Shared Memory Cache-Blocking", M, N, K, avg_ms);

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
