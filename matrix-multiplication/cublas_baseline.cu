/*
 * cuBLAS Baseline
 *
 * Runs cuBLAS SGEMM and reports timing/GFLOPS.  This is the performance
 * ceiling to compare custom kernels against.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm_utils.cuh"

int main()
{
    const int M = GEMM_M, N = GEMM_N, K = GEMM_K;

    float *h_A = (float *)malloc((size_t)M * K * sizeof(float));
    float *h_B = (float *)malloc((size_t)K * N * sizeof(float));

    init_random(h_A, M * K, 42u);
    init_random(h_B, K * N, 43u);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.f, beta = 0.f;

    printf("Matrix: %d x %d x %d\n", M, N, K);

    auto kernel = [&]() {
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
    };

    float avg_ms = benchmark_ms(kernel);
    print_results("cuBLAS Baseline (SGEMM)", M, N, K, avg_ms);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A); free(h_B);

    return 0;
}
