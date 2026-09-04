#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef M_OVERRIDE
#  define GEMM_M M_OVERRIDE
#else
#  define GEMM_M 4096
#endif
#ifdef N_OVERRIDE
#  define GEMM_N N_OVERRIDE
#else
#  define GEMM_N 4096
#endif
#ifdef K_OVERRIDE
#  define GEMM_K K_OVERRIDE
#else
#  define GEMM_K 4096
#endif

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d — code %d\n",               \
                    __FILE__, __LINE__, (int)st);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static inline void init_random(float *mat, int n, unsigned seed = 42u)
{
    srand(seed);
    for (int i = 0; i < n; ++i)
        mat[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

static inline void cublas_sgemm_ref(int M, int N, int K,
                                    const float *d_A, const float *d_B,
                                    float *d_C_ref)
{
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.f, beta = 0.f;
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, N,   
                             d_A, K,
                             &beta,
                             d_C_ref, N));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUBLAS(cublasDestroy(handle));
}

static inline bool verify_correctness(int M, int N,
                                      const float *d_C,
                                      const float *d_C_ref,
                                      float tol = 1e-2f,
                                      bool verbose = true)
{
    size_t bytes = (size_t)M * N * sizeof(float);
    float *h_C     = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);

    CHECK_CUDA(cudaMemcpy(h_C,     d_C,     bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.f, max_rel = 0.f;
    long long wrong = 0;
    for (int i = 0; i < M * N; ++i) {
        float err = fabsf(h_C[i] - h_C_ref[i]);
        float rel = err / (fabsf(h_C_ref[i]) + 1e-6f);
        if (err > max_err) max_err = err;
        if (rel > max_rel) max_rel = rel;
        if (err > tol) ++wrong;
    }

    bool ok = (wrong == 0);
    if (verbose) {
        if (ok)
            printf("  Correctness: PASS  (max_abs_err=%.6f, max_rel_err=%.6f)\n",
                   max_err, max_rel);
        else
            printf("  Correctness: FAIL  %lld / %d elements wrong "
                   "(max_abs_err=%.6f, tol=%.6f)\n",
                   wrong, M * N, max_err, tol);
    }

    free(h_C);
    free(h_C_ref);
    return ok;
}

template<typename KernelFn>
static inline float benchmark_ms(KernelFn fn,
                                  int warmup_runs  = 5,
                                  int timed_runs   = 20)
{
    for (int i = 0; i < warmup_runs; ++i) fn();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < timed_runs; ++i) fn();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / timed_runs;
}

static inline void print_results(const char *kernel_name,
                                  int M, int N, int K,
                                  float avg_ms)
{
    double gflops = 2.0 * (double)M * (double)N * (double)K
                    / (avg_ms * 1e6);
    printf("\n");
    printf("==================================================\n");
    printf("  %s\n", kernel_name);
    printf("==================================================\n");
    printf("  Matrix: %d x %d x %d\n", M, N, K);
    printf("  Avg kernel time : %.4f ms\n", avg_ms);
    printf("  Performance     : %.2f GFLOPS\n", gflops);
    printf("==================================================\n");
}
