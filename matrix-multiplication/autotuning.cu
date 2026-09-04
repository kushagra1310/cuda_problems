/*
 * Kernel 9: Autotuned Vectorized 2D Blocktiling
 *
 * Same algorithm as kernel 6 (vectorized SMEM+GMEM accesses, transposed As)
 * but all tile sizes are compile-time defines so the autotune.sh script can
 * sweep over many parameter combinations.
 *
 * Template parameters (set via -D flags at compile time):
 *   BM, BN  — block tile size in M and N  (must be multiples of TM, TN, 4)
 *   BK      — block tile size in K          (must be multiple of 4)
 *   TM, TN  — per-thread tile in M and N   (must be multiples of 4)
 *
 * Reference: https://siboehm.com/articles/22/CUDA-MMM  (Kernel 9)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

// ---- Default parameters (overridden by -D on the command line) ------------
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK   8
#endif
#ifndef TM
#define TM   8
#endif
#ifndef TN
#define TN   8
#endif

// Compile-time sanity checks
static_assert(BM % TM  == 0,  "BM must be divisible by TM");
static_assert(BN % TN  == 0,  "BN must be divisible by TN");
static_assert(BK % 4   == 0,  "BK must be divisible by 4 (float4 A loads)");
static_assert(BN % 4   == 0,  "BN must be divisible by 4 (float4 B loads)");
static_assert(TN % 4   == 0,  "TN must be divisible by 4 (float4 C stores)");
static_assert(TM % 4   == 0,  "TM must be divisible by 4 (float4 As loads)");

// ---------------------------------------------------------------------------
__global__ void sgemm_autotuned(int M, int N, int K,
                                 const float * __restrict__ A,
                                 const float * __restrict__ B,
                                 float       * __restrict__ C)
{
    // As stored transposed: [BK][BM]
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * blockDim.y;

    // GMEM-loading indices for A
    const int innerRowA  = tid / (BK / 4);
    const int innerColA  = tid % (BK / 4);
    const int rowStrideA = numThreads / (BK / 4);

    // GMEM-loading indices for B
    const int innerRowB  = tid / (BN / 4);
    const int innerColB  = tid % (BN / 4);
    const int rowStrideB = numThreads / (BN / 4);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM * TN] = {0.f};
    float regM[TM] = {0.f};
    float regN[TN] = {0.f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ---- load A tile into transposed SMEM (float4) -------------------
        for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            int gRow = cRow * BM + innerRowA + offset;
            int gCol = bkIdx + innerColA * 4;
            float4 tmp = {0.f, 0.f, 0.f, 0.f};
            if (gRow < M && gCol + 3 < K)
                tmp = reinterpret_cast<const float4 *>(
                          &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // ---- load B tile into SMEM (float4) --------------------------------
        for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            int gRow = bkIdx + innerRowB + offset;
            int gCol = cCol * BN + innerColB * 4;
            if (gRow < K && gCol + 3 < N)
                reinterpret_cast<float4 *>(
                    &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                    reinterpret_cast<const float4 *>(
                        &B[(innerRowB + offset) * N + innerColB * 4])[0];
            else {
                for (int x = 0; x < 4; ++x)
                    Bs[(innerRowB + offset) * BN + innerColB * 4 + x] =
                        (gRow < K && gCol + x < N)
                        ? B[(innerRowB + offset) * N + innerColB * 4 + x] : 0.f;
            }
        }

        __syncthreads();
        A += BK;
        B += BK * N;

        // ---- compute: register caches + outer-product ---------------------
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load TM floats from As (consecutive because As is transposed)
            // Using pragma unroll instead of hardcoded float4 for generality
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            #pragma unroll
            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
    }

    // ---- write back with float4 -------------------------------------------
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            int row = cRow * BM + threadRow * TM + i;
            int col = cCol * BN + threadCol * TN + j;
            if (row < M && col + 3 < N) {
                float4 out;
                out.x = threadResults[i * TN + j + 0];
                out.y = threadResults[i * TN + j + 1];
                out.z = threadResults[i * TN + j + 2];
                out.w = threadResults[i * TN + j + 3];
                reinterpret_cast<float4 *>(
                    &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = out;
            } else {
                for (int x = 0; x < 4; ++x) {
                    int c = col + x;
                    if (row < M && c < N)
                        C[(threadRow * TM + i) * N + threadCol * TN + j + x] =
                            threadResults[i * TN + j + x];
                }
            }
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
        sgemm_autotuned<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    };

    float avg_ms = benchmark_ms(kernel);
    print_results("Kernel 9: Autotuned", M, N, K, avg_ms);
    printf("RESULT,%.4f,%.4f\n", avg_ms, (2.0 * M * N * K) / (avg_ms * 1e6));
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
