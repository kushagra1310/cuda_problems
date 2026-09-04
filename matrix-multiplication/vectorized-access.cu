/*
 * Kernel 6: Vectorized SMEM and GMEM Accesses
 *
 * Builds on kernel 5 with two additions:
 *  1. As is stored *transposed* (BK × BM) so that the SMEM loads for regM
 *     access consecutive addresses → the compiler emits LDS.128 (128-bit
 *     shared-memory load) instead of scalar LDS.
 *  2. All GMEM loads/stores use float4 reinterpret casts to emit 128-bit
 *     LDG.E.128 / STG.E.128 instructions, improving memory throughput.
 *
 * Constraints (so float4 loads are legal):
 *   BK % 4 == 0,  BN % 4 == 0,  TM % 4 == 0,  TN % 4 == 0
 *
 * Reference: https://siboehm.com/articles/22/CUDA-MMM  (Kernel 6)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

#define BM  128
#define BN  128
#define BK    8
#define TM    8
#define TN    8

static_assert(BM % TM  == 0, "BM must be divisible by TM");
static_assert(BN % TN  == 0, "BN must be divisible by TN");
static_assert(BK % 4   == 0, "BK must be divisible by 4 for vectorised A loads");
static_assert(BN % 4   == 0, "BN must be divisible by 4 for vectorised B loads");
static_assert(TN % 4   == 0, "TN must be divisible by 4 for vectorised C stores");

// ---------------------------------------------------------------------------
__global__ void sgemm_vectorized(int M, int N, int K,
                                  const float * __restrict__ A,
                                  const float * __restrict__ B,
                                  float       * __restrict__ C)
{
    // As transposed → stored as [BK][BM]  (accessed as As[k][row])
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int threadRow = threadIdx.y;  // 0 .. BM/TM-1
    const int threadCol = threadIdx.x;  // 0 .. BN/TN-1
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * blockDim.y;

    // Loading indices for A — each thread loads a float4 from GMEM
    const int innerRowA = tid / (BK / 4); // row within BM
    const int innerColA = tid % (BK / 4); // 4-float group within BK
    // How many rows of A each thread-group covers per pass
    const int rowStrideA = numThreads / (BK / 4);

    // Loading indices for B
    const int innerRowB = tid / (BN / 4);
    const int innerColB = tid % (BN / 4);
    const int rowStrideB = numThreads / (BN / 4);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float threadResults[TM * TN] = {0.f};
    float regM[TM] = {0.f};
    float regN[TN] = {0.f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // ---- load A with float4, transpose into SMEM ----------------------
        for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            int gRow = cRow * BM + innerRowA + offset;
            int gCol = bkIdx + innerColA * 4;
            float4 tmp = {0.f, 0.f, 0.f, 0.f};
            if (gRow < M && gCol + 3 < K)
                tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
            // transpose: As[k][row] = A[row][bkIdx+k]
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        // ---- load B with float4 into SMEM ----------------------------------
        for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            int gRow = bkIdx + innerRowB + offset;
            int gCol = cCol * BN + innerColB * 4;
            if (gRow < K && gCol + 3 < N)
                reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                    reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
            else {
                // partial — zero pad
                for (int x = 0; x < 4; ++x)
                    Bs[(innerRowB + offset) * BN + innerColB * 4 + x] =
                        (gRow < K && gCol + x < N)
                        ? B[(innerRowB + offset) * N + innerColB * 4 + x] : 0.f;
            }
        }

        __syncthreads();
        A += BK;
        B += BK * N;

        // ---- compute with register caching (As is transposed) -------------
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // load TM entries of As (transposed): As[dotIdx][threadRow*TM + 0..TM-1]
            // these are now consecutive → LDS.128
            for (int i = 0; i < TM; ++i)
                regM[i] = As[dotIdx * BM + threadRow * TM + i];
            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += regM[i] * regN[j];
        }

        __syncthreads();
    }

    // ---- write back using float4 ------------------------------------------
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
                reinterpret_cast<float4 *>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = out;
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
        sgemm_vectorized<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    };

    float avg_ms = benchmark_ms(kernel);
    print_results("Kernel 6: Vectorized SMEM+GMEM", M, N, K, avg_ms);

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
