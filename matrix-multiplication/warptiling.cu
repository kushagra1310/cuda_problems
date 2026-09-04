#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#endif

#define WARPSIZE 32

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                              float *As, float *Bs, int innerRowA,
                              int innerColA, int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];

    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM,
          const int WN, const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void processFromSmem(float *regM, float *regN,
                                 float *threadResults, const float *As,
                                 const float *Bs, const uint warpRow,
                                 const uint warpCol,
                                 const uint threadRowInWarp,
                                 const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul, accumulating with register-cache locality
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

template <const int BM, const int BN, const int BK, const int WM,
          const int WN, const int WNITER, const int TM, const int TN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptilingKernel(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // move A, B to the start of this block's row/col
  A += cRow * BM * K;
  B += cCol * BN;
  // move C to the start of this warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // indices used for vectorized (float4) SMEM loads
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  // write results back to C, one warp-subtile at a time
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

void sgemm_warptiling(int M, int N, int K,
                       const float *__restrict__ A,
                       const float *__restrict__ B,
                       float *__restrict__ C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;

  dim3 blockDim(K10_NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));

  sgemmWarptilingKernel<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER,
                         K10_TM, K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

static bool verify_result(int M, int N, const float *h_ref,
                           const float *h_test, float tol = 1e-1f) {
  double max_abs_err = 0.0;
  for (int i = 0; i < M * N; ++i) {
    double err = fabs((double)h_ref[i] - (double)h_test[i]);
    if (err > max_abs_err) max_abs_err = err;
    if (err > tol) {
      if (i < 10) {
        fprintf(stderr, "Mismatch at %d: ref=%f test=%f\n", i, h_ref[i],
                h_test[i]);
      }
    }
  }
  printf("Max abs error vs cuBLAS: %f\n", max_abs_err);
  return max_abs_err <= tol;
}

int main() {
  const int M = GEMM_M, N = GEMM_N, K = GEMM_K;

  float *h_A = (float *)malloc((size_t)M * K * sizeof(float));
  float *h_B = (float *)malloc((size_t)K * N * sizeof(float));
  float *h_C = (float *)malloc((size_t)M * N * sizeof(float));
  float *h_C_ref = (float *)malloc((size_t)M * N * sizeof(float));

  init_random(h_A, M * K, 42u);
  init_random(h_B, K * N, 43u);

  float *d_A, *d_B, *d_C, *d_C_ref;
  CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_C_ref, (size_t)M * N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, h_A, (size_t)M * K * sizeof(float),
                         cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, (size_t)K * N * sizeof(float),
                         cudaMemcpyHostToDevice));

  cublas_sgemm_ref(M, N, K, d_A, d_B, d_C_ref);

  // correctness check
  CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));
  sgemm_warptiling(M, N, K, d_A, d_B, d_C);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(h_C, d_C, (size_t)M * N * sizeof(float),
                         cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, (size_t)M * N * sizeof(float),
                         cudaMemcpyDeviceToHost));

  bool ok = verify_result(M, N, h_C_ref, h_C);
  printf("Kernel 10: Warptiling — %s\n", ok ? "PASSED" : "FAILED");

  auto kernel = [&]() { sgemm_warptiling(M, N, K, d_A, d_B, d_C); };
  float avg_ms = benchmark_ms(kernel);
  print_results("Kernel 10: Warptiling", M, N, K, avg_ms);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  CHECK_CUDA(cudaFree(d_C_ref));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return 0;
}