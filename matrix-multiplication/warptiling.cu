#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "gemm_utils.cuh"

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#endif

#define WARPSIZE 32

// ---------------------------------------------------------------------
// Tunable parameters, overridable via -D at compile time (for
// param_sweep.sh). Template parameters use a K_ prefix so they never
// collide textually with these macros (see the earlier fix — passing
// -DBM=64 would otherwise substitute into the template parameter LIST
// itself if a template parameter were also literally named BM).
// ---------------------------------------------------------------------
#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 16
#endif
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 64
#endif
#ifndef WNITER
#define WNITER 4
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 4
#endif

// ---------------------------------------------------------------------
// Load a BM x BK tile of A and a BK x BN tile of B from global memory
// into shared memory, using float4 vectorized loads where the full
// 4-wide access is in-bounds, falling back to a scalar zero-padded load
// at the M/N/K edges.
//
// rowBase = cRow*BM   (this block's starting row in A / row in C — constant
//                       across the bkIdx loop)
// colBase = cCol*BN   (this block's starting col in B / col in C — constant)
// kBase   = bkIdx     (current position along the reduction dim — changes
//                       every iteration of the outer loop in the kernel)
//
// Without these checks, a block whose tile extends past M, N, or K (which
// happens whenever M/N/K aren't exact multiples of BM/BN/BK — e.g.
// 3000x1500x2048 with the default BM=BN=128) reads and writes past the
// allocated buffers instead of stopping at the real matrix edge.
// ---------------------------------------------------------------------
template <const int K_BM, const int K_BN, const int K_BK,
          const int K_rowStrideA, const int K_rowStrideB>
__device__ void loadFromGmem(int M, int N, int K, const float *A,
                              const float *B, float *As, float *Bs,
                              int innerRowA, int innerColA, int innerRowB,
                              int innerColB, int rowBase, int colBase,
                              int kBase) {
  for (uint offset = 0; offset + K_rowStrideA <= K_BM; offset += K_rowStrideA) {
    const int gRow = rowBase + innerRowA + (int)offset;
    const int gCol = kBase + innerColA * 4;

    float4 tmp = {0.f, 0.f, 0.f, 0.f};
    if (gRow < M && gCol + 3 < K) {
      tmp = reinterpret_cast<const float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
    } else if (gRow < M) {
      // partial-K tail: fetch whatever columns are still in-bounds, zero
      // the rest, one element at a time (rare — only the last K-tile of a
      // K not divisible by BK hits this path).
      float v[4] = {0.f, 0.f, 0.f, 0.f};
      for (int x = 0; x < 4; ++x)
        if (gCol + x < K) v[x] = A[(innerRowA + offset) * K + innerColA * 4 + x];
      tmp = {v[0], v[1], v[2], v[3]};
    }
    // else: gRow >= M, this row of the tile doesn't exist — leave zeroed.

    As[(innerColA * 4 + 0) * K_BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * K_BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * K_BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * K_BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + K_rowStrideB <= K_BK; offset += K_rowStrideB) {
    const int gRow = kBase + innerRowB + (int)offset;
    const int gCol = colBase + innerColB * 4;

    if (gRow < K && gCol + 3 < N) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * K_BN + innerColB * 4])[0] =
          reinterpret_cast<const float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    } else {
      for (int x = 0; x < 4; ++x) {
        float v = 0.f;
        if (gRow < K && gCol + x < N)
          v = B[(innerRowB + offset) * N + innerColB * 4 + x];
        Bs[(innerRowB + offset) * K_BN + innerColB * 4 + x] = v;
      }
    }
  }
}

// ---------------------------------------------------------------------
// Consume the current BK-deep SMEM tile — unchanged from before; this
// part only ever reads shared memory (which is now always fully and
// correctly populated, including zero-padding at the edges), so it needs
// no bounds checks of its own.
// ---------------------------------------------------------------------
template <const int K_BM, const int K_BN, const int K_BK, const int K_WM,
          const int K_WN, const int K_WMITER, const int K_WNITER,
          const int K_WSUBM, const int K_WSUBN, const int K_TM, const int K_TN>
__device__ void processFromSmem(float *regM, float *regN,
                                 float *threadResults, const float *As,
                                 const float *Bs, const uint warpRow,
                                 const uint warpCol,
                                 const uint threadRowInWarp,
                                 const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < K_BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < K_WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < K_TM; ++i) {
        regM[wSubRowIdx * K_TM + i] =
            As[(dotIdx * K_BM) + warpRow * K_WM + wSubRowIdx * K_WSUBM +
               threadRowInWarp * K_TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < K_WNITER; ++wSubColIdx) {
      for (uint i = 0; i < K_TN; ++i) {
        regN[wSubColIdx * K_TN + i] =
            Bs[(dotIdx * K_BN) + warpCol * K_WN + wSubColIdx * K_WSUBN +
               threadColInWarp * K_TN + i];
      }
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < K_WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < K_WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < K_TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < K_TN; ++resIdxN) {
            threadResults[(wSubRowIdx * K_TM + resIdxM) * (K_WNITER * K_TN) +
                          (wSubColIdx * K_TN) + resIdxN] +=
                regM[wSubRowIdx * K_TM + resIdxM] *
                regN[wSubColIdx * K_TN + resIdxN];
          }
        }
      }
    }
  }
}

template <const int K_BM, const int K_BN, const int K_BK, const int K_WM,
          const int K_WN, const int K_WNITER, const int K_TM, const int K_TN,
          const int K_NUM_THREADS>
__global__ void __launch_bounds__(K_NUM_THREADS)
    sgemmWarptilingKernel(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (K_BN / K_WN);
  const uint warpRow = warpIdx / (K_BN / K_WN);

  constexpr uint K_WMITER = (K_WM * K_WN) / (WARPSIZE * K_TM * K_TN * K_WNITER);
  constexpr uint K_WSUBM = K_WM / K_WMITER;
  constexpr uint K_WSUBN = K_WN / K_WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (K_WSUBN / K_TN);
  const uint threadRowInWarp = threadIdxInWarp / (K_WSUBN / K_TN);

  __shared__ float As[K_BM * K_BK];
  __shared__ float Bs[K_BK * K_BN];

  // Absolute tile origins — used both to offset the A/B pointers for the
  // fast in-bounds path AND as the M/N/K bounds-check reference in
  // loadFromGmem.
  const int rowBase = cRow * K_BM;
  const int colBase = cCol * K_BN;

  A += (size_t)rowBase * K;
  B += colBase;
  C += (size_t)(rowBase + warpRow * K_WM) * N + colBase + warpCol * K_WN;

  const uint innerRowA = threadIdx.x / (K_BK / 4);
  const uint innerColA = threadIdx.x % (K_BK / 4);
  constexpr uint rowStrideA = (K_NUM_THREADS * 4) / K_BK;
  const uint innerRowB = threadIdx.x / (K_BN / 4);
  const uint innerColB = threadIdx.x % (K_BN / 4);
  constexpr uint rowStrideB = K_NUM_THREADS / (K_BN / 4);

  float threadResults[K_WMITER * K_TM * K_WNITER * K_TN] = {0.0f};
  float regM[K_WMITER * K_TM] = {0.0f};
  float regN[K_WNITER * K_TN] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += K_BK) {
    loadFromGmem<K_BM, K_BN, K_BK, rowStrideA, rowStrideB>(
        M, N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB,
        rowBase, colBase, bkIdx);
    __syncthreads();
    processFromSmem<K_BM, K_BN, K_BK, K_WM, K_WN, K_WMITER, K_WNITER, K_WSUBM,
                     K_WSUBN, K_TM, K_TN>(
        regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += K_BK;
    B += (size_t)K_BK * N;
    __syncthreads();
  }

  // write results back to C, one warp-subtile at a time, with per-element
  // bounds checks against M/N so a tile that overhangs the matrix edge
  // (any block along the last row/col when M/N isn't a multiple of
  // BM/BN) only writes the elements that actually exist in C.
  for (uint wSubRowIdx = 0; wSubRowIdx < K_WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < K_WNITER; ++wSubColIdx) {
      float *C_interim = C + (size_t)(wSubRowIdx * K_WSUBM) * N + wSubColIdx * K_WSUBN;
      const int rowBlockBase = rowBase + warpRow * K_WM + wSubRowIdx * K_WSUBM;
      const int colBlockBase = colBase + warpCol * K_WN + wSubColIdx * K_WSUBN;

      for (uint resIdxM = 0; resIdxM < K_TM; ++resIdxM) {
        const int row = rowBlockBase + threadRowInWarp * K_TM + resIdxM;
        if (row >= M) continue;  // this thread-row is past the bottom edge

        for (uint resIdxN = 0; resIdxN < K_TN; resIdxN += 4) {
          const int col = colBlockBase + threadColInWarp * K_TN + resIdxN;
          const int i = (wSubRowIdx * K_TM + resIdxM) * (K_WNITER * K_TN) +
                        wSubColIdx * K_TN + resIdxN;
          float *dst = &C_interim[(threadRowInWarp * K_TM + resIdxM) * N +
                                   threadColInWarp * K_TN + resIdxN];

          if (col + 3 < N) {
            float4 tmp = reinterpret_cast<float4 *>(dst)[0];
            tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(dst)[0] = tmp;
          } else {
            // right edge: write only the columns that exist, one at a time
            for (int x = 0; x < 4; ++x) {
              if (col + x < N)
                dst[x] = alpha * threadResults[i + x] + beta * dst[x];
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------
// Host-side launcher. NUM_THREADS is derived from warp geometry.
// ---------------------------------------------------------------------
static_assert(BM % WM == 0, "BM must be divisible by WM");
static_assert(BN % WN == 0, "BN must be divisible by WN");
static_assert(BK % 4 == 0, "BK must be divisible by 4 (float4 A loads)");
static_assert(BN % 4 == 0, "BN must be divisible by 4 (float4 B loads)");
static_assert((WM * WN) % (32 * TM * TN * WNITER) == 0,
              "WM*WN must be divisible by 32*TM*TN*WNITER (WMITER must be integral)");

void sgemm_warptiling(int M, int N, int K,
                       const float *__restrict__ A,
                       const float *__restrict__ B,
                       float *__restrict__ C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  constexpr int numWarpsM = BM / WM;
  constexpr int numWarpsN = BN / WN;
  constexpr int NUM_THREADS = numWarpsM * numWarpsN * WARPSIZE;

  static_assert(NUM_THREADS <= 1024, "NUM_THREADS exceeds hardware block limit");
  static_assert((NUM_THREADS * 4) % BK == 0,
                "NUM_THREADS*4 must be divisible by BK (A load mapping)");
  static_assert(NUM_THREADS % (BN / 4) == 0,
                "NUM_THREADS must be divisible by BN/4 (B load mapping)");
  static_assert(BM % ((NUM_THREADS * 4) / BK) == 0,
                "BM must be divisible by rowStrideA — the boundary check "
                "handles M not being a multiple of BM, but the SMEM load "
                "loop itself still needs rowStrideA to tile BM evenly");
  static_assert(BK % (NUM_THREADS / (BN / 4)) == 0,
                "BK must be divisible by rowStrideB (same reasoning, for BK)");

  dim3 blockDim(NUM_THREADS);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  sgemmWarptilingKernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
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

  printf("BM=%d BN=%d BK=%d WM=%d WN=%d WNITER=%d TM=%d TN=%d\n",
         BM, BN, BK, WM, WN, WNITER, TM, TN);

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