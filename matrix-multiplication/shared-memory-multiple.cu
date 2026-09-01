#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define STRIDE 8
#define PERTHREAD 8
#define CHECK_CUDA(call)                                                   \
do {                                                                       \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)


__global__ void shared_mult(double* A, double* B, double* C,
                           int rows1, int mid, int cols2)
{
    __shared__ double As[BM][STRIDE];
    __shared__ double Bs[STRIDE][BN];
    
    double threadResults[PERTHREAD] = {0.0};

    for (int blockidx=0; blockidx < mid; blockidx+=STRIDE)
    {
        As[threadIdx.x][threadIdx.y]=A[(blockIdx.y*BM+threadIdx.x)*mid+blockidx+threadIdx.y];
        Bs[threadIdx.y][threadIdx.x]=B[(blockidx+threadIdx.y)*cols2+blockIdx.x*BN+threadIdx.x];
        
        __syncthreads();
        for (int k = 0; k < STRIDE; k++)
        {
            double btemp=Bs[k][threadIdx.x];
            for (int r = 0; r < PERTHREAD; r++)
            {
                threadResults[r]+=As[threadIdx.y*PERTHREAD+r][k]*btemp;
            }
        }
        __syncthreads();
    }
    for(int i=0; i<PERTHREAD; i++)
    {
        int row=blockIdx.y * BM + threadIdx.y*PERTHREAD+i;
        int col=blockIdx.x*BN+threadIdx.x;
        if (row < rows1 && col < cols2)
        {
            C[row * cols2 + col] = threadResults[i];
        }
    }
}


int main()
{
    int rows = 4096;
    int mid  = 4096;
    int cols = 4096;

    int num_threadsx = BN;
    int num_threadsy = BM/PERTHREAD;


    double* A = (double*)malloc(rows * mid * sizeof(double));
    double* B = (double*)malloc(mid * cols * sizeof(double));
    double* C = (double*)malloc(rows * cols * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Host memory allocation failed\n");
        return 1;
    }


    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < mid; j++)
        {
            A[i * mid + j] = i * 2.0 + j * 4.0;
        }
    }

    for (int i = 0; i < mid; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            B[i * cols + j] = i * 4.0 + j * 7.0;
        }
    }


    double *cA, *cB, *cC;

    CHECK_CUDA(cudaMalloc(&cA, rows * mid * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&cB, mid * cols * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&cC, rows * cols * sizeof(double)));


    CHECK_CUDA(cudaMemcpy(
        cA,
        A,
        rows * mid * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        cB,
        B,
        mid * cols * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    dim3 blockDim(num_threadsx, num_threadsy, 1);

    int num_blocksx =
        (cols + num_threadsx - 1) / BN;

    int num_blocksy =
        (rows + num_threadsy - 1) / BM;

    dim3 gridDim(num_blocksx, num_blocksy, 1);


    printf("Matrix size: %d x %d x %d\n", rows, mid, cols);
    printf("Block size: %d x %d\n", num_threadsx, num_threadsy);
    printf("Grid size: %d x %d\n", num_blocksx, num_blocksy);


    const int warmup_runs = 5;

    for (int i = 0; i < warmup_runs; i++)
    {
        shared_mult<<<gridDim, blockDim>>>(
            cA, cB, cC,
            rows, mid, cols
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    const int benchmark_runs = 20;

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < benchmark_runs; i++)
    {
        shared_mult<<<gridDim, blockDim>>>(
            cA, cB, cC,
            rows, mid, cols
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaGetLastError());


    float total_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    float avg_ms = total_ms / benchmark_runs;

    double total_flops =
        2.0 * (double)rows * (double)mid * (double)cols;

    double gflops =
        total_flops / (avg_ms * 1.0e6);


    printf("\n");
    printf("========================================\n");
    printf("Naive GEMM Benchmark\n");
    printf("========================================\n");
    printf("Average kernel time : %.4f ms\n", avg_ms);
    printf("Performance         : %.2f GFLOPS\n", gflops);
    printf("========================================\n");


    CHECK_CUDA(cudaMemcpy(
        C,
        cC,
        rows * cols * sizeof(double),
        cudaMemcpyDeviceToHost
    ));


    printf("C[0][0] = %f\n", C[0]);
    printf("C[100][100] = %f\n", C[100 * cols + 100]);


    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(cA));
    CHECK_CUDA(cudaFree(cB));
    CHECK_CUDA(cudaFree(cC));

    free(A);
    free(B);
    free(C);

    return 0;
}
