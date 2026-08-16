#include <stdio.h>

__global__ void reduce(double* arr, double* sums, int n)
{
    extern __shared__ double shared[];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i<n)
    {
        shared[threadIdx.x]=arr[i];
    }
    else
        shared[threadIdx.x] = 0.0f;
    __syncthreads();

    for(int stride = 1; stride<blockDim.x; stride<<=1)
    {
        if(threadIdx.x%(stride<<1)==0)
            shared[threadIdx.x]+=shared[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x == 0)
        sums[blockIdx.x] = shared[0];
}

int main()
{
    int n;
    n=10000000;

    int num_threads=256;
    int num_blocks=(n+num_threads-1)/num_threads;

    double* arr=(double*)malloc(n*sizeof(double));
    double* partial_sum=(double*)malloc(num_blocks*sizeof(double));

    for(int i=0; i<n; i++)
    {
        arr[i]=i;
    }

    double* carr, *csum;
    cudaMalloc(&carr, n*sizeof(double));
    cudaMalloc(&csum, (num_blocks)*sizeof(double));
    cudaMemcpy(carr,arr,n*sizeof(double),cudaMemcpyHostToDevice);
    
    double sum=0;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce<<<num_blocks,num_threads, 256*sizeof(double)>>>(carr, csum, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel time: %.6f ms\n", milliseconds);

    cudaMemcpy(partial_sum, csum, num_blocks*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<num_blocks; i++)
        sum+=partial_sum[i];
    printf("%f\n",sum);
    return 0;
}
