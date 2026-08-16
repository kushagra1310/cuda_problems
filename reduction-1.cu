#include <stdio.h>

__global__ void reduce(float* arr, float* sum, int n)
{
    extern __shared__ float shared[];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i<n)
    {
        atomicAdd(sum,arr[i]);
    }
}

int main()
{
    int n;
    n=10000000;
    float* arr=(float*)malloc(n*sizeof(float));
    for(int i=0; i<n; i++)
    {
        arr[i]=i;
    }
    float* carr, *csum;
    cudaMalloc(&carr, n*sizeof(float));
    cudaMalloc(&csum, sizeof(float));
    cudaMemcpy(carr,arr,n*sizeof(float),cudaMemcpyHostToDevice);
    float sum=0;
    cudaMemcpy(csum,&sum,sizeof(float),cudaMemcpyHostToDevice);

    int num_threads=256;
    int num_blocks=(n+num_threads-1)/num_threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    reduce<<<num_blocks,num_threads>>>(carr,csum,n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel time: %.6f ms\n", milliseconds);

    cudaMemcpy(&sum, csum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n",sum);
    return 0;
}
