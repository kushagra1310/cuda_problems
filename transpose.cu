#include <stdio.h>

__global__ void transpose(float* input, float* output, int m, int n)
{
    int x=threadIdx.x + blockDim.x*blockIdx.x;
    int y=threadIdx.y + blockDim.y*blockIdx.y;

    if(x<m && y<n)
    {
        output[x*n+y]=input[y*m+x];
    }
}

int main()
{
    int rows=5;
    int cols=6;
    int num_threads_x=16, num_threads_y=16;
    int num_blocks_x = (cols + num_threads_x - 1) / num_threads_x;
    int num_blocks_y = (rows + num_threads_y - 1) / num_threads_y;
    float* input_matrix = (float*)malloc(rows*cols*sizeof(float));
    float* output_matrix = (float*)malloc(rows*cols*sizeof(float));
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            input_matrix[i*cols+j]= 2*i+j;
        }
    }
    float* inputc, *outputc;
    cudaMalloc(&inputc, rows*cols*sizeof(float));
    cudaMalloc(&outputc, rows*cols*sizeof(float));

    cudaMemcpy(inputc, input_matrix, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(num_blocks_x, num_blocks_y);
    dim3 threads(num_threads_x, num_threads_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    transpose<<<blocks,threads>>>(inputc,outputc,cols,rows);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel time: %.6f ms\n", milliseconds);

    cudaMemcpy(output_matrix, outputc, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            printf("%f ",input_matrix[i*cols+j]);
        }
        printf("\n");
    }
    for(int i=0; i<cols; i++)
    {
        for(int j=0; j<rows; j++)
        {
            printf("%f ",output_matrix[i*rows+j]);
        }
        printf("\n");
    }
    
    return 0;
}