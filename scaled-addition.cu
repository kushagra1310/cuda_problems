#include <stdio.h>

__global__ void add(float* a, float* b, float* c, float alpha, int n)
{
    int i=blockIdx.x*blockDim.x +threadIdx.x;
    if(i<n)
    {
        c[i]=alpha*a[i]+b[i];
    }
}

int main()
{
    int n=20;
    float alpha=2.1;
    float* a=(float*)malloc(n*sizeof(float));
    float* b=(float*)malloc(n*sizeof(float));
    float* c=(float*)malloc(n*sizeof(float));

    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    float *ac, *bc, *cc;
    cudaMalloc(&ac, n*sizeof(float));
    cudaMalloc(&bc,n*sizeof(float));
    cudaMalloc(&cc,n*sizeof(float));

    cudaMemcpy(ac,a, n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bc,b, n*sizeof(float), cudaMemcpyHostToDevice);

    add<<<n/256+1,256>>>(ac,bc,cc,alpha,n);

    cudaMemcpy(c,cc,n*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0; i<n; i++)
    {
        printf("%f\n",c[i]);
    }
    return 0;
}