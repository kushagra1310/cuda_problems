#include <stdio.h>

__global__ void hello(float* a, float* b, float* c, int* n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<*n)
    {
        c[i]=a[i]+b[i];
    }
}

int main()
{
    int n=20;
    float* a=(float*)malloc(n*sizeof(float));
    float* b=(float*)malloc(n*sizeof(float));
    float* c=(float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }
    float *ac, *bc, *cc;
    int *nc;
    cudaMalloc(&ac, n*sizeof(float));
    cudaMalloc(&bc,n*sizeof(float));
    cudaMalloc(&cc,n*sizeof(float));
    cudaMalloc(&nc,sizeof(int));
 
    cudaMemcpy(ac, a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bc, b, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nc, &n, sizeof(int), cudaMemcpyHostToDevice);

    hello<<<1,100>>>(ac,bc,cc,nc);
    cudaMemcpy(c, cc, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<n; i++)
    {
        printf("%f\n",c[i]);
    }
    return 0;
}
