// Cuda code to compute the cosine similarity between two vectors

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to compute the cosine similarity between two vectors
__global__ void cosine_similarity(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] * b[index];
    }
}

// Function to compute the cosine similarity between two vectors
void cosine(float *a, float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;
    int size = n * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    cosine_similarity<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    // Copy data from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Main function
int main() {
    int n = 100000;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }

    cosine(a, b, c, n);

    for (int i = 0; i < n; i++) {
        printf("%f\n", c[i]);
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}