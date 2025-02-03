// CUDA C program to search for the neasrest vector given all the vectors
// First need to port the vectors fromm python to c using ctypes (python -> c -> cuda)

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8

// Sample c++ function to populate the host array
void populateHostArray(float h_arr[][100], int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_arr[i][j] = i + j;
        }
    }
}

// Function to calculate cosine similiarity 
float cosine_similarity(float *arr1, float *arr2, int m, int n) {
    
}

// Kernel
__global__ void findSimilar(float *arr, float *target, int m, int n, float *result) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        if (arr[row * n + col] != target[col]) {
            result[row] = 0;
        }
    }
}

int main() {
    
    // Number of vectors (m)
    int m = 1000; // Actual number ported from python
    // Number of dimensions (n)
    int n = 100; // Actual number ported from python

    // Sample host 2d array for now
    float h_arr[1000][100];

    // Populate the host array
    populateHostArray(h_arr, m, n);

    // Host query vector array
    float h_query[100];

    // Host result array
    float h_result[1000];

    // Populate the host query vector
    for (int i = 0; i < n; i++) {
        h_query[i] = (float)i;
    }

    // Create the pointers for device array
    float *d_arr, *d_target, *d_result;
    // Allocate memory on the device
    cudaMalloc((void**)&d_arr, m * n * sizeof(float));
    cudaMalloc((void**)&d_target, n * sizeof(float));
    cudaMalloc((void**)&d_result, m * sizeof(float));

    // Copy data from host to device memory
    cudaMemcpy(d_arr, h_arr, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr, h_arr, m*n*sizeof(float), cudaMemcpyHostToDevice);

    /*We have two options on how to get the result
    1. We define a boolean array, with the closest vector rows having the value 1.
    2. Is to store the result in a result array -> biit of a problem if we need to get multiple closest results
    We will have to go with the boolean array method. Or something better we will go with a float array, each position either 0 or the cosine similarity bw the query and the stored vector. i.e., if we need 3 closest results, we will aahvee 3 floats ini the array, m rest all 0.
    */

    // Defining the block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    return 0;
}