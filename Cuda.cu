#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void MatrixMultiplication(float* A, float* B, float* C, int height, int width, int depth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < depth) {
        float result = 0;
        for (int k = 0; k < width; k++) {
            result += A[row * width + k] * B[k * depth + col];
        }
        C[row * depth + col] = result;
    }
}

void printMatrix(float* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%f ", matrix[i * numCols + j]);
        }
        printf("\n");
    }
}

int main() {
    int numRowsA = 1024;
    int numColsA = 512;
    int numColsB = 2048;

    float *hostA = (float*)malloc(numRowsA * numColsA * sizeof(float));
    float *hostB = (float*)malloc(numColsA * numColsB * sizeof(float));
    float *hostC = (float*)malloc(numRowsA * numColsB * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < numRowsA * numColsA; i++) {
        hostA[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < numColsA * numColsB; i++) {
        hostB[i] = rand() / (float)RAND_MAX;
    }

    float *deviceA, *deviceB, *deviceC;
    cudaMalloc((void**)&deviceA, numRowsA * numColsA * sizeof(float));
    cudaMalloc((void**)&deviceB, numColsA * numColsB * sizeof(float));
    cudaMalloc((void**)&deviceC, numRowsA * numColsB * sizeof(float));

    cudaMemcpy(deviceA, hostA, numRowsA * numColsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numColsA * numColsB * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks(ceil(numColsB / (float)blockSize), ceil(numRowsA / (float)blockSize));

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MatrixMultiplication<<<numBlocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, numRowsA, numColsA, numColsB);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(hostC, deviceC, numRowsA * numColsB * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Elapsed time: %f ms\n", elapsedTime);

    free(hostA);
    free(hostB);
    free(hostC);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}
