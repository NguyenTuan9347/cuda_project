#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_IMAGESIZE 784  // 28x28 flattened size
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define SIZE 784

#define TRAIN_IMAGE "./fashion/train-images-idx3-ubyte"
#define TRAIN_LABEL "./fashion/train-labels-idx1-ubyte"
#define TEST_IMAGE "./fashion/t10k-images-idx3-ubyte"
#define TEST_LABEL "./fashion/t10k-labels-idx1-ubyte"
#define NUM_HIDDEN_LAYERS 2
#define TILE_k 32

#define CHECK(call)\
{\
    cudaError_t errorSync = call;\
    cudaError_t errorASync = cudaPeekAtLastError();\
    if (errorSync != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", errorSync,\
                cudaGetErrorString(errorSync));\
        exit(EXIT_FAILURE);\
    }\
    if (errorASync != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", errorASync,\
                cudaGetErrorString(errorASync));\
        exit(EXIT_FAILURE);\
    }\
}

// Allocate memory for a matrix
float* allocMatrix(int rowSize, int colSize = 1) {
    return (float*)malloc(rowSize * colSize * sizeof(float));
}

// Free allocated memory
void freeMatrix(float* matrix) {
    if (matrix) {
        free(matrix);
    }
}

// Initialize a random matrix
float* initRandomMatrix(int rowSize, int colSize = 1, float lower = 0.0, float upper = 1.0) {
    int size = rowSize * colSize;
    float* res = allocMatrix(rowSize, colSize);
    for (int i = 0; i < size; i++) {
        res[i] = lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (upper - lower)));
    }
    return res;
}

// Copy values from one matrix to another
void copyMatrix(float* dest, float* src, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

// Extract a row from the matrix
float* getRow(float* matrix, int rowIndex, int colSize) {
    float* row = allocMatrix(1, colSize);
    for (int i = 0; i < colSize; i++) {
        row[i] = matrix[rowIndex * colSize + i];
    }
    return row;
}

// Extract a column from the matrix
float* getCol(float* matrix, int colIndex, int rowSize, int colSize) {
    float* col = allocMatrix(rowSize, 1);
    for (int i = 0; i < rowSize; i++) {
        col[i] = matrix[i * colSize + colIndex];
    }
    return col;
}

// Get label for a given index
const char* getLabelByIdx(int idx) {
    switch (idx) {
        case 0: return "T-shirt/top";
        case 1: return "Trouser";
        case 2: return "Pullover";
        case 3: return "Dress";
        case 4: return "Coat";
        case 5: return "Sandal";
        case 6: return "Shirt";
        case 7: return "Sneaker";
        case 8: return "Bag";
        case 9: return "Ankle boot";
        default: return "Not exist label";
    }
}
