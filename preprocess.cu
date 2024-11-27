#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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
#define HIDDEN_SIZE 128

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

#define TRAIN_IMAGE "train-images-idx3-ubyte"
#define TRAIN_LABEL "train-labels-idx1-ubyte"
#define TEST_IMAGE "t10k-images-idx3-ubyte"
#define TEST_LABEL "t10k-labels-idx1-ubyte"
#define NUM_HIDDEN_LAYERS 2
#define TILE_k 32
// Reverse integer bytes for MNIST file format
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float* readLabels(const char* path, int* num_labels) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return nullptr;
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverseInt(magic_number);

    fread(num_labels, sizeof(*num_labels), 1, file);
    *num_labels = reverseInt(*num_labels);

    if (magic_number != 2049) {
        printf("Invalid magic number: %d. Expected 2049 for label file.\n", magic_number);
        fclose(file);
        return nullptr;
    }

    float* labels = (float*)malloc((*num_labels) * sizeof(float));
    for (int i = 0; i < *num_labels; ++i) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, file);
        labels[i] = (float)temp;
    }

    fclose(file);
    return labels;
}

float* readImages(const char* path, int* num_images, int* image_size) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return nullptr;
    }

    int magic_number = 0, n_rows = 0, n_cols = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverseInt(magic_number);

    fread(num_images, sizeof(*num_images), 1, file);
    *num_images = reverseInt(*num_images);

    fread(&n_rows, sizeof(n_rows), 1, file);
    n_rows = reverseInt(n_rows);

    fread(&n_cols, sizeof(n_cols), 1, file);
    n_cols = reverseInt(n_cols);

    if (magic_number != 2051 || n_rows != 28 || n_cols != 28) {
        printf("Invalid file format or dimensions. MNIST expects 28x28 images.\n");
        fclose(file);
        return nullptr;
    }

    *image_size = n_rows * n_cols;
    float* images = (float*)malloc((*num_images) * (*image_size) * sizeof(float));

    for (int i = 0; i < *num_images; ++i) {
        for (int j = 0; j < *image_size; ++j) {
            unsigned char temp = 0;
            fread(&temp, sizeof(temp), 1, file);
            images[i * (*image_size) + j] = (float)temp / 255.0f;  // Normalize to [0, 1]
        }
    }

    fclose(file);
    return images;
}

void displayImg(const float* image, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (image[r * cols + c] > 0.0f) {
                printf("* ");
            } else {
                printf("  ");
            }
        }
        printf("\n");
    }
}

const char* getLabelByIdx(int label) {
    static char buffer[10];
    snprintf(buffer, sizeof(buffer), "%d", label);
    return buffer;
}

int main() {
    int train_image_count, train_label_count, test_image_count, test_label_count;
    int image_size;

    float* train_images = readImages(TRAIN_IMAGE, &train_image_count, &image_size);
    float* train_labels = readLabels(TRAIN_LABEL, &train_label_count);

    if (!train_images || !train_labels) {
        printf("Failed to load MNIST data.\n");
        return 1;
    }

    const int hiddenSize = 128;
    const int numHiddenLayers = 2;
    const int outputSize = 10;
    const int epochs = 10;
    const float learningRate = 0.01f;

   

    return 0;
}
