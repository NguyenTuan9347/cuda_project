﻿#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <direct.h>

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

// Global variables
int NUM_HIDDEN_LAYERS = 2;
int MAX_IMAGESIZE = 784;  // 28x28 flattened size
int NUM_TRAIN = 60000;
int NUM_TEST = 10000;
int SIZE = 784;
float LR = 0.1;
char TRAIN_IMAGE[512] = "D:/gpu_project_course/fashion/train-images-idx3-ubyte";
char TRAIN_LABEL[512] = "D:/gpu_project_course/fashion/train-labels-idx1-ubyte";
char TEST_IMAGE[512] = "D:/gpu_project_course/fashion/t10k-images-idx3-ubyte";
char TEST_LABEL[512] = "D:/gpu_project_course/fashion/t10k-labels-idx1-ubyte";
int HIDDEN_SIZE = 128;
int OUTPUT_SIZE = 10;
char BEST_CHECKPOINT[512] = "None";
float BEST_ACCURACY = 0.0f;

void setConfigValue(const char* key, const char* value) {
    if (strcmp(key, "NUM_HIDDEN_LAYERS") == 0) {
        NUM_HIDDEN_LAYERS = atoi(value);
    } else if (strcmp(key, "MAX_IMAGESIZE") == 0) {
        MAX_IMAGESIZE = atoi(value);
    } else if (strcmp(key, "NUM_TRAIN") == 0) {
        NUM_TRAIN = atoi(value);
    } else if (strcmp(key, "NUM_TEST") == 0) {
        NUM_TEST = atoi(value);
    } else if (strcmp(key, "SIZE") == 0) {
        SIZE = atoi(value);
    } else if (strcmp(key, "LR") == 0) {
        LR = atof(value);
    } else if (strcmp(key, "TRAIN_IMAGE") == 0) {
        strncpy(TRAIN_IMAGE, value, sizeof(TRAIN_IMAGE));
    } else if (strcmp(key, "TRAIN_LABEL") == 0) {
        strncpy(TRAIN_LABEL, value, sizeof(TRAIN_LABEL));
    } else if (strcmp(key, "TEST_IMAGE") == 0) {
        strncpy(TEST_IMAGE, value, sizeof(TEST_IMAGE));
    } else if (strcmp(key, "TEST_LABEL") == 0) {
        strncpy(TEST_LABEL, value, sizeof(TEST_LABEL));
    } else if (strcmp(key, "HIDDEN_SIZE") == 0) {
        HIDDEN_SIZE = atoi(value);
    } else if (strcmp(key, "OUTPUT_SIZE") == 0) {
        OUTPUT_SIZE = atoi(value);
    } else if (strcmp(key, "BEST_CHECKPOINT") == 0) {
        strncpy(BEST_CHECKPOINT, value, sizeof(BEST_CHECKPOINT));
    }
}

void loadConfig(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening configuration file");
        return;
    }

    char line[512];  // Increase buffer size to accommodate longer lines with spaces
    while (fgets(line, sizeof(line), file)) {
        char key[128], value[384];  // Increase value size to handle longer paths

        // Parse each line in the form of key = value, handling spaces in the value
        if (sscanf(line, "%127[^=]=%383[^\n]", key, value) == 2) {
            // Trim leading and trailing spaces from value
            char *trimmed_value = value;

            // Remove leading spaces
            while (*trimmed_value == ' ') {
                trimmed_value++;
            }

            // Remove trailing spaces
            char *end = trimmed_value + strlen(trimmed_value) - 1;
            while (end > trimmed_value && *end == ' ') {
                *end = '\0';
                end--;
            }

            setConfigValue(key, trimmed_value);
        }
    }

    fclose(file);
}

void modifyConfig(const char* filename, const char* key, const char* newValue) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file '%s'\n", filename);
        return;
    }

    // Temporary array to store file content
    char lines[20][512]; // Assuming the file has at most 20 lines, each line has 512 characters max
    int line_count = 0;

    // Read all lines into memory
    while (fgets(lines[line_count], 512, file) != NULL) {
        line_count++;
    }
    fclose(file);

    // Search for the line with the variable to be modified
    int variable_found = 0;
    for (int i = 0; i < line_count; i++) {
        char current_variable[256];
        sscanf(lines[i], "%[^=]", current_variable);  // Extract the variable name up to the '='

        // Compare with the desired variable name
        if (strcmp(current_variable, key) == 0) {
            variable_found = 1;
            // Update the line with the new value
            snprintf(lines[i], 512, "%s=%s\n", key, newValue);
            break;
        }
    }

    if (!variable_found) {
        printf("Error: Variable '%s' not found in the file.\n", key);
        return;
    }

    // Write updated content back to the file
    file = fopen(filename, "w");
    if (!file) {
        printf("Error: Unable to open file for writing '%s'\n", filename);
        return;
    }

    for (int i = 0; i < line_count; i++) {
        fputs(lines[i], file);
    }

    fclose(file);
    printf("Variable '%s' successfully updated to '%s'.\n", key, newValue);
}

// Write weights & biases to file
void saveWANDB(float** hiddenWeights, float** bias, int featureSize, int outputSize, const char* filepath) {
    FILE *file = fopen(filepath, "w");
    
    if (file == NULL) {
        perror("Error opening configuration file");
        return;
    }

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int prevSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int currSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        for (int j = 0; j < prevSize * currSize; j++) {
            fprintf(file, "%lf ", hiddenWeights[i][j]);
        }
        fprintf(file, "\n");
    }

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int currSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        for (int j = 0; j < currSize; j++) {
            fprintf(file, "%lf ", bias[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// Load weights & biases for testing
void loadWANDB(float** hiddenWeights, float** bias, int featureSize, int outputSize, const char* filepath = nullptr) {
    FILE* file;

    // Auto-load best model
    if (filepath == nullptr) {
        // No best model found
        if (strcmp(BEST_CHECKPOINT, "None") == 0) {
            perror("No model file could be found!");
            return;
        }
        file = fopen(BEST_CHECKPOINT, "r");   
    }
    else {
        file = fopen(filepath, "r");
    }

    if (file == NULL) {
        perror("Error opening configuration file");
        return;
    }

    printf("Loading weights\n");
    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int prevSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int currSize = (i == NUM_HIDDEN_LAYERS) ? OUTPUT_SIZE : HIDDEN_SIZE;
        for (int j = 0; j < prevSize * currSize; j++) {
            float tmp = 0.0f;
            fscanf(file, "%lf", &tmp);
            hiddenWeights[i][j] = tmp;
        }
        printf("________________________________________\n");
    }

    printf("Loading biases\n");
    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int currSize = (i == NUM_HIDDEN_LAYERS) ? OUTPUT_SIZE : HIDDEN_SIZE;
        for (int j = 0; j < currSize; j++) {
            float tmp = 0.0f;
            fscanf(file, "%lf", &tmp);
            bias[i][j] = tmp;
        }
    }

    fclose(file);
}

float* allocMatrix(int rowSize, int colSize = 1) {
    return (float*)malloc(rowSize * colSize * sizeof(float));
}

void freeMatrix(float* matrix) {
    if (matrix) {
        free(matrix);
    }
}

float* initRandomMatrix(int rowSize, int colSize = 1, float lower = 0.0, float upper = 1.0) {
    int size = rowSize * colSize;
    float* res = allocMatrix(rowSize, colSize);
    for (int i = 0; i < size; i++) {
        res[i] = lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (upper - lower)));
    }
    return res;
}

float* initHeMatrix(int rowSize, int colSize = 1) {
    int size = rowSize * colSize;
    float* res = allocMatrix(rowSize, colSize);

    float limit = sqrt(6.0f / rowSize); 

    for (int i = 0; i < size; i++) {
        res[i] = -limit + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) / (2 * limit));
    }
    return res;
}

float* initFilledMatrix(int rowSize, int colSize = 1, float val=0.0) {
    int size = rowSize * colSize;
    float* res = allocMatrix(rowSize, colSize);
    for (int i = 0; i < size; i++) {
        res[i] = val;
    }
    return res;
}

// Copy values from one matrix to another
void copyMatrix(float* dest, float* src, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

// Initialize weights & biases
bool initWANDB(float** hiddenWeights, float** bias, int featureSize, int outputSize, bool test) {
    if (test) {
        if (strcmp(BEST_CHECKPOINT, "None") != 0) {
            loadWANDB(hiddenWeights, bias, featureSize, outputSize, BEST_CHECKPOINT);
            printf("Loading best model for testing\n");
        } else {
            perror("No model found!\n");
            return false;
        }
        return true;
    } else {
        for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
            int prevSize = (i == 0) ? featureSize : HIDDEN_SIZE;
            int currSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;

            hiddenWeights[i] = initHeMatrix(prevSize, currSize);
            bias[i] = initFilledMatrix(currSize, 1,0.0);
            printf("Layer %d initialized: (%d, %d)\n", i, prevSize, currSize);
        }
        return true;
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



void printMatrix(float* matrix, int rowSize, int colSize) {
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            printf("%.2f ", matrix[i * colSize + j]);
        }
        printf("\n");
    }
    printf("\n");
}

float calChange(float* a, float* b, int rowSize, int colSize, float (*binary)(float& a, float& b)) {
    float sum = 0.0;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            int idx = i * colSize + j;
            sum += binary(a[idx], b[idx]);
        }
        
    }
    return sum;
}

void displayImg(const float* image, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (image[r * cols + c] > 0.0f) {
                printf("* ");
            }
            else {
                printf("  ");
            }
        }
        printf("\n");
    }
}

void displayBlank(const float* image, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float tmp = image[r * cols + c];
        };
    }
}

float** readImages(const char* path, int* num_images, int* image_size, int batchSize) {
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
    int numBatch = (*num_images - 1) / batchSize + 1;
    float** images = (float**)malloc(numBatch * sizeof(float*));
    for (int batchIdx = 0; batchIdx < numBatch; batchIdx++) {
        int end = ((batchIdx + batchSize) > *num_images) ? (*num_images - batchIdx) : batchSize;

        images[batchIdx] = (float*)malloc((*image_size) * end * sizeof(float));
        for (int i = 0; i < end; i++) {
            for (int j = 0; j < *image_size; ++j) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                images[batchIdx][i * (*image_size) + j] = (float)temp;
            }
        }
    }
    

    fclose(file);
    return images;
}

void softmax(float* input, float* output, int batchSize, int outputSize) {
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int buffer = batchIdx * outputSize;

        float max_val = input[buffer];
        for (int i = 1; i < outputSize; i++) {
            max_val = fmax(max_val, input[buffer + i]);
        }

        float sum = 0.0;
        for (int i = 0; i < outputSize; i++) {
            output[buffer + i] = exp(input[buffer + i] - max_val);
            sum += output[buffer + i];
        }

        for (int i = 0; i < outputSize; i++) {
            output[buffer + i] /= sum;
        }
    }
}

/*
    n is batch size
    d is 728
    Input size : n x d


*/


__global__ void matrixMultiKernel(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float s_A[TILE_k][TILE_k];
    __shared__ float s_B[TILE_k][TILE_k];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;

    for (int batch_idx = 0; batch_idx < (n + TILE_k - 1) / TILE_k; batch_idx++) {
        int A_col = batch_idx * TILE_k + threadIdx.x;
        int B_row = batch_idx * TILE_k + threadIdx.y;

        // Load tiles into shared memory
        s_A[threadIdx.y][threadIdx.x] = (row < m && A_col < n) ? A[row * n + A_col] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] = (col < k && B_row < n) ? B[B_row * k + col] : 0.0f;

        __syncthreads();

        // Perform partial matrix multiplication
        for (int i = 0; i < TILE_k; i++) {
            s += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < m && col < k) {
        C[row * k + col] = s;
    }
}

// Matrix multiplication wrapper
void matrixMultiplication(float* A, int m, int n, float* B, int k, float* C, bool useDevice = false, dim3 blockSize = dim3(1)) {
    if (!useDevice) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int t = 0; t < n; t++) {
                    sum += A[i * n + t] * B[t * k + j];
                }
                C[i * k + j] = sum;
            }
        }
    }
    else {
        float* d_A, * d_B, * d_C;
        cudaMalloc((void**)&d_A, m * n * sizeof(float));
        cudaMalloc((void**)&d_B, n * k * sizeof(float));
        cudaMalloc((void**)&d_C, m * k * sizeof(float));

        cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        matrixMultiKernel <<<gridSize, blockSize >>> (d_A, d_B, d_C, m, n, k);

        cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

float* transpose(float* matrix, int rowSize, int colSize, bool useDevice = false) {
    float* output = allocMatrix(colSize, rowSize);
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                int matrixIdx = i * colSize + j;  
                int transposedIdx = j * rowSize + i; 
                output[transposedIdx] = matrix[matrixIdx];
            }
        }
    }
    return output;
}


float applyRelu(float& a) {
    return a > 0 ? a : 0;
}


void computeGradientForOutputLayer(float* output, float* gradOutput, float* targetLabels, int batchSize, int outputSize = 10) {
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            gradOutput[i * outputSize + j] = output[i * outputSize + j];
        }
        gradOutput[i * outputSize + (int)targetLabels[i]] -= 1.0;
    }
}


void computeGradientForBias(float* gradToLoss, float* gradBias, int batchSize, int outputSize = 10, bool useDevice = false) {
    if (!useDevice) {
        for (int i = 0; i < outputSize; i++) {
            gradBias[i] = 0.0;
        }
        for (int j = 0; j < batchSize; j++) {
            for (int i = 0; i < outputSize; i++) {
                gradBias[i] += gradToLoss[j * outputSize + i];
            }
        }
        for (int i = 0; i < outputSize; i++) {
            gradBias[i] /= batchSize;
        }
    }
}


float retainPositive(float& org, float& dest) {
    return dest > 0 ? org : 0.0;
}

float multiply(float& a, float& b) {
    return a * b;
}

void elementWiseUnary(float* a, float* c, int rowSize, int colSize, float (*unary)(float&), bool useDevice = false) {
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                int idx = i * colSize + j;
                c[idx] = unary(a[idx]);          
            }
        }
    }
}

void elementWiseBinary(float* a, float* b, float* c, int rowSize, int colSize, float (*binary)(float&, float&), bool useDevice = false) {
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                int idx = i * colSize + j;
                c[idx] = binary(a[idx], b[idx]);
            }
        }
    }
}

float addition(float& a, float& b) {
    return a + b;
}

void forward(float* input, float** hiddenWeights, float** activations, float** bias, float* output, float** Z, float* zOutput, 
    int outputSize, int batchSize, bool useDevice = false, int featureSize = 784) {
    float* currentInput = input;
    int currentInputSize = featureSize;
    dim3 blockSize = dim3(1);
    if (useDevice) {
        blockSize.x = 128;
        blockSize.y = 128;
    }
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        matrixMultiplication(currentInput, batchSize, currentInputSize, hiddenWeights[i], HIDDEN_SIZE, Z[i], useDevice, blockSize);
        printMatrix(currentInput, batchSize, currentInputSize);
        printMatrix(hiddenWeights[i], currentInputSize, HIDDEN_SIZE);

        for (int j = 0; j < batchSize; j++) {
            elementWiseBinary(&activations[i][j * HIDDEN_SIZE], bias[i], &Z[i][j * HIDDEN_SIZE], HIDDEN_SIZE, 1, addition);
        }
        printMatrix(Z[i], batchSize, HIDDEN_SIZE);

        elementWiseUnary(Z[i], activations[i], batchSize, HIDDEN_SIZE, applyRelu);
        printMatrix(activations[i], batchSize, HIDDEN_SIZE);

        currentInputSize = HIDDEN_SIZE;
        currentInput = activations[i];
    }
    
    matrixMultiplication(currentInput, batchSize, HIDDEN_SIZE, hiddenWeights[NUM_HIDDEN_LAYERS], outputSize, zOutput, useDevice, blockSize);

    for (int j = 0; j < batchSize; j++) {
        elementWiseBinary(&zOutput[j * outputSize], bias[NUM_HIDDEN_LAYERS], &zOutput[j * outputSize], outputSize, 1, addition);
    }

    softmax(zOutput, output,batchSize, outputSize);


}


float updateWeight(float& org, float& grad) {
    return org - LR * grad;
}

void backward(float* input, float* output, float* targetLabels, float** hiddenWeights, float** activations,
    float** bias,float** Z,float* zOutput , int batchSize, bool useDevice = false, int featureSize = 784, int outputSize = OUTPUT_SIZE) {
    // Allocate gradients
    float* gradOutput = allocMatrix(batchSize, outputSize);

    float** gradWeights = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    float** gradBias = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int prevSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int currSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        gradWeights[i] = allocMatrix(prevSize, currSize);
        gradBias[i] = allocMatrix(currSize, 1);
    }

    computeGradientForOutputLayer(output, gradOutput, targetLabels, batchSize, outputSize);
    float* gradientToLoss = gradOutput;

    dim3 blockSize(128, 128);
    if (!useDevice) blockSize = dim3(1);

    for (int layer = NUM_HIDDEN_LAYERS; layer >= 0; layer--) {
        int currSize = (layer == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        int prevSize = (layer == 0) ? featureSize : HIDDEN_SIZE;

        float* activationsTransposed = (layer == 0) ? transpose(input, batchSize, featureSize) : transpose(activations[layer - 1], batchSize, prevSize);

        matrixMultiplication(activationsTransposed, prevSize, batchSize, gradientToLoss, currSize, gradWeights[layer], useDevice, blockSize);
        free(activationsTransposed);
        computeGradientForBias(gradientToLoss, gradBias[layer], batchSize, currSize);

        if (layer == 0) break;

        float* weightsTransposed = transpose(hiddenWeights[layer], prevSize, currSize);
        float* previousGradient = allocMatrix(batchSize, prevSize);

        matrixMultiplication(gradientToLoss, batchSize, currSize, hiddenWeights[layer], prevSize, previousGradient, useDevice, blockSize);

        //For derivative of ReLu is a > 0 ? 1.0 : 0.0, and compute of element wise. So it would better to combine the two operation into 1
        elementWiseBinary(previousGradient, Z[layer - 1], previousGradient, batchSize, prevSize, retainPositive); 
        
        free(weightsTransposed);
        if(layer < NUM_HIDDEN_LAYERS) 
            free(gradientToLoss);
        gradientToLoss = previousGradient;
    }

    for (int layer = 0; layer <= NUM_HIDDEN_LAYERS; layer++) {
        int currSize = (layer == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        int prevSize = (layer == 0) ? featureSize : HIDDEN_SIZE;
        elementWiseBinary(hiddenWeights[layer], gradWeights[layer], hiddenWeights[layer], prevSize, currSize, updateWeight);
        elementWiseBinary(bias[layer], gradBias[layer], bias[layer], currSize, 1, updateWeight);
    }

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        freeMatrix(gradWeights[i]);
        freeMatrix(gradBias[i]);
    }
    free(gradWeights);
    free(gradBias);
    free(gradOutput);
}



float calculateCrossEntropyLoss(float* output, float* trueLabels, int batchSize, int numClasses) {
    float totalLoss = 0.0;
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int label = (int)trueLabels[batchIdx];
        float predicted_prob = output[numClasses * batchIdx + label];

        totalLoss -= log(predicted_prob + 1e-15);
    }
    return totalLoss;
}

float calculateAccuracy(float* output, float* trueLabels, int batchSize, int numClasses) {
    int correct = 0;
    int labels[] = { 0,0,0,0,0,0,0,0,0,0 };
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int truth = (int)trueLabels[batchIdx];
        int label = 0;
        for (int j = 0; j < numClasses; j++) {
            label = (output[batchIdx * numClasses + j] > output[batchIdx * numClasses +  label]) ? j : label;
        }
        
        // DEBUG print
      /*   printf("\nLabel: %d - Truth: %d - Prob: %.2f\n", label, truth, output[batchIdx * numClasses + label]);
         for (int i = 0; i < numClasses; i++) {
             printf("%.2lf ", output[numClasses * batchIdx + i]);
         }
         _sleep(50);*/
        if (label == truth) {
            correct += 1;
        }
        labels[label] += 1;
    }

    //for (int i = 0; i < numClasses; i++) {
    //    printf("Guess label %d for %d times, ", i, labels[i]);
    //}
    //printf("\n");
    return correct * 1.0;
}

void train(float** dataset, float* labels, float** hiddenWeights, float** bias, int epochSize, int batchSize, int featureSize, int totalSize, const char* configFile,int step_save = 5, int outputSize = 10) {
    for (int epoch = 0; epoch < epochSize; epoch++) {
        float totalLoss = 0.0;
        float totalAccuracy = 0.0;
        int numBatch = (totalSize - 1) / batchSize + 1;
        for (int batchIdx = 0; batchIdx < numBatch; batchIdx++) {
            int startIdx = batchIdx * batchSize;
            int end = ((startIdx + batchSize) > totalSize) ? (totalSize - startIdx) : batchSize;
            float* batchLabels = &labels[startIdx];
            
            float** activations = (float**)malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
            for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
                activations[i] = allocMatrix(end, HIDDEN_SIZE);
            }

            float** Z = (float**)malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
            for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
                Z[i] = allocMatrix(end, HIDDEN_SIZE);
            }

            float* zOutput = allocMatrix(end, outputSize);
            float* output = allocMatrix(end, outputSize);
            forward(dataset[batchIdx], hiddenWeights, activations, bias, output,Z, zOutput, outputSize, end, true);

            totalLoss += calculateCrossEntropyLoss(output, batchLabels, end, outputSize);
            totalAccuracy += calculateAccuracy(output, batchLabels, end, outputSize);

            backward(dataset[batchIdx], output, batchLabels, hiddenWeights, activations, bias, Z, zOutput, end, true);

            free(output);
            for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
                free(activations[i]);
            }
            free(activations);
        }
        totalLoss /= totalSize;
        totalAccuracy /= totalSize;
        if ((epoch + 1) % step_save == 0) {
            char saveFile[256];
            snprintf(saveFile, sizeof(saveFile), "./checkpoints/wandb_%d.txt", epoch);
            saveWANDB(hiddenWeights, bias, featureSize, outputSize, saveFile);
            if (fabs(totalAccuracy - BEST_ACCURACY) > 1e-4f) { // Make sure the difference it correct since this one is floating point
                saveWANDB(hiddenWeights, bias, featureSize, outputSize, "best.txt");
                modifyConfig(configFile, "BEST_CHECKPOINT", "best.txt");
                char accuracyStr[10];
                snprintf(accuracyStr, sizeof(accuracyStr), "%lf", totalAccuracy);
                modifyConfig(configFile, "BEST_ACCURACY", accuracyStr);
            }
        }
        
        printf("Epoch %d, Loss: %.4f, Accuracy: %.4f\n", epoch + 1, totalLoss, totalAccuracy);
    }

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        free(hiddenWeights[i]);
        free(bias[i]);
    }
    free(hiddenWeights);
    free(bias);
}


int main(int argc, char *argv[]) {
    // Pass runtime arguments to choose config file or testing mode
    // Default run also works. Example command run:
    // ./a.exe test config.txt
    bool test = false;
    char configFile[256] = "config.txt";
    if (argc > 1) {
        if (strcmp(argv[1], "test") == 0) {
            test = true;
        }
        if (argc > 2) {
            strcpy(configFile, argv[2]);
        }
    }

    // Create checkpoint folder
    if (_mkdir("./checkpoints") == 0) {
        printf("Checkpoint directory created: %s\n", "./checkpoints");
    } else if (errno == EEXIST) {
        printf("Directory already exists!\n");
    } else {
        perror("Error creating directory!\n");
    }

    loadConfig(configFile);
    int train_image_count, train_label_count;
    int image_size;

    const int epochs = 1000;
    const int batchSize = 3;

    float** hiddenWeights = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    float** bias = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    float** train_images = readImages(TRAIN_IMAGE, &train_image_count, &image_size, batchSize);
    float* train_labels = readLabels(TRAIN_LABEL, &train_label_count);
    if (!train_images || !train_labels) {
        printf("Failed to load Fashion MNIST data.\n");
        return 1;
    }

    // Set test = true to load weights & biases from file
    bool check = initWANDB(hiddenWeights, bias, image_size, OUTPUT_SIZE, test);
    if (!check) {
        perror("Error intializing weights & biases.\nTerminating program...\n");
        free(train_images);
        free(train_labels);
        return 1;
    }
    
    if (test) {
        // TODO: Test function
    } else {
        train(train_images, train_labels, hiddenWeights, bias, epochs, batchSize, image_size, train_image_count, configFile,10);
    }
    
    for (int i = 0; i < (train_image_count - 1 / batchSize) + 1; i++) {
        free(train_images[i]);
    }
    free(train_images);
    free(train_labels);

    // Test matrix multiplication
    // float a[6] = {2, 3, 4, 5, 6, 7};
    // float b[6] = {7, 8, 9, 10, 11, 12};
    // float c[9];
    // float real_c[9];
    // matrixMultiplication(a, 3, 2, b, 3, c, false);
    // matrixMultiplication(a, 3, 2, b, 3, real_c, false);
    // printMatrix(c, 3, 3);
    // printMatrix(real_c, 3, 3);

    // Test loss function
    // float a[20] = {0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11, 0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11};
    // float b[2] = {0, 2};
    // printf("%.2lf", calculateCrossEntropyLoss(a, b, 2, 10));

    // Test softmax
    // float a[5] = {1.3, 5.1, 2.2, 0.7, 1.1};
    // softmax(a, 5);
    // for (int i = 0; i < 5; i++) {
    //     printf("%.2lf ", a[i]);
    // }

    // Test accuracy
    // float a[20] = {0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11, 0.10, 
    //                 0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11, 0.10};
    // float b[2] = {0, 2};
    // printf("%.2lf", calculateAccuracy(a, b, 2, 10));

    // Test load/save weights
    // float** hiddenWeights = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    // float** bias = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    // for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
    //     int prevSize = (i == 0) ? 784 : HIDDEN_SIZE;
    //     int currSize = (i == NUM_HIDDEN_LAYERS) ? 10 : HIDDEN_SIZE;

    //     hiddenWeights[i] = initRandomMatrix(prevSize, currSize, -0.5, 0.5);
    //     bias[i] = initRandomMatrix(currSize,1, 0.0, 0.0);
    //     printf("At layer %d: (%d,%d)\n", i, prevSize, currSize);
    // }
    // loadWANDB(hiddenWeights, bias, 784, "wandb.txt");
    // printf("Finished loading\n");
    // saveWANDB(hiddenWeights, bias, 784, "test.txt");
    return 0;
}