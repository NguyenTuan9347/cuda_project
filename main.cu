#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_HIDDEN_LAYERS 2
#define TILE_k 32
#define MAX_IMAGESIZE 784  // 28x28 flattened size
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define SIZE 784
#define LR 0.001
#define TRAIN_IMAGE "D:/gpu_project_course/fashion/train-images-idx3-ubyte"
#define TRAIN_LABEL "D:/gpu_project_course/fashion/train-labels-idx1-ubyte"
#define TEST_IMAGE "D:/gpu_project_course/fashion/t10k-images-idx3-ubyte"
#define TEST_LABEL "D:/gpu_project_course/fashion/t10k-labels-idx1-ubyte"
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


// Apply ReLU activation
void applyRelu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

// Compute softmax activation
void softmax(float* input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i]);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Compute loss (cross-entropy)
float computeLoss(float* output, int label, int size) {
    float loss = 0.0f;
    float *labels = (float*) malloc(sizeof(float)* size);
    for(int i =0;i< size;i++) labels[i] = 0.0;
    labels[label] = 1.0;
    for (int i = 0; i < size; i++) {
        loss -= labels[i] * logf(output[i]);
    }
    free(labels);
    return loss;
}

void applyActivationDerivative(float* gradient, float* activation, int size) {
    for (int i = 0; i < size; i++) {
        gradient[i] *= (activation[i] > 0) ? 1 : 0;
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
    } else {
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, m * n * sizeof(float));
        cudaMalloc((void**)&d_B, n * k * sizeof(float));
        cudaMalloc((void**)&d_C, m * k * sizeof(float));

        cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        matrixMultiKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

        cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

float* transpose(float* matrix, int rowSize, int colSize, bool useDevice=false){
    float *output = initRandomMatrix(colSize, rowSize);
    
    if(!useDevice){
        for(int i=0;i < rowSize;i++){
            for(int j=0;j < colSize; j++){
                output[j * rowSize + i] = matrix[i * rowSize + j];
            }
        }
    }
    
    return output;
}

void computeGradientForOutputLayer(float* output, float* gradOutput, float* targetLabels, int sampleSize, int outputSize=10, bool useDevice=false) {
    if (!useDevice) {
        for (int i = 0; i < sampleSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                gradOutput[i * outputSize + j] = output[i * outputSize + j];
            }
            gradOutput[i * outputSize + (int)targetLabels[i]] -= 1.0;
        }
    }
}

void computeGradientForOutputBias(float* gradOutput, float* gradBias, int sampleSize, int outputSize=10, bool useDevice=false) {
    if (!useDevice) {
        for (int j = 0; j < outputSize; j++) {
            gradBias[j] = 0.0;
            for (int i = 0; i < sampleSize; i++) {
                gradBias[j] += gradOutput[i * outputSize + j];
            }
        }
    }
}


float computeDerivativeHiddenLayer(float &a) {
    return a > 0 ? a : 0;
}

float multiply(float& a, float& b) {
    return a * b;
}

void elementWiseUnary(float* a, float* c, int rowSize, int colSize, float (*unary)(float&), bool useDevice=false) {
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                c[i * colSize + j] = unary(a[i * colSize + j]);
            }
        }
    }
}

void elementWiseBinary(float* a, float* b, float* c, int rowSize, int colSize, float (*binary)(float&, float&), bool useDevice=false) {
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                c[i * colSize + j] = binary(a[i * colSize + j], b[i * colSize + j]);
            }
        }
    }
}

float addition(float& a, float& b){
    return a + b;
}

void forward(float* input, float** hiddenWeights, float** activations, float** bias, float* output, int outputSize, int sampleSize, int featureSize=728, bool useDevice = false, dim3 blockSize = dim3(1)) {
    float* currentInput = input;
    int currentInputSize = featureSize;
    
    for (int i = 0; i < NUM_HIDDEN_LAYERS - 1; i++) {
        matrixMultiplication(currentInput, sampleSize, currentInputSize, hiddenWeights[i], HIDDEN_SIZE, activations[i], useDevice, blockSize);
        elementWiseBinary(activations[i], bias[i], activations[i],sampleSize, HIDDEN_SIZE,  addition);
        applyRelu(activations[i], HIDDEN_SIZE);
        
        currentInputSize = HIDDEN_SIZE;
        currentInput = activations[i];
    }
    
    matrixMultiplication(currentInput, 1, HIDDEN_SIZE, 
                         hiddenWeights[NUM_HIDDEN_LAYERS-1], outputSize, 
                         output, useDevice, blockSize);
    
    softmax(output, outputSize);
}


float updateWeight(float& org, float& grad) {
    return org - LR * grad;
}


void backward(float* input, float* output, float* targetLabels, float** hiddenWeights, float** activations, float** bias, int sampleSize, int featureSize = 728, int outputSize = 10) {
    float* gradOutput = (float*)malloc(sampleSize * outputSize * sizeof(float));
    float** gradWeights = (float**)malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
    float** gradBias = (float**)malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
    
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        int rowSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int colSize = (i == NUM_HIDDEN_LAYERS - 1) ? outputSize : HIDDEN_SIZE;
        gradWeights[i] = (float*)malloc(rowSize * colSize * sizeof(float));
        gradBias[i] = (float*)malloc(colSize * sizeof(float));
    }
    
    computeGradientForOutputLayer(output, gradOutput, targetLabels, sampleSize, outputSize);

    float* gradientToLoss = gradOutput;

    for (int i = NUM_HIDDEN_LAYERS - 1; i >= 0; i--) {
        int activationColSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int weightColSize = (i == NUM_HIDDEN_LAYERS - 1) ? outputSize : HIDDEN_SIZE;
        int weightRowSize = (i == 0) ? featureSize : HIDDEN_SIZE;

        float* activationsTransposed = transpose(activations[i], sampleSize, activationColSize);
        
        matrixMultiplication(activationsTransposed, activationColSize, sampleSize, gradientToLoss, weightColSize, gradWeights[i]);
        
        computeGradientForOutputBias(gradientToLoss, gradBias[i], sampleSize, weightColSize);
        
        // If not the first layer, propagate gradient backwards
        if (i > 0) {
            float* weightsTransposed = transpose(hiddenWeights[i], weightRowSize, weightColSize);
            float* derivativeOfActivation = (float*)malloc(sampleSize * activationColSize * sizeof(float));
            
            float* previousGradient = (float*)malloc(sampleSize * activationColSize * sizeof(float));
            matrixMultiplication(gradientToLoss, sampleSize, weightColSize, weightsTransposed, activationColSize, previousGradient);

            elementWiseUnary(activations[i - 1], derivativeOfActivation, sampleSize, activationColSize, computeDerivativeHiddenLayer);
            elementWiseBinary(previousGradient, derivativeOfActivation, previousGradient, sampleSize, activationColSize, multiply);

            gradientToLoss = previousGradient;

            free(weightsTransposed);
            free(derivativeOfActivation);
        }
        
        free(activationsTransposed);
    }
    
    //Update weights && bias
    for (int i = NUM_HIDDEN_LAYERS - 1; i >= 0; i--) {
        int activationColSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int weightColSize = (i == NUM_HIDDEN_LAYERS - 1) ? outputSize : HIDDEN_SIZE;
        int weightRowSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        elementWiseBinary(hiddenWeights[i], gradWeights[i], hiddenWeights[i], weightRowSize, weightColSize, updateWeight);
        elementWiseBinary(bias[i], gradBias[i], bias[i], sampleSize, activationColSize, updateWeight);
    }
    
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        free(gradWeights[i]);
        free(gradBias[i]);
    }

    free(gradWeights);
    free(gradBias);
    free(gradOutput);
}




void train(float** dataset, float* labels, int epochSize, int sampleSize, int inputSize, int outputSize=10) {
    float** hiddenWeights = (float**) malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
    float** activations = (float**) malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
    float** bias = (float**) malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        int prevSize = (i == 0) ? inputSize : HIDDEN_SIZE;
        int currSize = (i == NUM_HIDDEN_LAYERS - 1) ? outputSize : HIDDEN_SIZE;
        
        hiddenWeights[i] = (float*) malloc(prevSize * currSize * sizeof(float));
        activations[i] = (float*) malloc(currSize * sizeof(float));
        bias[i] = (float*) malloc(currSize * sizeof(float)); 
        
        for (int j = 0; j < prevSize * currSize; j++) {
            hiddenWeights[i][j] = (float)rand() / RAND_MAX - 0.5f;
        }
    }
    printf("Finish initialize weights and bias\n");
    // Training loop
    for (int epoch = 0; epoch < epochSize; epoch++) {
        float totalLoss = 0.0f;
        for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
            float* sample = dataset[sampleIdx];
            float* output = (float*) malloc(outputSize * sizeof(float) * sampleSize);
            forward(sample, hiddenWeights, activations, bias, output, outputSize, sampleSize);
            backward(sample, output, &labels[sampleIdx], hiddenWeights, activations, bias, sampleSize);
            
            free(output);
        }
        printf("Epoch %d, Loss: %.4f\n", epoch + 1, totalLoss / sampleSize);

    }

    // Memory cleanup
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        free(hiddenWeights[i]);
        free(activations[i]);
    }
    free(hiddenWeights);
    free(activations);
}

int main() {
    int train_image_count, train_label_count;
    int image_size;
    float* train_images = readImages(TRAIN_IMAGE, &train_image_count, &image_size);
    float* train_labels = readLabels(TRAIN_LABEL, &train_label_count);
    if (!train_images || !train_labels) {
        printf("Failed to load MNIST data.\n");
        return 1;
    }

    const int outputSize = 10;
    const int epochs = 100;

    // Prepare dataset as 2D array of pointers
    float** dataset = (float**) malloc(train_image_count * sizeof(float*));
    for (int i = 0; i < train_image_count; i++) {
        dataset[i] = train_images + i * image_size;
    }
    printf("Loaded Fashion MNIST data.\n");
    // Train the neural network

    printf("Trainning start\n");
    train(dataset, train_labels, epochs, train_image_count, image_size, outputSize);
    printf("Trainning end\n");

    // Cleanup
    free(dataset);
    free(train_labels);
    free(train_images);
    free(train_labels);
    
    return 0;
}
