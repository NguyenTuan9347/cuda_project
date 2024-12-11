#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

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
int NUM_HIDDEN_LAYERS = 3;
int MAX_IMAGESIZE = 784;  // 28x28 flattened size
int NUM_TRAIN = 60000;
int NUM_TEST = 10000;
int SIZE = 784;
double LR = 0.01;
char TRAIN_IMAGE[512] = "D:/gpu_project_course/fashion/train-images-idx3-ubyte";
char TRAIN_LABEL[512] = "D:/gpu_project_course/fashion/train-labels-idx1-ubyte";
char TEST_IMAGE[512] = "D:/gpu_project_course/fashion/t10k-images-idx3-ubyte";
char TEST_LABEL[512] = "D:/gpu_project_course/fashion/t10k-labels-idx1-ubyte";
int HIDDEN_SIZE = 128;
int OUTPUT_SIZE = 10;

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

// Allocate memory for a matrix
double* allocMatrix(int rowSize, int colSize = 1) {
    return (double*)malloc(rowSize * colSize * sizeof(double));
}

// Free allocated memory
void freeMatrix(double* matrix) {
    if (matrix) {
        free(matrix);
    }
}

// Initialize a random matrix
double* initRandomMatrix(int rowSize, int colSize = 1, double lower = 0.0, double upper = 1.0) {
    int size = rowSize * colSize;
    double* res = allocMatrix(rowSize, colSize);
    for (int i = 0; i < size; i++) {
        res[i] = lower + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (upper - lower)));
    }
    return res;
}

// Copy values from one matrix to another
void copyMatrix(double* dest, double* src, int size) {
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

double* readLabels(const char* path, int* num_labels) {
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

    double* labels = (double*)malloc((*num_labels) * sizeof(double));
    for (int i = 0; i < *num_labels; ++i) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, file);
        labels[i] = (double)temp;
    }

    fclose(file);
    return labels;
}

double* readImages(const char* path, int* num_images, int* image_size) {
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
    double* images = (double*)malloc((*num_images) * (*image_size) * sizeof(double));

    for (int i = 0; i < *num_images; ++i) {
        for (int j = 0; j < *image_size; ++j) {
            unsigned char temp = 0;
            fread(&temp, sizeof(temp), 1, file);
            images[i * (*image_size) + j] = (double)temp / 255.0;

        }
    }

    fclose(file);
    return images;
}


void printMatrix(double* matrix, int rowSize, int colSize) {
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            printf("%d ", matrix[i * colSize + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void displayImg(const double* image, int rows, int cols) {
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



// Compute softmax activation
void softmax(double* input, int sampleSize, int outputSize) {
    for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
        int buffer = sampleIdx * outputSize;

        double max_val = input[buffer];
        for (int i = 1; i < outputSize; i++) {
            if (input[buffer + i] > max_val) {
                max_val = input[buffer + i];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < outputSize; i++) {
            double tmp = exp(input[buffer + i] - max_val);
            sum += tmp;
        }

        for (int i = 0; i < outputSize; i++) {
            double tmp = exp(input[buffer + i] - max_val);

            input[buffer + i] = (tmp / sum);
        }
    }
}

// Compute loss (cross-entropy)
double computeLoss(double* output, int label, int size) {
    double loss = 0.0f;
    double* labels = (double*)malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) labels[i] = 0.0;
    labels[label] = 1.0;
    for (int i = 0; i < size; i++) {
        loss -= labels[i] * logf(output[i]);
    }
    free(labels);
    return loss;
}

void applyActivationDerivative(double* gradient, double* activation, int size) {
    for (int i = 0; i < size; i++) {
        gradient[i] *= (activation[i] > 0) ? 1 : 0;
    }
}

/*
    n is batch size
    d is 728
    Input size : n x d


*/


__global__ void matrixMultiKernel(double* A, double* B, double* C, int m, int n, int k) {
    __shared__ double s_A[TILE_k][TILE_k];
    __shared__ double s_B[TILE_k][TILE_k];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double s = 0.0f;

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
void matrixMultiplication(double* A, int m, int n, double* B, int k, double* C, bool useDevice = false, dim3 blockSize = dim3(1)) {
    if (!useDevice) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0.0f;
                for (int t = 0; t < n; t++) {
                    sum += A[i * n + t] * B[t * k + j];
                }
                C[i * k + j] = sum;
            }
        }
    }
    else {
        double* d_A, * d_B, * d_C;
        cudaMalloc((void**)&d_A, m * n * sizeof(double));
        cudaMalloc((void**)&d_B, n * k * sizeof(double));
        cudaMalloc((void**)&d_C, m * k * sizeof(double));

        cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);

        dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        matrixMultiKernel <<<gridSize, blockSize >>> (d_A, d_B, d_C, m, n, k);

        cudaMemcpy(C, d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

double* transpose(double* matrix, int rowSize, int colSize, bool useDevice = false) {
    double* output = allocMatrix(colSize, rowSize);
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


// Apply ReLU activation
double applyRelu(double& a) {
    return a >= 0 ? a : 0;
}


void computeGradientForOutputLayer(double* output, double* gradOutput, double* targetLabels, int sampleSize, int outputSize = 10, bool useDevice = false) {
    if (!useDevice) {
        for (int i = 0; i < sampleSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                gradOutput[i * outputSize + j] = output[i * outputSize + j];
            }
            gradOutput[i * outputSize + (int)targetLabels[i]] -= 1.0;
        }
    }
}

void computeGradientForBias(double* gradOutput, double* gradBias, int sampleSize, int outputSize = 10, bool useDevice = false) {
    if (!useDevice) {
        for (int j = 0; j < outputSize; j++) {
            gradBias[j] = 0.0;
            for (int i = 0; i < sampleSize; i++) {
                gradBias[j] += gradOutput[i * outputSize + j];
            }
        }
    }
}


double computeDerivativeHiddenLayer(double& a) {
    return a > 0 ? a : 0;
}

double multiply(double& a, double& b) {
    return a * b;
}

void elementWiseUnary(double* a, double* c, int rowSize, int colSize, double (*unary)(double&), bool useDevice = false) {
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                int idx = i * colSize + j;
                c[idx] = unary(a[idx]);            
            }
        }
    }
}

void elementWiseBinary(double* a, double* b, double* c, int rowSize, int colSize, double (*binary)(double&, double&), bool useDevice = false) {
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                int idx = i * colSize + j;
                c[idx] = binary(a[idx], b[idx]);
            }
        }
    }
}

double addition(double& a, double& b) {
    return a + b;
}

void forward(double* input, double** hiddenWeights, double** activations, double** bias, double* output, 
        int outputSize, int sampleSize, bool useDevice = false, dim3 blockSize = dim3(1), int featureSize = 728) {
    double* currentInput = input;
    int currentInputSize = featureSize;

    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        matrixMultiplication(currentInput, sampleSize, currentInputSize, hiddenWeights[i], HIDDEN_SIZE, activations[i], useDevice, blockSize);
        
        for (int j = 0; j < sampleSize; j++) {
            elementWiseBinary(&activations[i][j * HIDDEN_SIZE], bias[i], &activations[i][j * HIDDEN_SIZE], HIDDEN_SIZE, 1, addition);
        }
        
        elementWiseUnary(activations[i], activations[i], sampleSize, HIDDEN_SIZE, applyRelu);

        currentInputSize = HIDDEN_SIZE;
        currentInput = activations[i];
    }

    matrixMultiplication(currentInput, sampleSize, HIDDEN_SIZE, hiddenWeights[NUM_HIDDEN_LAYERS], outputSize, output, useDevice, blockSize);

    softmax(output,sampleSize, outputSize);
}


double updateWeight(double& org, double& grad) {
    return org - LR * grad;
}

void backward(double* input, double* output, double* targetLabels, double** hiddenWeights, double** activations, 
        double** bias, int sampleSize, bool useDevice = false, dim3 blockSize = dim3(1), int featureSize = 728, int outputSize = OUTPUT_SIZE) {
    double* gradOutput = (double*)malloc(sampleSize * outputSize * sizeof(double));
    double** gradWeights = (double**)malloc((NUM_HIDDEN_LAYERS+1) * sizeof(double*));
    double** gradBias = (double**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(double*));
    int activationColSize = HIDDEN_SIZE;

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int prevSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int currSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        gradWeights[i] = (double*)allocMatrix(prevSize, currSize);
        gradBias[i] = (double*)allocMatrix(currSize, 1);
    }

    computeGradientForOutputLayer(output, gradOutput, targetLabels, sampleSize, outputSize);

    double* gradientToLoss = gradOutput;

    double* outputTranposed = transpose(output, sampleSize, outputSize);

    matrixMultiplication(outputTranposed, outputSize, sampleSize, gradientToLoss, outputSize, gradWeights[NUM_HIDDEN_LAYERS],useDevice,blockSize);

    computeGradientForBias(gradientToLoss, gradBias[NUM_HIDDEN_LAYERS], sampleSize, outputSize);
    
    free(outputTranposed);

    double* weightsTransposed = transpose(hiddenWeights[NUM_HIDDEN_LAYERS], activationColSize, outputSize);
    double* derivativeOfActivation = allocMatrix(sampleSize, activationColSize);
    double* previousGradient = allocMatrix(sampleSize, activationColSize);

    matrixMultiplication(gradientToLoss, sampleSize, outputSize, weightsTransposed, activationColSize, previousGradient, useDevice, blockSize);

    elementWiseUnary(activations[NUM_HIDDEN_LAYERS - 1], derivativeOfActivation, sampleSize, activationColSize, computeDerivativeHiddenLayer);
    elementWiseBinary(previousGradient, derivativeOfActivation, previousGradient, sampleSize, activationColSize, multiply);


    gradientToLoss = previousGradient;
    free(weightsTransposed);
    free(derivativeOfActivation);
    

    for (int i = NUM_HIDDEN_LAYERS-1; i >= 0; i--) {
        int weightColSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        int weightRowSize = (i == 0) ? featureSize : HIDDEN_SIZE;

        double* activationsTransposed = transpose(activations[i], sampleSize, activationColSize);

        matrixMultiplication(activationsTransposed, activationColSize, sampleSize, gradientToLoss, weightColSize, gradWeights[i]);

        computeGradientForBias(gradientToLoss, gradBias[i], sampleSize, weightColSize);

        if (i > 0) {
            weightsTransposed = transpose(hiddenWeights[i], weightRowSize, weightColSize);
            derivativeOfActivation = allocMatrix(sampleSize, activationColSize);
            previousGradient = allocMatrix(sampleSize, activationColSize);

            matrixMultiplication(gradientToLoss, sampleSize, weightColSize, weightsTransposed, activationColSize, previousGradient, useDevice, blockSize);

            elementWiseUnary(activations[i - 1], derivativeOfActivation, sampleSize, activationColSize, computeDerivativeHiddenLayer);
            elementWiseBinary(previousGradient, derivativeOfActivation, previousGradient, sampleSize, activationColSize, multiply);

            free(gradientToLoss);

            gradientToLoss = previousGradient;

            free(weightsTransposed);
            free(derivativeOfActivation);

        }

        free(activationsTransposed);
    }
    
      
    // Update weights and bias
    for (int i = NUM_HIDDEN_LAYERS; i >= 0; i--) {
        int weightColSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        int weightRowSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        elementWiseBinary(hiddenWeights[i], gradWeights[i], hiddenWeights[i], weightRowSize, weightColSize, updateWeight);
        elementWiseBinary(bias[i], gradBias[i], bias[i], 1, weightColSize, updateWeight);
    }

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        freeMatrix(gradWeights[i]);
        freeMatrix(gradBias[i]);
    }
    free(gradWeights);
    free(gradBias);
    free(gradOutput);
}

double calculateCrossEntropyLoss(double* output, double* trueLabels, int sampleSize, int numClasses) {
    double totalLoss = 0.0f;
    for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
        double sampleLoss = 0.0f;
        int label = (int)trueLabels[sampleIdx];
        for (int j = 0; j < numClasses; j++) {
            if(label == j)
                sampleLoss -= log(output[numClasses * sampleIdx + j] + 1e-15f);
        }
        totalLoss += sampleLoss;
    }
    return totalLoss / sampleSize;
}

double calculateAccuracy(double* output, double* trueLabels, int sampleSize, int numClasses) {
    int correct = 0;
    for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
        int truth = (int)trueLabels[sampleIdx];
        double maxPred = 0.0f;
        int label = 0;
        for (int j = 0; j < numClasses; j++) {
            double pred = output[numClasses * sampleIdx + j];
            if (pred > maxPred) {
                maxPred = pred;
                label = j;
            }
        }
        
        // DEBUG print
        // printf("\nLabel: %d - Truth: %d - Prob: %.2f\n", label, truth, maxPred);
        // for (int i = 0; i < numClasses; i++) {
        //     printf("%.2lf ", output[numClasses * sampleIdx + i]);
        // }
        // _sleep(50);
        if (label == truth) {
            correct += 1;
        }
    }
    return (correct * 1.0) / sampleSize;
}

void train(double** dataset, double* labels, int epochSize, int sampleSize, int featureSize, int totalSize, int outputSize = 10) {
    double** hiddenWeights = (double**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(double*));
    double** bias = (double**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(double*));
    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int prevSize = (i == 0) ? featureSize : HIDDEN_SIZE;
        int currSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;

        hiddenWeights[i] = initRandomMatrix(prevSize, currSize, -0.5, 0.5);
        bias[i] = initRandomMatrix(currSize,1, -0.5, 0.5);
        printf("At layer %d: (%d,%d)\n", i, prevSize, currSize);
    }
    double totalLoss = 0.0;

    for (int epoch = 0; epoch < epochSize; epoch++) {
        int batchSize = 0;
        for (int sampleIdx = 0; sampleIdx * sampleSize < totalSize; sampleIdx++) {
            int sampleTrueIdx = sampleIdx * sampleSize;
            int end = (sampleTrueIdx + sampleSize) > totalSize ? totalSize - sampleTrueIdx : sampleSize;
            double* sample = dataset[sampleTrueIdx];
            double** activations = (double**)malloc((NUM_HIDDEN_LAYERS) * sizeof(double*));
            for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
                activations[i] = allocMatrix(end, HIDDEN_SIZE);
            }

            double* output = allocMatrix(end, outputSize);
            dim3 blockSize(256);
            forward(sample, hiddenWeights, activations, bias, output, outputSize, end, true, blockSize);
            
            double loss = calculateCrossEntropyLoss(output, &labels[sampleTrueIdx], end, outputSize);
            // Per batch accuracy (not total)
            double accuracy = calculateAccuracy(output, &labels[sampleTrueIdx], end, outputSize);
            printf("Epoch %d, Batch ID %d, Loss: %.4f, Acc: %.4f\n", epoch + 1, sampleIdx + 1, loss, accuracy);

            backward(sample, output, &labels[sampleTrueIdx], hiddenWeights, activations, bias, end, true, blockSize);

            for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
                free(activations[i]);
            }
            free(activations);
            free(output);
            batchSize++;
        }
    }

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        free(hiddenWeights[i]);
        free(bias[i]);
    }
    free(hiddenWeights);
    free(bias);
}

int main() {
    //loadConfig("config.txt");
    int train_image_count, train_label_count;
    int image_size;
    double* train_images = readImages(TRAIN_IMAGE, &train_image_count, &image_size);
    double* train_labels = readLabels(TRAIN_LABEL, &train_label_count);
    if (!train_images || !train_labels) {
        printf("Failed to load Fashion MNIST data.\n");
        return 1;
    }

    const int epochs = 100;

    double** dataset = (double**)malloc(train_image_count * sizeof(double*));
    for (int i = 0; i < train_image_count; i++) {
        dataset[i] = train_images + i * image_size;
    }

    train(dataset, train_labels, epochs, 5000, image_size, train_image_count);

    free(dataset);
    free(train_images);
    free(train_labels);

    // Test matrix multiplication
    // double a[6] = {2, 3, 4, 5, 6, 7};
    // double b[6] = {7, 8, 9, 10, 11, 12};
    // double c[9];
    // double real_c[9];
    // matrixMultiplication(a, 3, 2, b, 3, c, false);
    // matrixMultiplication(a, 3, 2, b, 3, real_c, false);
    // printMatrix(c, 3, 3);
    // printMatrix(real_c, 3, 3);

    // Test loss function
    // double a[20] = {0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11, 0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11};
    // double b[2] = {0, 2};
    // printf("%.2lf", calculateCrossEntropyLoss(a, b, 2, 10));

    // Test softmax
    // double a[5] = {1.3, 5.1, 2.2, 0.7, 1.1};
    // softmax(a, 5);
    // for (int i = 0; i < 5; i++) {
    //     printf("%.2lf ", a[i]);
    // }

    // Test accuracy
    // double a[20] = {0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11, 0.10, 
    //                 0.10, 0.09, 0.11, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11, 0.10};
    // double b[2] = {0, 2};
    // printf("%.2lf", calculateAccuracy(a, b, 2, 10));
    return 0;
}