#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_K 32


struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

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
float LR = 1e-4;
char TRAIN_IMAGE[512] = "content/fashion/train-images-idx3-ubyte";
char TRAIN_LABEL[512] = "content/fashion/train-labels-idx1-ubyte";
char TEST_IMAGE[512] = "content/fashion/t10k-images-idx3-ubyte";
char TEST_LABEL[512] = "content/fashion/t10k-labels-idx1-ubyte";
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
    } else if (strcmp(key, "BEST_ACCURACY") == 0) {
        sscanf(value, "%f", &BEST_ACCURACY);
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
            fprintf(file, "%f ", hiddenWeights[i][j]);
        }
        fprintf(file, "\n");
    }

    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int currSize = (i == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        for (int j = 0; j < currSize; j++) {
            fprintf(file, "%f ", bias[i][j]);
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
            fscanf(file, "%f", &tmp);
            hiddenWeights[i][j] = tmp;
        }
    }

    printf("Loading biases\n");
    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        int currSize = (i == NUM_HIDDEN_LAYERS) ? OUTPUT_SIZE : HIDDEN_SIZE;
        for (int j = 0; j < currSize; j++) {
            float tmp = 0.0f;
            fscanf(file, "%f", &tmp);
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

    float limit = sqrt(2.0f / rowSize); 

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

//Check if the image is valid or not
void displayBlank(const float* image, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float tmp = image[r * cols + c];
            tmp += 1;
        }
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
                images[batchIdx][i * (*image_size) + j] = (float)temp / 255.0;
            }
        }
    }
    

    fclose(file);
    return images;
}

__global__ void softmaxKernel(float* d_in, float* d_out, int batchSize, int outputSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < batchSize) {
        int buffer = idx * outputSize;

        float max_val = d_in[buffer];
        for (int i = 1; i < outputSize; i++) {
            max_val = max_val >= d_in[buffer + i] ? max_val : d_in[buffer + i] ;
        }

        float sum = 0.0f;
        for (int i = 0; i < outputSize; i++) {
            d_out[buffer + i] = __expf(d_in[buffer + i] - max_val);
            sum += d_out[buffer + i];
        }

        for (int i = 0; i < outputSize; i++) {
            d_out[buffer + i] /= sum;
        }
    }
}


void softmax(float* input, float* output, int batchSize, int outputSize, bool useDevice, dim3 blockSize) {
    if(!useDevice){    
        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            int buffer = batchIdx * outputSize;

            float max_val = input[buffer];
            for (int i = 1; i < outputSize; i++) {
                max_val = fmax(max_val, input[buffer + i]);
            }

            float sum = 0.0;
            for (int i = 0; i < outputSize; i++) {
                output[buffer + i] = expf(input[buffer + i] - max_val);
                sum += output[buffer + i];
            }

            for (int i = 0; i < outputSize; i++) {
                output[buffer + i] /= sum;
            }
        }
    } else {
        int eleSize = batchSize * outputSize; 
        float* d_in, * d_out;
        CHECK(cudaMalloc((void**)&d_in, eleSize * sizeof(float)));
        CHECK(cudaMalloc((void**)&d_out, eleSize* sizeof(float)));

        CHECK(cudaMemcpy(d_in, input, eleSize * sizeof(float), cudaMemcpyHostToDevice));
        blockSize = dim3(32); // Because working on a 1D grid it better to do this way
        dim3 gridSize((batchSize- 1) / blockSize.x + 1);
        softmaxKernel<<<gridSize, blockSize>>>(d_in, d_out, batchSize, outputSize);

        CHECK(cudaMemcpy(output, d_out, eleSize * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_in);
        cudaFree(d_out);
    }
}

__global__ void matrixMultiKernel(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float s_A[TILE_K][TILE_K];
    __shared__ float s_B[TILE_K][TILE_K];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;

    for (int batch_idx = 0; batch_idx < (n + TILE_K - 1) / TILE_K; batch_idx++) {
        int A_col = batch_idx * TILE_K + threadIdx.x;
        int B_row = batch_idx * TILE_K + threadIdx.y;

        s_A[threadIdx.y][threadIdx.x] = (row < m && A_col < n) ? A[row * n + A_col] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] = (col < k && B_row < n) ? B[B_row * k + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_K; i++) {
            s += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = s;
    }
}


__global__ void updateWeightKernel(float* hiddenWeight, float* grad, int rowSize, int colSize,float d_LR) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * colSize + j;

    if (i < rowSize && j < colSize) {
        hiddenWeight[idx] -= d_LR * grad[idx];
    }
}

void initUpdateStream(cudaStream_t* stream, float* hiddenWeight, float* grad, float* d_hiddenWeight, float* d_grad, int rowSize, int colSize, dim3 blockSize, float LR) {
    CHECK(cudaStreamCreate(stream));
    
    int totalSize = rowSize * colSize;
            
    CHECK(cudaMemcpyAsync(d_hiddenWeight, hiddenWeight, totalSize * sizeof(float), cudaMemcpyHostToDevice, *stream));
    CHECK(cudaMemcpyAsync(d_grad, grad, totalSize * sizeof(float), cudaMemcpyHostToDevice, *stream));

    dim3 gridSize((colSize + blockSize.x - 1) / blockSize.x, (rowSize + blockSize.y - 1) / blockSize.y);

    updateWeightKernel<<<gridSize, blockSize, 0, *stream>>>(d_hiddenWeight, d_grad, rowSize, colSize, LR);

    CHECK(cudaMemcpyAsync(hiddenWeight, d_hiddenWeight, totalSize * sizeof(float), cudaMemcpyDeviceToHost, *stream));

}

void updateWeights(float** hiddenWeights, float** grads, int featureSize, int outputSize, float LR, bool isUpdateWeight, bool useDevice = false, dim3 blockSize = dim3(1)) {
    if (!useDevice) {
        for (int layer = 0; layer <= NUM_HIDDEN_LAYERS; layer++) {
            int currSize = (layer == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
            int prevSize = (layer == 0) ? featureSize : HIDDEN_SIZE;

             if(!isUpdateWeight) { // to update bias also
                prevSize = currSize;
                currSize = 1;
            }
            
            float* hiddenWeight = hiddenWeights[layer];
            float* grad = grads[layer];

            for (int i = 0; i < prevSize; i++) {
                for (int j = 0; j < currSize; j++) {
                    int idx = i * currSize + j;
                    hiddenWeight[idx] -= LR * grad[idx];
                }
            }
        }
    } else {
        int nStreams = NUM_HIDDEN_LAYERS + 1;

        cudaStream_t* streams = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));
        float** d_hiddenWeights = (float**)malloc(nStreams * sizeof(float*));
        float** d_grads = (float**)malloc(nStreams * sizeof(float*));

        for (int layer = 0; layer <= NUM_HIDDEN_LAYERS; layer++) {
            int currSize = (layer == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
            int prevSize = (layer == 0) ? featureSize : HIDDEN_SIZE;
            if(!isUpdateWeight) { // to update bias also
                prevSize = currSize;
                currSize = 1;
            }

            CHECK(cudaMalloc((void**)&d_hiddenWeights[layer], currSize * prevSize * sizeof(float)));
            CHECK(cudaMalloc((void**)&d_grads[layer], prevSize * currSize * sizeof(float)));
            initUpdateStream(&streams[layer], hiddenWeights[layer], grads[layer], d_hiddenWeights[layer], d_grads[layer], prevSize, currSize, blockSize, LR);
        }

        for (int layer = 0; layer <= NUM_HIDDEN_LAYERS; layer++) {
            CHECK(cudaStreamSynchronize(streams[layer]));
            CHECK(cudaFree(d_hiddenWeights[layer]));
            CHECK(cudaFree(d_grads[layer]));
            CHECK(cudaStreamDestroy(streams[layer]));
        }
        
        free(streams);
        free(d_hiddenWeights);
        free(d_grads);
    }
}


void matrixMultiplication(float* A, int m, int n, float* B, int k, float* C, bool useDevice = false, dim3 blockSize = dim3(16, 16)) {
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
        CHECK(cudaMalloc((void**)&d_A, m * n * sizeof(float)));
        CHECK(cudaMalloc((void**)&d_B, n * k * sizeof(float)));
        CHECK(cudaMalloc((void**)&d_C, m * k * sizeof(float)));

        CHECK(cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice));

        dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
        matrixMultiKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, m, n, k);

        CHECK(cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

__global__ void transpose_kernel(float* input, float* output, int iRowSize, int iColSize)
{
    __shared__ float s_blkData[TILE_K][TILE_K + 1];

    int iR = blockIdx.x * blockDim.x + threadIdx.x;
    int iC = blockIdx.y * blockDim.y + threadIdx.y;
    if (iR < iRowSize && iC < iColSize) {
        s_blkData[threadIdx.x][threadIdx.y] = input[iR * iColSize + iC];
    }

    __syncthreads();
        
    int oR = blockIdx.y * blockDim.y + threadIdx.y;
    int oC = blockIdx.x * blockDim.x + threadIdx.x;
    if (oR < iColSize && oC < iRowSize) {
        output[oR * iRowSize + oC] = s_blkData[threadIdx.x][threadIdx.y];
    }
}

float* transpose(float* matrix, int rowSize, int colSize, bool useDevice = false, dim3 blockSize = dim3(1)) {
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
    else {
        float* d_in, * d_out;
        size_t nBytes = rowSize * colSize * sizeof(float);
        CHECK(cudaMalloc((void**)&d_in, nBytes));
        CHECK(cudaMalloc((void**)&d_out, nBytes));

        CHECK(cudaMemcpy(d_in, matrix, nBytes, cudaMemcpyHostToDevice));

        dim3 gridSize((colSize - 1) / blockSize.x + 1, (rowSize - 1) / blockSize.y + 1);

        transpose_kernel<< <gridSize, blockSize >> > (d_in, d_out, rowSize,colSize);

        CHECK(cudaDeviceSynchronize();)
        CHECK(cudaMemcpy(output, d_out, nBytes, cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));

    }
    return output;
}


void computeGradientForOutputLayer(float* output, float* gradOutput, float* targetLabels, int batchSize, int outputSize = 10) {
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            gradOutput[i * outputSize + j] = output[i * outputSize + j];
        }
        gradOutput[i * outputSize + (int)targetLabels[i]] -= 1.0;
    }
}
__global__ void computeGradBiasKernel(float* d_gradBias, float* d_gradToLoss, int batchSize, int outputSize) {
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.x * blockIdx.x + threadIdx.y;
    if (rowIdx < batchSize && colIdx < outputSize) {
        float addVal = d_gradToLoss[colIdx * outputSize + rowIdx];
        atomicAdd(&d_gradBias[colIdx], addVal);
    }
}

void computeGradientForBias(float* gradToLoss, float* gradBias, int batchSize, int outputSize, bool useDevice=false, dim3 blockSize= dim3(1)) {
    for (int i = 0; i < outputSize; i++) {
        gradBias[i] = 0.0;
    }

    if (!useDevice) {
        for (int j = 0; j < batchSize; j++) {
            for (int i = 0; i < outputSize; i++) {
                gradBias[i] += gradToLoss[j * outputSize + i];
            }
        }
    } else {
        int totalSize = batchSize * outputSize;
        float* d_gradBias;
        float* d_gradToLoss;

        CHECK(cudaMalloc((void**)&d_gradBias, totalSize * sizeof(float)));
        CHECK(cudaMalloc((void**)&d_gradToLoss, totalSize * sizeof(float)));

        cudaMemcpy(d_gradBias, gradBias, totalSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gradToLoss, gradToLoss, totalSize * sizeof(float), cudaMemcpyHostToDevice);

        dim3 gridSize((batchSize - 1) / blockSize.x + 1, (outputSize - 1) / blockSize.y + 1);

        computeGradBiasKernel<<<gridSize, blockSize>>>(d_gradBias, d_gradToLoss, batchSize, outputSize);

        cudaMemcpy(gradBias, d_gradBias, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_gradBias);
        cudaFree(d_gradToLoss);
    }

    for (int i = 0; i < outputSize; i++) {
        gradBias[i] /= batchSize;
    }
}

__global__ void reluKernel(float* a, float* c, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalSize) {
        c[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
    }
}

void applyRelu(float* a, float* c, int rowSize, int colSize, bool useDevice = false, dim3 blockSize = dim3(1)) {
    if (!useDevice) {
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                int idx = i * colSize + j;
                c[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
            }
        }
    } else {
        int totalSize = rowSize * colSize;
        float* d_a, *d_c;

        CHECK(cudaMalloc((void**)&d_a, totalSize * sizeof(float)));
        CHECK(cudaMalloc((void**)&d_c, totalSize * sizeof(float)));

        cudaMemcpy(d_a, a, totalSize * sizeof(float), cudaMemcpyHostToDevice);

        int gridSize = (totalSize + blockSize.x - 1) / blockSize.x;

        reluKernel<<<gridSize, blockSize>>>(d_a, d_c, totalSize);

        cudaMemcpy(c, d_c, totalSize * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_c);
    }
}

float retainPositive(float org, float dest) {
    return dest > 0 ? org : 0.0;
}

float multiply(float a, float b) {
    return a * b;
}

float addition(float a, float b) {
    return a + b;
}

typedef float (*op_func_p) (float, float);

__device__ float retainPositiveDevice(float org, float dest) {
    return dest > 0 ? org : 0.0;
}

__device__ float multiplyDevice(float a, float b) {
    return a * b;
}

__device__ float additionDevice(float a, float b) {
    return a + b;
}
__device__ float updateWeightDevice(float org, float grad) {
    return org - 0.00005 * grad;
}
__device__ __constant__ op_func_p h_addition = additionDevice;
__device__ __constant__ op_func_p h_multiply = multiplyDevice;
__device__ __constant__ op_func_p h_retain_positive = retainPositiveDevice;
__device__ __constant__ op_func_p h_update_weight = updateWeightDevice;

__device__ op_func_p getFunc(int opCode) {
    switch (opCode) {
    case 0:
        return h_addition;
    case 1:
        return h_multiply;
    case 2:
        return h_retain_positive;
    case 3:
        return h_update_weight;
    default:
        return h_addition;
    }
}


__global__ void binaryKernel(float* a, float* b, float* c, int rowSize, int colSize, int opCode) {
    op_func_p binary = getFunc(opCode);
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rowSize && col < colSize) {
        int i = row * colSize + col;
        c[i] = binary(a[i], b[i]);
    }

}

void elementWiseBinary(float* a, float* b, float* c, int rowSize, int colSize, float (*binary)(float, float), int opCode, dim3 blockSize = dim3(1), bool useDevice = false) {
    if (!useDevice) {
        // printf("Not using device\n");
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                int idx = i * colSize + j;
                c[idx] = binary(a[idx], b[idx]);
            }
        }
    }
    else {
        float *d_a, *d_b, *d_c;
        size_t bytes = rowSize * colSize * sizeof(float);
        cudaMalloc((void**)&d_a, bytes);
        cudaMalloc((void**)&d_b, bytes);
        cudaMalloc((void**)&d_c, bytes);
        
        cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
        dim3 gridSize(((colSize) + blockSize.x - 1) / blockSize.x, ((rowSize) + blockSize.y - 1) / blockSize.y);
        binaryKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, rowSize, colSize, opCode);
        cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
}


void forward(float* input, float** hiddenWeights, float** activations, float** bias, float* output, float** Z, float* zOutput, 
    int outputSize, int batchSize, bool useDevice = false, int featureSize = 784) {
    float* currentInput = input;
    int currentInputSize = featureSize;
    dim3 blockSize = dim3(1);
    if (useDevice) {
        blockSize.x = 32;
        blockSize.y = 32;
    }
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        matrixMultiplication(currentInput, batchSize, currentInputSize, hiddenWeights[i], HIDDEN_SIZE, Z[i], useDevice, blockSize);

        for (int j = 0; j < batchSize; j++) {
            elementWiseBinary(&Z[i][j * HIDDEN_SIZE], bias[i], &Z[i][j * HIDDEN_SIZE], HIDDEN_SIZE, 1, addition, 0, blockSize, false);
        }

        applyRelu(Z[i], activations[i], batchSize, HIDDEN_SIZE,useDevice, blockSize);

        currentInputSize = HIDDEN_SIZE;
        currentInput = activations[i];
    }
    
    matrixMultiplication(currentInput, batchSize, HIDDEN_SIZE, hiddenWeights[NUM_HIDDEN_LAYERS], outputSize, zOutput, useDevice, blockSize);

    for (int j = 0; j < batchSize; j++) {
        elementWiseBinary(&zOutput[j * outputSize], bias[NUM_HIDDEN_LAYERS], &zOutput[j * outputSize], outputSize, 1, addition, 0, blockSize, false);
    }

    softmax(zOutput, output, batchSize, outputSize,useDevice, blockSize);
}


float decayedLR(float orgLR, float decayRate, int step, int decaySteps){
    return orgLR * (float)pow(decayRate, ((step * 1.0)/ decaySteps));
}

float lrDecay(float beta1, float beta2, int epoch, int epochDrop) {
    epoch = 5;
    printf("%f ", (1.0 - powf(beta2, epoch * 1.0)));
    printf("%f ", (1.0 - powf(beta1, epoch * 1.0)));
    printf("%f \n", sqrt((1.0 - powf(beta2, epoch * 1.0)) / (1.0 - powf(beta1, epoch * 1.0))));
    return LR * sqrt((1.0 - powf(beta2, epoch * 1.0)) / (1.0 - powf(beta1, epoch * 1.0)));
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

    dim3 blockSize(32, 32);
    if (!useDevice) blockSize = dim3(1);


    for (int layer = NUM_HIDDEN_LAYERS; layer >= 0; layer--) {
        int currSize = (layer == NUM_HIDDEN_LAYERS) ? outputSize : HIDDEN_SIZE;
        int prevSize = (layer == 0) ? featureSize : HIDDEN_SIZE;

        float* activationsTransposed = (layer == 0) ? transpose(input, batchSize, featureSize,useDevice, blockSize) : transpose(activations[layer - 1], batchSize, prevSize, useDevice, blockSize);

        matrixMultiplication(activationsTransposed, prevSize, batchSize, gradientToLoss, currSize, gradWeights[layer], useDevice, blockSize);
      
        free(activationsTransposed);
        computeGradientForBias(gradientToLoss, gradBias[layer], batchSize, currSize, useDevice, blockSize);

        if (layer == 0) break;

        float* weightsTransposed = transpose(hiddenWeights[layer], prevSize, currSize, useDevice, blockSize);
        float* previousGradient = allocMatrix(batchSize, prevSize);

        matrixMultiplication(gradientToLoss, batchSize, currSize, hiddenWeights[layer], prevSize, previousGradient, useDevice, blockSize);
 

        //For derivative of ReLu is a > 0 ? 1.0 : 0.0, and compute of element wise. So it would better to combine the two operation into 1
        elementWiseBinary(previousGradient, Z[layer - 1], previousGradient, batchSize, prevSize, retainPositive, 2, blockSize, false); 

        free(weightsTransposed);
        if(layer < NUM_HIDDEN_LAYERS) 
            free(gradientToLoss);
        gradientToLoss = previousGradient;
    }

    updateWeights(hiddenWeights, gradWeights, featureSize, outputSize, LR, true, useDevice, blockSize);
    updateWeights(bias, gradBias, featureSize, outputSize, LR, false, useDevice, blockSize);

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
        if (label == truth) {
            correct += 1;
        }
        labels[label] += 1;
    }
    return correct * 1.0;
}

void train(float** dataset, float* labels, float** hiddenWeights, float** bias, int epochSize, int batchSize, int featureSize, int totalSize, const char* configFile,bool useDevice,int step_save = 5, int outputSize = 10) {
    int decaySteps = 100000;
    float decayRate = 0.9995;
    int steps = 0;
    float totalTime = 0.0f;
    for (int epoch = 0; epoch < epochSize; epoch++) {
        double totalLoss = 0.0;
        double totalAccuracy = 0.0;
        int numBatch = (totalSize - 1) / batchSize + 1;
        GpuTimer timer;
        timer.Start();
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
            forward(dataset[batchIdx], hiddenWeights, activations, bias, output,Z, zOutput, outputSize, end, useDevice);

            totalLoss += calculateCrossEntropyLoss(output, batchLabels, end, outputSize);
            totalAccuracy += calculateAccuracy(output, batchLabels, end, outputSize);

            backward(dataset[batchIdx], output, batchLabels, hiddenWeights, activations, bias, Z, zOutput, end, useDevice);
            steps++;

            free(output);
            free(zOutput);
            for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
                free(activations[i]);
                free(Z[i]);
            }
            free(activations);
            free(Z);
        }
        timer.Stop();
        totalLoss /= totalSize;
        totalAccuracy /= totalSize;
        LR = decayedLR(LR, decayRate, steps, decaySteps);

        if ((epoch + 1) % step_save == 0) {
            char saveFile[256];
            snprintf(saveFile, sizeof(saveFile), "./checkpoints/wandb_%d.txt", epoch);
            saveWANDB(hiddenWeights, bias, featureSize, outputSize, saveFile);
            if (fabs(totalAccuracy - BEST_ACCURACY) > 1e-4f) { // Make sure the difference it correct since this one is floating point
                saveWANDB(hiddenWeights, bias, featureSize, outputSize, "best.txt");
                modifyConfig(configFile, "BEST_CHECKPOINT", "best.txt");
                char accuracyStr[10];
                snprintf(accuracyStr, sizeof(accuracyStr), "%f", totalAccuracy);
                BEST_ACCURACY = totalAccuracy;
                modifyConfig(configFile, "BEST_ACCURACY", accuracyStr);
            }
        }
        float time = timer.Elapsed();
        printf("Epoch %d, Loss: %.4f, Accuracy: %.4f, Time (seconds): %.4f\n", epoch + 1, totalLoss, totalAccuracy, time / 1000);
        totalTime += time;
   }
    
    printf("Processing time (%s): %f s\n\n", useDevice == true? "use device" : "use host", ((totalTime / epochSize) / 1000));
    
    for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
        free(hiddenWeights[i]);
        free(bias[i]);
    }
    free(hiddenWeights);
    free(bias);
}


// Test model
void test(float** dataset, float* labels, float** hiddenWeights, float** bias, int featureSize, int batchSize, int totalSize, int outputSize = 10, bool useDevice = false) {
    double totalAccuracy = 0.0f;
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
        forward(dataset[batchIdx], hiddenWeights, activations, bias, output,Z, zOutput, outputSize, end, useDevice);

        totalAccuracy += calculateAccuracy(output, batchLabels, end, outputSize);

        free(output);
        free(zOutput);
        for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
            free(activations[i]);
            free(Z[i]);
        }
        free(activations);
        free(Z);

    }
    totalAccuracy /= totalSize;
    printf("Total accuracy: %f\n", totalAccuracy);
}

int main(int argc, char *argv[]) {
    // Pass runtime arguments to choose config file or testing mode
    // Default run also works. Example command run:
    // ./a.exe test config.txt
    bool runTest = false;
    char configFile[256] = "config.txt";
    if (argc > 1) {
        if (strcmp(argv[1], "test") == 0) {
            runTest = true;
        }
        if (argc > 2) {
            strcpy(configFile, argv[2]);
        }
    }

    loadConfig(configFile);
    int train_image_count, train_label_count, test_image_count, test_label_count;
    int image_size;
    bool useDevice = true;
    const int epochs = 3;
    const int batchSize = 32 * 10;

    float** hiddenWeights = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    float** bias = (float**)malloc((NUM_HIDDEN_LAYERS + 1) * sizeof(float*));
    // Set runTest = true to load weights & biases from file
    if (runTest) {
        float** test_images = readImages(TEST_IMAGE, &test_image_count, &image_size, batchSize);
        float* test_labels = readLabels(TEST_LABEL, &test_label_count);
        if (!test_images || !test_labels) {
            printf("Failed to load Fashion MNIST training data.\n");
            return 1;
        }
        for (int i = 0; i <= NUM_HIDDEN_LAYERS; i++) {
            int prevSize = (i == 0) ? image_size : HIDDEN_SIZE;
            int currSize = (i == NUM_HIDDEN_LAYERS) ? OUTPUT_SIZE : HIDDEN_SIZE;

            hiddenWeights[i] = initHeMatrix(prevSize, currSize);
            bias[i] = initFilledMatrix(currSize,1, 0.0);
            printf("At layer %d: (%d,%d)\n", i, prevSize, currSize);
        }
        bool check = initWANDB(hiddenWeights, bias, image_size, OUTPUT_SIZE, runTest);
        if (!check) {
            perror("Error intializing weights & biases.\nTerminating program...\n");
            free(test_images);
            free(test_labels);
            return 1;
        }
        
        test(test_images, test_labels, hiddenWeights, bias, image_size, batchSize, test_image_count,10);
        for (int i = 0; i < (test_image_count - 1 / batchSize) + 1; i++) {
            free(test_images[i]);
        }
        free(test_images);
        free(test_labels);
    } 
    else {
        float** train_images = readImages(TRAIN_IMAGE, &train_image_count, &image_size, batchSize);
        float* train_labels = readLabels(TRAIN_LABEL, &train_label_count);
        if (!train_images || !train_labels) {
            printf("Failed to load Fashion MNIST training data.\n");
            return 1;
        }
        bool check = initWANDB(hiddenWeights, bias, image_size, OUTPUT_SIZE, runTest);
        if (!check) {
            perror("Error intializing weights & biases.\nTerminating program...\n");
            free(train_images);
            free(train_labels);
            return 1;
        }

        train(train_images, train_labels, hiddenWeights, bias, epochs, batchSize, image_size, train_image_count, configFile,useDevice,10);
        int numBatch = (train_image_count - 1) / batchSize + 1;

        for (int i = 0; i < numBatch; i++) {
            free(train_images[i]);
        }
        free(train_images);
        free(train_labels);
    }

    return 0;
}