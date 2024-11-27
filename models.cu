#include "preprocess.cu"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// Define TILE_k for shared memory tiling
#define TILE_k 16  // Adjust this value based on hardware capabilities

// CUDA kernel for matrix multiplication with tiling
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

// Forward pass
void forward(float* input, int inputSize, float** hiddenWeights, float** hiddenLayers, float* output, int outputSize, bool useDevice = false, dim3 blockSize = dim3(1)) {
    float* currentInput = input;
    int currentInputSize = inputSize;

    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        matrixMultiplication(currentInput,1 , currentInputSize, hiddenWeights[i], NUM_HIDDEN_LAYERS, hiddenLayers[i], useDevice, blockSize);
        applyRelu(hiddenLayers[i], HIDDEN_SIZE);
        currentInputSize = HIDDEN_SIZE;
        currentInput = hiddenLayers[i];
    }

    matrixMultiplication(currentInput, 1, HIDDEN_SIZE, hiddenWeights[NUM_HIDDEN_LAYERS-1], outputSize, output, useDevice, blockSize);
    softmax(output, outputSize);
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

// Backward pass
void backward(float* sample, int label, float** hiddenWeights, float** hiddenLayers, float* output, int outputSize, bool useDevice, dim3 blockSize) {

}

// Update weights using gradient descent
void updateWeights(float** hiddenWeights,float** hiddenLayers,float lr) {
   
}

void train(float** dataset, float* labels, int epochSize, int sampleSize, int outputSize, bool useDevice = false, dim3 blockSize = dim3(1), float lr = 0.01f) {
    // Allocate memory for hidden layers, weights, and biases
    float** hiddenLayers = (float**) malloc(NUM_HIDDEN_LAYERS * sizeof(float*));
    float** hiddenWeights = (float**) malloc(NUM_HIDDEN_LAYERS * sizeof(float*));

    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        hiddenLayers[i] = initRandomMatrix(HIDDEN_SIZE);
        hiddenWeights[i] = initRandomMatrix(i == 0 ? SIZE : HIDDEN_SIZE, i == NUM_HIDDEN_LAYERS - 1 ? outputSize : HIDDEN_SIZE);
    }

    // Training loop
    for (int epoch = 0; epoch < epochSize; epoch++) {
        float totalLoss = 0.0f;  // Accumulate loss for reporting
        for (int sampleIdx = 0; sampleIdx < sampleSize; sampleIdx++) {
            float* sample = dataset[sampleIdx];  // Input data sample
            float* output = (float*) malloc(sizeof(float) * outputSize);

            // Forward pass to compute output
            forward(sample, SIZE, hiddenWeights, hiddenLayers, output, outputSize, useDevice, blockSize);

            // Compute the loss for this sample
            float loss = computeLoss(output, labels[sampleIdx], outputSize);
            totalLoss += loss;  // Add the loss for this sample to total loss

            // Backpropagation to calculate gradients (assuming a backward function exists)
            backward(sample, labels[sampleIdx], hiddenWeights, hiddenLayers, output, outputSize, useDevice, blockSize);

            // Update the weights and biases using gradient descent or another optimizer
            updateWeights(hiddenWeights, hiddenLayers, lr);

            free(output);
        }

        // Print the average loss for this epoch
        printf("Epoch %d, Loss: %.4f\n", epoch + 1, totalLoss / sampleSize);
    }

    // Free allocated memory for weights and biases
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        freeMatrix(hiddenLayers[i]);
    }
    free(hiddenLayers);
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        freeMatrix(hiddenWeights[i]);
    }
    free(hiddenWeights);
}

