#include "preprocess.cu"

// CUDA kernel for matrix multiplication with tiling
__global__ void matrixMultiKernel(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float s_A[TILE_k][TILE_k];
    __shared__ float s_B[TILE_k][TILE_k];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;

    int batch_size = (k - 1) / TILE_k + 1;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int batchStartIdx = batch_idx * TILE_k;

        if (row < m && batchStartIdx + threadIdx.x < n) {
            s_A[threadIdx.y][threadIdx.x] = A[row * n + batchStartIdx + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < k && batchStartIdx + threadIdx.y < n) {
            s_B[threadIdx.y][threadIdx.x] = B[(batchStartIdx + threadIdx.y) * k + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_k; ++i) {
            s += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

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

        dim3 gridSize((k - 1) / blockSize.x + 1, (m - 1) / blockSize.y + 1);
        matrixMultiKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

        cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

// Apply ReLU activation function
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
void forward(float* input, int inputSize, float** hiddenWeights, float** hiddenLayers, int numHiddenLayers, int hiddenSize, float* outputWeights, float* outputLayer, int outputSize) {
    float* currentInput = input;

    for (int i = 0; i < numHiddenLayers; i++) {
        matrixMultiplication(hiddenWeights[i], hiddenSize, inputSize, currentInput, 1, hiddenLayers[i]);
        applyRelu(hiddenLayers[i], hiddenSize);
        currentInput = hiddenLayers[i];
    }

    matrixMultiplication(outputWeights, outputSize, hiddenSize, currentInput, 1, outputLayer);
    softmax(outputLayer, outputSize);
}

void backward(float* output, int outputSize, int trueLabel, float* gradOutput, float* gradHidden, float learningRate) {
   //TODO
}


void train(float** trainData, float* labels, float lr){
    //TODO
}