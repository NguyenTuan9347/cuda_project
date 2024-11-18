#include "preprocess.cu"


__global__ void matrixMultiKernel(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_k][TILE_k];
	__shared__ float s_B[TILE_k][TILE_k];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;

    int batch_size = (k - 1) / TILE_k + 1;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int batchStartIdx = batch_idx * TILE_k;
        if (row < m && batchStartIdx + threadIdx.x < n) {
            s_A[threadIdx.y][threadIdx.x] = A[(batchStartIdx + threadIdx.x) + row * n];
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

void matrixMultiplication(Matrix* A, Matrix* B, Matrix* C, bool useDevice = false, dim3 blockSize = dim3(1),int kernelType=1)
{
    if (useDevice == false)
    {
        for(int i=0; i < A->rowSize;i++){
            for(int j = 0;j < B->colSize;j++){
                int index = i * B->colSize + j;
                C->val[index] = 0;
                for(int t=0;t < A->colSize;t++){
                    C->val[index] += A->val[i * A->colSize + t] * B->val[t * B->colSize + j];
                }
            }
        }
    }
    else 
    {
        float* d_A, * d_B, * d_C;
        int m = A->rowSize, n = A->colSize, k = B->colSize;

        cudaMalloc((void**)&d_A, m * n * sizeof(float));
        cudaMalloc((void**)&d_B, k * n * sizeof(float));
        cudaMalloc((void**)&d_C, k * m * sizeof(float));
        cudaMemcpy(d_A, A->val, sizeof(float) * m * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B->val, sizeof(float) * k * n, cudaMemcpyHostToDevice);
        dim3 gridSize((k-1) / blockSize.x + 1, (m-1) / blockSize.y + 1); 
		
        matrixMultiKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

        cudaMemcpy(C->val,d_C, sizeof(float) * k * m, cudaMemcpyDeviceToHost);
        
        cudaFree(d_C);
        cudaFree(d_B);
        cudaFree(d_A);
		printf("Grid size: %d * %d, block size: %d * %d\n", 
			gridSize.x,gridSize.y, blockSize.x,blockSize.y);

    }
}

Matrix* softmax(Matrix *input) {
    if(input->colSize > 1) return nullptr;
    Matrix* res = allocMatrix(input->rowSize, input->colSize);
    float sum = 0.0f;
    int size = input->rowSize;
    for(int i = 0;i < size;i++){
        sum += expf(input->val[i]);
    } 
    for (int i = 0; i < size; i++){
        res->val[i] = input->val[i] / sum;       
    }
    return res;
}



void applyRelu(Matrix* layer){
    int size = layer->colSize * layer->rowSize;
    for(int i=0; i< size;i++){
        layer->val[i] = layer->val[i] > 0 ? layer->val[i] : 0; 
    }
}

void forward(Matrix* input, Matrix** hiddenWeights, Matrix** hiddenLayers, int numHiddenLayers, Matrix* outputWeights, Matrix* outputLayer) {
    Matrix* currentInput = input;

    for (int i = 0; i < numHiddenLayers; i++) {
        matrixMultiplication(hiddenWeights[i], currentInput, hiddenLayers[i]);

        applyRelu(hiddenLayers[i]);  // Apply ReLU to each hidden layer

        currentInput = hiddenLayers[i];
    }

    matrixMultiplication(outputWeights, currentInput, outputLayer);

    Matrix* softmaxOutput = softmax(outputLayer);
    copyMatrix(outputLayer, softmaxOutput);
    freeMatrix(softmaxOutput);
}
void crossEntropyLossGradient(Matrix* output, int trueLabel, Matrix* gradOutput) {
    copyMatrix(gradOutput, output); 
    gradOutput->val[trueLabel] -= 1.0;
}

void reluDerivative(Matrix* layer, Matrix* gradLayer) {
    int size = layer->rowSize * layer->colSize;
    for (int i = 0; i < size; i++) {
        gradLayer->val[i] = layer->val[i] > 0 ? 1.0 : 0.0;
    }
}

void backward(Matrix* input, Matrix** hiddenWeights, Matrix** hiddenLayers, Matrix* outputWeights, Matrix* outputLayer, int trueLabel, float learningRate, int numHiddenLayers) {

    Matrix* gradOutput = allocMatrix(outputLayer->rowSize, outputLayer->colSize);
    crossEntropyLossGradient(outputLayer, trueLabel, gradOutput);


    Matrix* gradOutputWeights = allocMatrix(outputWeights->rowSize, outputWeights->colSize);
    matrixMultiplication(gradOutput, hiddenLayers[numHiddenLayers - 1], gradOutputWeights, false, dim3(1));


    Matrix* gradHiddenLayer = allocMatrix(hiddenLayers[numHiddenLayers - 1]->rowSize, hiddenLayers[numHiddenLayers - 1]->colSize);
    matrixMultiplication(outputWeights, gradOutput, gradHiddenLayer, true, dim3(1));
    reluDerivative(hiddenLayers[numHiddenLayers - 1], gradHiddenLayer);


    for (int i = numHiddenLayers - 1; i >= 0; i--) {
        Matrix* gradHiddenWeights = allocMatrix(hiddenWeights[i]->rowSize, hiddenWeights[i]->colSize);
        matrixMultiplication(gradHiddenLayer, i > 0 ? hiddenLayers[i - 1] : input, gradHiddenWeights, false, dim3(1));

        for (int j = 0; j < hiddenWeights[i]->rowSize * hiddenWeights[i]->colSize; j++) {
            hiddenWeights[i]->val[j] -= learningRate * gradHiddenWeights->val[j];
        }

        if (i > 0) {
            Matrix* prevGradHiddenLayer = allocMatrix(hiddenLayers[i - 1]->rowSize, hiddenLayers[i - 1]->colSize);
            matrixMultiplication(hiddenWeights[i], gradHiddenLayer, prevGradHiddenLayer, true, dim3(1));
            reluDerivative(hiddenLayers[i - 1], prevGradHiddenLayer);
            freeMatrix(gradHiddenLayer);
            gradHiddenLayer = prevGradHiddenLayer;
        }

        freeMatrix(gradHiddenWeights);
    }

    for (int j = 0; j < outputWeights->rowSize * outputWeights->colSize; j++) {
        outputWeights->val[j] -= learningRate * gradOutputWeights->val[j];
    }

    freeMatrix(gradOutput);
    freeMatrix(gradOutputWeights);
    freeMatrix(gradHiddenLayer);
}

void train(Matrix* trainData, Matrix* trainLabels, int numClasses, int numEpochs, float learningRate) {
    int inputSize = trainData->colSize;
    int hiddenSize = 64;
    int outputSize = numClasses;

    Matrix* hiddenWeights[NUM_HIDDEN_LAYERS];
    Matrix* hiddenLayers[NUM_HIDDEN_LAYERS];
    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        hiddenWeights[i] = initRandomMatrix(hiddenSize, i == 0 ? inputSize : hiddenSize);
        hiddenLayers[i] = allocMatrix(hiddenSize, 1);
    }

    Matrix* outputWeights = initRandomMatrix(outputSize, hiddenSize);
    Matrix* outputLayer = allocMatrix(outputSize, 1);

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        for (int i = 0; i < trainData->rowSize; i++) {
            Matrix* input = trainData->getRow(i);

            int trueLabel = (int)trainLabels->val[i];

            forward(input, hiddenWeights, hiddenLayers, NUM_HIDDEN_LAYERS, outputWeights, outputLayer);

            backward(input, hiddenWeights, hiddenLayers, outputWeights, outputLayer, trueLabel, learningRate, NUM_HIDDEN_LAYERS);

            freeMatrix(input);
        }
    }

    for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        freeMatrix(hiddenWeights[i]);
        freeMatrix(hiddenLayers[i]);
    }
    freeMatrix(outputWeights);
    freeMatrix(outputLayer);
}