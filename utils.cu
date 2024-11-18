#include <stdio.h>
#include <stdlib.h>

#define MAX_IMAGESIZE 784  // 28x28 flattened size
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define SIZE 784

#define TRAIN_IMAGE "./fashion/train-images-idx3-ubyte"
#define TRAIN_LABEL "./fashion/train-labels-idx1-ubyte"
#define TEST_IMAGE "./fashion/t10k-images-idx3-ubyte"
#define TEST_LABEL "./fashion/t10k-labels-idx1-ubyte"
#define NUM_HIDDEN_LAYERS 3 
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

struct Matrix{
    float *val;
    int rowSize;
    int colSize;
    Matrix* getRow(int rowIndex){
        Matrix* row = (Matrix*)malloc(sizeof(Matrix));
		row->rowSize = this->colSize;
		row->colSize = 1;
		row->val = (float*)malloc((size_t)(this->colSize * sizeof(float)));
		for(int i=0;i< this->colSize;i++){
            row->val[i] = this->val[rowIndex * this->colSize + i];
        }
        return row;
    }
    Matrix* getCol(int colIndex){
        Matrix* col = (Matrix*)malloc(sizeof(Matrix));
		col->rowSize = this->rowSize;
		col->colSize = 1;
		col->val = (float*)malloc((size_t)(this->rowSize * sizeof(float)));
        for(int i=0;i< this->rowSize;i++){
            col->val[i] = this->val[i * this->colSize + colIndex];
        }
        return col;
    }
};

typedef struct Matrix Matrix;


Matrix* allocMatrix(int rowSize, int colSize= 1)
{
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rowSize = rowSize;
    matrix->colSize = colSize;
    matrix->val = (float*)malloc((size_t)(colSize *  rowSize * sizeof(float)));
    return matrix;
}



void copyMatrix(Matrix* dest, Matrix* org){
    if(org->colSize != dest->colSize || org->rowSize != dest->rowSize) return;
    int size = 0;
    for(int i=0;i < size;i++){
        dest[i] = org[i];
    }
}

Matrix* initRandomMatrix(int rowSize, int colSize= 1, float lower= 0.0, float upper= 1.0){
    int size = rowSize * colSize;
    Matrix* res = allocMatrix(rowSize, colSize);
    for(int i=0;i < size;i++){
        res->val[i] =  lower + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(upper-lower)));
    }
    return res;
}

void freeMatrix(Matrix* matrix) {
    if (matrix) {
        free(matrix->val);
        free(matrix);
    }
}

const char* getLabelByIdx(int idx){
	switch (idx)
	{
		case 0:
			return "T-shirt/top";
		case 1:
			return "Trouser";
		case 2:
			return "Pullover";
		case 3:
			return "Dress";
		case 4:
			return "Coat";
		case 5:
			return "Sandal";
		case 6:
			return "Shirt";
		case 7:
			return "Sneaker";
		case 8:
			return "Bag";
		case 9:
			return "Ankle boot";
		default:
			return "Not exist label";
	}
	return "";
}