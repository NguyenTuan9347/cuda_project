# **MULTILAYER PERCEPTRON OPTIMIZATION APPLYING PARALLEL COMPUTING ON GPU**

## **I. Introduction**

### **0. Group members**

| Student                | ID        | Email                          |
|------------------------|-----------|--------------------------------|
| Nguyen Hoang Anh Tuan  | 21127471  | nhatuan21@clc.fitus.edu.vn     |
| Phan Huy Duc Tai       | 21127687  | phdtai21@clc.fitus.edu.vn      |
| Ho Dinh Duy Luc        | 21127351  | hddluc21@clc.fitus.edu.vn        |


### **1. Background**
Deep learning has revolutionized domains such as image recognition, natural language processing, and robotics by utilizing artificial neural networks to learn complex patterns. These models excel in tasks like image classification, where traditional programming methods struggle.

The MNIST dataset has historically been a benchmark for image classification research. However, its simplicity limits its relevance for modern deep learning challenges. To address this, Zalando introduced the **Fashion-MNIST** dataset—a more complex and practical benchmark for image classification tasks.


![Screenshot 2024-12-21 114956](https://github.com/user-attachments/assets/6ecf5578-b0cb-41cc-bfcd-9a4e2c9c7075)

**Figure:**  Class names and example images in Fashion-MNIST dataset.




Fashion-MNIST features:
- 70,000 grayscale images (60,000 for training, 10,000 for testing).
- 28×28 pixel resolution.
- 10 fashion categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

Using Fashion-MNIST in this project ensures both complexity and real-world relevance, providing a rigorous evaluation of CUDA-accelerated neural networks.


### **2. Problem Statement**
Training neural networks on large datasets like Fashion-MNIST is computationally demanding. Sequential CPU-based training is inefficient, limiting scalability and real-time applications. This project tackles these challenges by implementing a CUDA-accelerated neural network for efficient training and inference.

- **Input**: Grayscale images from Fashion-MNIST, preprocessed into a 1D array of 784 pixel intensity values.
- **Output**: Predicted clothing category, represented as:
  - **Class Scores**: A probability vector for 10 categories.
  - **Predicted Label**: The index with the highest probability.
  
#### **Key Challenges**
- **High Computational Demand**: Intensive operations like matrix multiplication and backpropagation.
- **Data Throughput**: Inefficient sequential processing.
- **Hardware Utilization**: Underutilized hardware resources in CPU implementations.

#### **Objective**
Implement and optimize a neural network using CUDA C, focusing on:
- Efficient parallel computation on GPUs.
- Performance comparison between CPU-only and hybrid CPU-GPU implementations.
- Documentation of implementation, optimization techniques, and results.


## **II. Project Goals**
1. Develop and train a neural network to classify images from Fashion-MNIST.
2. Implement two versions of the model:
   - CPU-only implementation.
   - Hybrid CPU-GPU implementation using CUDA for intensive operations.
3. Optimize GPU performance using techniques like:
   - Efficient memory management.
   - Kernel optimization.
4. Evaluate and compare the performance of the two implementations.


## **III. Setup Instructions**

### **0. Project structure**

```markdown
# Project Structure
.cuda_project
├── readme.md :   A detailed description of the project, including goals, setup instructions, and usage guidelines.
├── project_contribution.sheet :   A sheet documenting the contributions of each team member to the project.
├── config.txt :   Contains configuration settings such as dataset paths, training parameters, and environment setup instructions.
├── kernel_colab.cu : CUDA implementation for core computations or neural network training tasks.
├── fashion
    ├── train-images-idx3-ubyte : Training set images.  
    ├── train-labels-idx1-ubyte : Training set labels.
    ├── t10k-images-idx3-ubyte : Test set images.  
    └── t10k-labels-idx1-ubyte : Test set labels.
└── report.ipynb :   A  jupyter notebook presented detailed the process developing the project.
```

### **1. Requirements**
- **Hardware**: NVIDIA GPU with CUDA support.
- **Software**:
  - CUDA Toolkit.
  - Visual Studio (for CUDA development).
  - Google Colab.

### **2. Dataset**
Download the Fashion-MNIST dataset from [Zalando's repository](https://github.com/zalandoresearch/fashion-mnist). The dataset is already included in the resporatory so you do not need to downloaded them.

### **3. Installation**
1. Clone this repository:
   ```bash
    !git clone https://github.com/NguyenTuan9347/cuda_project.git
   ```
2. Set up the environment:
   - Install CUDA Toolkit and ensure `nvcc` is accessible.

### **4. Compilation**
Compile the CUDA code using the following command:
```bash
!nvcc cuda_project/kernel_colab.cu -o main.exe -diag-suppress 177
```

### **5. Running the Program**
The programe accept 3 params
```bash
1. Train/ test mode:
  - test: test mode option.
  - other: train mode option.

2. config file path
  - a path to convey configuration file to the model. This file needs to be in '.txt' format

3. Using device flag:
  - true: to accelerate with GPU.
  - other: using CPU only.
```

For example, execute the compiled program with `train` mode, with `config.txt` and using `device`
```bash
!nvprof ./main.exe train config.txt true
```
The `config.txt` has the belowed format:
```markdown
NUM_HIDDEN_LAYERS=2
MAX_IMAGESIZE=784
NUM_TRAIN=60000
NUM_TEST=10000
SIZE=784
LR=0.00005
HIDDEN_SIZE=128
OUTPUT_SIZE=10
BEST_CHECKPOINT=/content/cuda_project/best.txt
BEST_ACCURACY=0.956433
```
## **IV. Demo:**
 The `V. Result` in **Report.ipynb** can be used as an Demo for the project.

## **V. Features**
1. **Neural Network Training**: Classifies Fashion-MNIST images using a multilayer perceptron.
2. **GPU Acceleration**: Leverages CUDA for operations like matrix multiplication and activation functions.
3. **Performance Analysis**: Compares CPU and GPU implementations for training efficiency.
4. **Extensibility**: Modular design for experimenting with different architectures or datasets.


## **VI. Results**
The project achieves:
- **Training Time Reduction**: GPU-accelerated training reduces time by over 96%.
- **Comparable Accuracy**: Both implementations achieve consistent improvements in classification accuracy.
- **Insights into Optimization**: Profiling highlights areas for further GPU utilization improvement.

<img width="2962" alt="Performance Comparison Table CUDA-Based Neural Network Training" src="https://github.com/user-attachments/assets/f02ed9f7-57bc-491d-8ebe-53d352909ed3" />


## **VII. Conclusion**
This project demonstrates the effectiveness of GPU acceleration in neural network training. CUDA-enabled parallel computing significantly reduces computational time while maintaining accuracy. While the implementation highlights areas for optimization, it serves as a foundation for exploring more advanced deep learning techniques.

# VIII. Reference:
1. Desai, Y. (1970, January 1). Tiled matrix multiplication using shared memory in Cuda. TILED Matrix Multiplication Using Shared Memory in CUDA. https://www.cstechera.com/2016/03/tiled-matrix-multiplication-using-shared-memory-in-cuda.html

2. An efficient matrix transpose in CUDA C/C++. NVIDIA Technical Blog. (2022, August 21). https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

3. Fashion-MNIST: A novel image dataset for benchmarking ... (n.d.). https://arxiv.org/pdf/1708.07747

4. Vu, T. (2017, February 24). Bài 14: Multi-layer Perceptron VÀ backpropagation. Tiep Vu’s blog. https://machinelearningcoban.com/2017/02/24/mlp/

5. Zalandoresearch. (n.d.). Zalandoresearch/fashion-mnist: A mnist-like fashion product database. benchmark. GitHub. https://github.com/zalandoresearch/fashion-mnist
6. Teacher Phạm Trọng Nghĩa, CSC14120 - Parallel Programming( semester I, 2024-2025)'s materials,  FACULTY OF INFORMATION TECHNOLOGY - VNUHCM-UNIVERSITY OF SCIENCE.

## **IX. Acknowledgments**
- Special thanks to Zalando for the Fashion-MNIST dataset and NVIDIA for the CUDA Toolkit.
- We also want to give our sincerest thanks to the most dedicated and helpful teachers **Phạm Trọng Nghĩa**, **Nguyễn Trần Duy Minh** and **Nguyễn Thanh Tình** for their unwavering support, insightful guidance, and the valuable knowledge they imparted throughout our learning journey.




