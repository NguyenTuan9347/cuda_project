#include "utils.cu"

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


Matrix* read_mnist_labels(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return nullptr;
    }

    int magic_number = 0;
    int number_of_labels = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverseInt(magic_number);

    fread(&number_of_labels, sizeof(number_of_labels), 1, file);
    number_of_labels = reverseInt(number_of_labels);
    
    Matrix* labels = allocMatrix(number_of_labels, 1);

    if (magic_number != 2049) {
        printf("Invalid magic number: %d. Expected 2049 for label file.\n", magic_number);
        fclose(file);
        return nullptr;
    }
    
    labels->rowSize = number_of_labels;
    labels->colSize = 1; 
    
    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, file);
        labels->val[i] = (float)temp;  
    }

    fclose(file);

    return labels;
}

Matrix* read_mnist_images(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return nullptr;
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverseInt(magic_number);

    fread(&number_of_images, sizeof(number_of_images), 1, file);
    number_of_images = reverseInt(number_of_images);

    fread(&n_rows, sizeof(n_rows), 1, file);
    n_rows = reverseInt(n_rows);

    fread(&n_cols, sizeof(n_cols), 1, file);
    n_cols = reverseInt(n_cols);


    if (n_rows != 28 || n_cols != 28) {
        printf("Unexpected image dimensions: %d x %d. MNIST expects 28x28.\n", n_rows, n_cols);
        fclose(file);
        return nullptr;
    }

    Matrix* images = allocMatrix(number_of_images, n_cols * n_rows);

    for (int i = 0; i < number_of_images; ++i) {
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                images->val[i * (n_rows * n_cols) + r * n_cols + c] = (float)(temp) / 255.0f;  // Normalize to [0, 1]
            }
        }
    }

    fclose(file);

    return images;
}

int main() {
    Matrix* train_images = read_mnist_images(TRAIN_IMAGE); 
    Matrix* test_images = read_mnist_images(TEST_IMAGE);
    Matrix* train_labels = read_mnist_labels(TRAIN_LABEL);
    Matrix* test_labels = read_mnist_labels(TEST_LABEL);
    
    int upper_size = 10;
    // Display the first 10 images for verification
    for (int k = 0; k < upper_size; k++) {
        printf("Image %d:\n", k + 1);
        Matrix *img = train_images->getRow(k);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                if (img->val[r * 28 + c] > 0.0)
                    printf("* ");
                else
                    printf(" ");
            }
            printf("\n");
            
        }
        printf("\nLabel: %s\n", getLabelByIdx((int)train_labels->val[k]));
        printf("********************************************************\n");
    }

    freeMatrix(train_images);
    freeMatrix(test_images);
    freeMatrix(test_labels);
    freeMatrix(train_labels);
    return 0;
}
