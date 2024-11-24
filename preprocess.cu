#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define TRAIN_IMAGE "train-images-idx3-ubyte"
#define TRAIN_LABEL "train-labels-idx1-ubyte"
#define TEST_IMAGE "t10k-images-idx3-ubyte"
#define TEST_LABEL "t10k-labels-idx1-ubyte"

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

const char* getLabelByIdx(int label) {
    static char buffer[10];
    snprintf(buffer, sizeof(buffer), "%d", label);
    return buffer;
}

int main() {
    int train_image_count, train_label_count, test_image_count, test_label_count;
    int image_size;

    float* train_images = readImages(TRAIN_IMAGE, &train_image_count, &image_size);
    float* test_images = readImages(TEST_IMAGE, &test_image_count, &image_size);
    float* train_labels = readLabels(TRAIN_LABEL, &train_label_count);
    float* test_labels = readLabels(TEST_LABEL, &test_label_count);

    if (!train_images || !test_images || !train_labels || !test_labels) {
        printf("Failed to load MNIST data.\n");
        free(train_images);
        free(test_images);
        free(train_labels);
        free(test_labels);
        return 1;
    }

    int upper_size = 10;
    for (int k = 0; k < upper_size; ++k) {
        printf("Image %d:\n", k + 1);
        displayImg(&train_images[k * image_size], 28, 28);
        printf("\nLabel: %s\n", getLabelByIdx((int)train_labels[k]));
        printf("********************************************************\n");
    }

    free(train_images);
    free(test_images);
    free(train_labels);
    free(test_labels);

    return 0;
}
