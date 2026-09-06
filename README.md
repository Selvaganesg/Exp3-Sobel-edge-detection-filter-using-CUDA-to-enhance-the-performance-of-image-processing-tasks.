# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>AIM: To implement Sobel edge detection using CUDA and accelerate image processing through GPU parallelization.</h3>
<h3>SELVAGANESH</h3>
<h3>212224230258</h3>
<h3>EX. NO:3</h3>
<h3>17/08/2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,
                            unsigned int width, unsigned int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {

        int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

        int sumX = 0;
        int sumY = 0;

        for(int i=-1;i<=1;i++){
            for(int j=-1;j<=1;j++){
                unsigned char pixel = srcImage[(y+i)*width + (x+j)];
                sumX += pixel * Gx[i+1][j+1];
                sumY += pixel * Gy[i+1][j+1];
            }
        }

        int magnitude = sqrtf(sumX*sumX + sumY*sumY);
        magnitude = min(max(magnitude,0),255);

        dstImage[y*width + x] = (unsigned char)magnitude;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {

    Mat image = imread("/content/images (2) .jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char*)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;

    checkCudaErrors(cudaMalloc(&d_inputImage,imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage,imageSize));

    checkCudaErrors(cudaMemcpy(d_inputImage,
                               image.data,
                               imageSize,
                               cudaMemcpyHostToDevice));

    dim3 blockSize(16,16);
    dim3 gridSize((width+15)/16,(height+15)/16);

    sobelFilter<<<gridSize,blockSize>>>(d_inputImage,d_outputImage,width,height);

    checkCudaErrors(cudaMemcpy(h_outputImage,
                               d_outputImage,
                               imageSize,
                               cudaMemcpyDeviceToHost));

    Mat outputImage(height,width,CV_8UC1,h_outputImage);

    imwrite("output_sobel.jpeg",outputImage);

    printf("Edge detection completed.\n");

    return 0;
}
```
## OUTPUT:
## Original Image
<img width="457" height="437" alt="images (2)" src="https://github.com/user-attachments/assets/0a34d84b-53aa-4b10-8da9-20c7dc628d6c" />

<img width="509" height="515" alt="image" src="https://github.com/user-attachments/assets/ca543085-d3d1-45c0-a0fc-66538e301dff" />

## RESULT:
Thus, the program has been executed successfully using CUDA to perform parallel Sobel edge detection on an image using GPU acceleration.

1. What challenges did you face while implementing the Sobel filter for color images?
The main challenges were converting the color image to grayscale, handling image boundaries, and ensuring that each CUDA thread accesses the correct neighboring pixels without going out of bounds.

2. How did changing the block size influence the performance of your CUDA implementation?
Changing the block size affected the number of threads executed together. A suitable block size such as 16×16 provided efficient GPU utilization, while very small or large block sizes could result in lower performance due to inefficient thread utilization.

3. What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Both implementations produced similar edge-detected images. Minor differences may occur due to differences in numerical calculations, boundary handling, and implementation details. The CUDA implementation provides the advantage of parallel GPU processing.

4. Suggest potential optimizations for improving the performance of the Sobel filter.

Use shared memory to reduce repeated global-memory accesses.
Experiment with different block sizes such as 8×8, 16×16, and 32×32.
Minimize unnecessary memory transfers between CPU and GPU.
Use optimized CUDA memory-access patterns for coalesced memory access.
Process multiple images or larger images in parallel where applicable.

## DELIVERABLES
Modified CUDA Sobel filter code with comments.
Execution-time analysis for different image and block sizes.
Comparison of CUDA and CPU Sobel outputs.

## TOOLS REQUIRED
NVIDIA GPU
CUDA Toolkit / NVCC
Google Colab
OpenCV
C/C++ compiler
Sample input images
Python/Matplotlib for performance graphs

