/*
    Histogram equalization GPU
*/

#include <iostream>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <string>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// Cuda
#include "common.h"
#include <cuda_runtime.h>

#define SIZE 256
#define PIXEL_SIZE 255

/////////////// FUNCTION DECLARATIONS ///////////////

__global__ void normalize(int * histogram , int * histogram_temp, float totalSize);
__global__ void histogramCreate(unsigned char* input, unsigned char* output, int width, int height, int step, int * histogram_temp);
__global__ void createImage(unsigned char* input, unsigned char* output, int width, int height, int step, int * histogram_temp);
void histogramEqualizationGPU(const cv::Mat& input, cv::Mat& output);

/////////////// FUNCTION DEFINITIONS ///////////////

// Main histogram function
void histogramEqualizationGPU(const cv::Mat& input, cv::Mat& output){

    size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;
	//size_t histogram_tempSize = SIZE * sizeof(int);
    float totalSize = output.rows * output.cols;

    int * histogram = {};
    int * histogram_temp = {};
	unsigned char *d_input;
    unsigned char *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_output, grayBytes), "CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&histogram, SIZE * sizeof(int)),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&histogram_temp, SIZE * sizeof(int)),"CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("equalization_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	auto start_cpu =  std::chrono::high_resolution_clock::now();

	histogramCreate<<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), histogram_temp);
	normalize<<<grid, block >>>(histogram, histogram_temp, totalSize);
	createImage<<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step),histogram_temp);

	auto end_cpu =  std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("Time elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
    SAFE_CALL(cudaFree(histogram), "CUDA Free Failed");
    SAFE_CALL(cudaFree(histogram_temp), "CUDA Free Failed");
}

__global__ void createImage(unsigned char* input, unsigned char* output, int width, int height, int step, int * histogram_temp){
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int tid = yIndex * step + xIndex;

	if(xIndex < width && yIndex < height){
        int finalImage = input[tid];
		output[tid] = histogram_temp[finalImage];
	}
}

// Calculate the histogram
__global__ void histogramCreate(unsigned char* input, unsigned char* output, int width, int height, int step, int * histogram_temp){

    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	const int tid = yIndex * step + xIndex;

	__shared__ int histogram[SIZE];

	int index = threadIdx.x + threadIdx.y * blockDim.x;

	if(index < SIZE){
        histogram[index] = 0;
    }

    __syncthreads();

	if(xIndex < width && yIndex < height){
        atomicAdd(&histogram[input[tid]], 1);
	}

	__syncthreads();

	if(index < SIZE){
        atomicAdd(&histogram_temp[index], histogram[index]);
    }
}

// Function to normalize
__global__ void normalize(int * histogram , int * histogram_temp, float totalSize){

    unsigned int index = threadIdx.x + threadIdx.y * blockDim.x;

	float normal = 0.0;
    if (index < SIZE && blockIdx.x == 0 && blockIdx.y == 0){
        for(int i = 0; i <= index; i++){
            normal += histogram[i];
        }
        histogram_temp[index] = (int)floor(normal * (PIXEL_SIZE / (totalSize-1)));
    }
}

// Main
int main(int argc, char *argv[]) {
    std::cout << "\n";
    std::cout << "---------- HISTOGRAM EQUALIZATION GPU ----------" << "\n";
    std::cout << "\n";

    std::string imagePath; // path

    if(argc < 2){
        imagePath = "Images/dog1.jpeg";
    } else {
        imagePath = argv[1];
    }

    // Read input image from the disk
	cv::Mat input = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (input.empty()){ // if no image found
        std::cout << "Image Not Found!" << std::endl;
		std::cin.get();
		return -1;
	}

    // Declare bw and final output
    cv::Mat input_BW(input.rows, input.cols, CV_8UC1);
	cv::Mat output(input.rows, input.cols, CV_8UC1);

    cv::cvtColor(input, input_BW, cv::COLOR_BGR2GRAY); // Convert to bw

	output = input_BW.clone(); // clone the bw to output

    // auto start_cpu =  std::chrono::high_resolution_clock::now();
	histogramEqualizationGPU(input_BW, output); // Calls main function
	// auto end_cpu =  std::chrono::high_resolution_clock::now();
	// std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	// printf("Time elapsed: %f ms\n", duration_ms.count());

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	cv::resizeWindow("Input", 800, 600);
	cv::resizeWindow("Output", 800, 600);

	//Show the input and output
	imshow("Input", input_BW);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

    return 0;
}
