/*
    Histogram equalization CPU
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

#define SIZE 256 // size constant

/////////////// FUNCTION DECLARATIONS ///////////////

void histogramEqualizationCPU(const cv::Mat& input, cv::Mat& output);

/////////////// FUNCTION DEFINITIONS ///////////////

// Function to equalize
void histogramEqualizationCPU(const cv::Mat& input, cv::Mat& output){

    float size = input.rows * input.cols;
    int histogram[SIZE] = {};
    unsigned int index;

    // Histogram
    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++){
            index = (int)input.at<uchar>(y,x);
            histogram[index]++;
        }
    }

    int histogram_norm[SIZE] = {}; // for the normalization

    // Normalize
    for(int y = 0; y < SIZE; y++){
        for(int x = 0; x <= y; x++){
            histogram_norm[y] += histogram[x];
        }
    }
    for(int i = 0; i < SIZE; i++){
        histogram_norm[i] = histogram_norm[i] * (SIZE / size);
    }

    // Fill output
    for (int y = 0; y < input.rows; y++){
        for (int x = 0; x < input.cols; x++){
            index = (int)input.at<uchar>(y,x);
            output.at<uchar>(y,x) = histogram_norm[index];
        }
    }
}

// Main
int main(int argc, char *argv[]) {
    std::cout << "\n";
    std::cout << "---------- HISTOGRAM EQUALIZATION CPU ----------" << "\n";
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

    auto start_cpu =  std::chrono::high_resolution_clock::now();
	histogramEqualizationCPU(input_BW, output); // Calls main function
	auto end_cpu =  std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("Time elapsed: %f ms\n", duration_ms.count());

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input_BW);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

    return 0;
}
