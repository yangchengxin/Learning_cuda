#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#include "utils.hpp"
#include "timer.hpp"
#include "preprocess.hpp"

using namespace std;

int main()
{
    Timer timer;

    string file_path     = "";
    string output_prefix = "";
    string output_path   = "";

    cv::Mat input = cv::imread(file_path);
    int dst_h = 500;
    int dst_w = 250;
    int tactis;

    tactis = 2;
    cv::Mat resized_gpu;
    resized_gpu = preprocess_gpu(input, dst_h, dst_w, timer, tactis);
    cv::namedWindow("test", cv::WINDOW_NORMAL);
    cv::imshow("test", resized_gpu);
    cv::waitKey(0);
    
    return 0;
}