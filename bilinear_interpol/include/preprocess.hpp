// #define __PREPROCESS__
#ifndef __PREPROCESS__
#define __PREPROCESS__

#include "opencv2/opencv.hpp"
#include "timer.hpp"

cv::Mat preprocess_cpu(cv::Mat &src, const int& dst_h, const int& dst_w, Timer timer, int choice);
cv::Mat preprocess_gpu(cv::Mat &src, const int& dst_h, const int& dst_w, Timer timer, int choice);
void resize_bilinear_gpu(uint8_t* dst, uint8_t* src, int dst_h, int dst_w, int src_h, int src_w, int tactis);

#endif