#include <chrono>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "timer.hpp"
#include "utils.hpp"

Timer::Timer()
{
    _timeElasped = 0;
    _cstart      = std::chrono::high_resolution_clock::now();
    _cstop       = std::chrono::high_resolution_clock::now();
    cudaEventCreate(&_gstart);
    cudaEventCreate(&_gstop);
}

Timer::~Timer()
{
    cudaFree(_gstart);
    cudaFree(_gstop);
}

void Timer::start_gpu()
{
    cudaEventRecord(_gstart, 0);
}

void Timer::stop_gpu()
{
    cudaEventRecord(_gstop, 0);
}

void Timer::start_cpu()
{
    _cstart = std::chrono::high_resolution_clock::now();
}

void Timer::stop_cpu()
{
    _cstop = std::chrono::high_resolution_clock::now();
}

void Timer::duration_gpu(std::string msg)
{
    CUDA_CHECK(cudaEventSynchronize(_gstart));
    CUDA_CHECK(cudaEventSynchronize(_gstop));
    cudaEventElapsedTime(&_timeElasped, _gstart, _gstop);

    LOG("%-60s uses %.6lf ms", msg.c_str(), _timeElasped);
}