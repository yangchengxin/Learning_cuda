#include "cuda_runtime_api.h"
#include "stdio.h"
#include <iostream>
// #include "preprocess.hpp"

#include "utils.hpp"

/* 最邻近插值法 */
/**
 * @brief 
 * 
 * @param tar       插值后的图   
 * @param src       原图
 * @param tarw      插值后的w
 * @param tarh      插值后的h
 * @param srcw      原图的w
 * @param srch      原图的h
 * @param scaled_w  w变换尺度
 * @param scaled_h  h变换尺度
 * @return __global__ 
 */
__global__ void resize_nearest_BGR2RGB_kernel(
    uint8_t* tar, uint8_t* src,
    int tarW, int tarH,
    int srcW, int srcH,
    float scaled_w, float scaled_h
)
{
    /* 插值后的图的像素索引 */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* 反向映射法 */ 
    /* round四舍五入到整数位，向上取整 */
    int src_y = round((float)y * scaled_h);
    int src_x = round((float)x * scaled_w);

    if(src_x < 0 || src_y < 0 || src_x > srcW || src_y > srcH)
    {
        /* 对于越界的部分不做处理 */
    }
    else
    {
        /* 三通道彩图，计算插值图的像素索引 */
        int tarIdx = (y * tarW + x) * 3;
        /* 三通道彩图，计算原图的像素索引 */
        int srcIdx = (src_y * srcW + src_x) * 3;

        /* 最邻近插值 + BGR 2 RGB */
        tar[tarIdx + 0] = src[srcIdx + 2];
        tar[tarIdx + 1] = src[srcIdx + 1];
        tar[tarIdx + 2] = src[srcIdx + 0];
    }
}

/* 二次线性插值法 */
/**
 * @brief 用cuda高并发编程来实现一个resize函数，使用的是二次线性插值方法，最后实现BGR2RGB
 * 
 * @param tar       插值后的图   
 * @param src       原图
 * @param tarw      插值后的w
 * @param tarh      插值后的h
 * @param srcw      原图的w
 * @param srch      原图的h
 * @param scaled_w  w变换尺度
 * @param scaled_h  h变换尺度
 * @return __global__ 
 */

__global__ void resize_bilinear_BGR2RGB_kernel(
    uint8_t*tar, uint8_t* src,
    int tarW, int tarH,
    int srcW, int srcH,
    float scaled_w, float scaled_h
)
{
    /* 插值后的图的像素索引 */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* 反向映射法 */ 
    /* floor向下取整 */
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if(src_x1 < 0 || src_y1 < 0 || src_x1 > srcW || src_y1 > srcH)
    {
        /* 对于越界的部分不做处理 */
    }
    else
    {
        /* 计算映射点的坐标与左下点的距离差 */
        float th = ((y + 0.5) * scaled_h - 0.5) - src_y1;
        float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1;

        float rightdown_ratio = (1.0 - tw) * (1.0 - th);
        float leftdown_ratio  = tw * (1.0 - th);
        float rightup_ratio   = (1.0 - tw) * th;
        float leftup_ratio    = tw * th;


        /* 计算原图上四个点的坐标索引 */
        int src_leftup      = (src_y1 * srcW + src_x1) * 3;
        int src_rightup     = (src_y1 * srcW + src_x2) * 3;
        int src_leftdown    = (src_y2 * srcW + src_x1) * 3;
        int src_rightdown   = (src_y2 * srcW + src_x2) * 3; 

        /* 计算插值图上点的坐标索引 */
        int tar_pixel = (y * tarW + x) * 3;

        /* 双线性插值 + BGR2RGB */
        tar[tar_pixel + 0] = round(leftup_ratio * src[src_rightdown + 2] +
                                    leftdown_ratio  * src[src_rightup + 2] +
                                    rightup_ratio   * src[src_leftdown + 2] +
                                    rightdown_ratio * src[src_leftup + 2]);
        
        tar[tar_pixel + 1] = round(leftup_ratio * src[src_rightdown + 1] +
                                    leftdown_ratio  * src[src_rightup + 1] +
                                    rightup_ratio   * src[src_leftdown + 1] +
                                    rightdown_ratio * src[src_leftup + 1]);

        tar[tar_pixel + 2] = round(leftup_ratio * src[src_rightdown + 0] +
                                    leftdown_ratio  * src[src_rightup + 0] +
                                    rightup_ratio   * src[src_leftdown + 0] +
                                    rightdown_ratio * src[src_leftup + 0]);
    }
}

__global__ void resize_bilinear_BGR2RGB_center_kernel(
    uint8_t*tar, uint8_t* src,
    int tarW, int tarH,
    int srcW, int srcH,
    float scaled_w, float scaled_h
)
{
    /* 插值后的图的像素索引 */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* 反向映射法 */ 
    /* floor向下取整 */
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if(src_x1 < 0 || src_y1 < 0 || src_x1 > srcW || src_y1 > srcH)
    {
        /* 对于越界的部分不做处理 */
    }
    else
    {
        /* 计算映射点的坐标与左下点的距离差 */
        float th = ((y + 0.5) * scaled_h - 0.5) - src_y1;
        float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1;

        float rightdown_ratio = (1.0 - tw) * (1.0 - th);
        float leftdown_ratio  = tw * (1.0 - th);
        float rightup_ratio   = (1.0 - tw) * th;
        float leftup_ratio    = tw * th;


        /* 计算原图上四个点的坐标索引 */
        int src_leftup      = (src_y1 * srcW + src_x1) * 3;
        int src_rightup     = (src_y1 * srcW + src_x2) * 3;
        int src_leftdown    = (src_y2 * srcW + src_x1) * 3;
        int src_rightdown   = (src_y2 * srcW + src_x2) * 3; 

        /* 计算插值图上点的坐标索引 */
        y = y + int(tarH / 2) - int(srcH / scaled_h / 2);

        int tar_pixel = (y * tarW + x) * 3;

        /* 双线性插值 + BGR2RGB */
        tar[tar_pixel + 0] = round(leftup_ratio * src[src_rightdown + 2] +
                                    leftdown_ratio  * src[src_rightup + 2] +
                                    rightup_ratio   * src[src_leftdown + 2] +
                                    rightdown_ratio * src[src_leftup + 2]);
        
        tar[tar_pixel + 1] = round(leftup_ratio * src[src_rightdown + 1] +
                                    leftdown_ratio  * src[src_rightup + 1] +
                                    rightup_ratio   * src[src_leftdown + 1] +
                                    rightdown_ratio * src[src_leftup + 1]);

        tar[tar_pixel + 2] = round(leftup_ratio * src[src_rightdown + 0] +
                                    leftdown_ratio  * src[src_rightup + 0] +
                                    rightup_ratio   * src[src_leftdown + 0] +
                                    rightdown_ratio * src[src_leftup + 0]);
    }
}



void resize_bilinear_gpu(
    uint8_t* d_tar, uint8_t* d_src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    int tactis)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid(tarW / 16 + 1, tarH / 16 + 1);

    /* 计算宽高的变换比例 原图/插值图 */
    float scaled_h = (float)srcH / tarH;
    float scaled_w = (float)srcW / tarW;

    if(tactis > 1)
    {
        float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);
        scaled_h = scale;
        scaled_w = scale;
    }

    switch(tactis)
    {
    case 0:
        resize_nearest_BGR2RGB_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
        break;
    case 1:
        resize_bilinear_BGR2RGB_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
        break;
    case 2:
        resize_bilinear_BGR2RGB_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
        break;
    case 3:
        resize_bilinear_BGR2RGB_center_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
        break;
    default:
        break;
    }
}
