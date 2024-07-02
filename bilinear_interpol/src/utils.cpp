#include "utils.hpp"
#include <math.h>
#include <random>

void initMatrix(float* data, int size, int low, int high, int seed)
{
    srand(seed);
    for(int i = 0;i < size;i++)
    {
        data[i] = (float)i;
    }
}

void printMat(float* data, int size)
{
    for(int i = 0;i < size;i++)
    {
        printf("%.8lf", data[i]);
        if(i != size - 1)
        {
            printf(", ");
        }
        else
        {
            printf("\n");
        }
    }
}

void compareMat(float* h_data, float* d_data, int size)
{
    double precision = 1.0E-4;
    for(int i = 0;i < size;i++)
    {
        if(abs(h_data[i] - d_data[i] > precision))
        {
            int y = i / size;
            int x = i % size;
            printf("Matmul result is different\n");
            printf("cpu: %.8lf, gpu: %.8lf, cord:[x=%d, y=%d]\n", h_data[i], d_data[i], x, y);
            break;
        }
    }
}

std::string getPrefix(std::string file_path)
{
    std::string prefix;
    int pos1 = file_path.find('/');
    int pos2 = file_path.find('.');
    prefix = file_path.substr(pos1, pos2 - pos1);
    return prefix;
}