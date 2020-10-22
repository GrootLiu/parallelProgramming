#include "cuda_runtime.h"
#include <iostream>
#include <random>
#include <math.h>
#include <algorithm>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>


#define ROWLEN 1024*5
#define COLLEN 1024*5
#define MAX 10

//N = arraySize
//#define n 512
//#define  multiple 3
//#define N 1024

//S = threadNum
#define S 32

#define RL ROWLEN / S
#define CL COLLEN / S


clock_t start, stop; //clock_t为clock()函数返回的变量类型
double duration;

using namespace std;

//Use the GPU to calculate the KNN's answer
__global__ void getDistanceGPU(double trainSet[COLLEN][ROWLEN], double* testData, double* dis)
{

    int xid = threadIdx.x + blockDim.x * blockIdx.x;
    int yid = threadIdx.y + blockDim.y * blockIdx.y;

    int row = yid;
    int col = xid;

    if (col < ROWLEN && row < COLLEN)
    {
        double temp = 0;
        for (int i = 0; i < ROWLEN; i++)
        {
            temp += pow((trainSet[row][i] - testData[i]), 2);
        }
        dis[row] = sqrt(temp);
    }
}

void gpuCal(double a[ROWLEN][COLLEN], double b[ROWLEN], double c[COLLEN])
{
    double (*dev_a)[ROWLEN];
    double* dev_b;
    double* dev_c;

    //在GPU中开辟空间
    cudaMalloc((void**)&dev_a, ROWLEN * COLLEN * sizeof(double));
    cudaMalloc((void**)&dev_b, ROWLEN * sizeof(double));
    cudaMalloc((void**)&dev_c, COLLEN * sizeof(double));

    //将CPU内容复制到GPU
    cudaMemcpy(dev_a, a, ROWLEN * COLLEN * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, ROWLEN * sizeof(double), cudaMemcpyHostToDevice);

    //声明时间Event
    float time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //GPU开始计算
    dim3 threadsPerBlock(S, S);
    dim3 blocksPerGrid(RL, CL);
    getDistanceGPU << <blocksPerGrid, threadsPerBlock >> > (dev_a, dev_b, dev_c);
    //结束计时
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    //计算时间差
    cudaEventElapsedTime(&time, start, stop);
    //将内容拷贝回CPU
    cudaMemcpy(c, dev_c, COLLEN * sizeof(double), cudaMemcpyDeviceToHost);

   /* for (int j = 0; j < COLLEN; j++)
    {
        printf("%f ", c[j]);
    }
    printf("\n");*/
    printf("GPU: spendTime: %fms\n\n\n", time);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

//Calculate the distance between testData and dataSet[i]
double getDistance(double* d1, int* d2);

//calculate all the distance between testData and each training data
void getAllDistance(double trainSet[ROWLEN][COLLEN], double* testData, double* discard_block);

// Randomly generated training set 
void randNum(double trainSet[ROWLEN][COLLEN], int rlen, int clen);

//Randomly generated testDate
void randNum(double* testData, int clen);

//Print the trainSet
void print(double trainSet[ROWLEN][COLLEN], int rlen, int clen);

//Print the testSet
void print(double* testData, int clen);

int main(int argc, char const* argv[])
{
    double (*trainSet)[ROWLEN];
    double* testData;
    double* dis;
    trainSet = new double[ROWLEN][COLLEN];
    testData = new double[COLLEN];
    dis = new double[ROWLEN];
    randNum(trainSet, ROWLEN, COLLEN);
    randNum(testData, COLLEN);

    gpuCal(trainSet, testData, dis);
    getAllDistance(trainSet, testData, dis);

    cout << "-----------------trainSet----------------------------" << endl;
    //print(trainSet, ROWLEN, COLLEN);
    cout << "-----------------testSet----------------------------" << endl;
    //print(testData, COLLEN);
    cout << "-----------------dis-------------------------------" << endl;
    print(dis, COLLEN);
    sort(dis, dis + COLLEN);
    print(dis, COLLEN);
    return 0;
}


//Calculate the distance between trainSet and testData
double getDistance(double* d1, double* d2)
{
    double dis = 0;
    for (int i = 0; i < COLLEN; i++)
    {
        dis += pow((d1[i] - d2[i]), 2);
    }
    return sqrt(dis);
}

//calculate all the distance between testData and each training data
void getAllDistance(double trainSet[ROWLEN][COLLEN], double* testData, double* dis)
{
    start = clock();
    //******************************
    //*这里写你所要测试运行时间的程序 *
    for (int i = 0; i < ROWLEN; i++)
    {
        dis[i] = getDistance(trainSet[i], testData);
    }
    //******************************
    stop = clock();
    duration = (double)(stop - start) / CLK_TCK; //CLK_TCK为clock()函数的时间单位，即时钟打点
    printf("CPU: spendTime: %fms\n\n\n", duration * 1000);
    
}

// Randomly generated training set 
void randNum(double trainSet[ROWLEN][COLLEN], int rlen, int clen)
{
    for (int i = 0; i < rlen; i++)
    {
        for (int j = 0; j < clen; j++)
        {
            trainSet[i][j] = rand() % MAX;
        }
    }
}

//Randomly generated testDatd
void randNum(double* testData, int clen)
{
    for (int i = 0; i < clen; i++)
    {
        testData[i] = rand() % MAX;
    }
}
//Print the trainSet
void print(double trainSet[ROWLEN][COLLEN], int rlen, int clen)
{
    for (int i = 0; i < rlen; i++)
    {
        for (int j = 0; j < clen; j++)
        {
            cout << trainSet[i][j] << " ";
        }
        cout << endl;
    }
}

//Print the testSet
void print(double* testData, int clen)
{
    for (int i = 0; i < clen; i++)
    {
        cout << testData[i] << " ";
    }
    cout << endl;
}


