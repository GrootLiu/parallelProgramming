#include <iostream>
#include <random>
#include <math.h>
#include <algorithm>

#define ROWLEN 5
#define COLLEN 5
#define MAX 10

using namespace std;

//Calculate the distance between testData and dataSet[i]
double getDistance(double* d1, int* d2);

//calculate all the distance between testData and each training data
void getAllDistance(double** trainSet, double* testData, double* discard_block);

// Randomly generated training set 
void randNum(double** trainSet, int rlen, int clen);

//Randomly generated testDate
void randNum(double* testData, int clen);

//Print the trainSet
void print(double** trainSet, int rlen, int clen);

//Print the testSet
void print(double* testData, int clen);

int main(int argc, char const *argv[])
{
    double** trainSet;
    double* testData;
    double* dis;
    trainSet = new double*[ROWLEN];
    testData = new double[COLLEN];
    dis = new double[ROWLEN];
    randNum(trainSet, ROWLEN, COLLEN);
    randNum(testData, COLLEN);

    getAllDistance(trainSet, testData, dis);

    cout<<"-----------------trainSet----------------------------"<<endl;
    print(trainSet, ROWLEN, COLLEN);
    cout<<"-----------------testSet----------------------------"<<endl;
    print(testData, COLLEN);
    cout<<"-----------------dis-------------------------------"<<endl;
    print(dis, COLLEN);
    sort(dis,dis+COLLEN);
    print(dis, COLLEN);
    return 0;
}


//Calculate the distance between trainSet and testData
double getDistance(double* d1, double* d2)
{
    double dis = 0;
    for (int i = 0; i < COLLEN; i++)
    {
        dis += pow( (d1[i] - d2[i]) , 2);
    }
    return sqrt(dis);
}

//calculate all the distance between testData and each training data
void getAllDistance(double** trainSet, double* testData, double* dis)
{
    for (int i = 0; i < ROWLEN; i++)
    {
        dis[i] = getDistance(trainSet[i], testData);
    }
}

// Randomly generated training set 
void randNum(double** trainSet, int rlen, int clen)
{
    for (int i = 0; i < rlen; i++)
    {
        trainSet[i] = new double[clen];
    }
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
        testData[i]= rand() % MAX;
    }
}
//Print the trainSet
void print(double** trainSet, int rlen, int clen)
{
    for (int i = 0; i < rlen; i++)
    {
        for (int j = 0; j < clen; j++)
        {
            cout<<trainSet[i][j]<<" ";
        }
        cout<<endl;
    }
}

//Print the testSet
void print(double* testData, int clen)
{
    for (int i = 0; i < clen; i++)
    {
        cout<<testData[i]<<" ";
    }
    cout<<endl;
}


//Use the GPU to calculate the KNN's answer
__globle__ 