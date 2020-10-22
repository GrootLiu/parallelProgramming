#include<iostream>
#include<map>
#include<vector>
#include<stdio.h>
#include<cmath>
#include<cstdlib>
#include<algorithm>
#include<fstream>

using  namespace std;

//行数
#define rowLen 12
//列数
#define colLen 2


typedef pair<int,double>  PAIR;
ifstream fin;
ofstream fout;

class KNN
{
private:
    //k是k阶
    int k;
    //数据集
    double dataSet[rowLen][colLen];
    //数据类别
    char lables[rowLen];
    //测试数据
    double testData[colLen];
    //index代表第几个数据，instance代表第index个训练数据和测试数据的距离
    map<int, double> map_idx_dis;
    map<char, int> map_idx_freq;

    double getDistance(double *d1, double *d2);

    struct CmpByValue
    {
        bool operator() (const pair<int ,double>& lhs,const pair<int ,double>& rhs)
        {
            return lhs.second < rhs.second;
        }
    };
    
    
public:
    KNN(int k);
    ~KNN();

    void getAllDistance();
    void getMaxFreqLable();

};

//init the KNN function
KNN::KNN(int k)
{
    this->k = k;
    fin.open("data.txt");
    
    if (!fin)
    {
        cout<<"can not open the file 'data.txt'"<<endl;
        exit(1);
    }

    for (int i = 0; i < rowLen; i++)
    {
        for (int j = 0; j < colLen; j++)
        {
            fin>>dataSet[i][j];
        }
        fin>>lables[i];
    }
    
    cout<<"please input the test data: "<<endl;
    for (int i = 0; i < colLen; i++)
    {
        cin>>testData[i];
    }
}

KNN::~KNN()
{
}


/*
* calculate the the Euclidean distance between test data and dataSet[i]
*/
double KNN::getDistance(double *d1, double *d2)
{
    double sum = 0;
    for (int i = 0; i < colLen; i++)
    {
        sum += pow((d1[i] - d2[i]), 2);
    }
    return sqrt(sum);
}


/*
* calculate all the distance between test data and each training data
*/
void KNN::getAllDistance()
{
    for (int i = 0; i < rowLen; i++)
    {
        map_idx_dis[i] = getDistance(dataSet[i], testData);
    }
    map<int, double>::iterator it = map_idx_dis.begin();
    while (it != map_idx_dis.end())
    {
        cout<<"index = "<<it->first<<" distance = "<<it->second<<endl;
        it++;
    }
    
}


/*
* check which label the test belong to and classify the test data
*/
void KNN::getMaxFreqLable()
{
    //transform the map_index_dis to vec_index_dis
	vector<PAIR> vec_idx_dis( map_idx_dis.begin(),map_idx_dis.end() );
	//sort the vec_index_dis by distance from low to high to get the nearest data
	sort(vec_idx_dis.begin(),vec_idx_dis.end(),CmpByValue());
    for (int i = 0; i < k; i++)
    {
        cout<<"the index = "<<vec_idx_dis[i].first<<" the distance = "<<vec_idx_dis[i].second;
        map_idx_freq[ lables[vec_idx_dis[i].first] ]++;
    }

    map<char, int>::iterator it = map_idx_freq.begin();

    //A或B的最多出现次数
    int max_freq = 0;
    char lable;
    while ( it != map_idx_freq.end())
    {
        if (it->second > max_freq)
        {
            max_freq = it->second;
            lable = it->first;
        }
        it++;
    }
    cout<<"The test data belongs to the "<<lable<<" label"<<endl;
}

int main(int argc, char const *argv[])
{
    int k = 0;
    cout<<"please input the k :"<<endl;

    cin>>k;

    KNN Knn(k);

    Knn.getAllDistance();
    Knn.getMaxFreqLable();
    
    return 0;
}
