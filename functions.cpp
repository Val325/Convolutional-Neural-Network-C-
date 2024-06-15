#include <iostream>
#include <vector>
#include <math.h> 
#include <limits.h>
/*
void multiplyMatrix(int mat1[][C1], int mat2[][C2])
{
    int sizeRow;
    int sizeColumn;
    int rslt[R1][C2];
    

    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            rslt[i][j] = 0;

            for (int k = 0; k < R2; k++) {
                rslt[i][j] += mat1[i][k] * mat2[k][j];
            }
        }

    }
}
*/
void Nothing(){
    std::cout << "nothing" << std::endl;
}
std::vector<float> sigmoid(const std::vector<float>& data) {
    
    /*  
        Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */
    
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<float> output(VECTOR_SIZE);
    
    
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[i] = 1 / (1 + exp(-data[i]));
    }
    
    return output;
}

float sigmoid(const float& data) {
    float output;
    output = 1 / (1 + exp(-data));
    
    return output;
}

std::vector<float> derivativeSigmoid(const std::vector<float>& data) {
    /*  
        Returns the value of the derivative sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: derivative sigmoid 
    */ 
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<float> output(VECTOR_SIZE);
    
    
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[i] = sigmoid(data[i]) * (1 - sigmoid(data[i])); 
    }
    
    return output;
}

float derivativeSigmoid(float& data) {
    /*  
        Returns the value of the derivative sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: derivative sigmoid 
    */ 
    float output; 
    output = sigmoid(data) * (1 - sigmoid(data)); 
    
    return output;
}

//
// Relu
//

std::vector<float> Relu(const std::vector<float>& data) {

    const unsigned long VECTOR_SIZE = data.size();
    std::vector<float> output(VECTOR_SIZE);
    
    
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        if (data[i] < 0) output[i] = 0;
        if (data[i] >= 0) output[i] = data[i]; 
    }
    
    return output;
}

float Relu(const float& data) {
    float output;
    if (data < 0) output = 0;
    if (data >= 0) output = data; 
    return output;
}

std::vector<float> derivativeRelu(const std::vector<float>& data) {
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<float> output(VECTOR_SIZE);
     
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        if (data[i] < 0) output[i] = 0;
        if (data[i] >= 0) output[i] = 1;
    }
    
    return output;
}

float derivativeRelu(float& data) {
    float output;
    if (data < 0) output = 0;
    if (data >= 0) output = 1; 
    return output;
}

//
//
//

std::vector<float> softmax(const std::vector<float>& data) { 
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<float> output(VECTOR_SIZE);

    float sum = 0; 
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        sum += data[i]; 
    }


    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[i] = data[i] / sum;
    }
    
    return output;
}

//https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
std::vector<std::vector<float>> softmaxDerivative(const std::vector<float>& data) { 
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<float> output(VECTOR_SIZE);
    std::vector<float> softmaxData(VECTOR_SIZE);
    
    softmaxData = softmax(data); 
    std::vector<std::vector<float>> softmaxJacobian(VECTOR_SIZE, std::vector<float>(VECTOR_SIZE, 0)); 
    softmaxJacobian.resize(VECTOR_SIZE);

    for(unsigned i = 0; i != VECTOR_SIZE-1; ++i ) {
       for(unsigned j = 0; j != VECTOR_SIZE-1; ++j ) {
            if (i == j){
                softmaxJacobian[i][j] = softmaxData[i] * (1 - softmaxData[i]);
            }else{
                softmaxJacobian[i][j] = -softmaxData[i] * softmaxData[j]; 
            }
        } 
    }
    
    return softmaxJacobian;
}

std::vector<std::vector<int>> NormalizeImage(std::vector<std::vector<int>> image, int min, int max){
    int sizeX = image.size(); 
    int sizeY = image[0].size();
    std::vector<std::vector<int>> output(sizeX, std::vector<int>(sizeY, 0)); 
    for(unsigned int i = 0; i != sizeX; ++i ) {
       for(unsigned int j = 0; j != sizeY; ++j ) {
            output[i][j] = 255 * (image[i][j] - min) / (max-min);
            //std::cout << "output[i][j]: " << output[i][j] << std::endl;
        } 
    }
    return output; 
}



std::vector<std::vector<int>> NormalizeImage(std::vector<std::vector<float>> image, float min, float max){
    int sizeX = image.size(); 
    int sizeY = image[0].size();
    std::vector<std::vector<int>> output(sizeX, std::vector<int>(sizeY, 0)); 
    for(unsigned int i = 0; i != sizeX; ++i ) {
       for(unsigned int j = 0; j != sizeY; ++j ) {
            output[i][j] = (int)(255 * (image[i][j] - min) / (max-min));
            //std::cout << "output[i][j]: " << output[i][j] << std::endl;
        } 
    }
    return output; 
}
template <class T>
std::vector<std::vector<T>> NormalizeImage(std::vector<std::vector<T>> image, T span, T min, T max){
    int sizeX = image.size(); 
    int sizeY = image[0].size();
    std::vector<std::vector<T>> output(sizeX, std::vector<T>(sizeY, 0)); 
    for(unsigned int i = 0; i != sizeX; ++i ) {
       for(unsigned int j = 0; j != sizeY; ++j ) {
            output[i][j] = (span * (image[i][j] - min) / (max-min));
            //std::cout << "output[i][j]: " << output[i][j] << std::endl;
        } 
    }
    return output; 
}
template <class T>
std::vector<T> NormalizeImage(std::vector<T> imageflat, T span, T min, T max){
    int size = imageflat.size(); 
    std::vector<T> output;
    output.resize(size);
    for(unsigned int i = 0; i != size; ++i ) {
       output[i] = (span * (imageflat[i] - min) / (max-min));

    }
    return output; 
}
template <class T>
inline T NormalizeImage(T num, T span, T min, T max){
    return (span * (num - min) / (max-min)); 
}

int FindMaxElem(std::vector<std::vector<int>> poolMax){
    int maxNumber = INT_MIN;
    for (int i = 0; i < poolMax[0].size(); ++i)              // rows
    {
        for (int j = 0; j < poolMax.size(); ++j)          // columns
        {
            if (poolMax[i][j] > maxNumber) {
                maxNumber = poolMax[i][j];
            }
        }
    }
    return maxNumber;
}
template <class T>
T FindMin(std::vector<T> array){
    T minNumber = INT_MAX;
    for (int i = 0; i < array.size(); ++i)              // rows
    {
        if (array[i] < minNumber) {
            minNumber = array[i];
        }

    }
    return minNumber;
}
template <class T>
T FindMax(std::vector<T> array){
    T maxNumber = INT_MIN;
    for (int i = 0; i < array.size(); ++i)              // rows
    {
        if (array[i] > maxNumber) {
            maxNumber = array[i];
        }

    }
    return maxNumber;
}
std::vector<int> FindMaxIndex(std::vector<std::vector<int>> poolMax){
    int maxNumber = INT_MIN;
    int iOffset;
    int jOffset;
    std::vector<int> maxData;
    for (int i = 0; i < poolMax[0].size(); ++i)              // rows
    {
        for (int j = 0; j < poolMax.size(); ++j)          // columns
        {
            if (poolMax[i][j] > maxNumber) {
                maxNumber = poolMax[i][j];
                iOffset = i;
                jOffset = j;
            }
        }
    }
    //offset from  
    maxData.push_back(iOffset); 
    maxData.push_back(jOffset); 
    return maxData;
}
int FindAverageElem(std::vector<std::vector<int>> poolAvg){
    int AvgNumber = INT_MIN;
    int totalNum = poolAvg[0].size() * poolAvg.size();
    int sum = 0;
    for (int i = 0; i < poolAvg[0].size(); ++i)              // rows
    {
        for (int j = 0; j < poolAvg.size(); ++j)          // columns
        {
               sum += poolAvg[i][j];
        }
    }
    return round(sum / totalNum);
}
