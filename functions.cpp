#include <iostream>
#include <vector>
#include <math.h> 
#include <limits.h>
#include <cmath>

float MSEloss(std::vector<float> X, int Y){
    int sizeOutput = X.size();
    float sum = 0;
    for (int i = 0; i < sizeOutput; i++) {
        sum += std::pow(((float)Y - X[i]), 2);
    }
    sum = sum / sizeOutput;
    return sum;
}
std::vector<float> MSElossDerivative(std::vector<float> X, int Y){
    int sizeOutput = X.size();
    std::vector<float> output;
    for (int i = 0; i < sizeOutput; i++) {
       float deriv = 2*((float)Y - X[i])/sizeOutput;
       output.push_back(deriv);
    }
    return output;
}
std::vector<std::vector<float>> multiplyMatrix(std::vector<float> input, std::vector<std::vector<float>> mat2)
{
    int sizeRowFirst = input.size();
    int sizeColumn = 1;
    int sizeRowTwo = mat2[0].size();
    std::vector<std::vector<float>> output(sizeRowFirst, std::vector<float>(sizeColumn, 0));
 

    for (int i = 0; i < sizeRowFirst; i++) {
        for (int j = 0; j < sizeColumn; j++) {
            
            for (int k = 0; k < sizeRowTwo; k++) {
               output[i][j] += input[i] * mat2[k][j]; 
            }
        }

    }
    return output; 
}

std::vector<std::vector<float>> multiplyMatrix(std::vector<std::vector<float>> mat1, std::vector<std::vector<float>> mat2)
{
    int sizeRowFirst = mat1[0].size();
    int sizeColumn = mat2.size();
    int sizeRowTwo = mat2[0].size();
    std::vector<std::vector<float>> output(sizeRowFirst, std::vector<float>(sizeColumn, 0));
 

    for (int i = 0; i < sizeRowFirst; i++) {
        for (int j = 0; j < sizeColumn; j++) {
            
            for (int k = 0; k < sizeRowTwo; k++) {
               output[i][j] += mat1[i][k] * mat2[k][j]; 
            }
        }

    }
    return output; 
}

std::vector<std::vector<float>> transposeMatrix(std::vector<std::vector<float>> matrix)
{
    int sizeRow = matrix[0].size();
    int sizeColumn = matrix.size();
    
    std::vector<std::vector<float>> output(sizeRow, std::vector<float>(sizeColumn, 0));
    for (int i = 0; i < sizeRow; i++) 
        for (int j = 0; j < sizeColumn; j++) 
            output[i][j] = matrix[j][i];  

    return output; 
}

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

std::vector<float> dotNN(std::vector<float> input, std::vector<std::vector<float>> weights, std::vector<float> bias)
{

    std::vector<float> output;
 

    for (int i = 0; i < weights[0].size(); i++) {
        float neuron = 0;
        for (int j = 0; j < weights.size(); j++) {
            for (int k = 0; k < weights[0].size(); k++) {
                neuron += input[i] * weights[i][j] + bias[i];
            }
        }
        output.push_back(sigmoid(neuron));
    }
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

std::vector<float> dotNNSoftmax(std::vector<float> input, std::vector<std::vector<float>> weights)
{

    std::vector<float> output;
    std::cout << "weights[0].size()" << weights[0].size() << std::endl; 
    std::cout << "weights.size()" << weights.size() << std::endl; 

    for (int i = 0; i < weights[0].size(); i++) {
        float neuron = 0;
        for (int j = 0; j < weights.size(); j++) {
            for (int k = 0; k < weights[0].size(); k++) {
                neuron += input[i] * weights[i][j];
            }
        }
        output.push_back(neuron);
    }
    return softmax(output); 
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
