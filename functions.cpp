#include <iostream>
#include <vector>
#include <math.h> 
#include <limits.h>

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
            std::cout << "output[i][j]: " << output[i][j] << std::endl;
        } 
    }
    return output; 
}

std::vector<std::vector<int>> ConvolutionRed(std::vector<std::vector<std::vector<unsigned char>>> image){
    //For normalization
    int minNumber = INT_MAX;
    int maxNumber = INT_MIN;

    int redSize = image[0].size();
    int greenSize = image[1].size();
    int blueSize = image[2].size();

    int padding = 0;
    int stride = 2;
    int filter = 5;

    int sizeX = image[0].size(); 
    int sizeY = image[0][0].size();
    //std::cout << "sizeX: " << sizeX << std::endl;
    //std::cout << "sizeY: " << sizeY << std::endl;
    std::vector<std::vector<int>> kernel{ 
        { 0, 0, 3, 0, 0 }, 
        { 0, 0, 3, 0, 0 }, 
        { 2, 3, 4, 2, 1 },
        { 0, 0, 3, 0, 0 }, 
        { 0, 0, 3, 0, 0 }, 
    };
    int convX = ((sizeX - filter + 2 * padding) / stride) + 1;
    int convY = ((sizeY - filter + 2 * padding) / stride) + 1;

    std::vector<std::vector<int>> output(convX, std::vector<int>(convY, 0)); 

    for(int x = 0; x != convX; ++x) {
       for(int y = 0; y != convY; ++y) {
            int sumKernel = 0;
            int itreationKernel = 0;
            //std::cout << "-------------------------------------------" << std::endl;
            //std::cout << "x: " << x << " y: " << y << std::endl;
            for(int yKernel = 0; yKernel != kernel.size(); ++yKernel ) {
                for(int xKernel = 0; xKernel != kernel[0].size(); ++xKernel ) {
                    //std::cout << "-------------------------------------------" << std::endl;
                    //std::cout << "x+xKernel: " << x+xKernel+stride << std::endl;
                    //std::cout << "y+yKernel: " << y+yKernel+stride << std::endl;
                    //std::cout << "image[0][x+xKernel][y+yKernel]: " << (int)image[0][x+xKernel][y+yKernel] << std::endl;
                    //std::cout << "kernel[xKernel][yKernel]: " << kernel[xKernel][yKernel] << std::endl;
                    sumKernel += image[0][x+xKernel+stride][y+yKernel+stride] * kernel[xKernel][yKernel];
                    //std::cout << "sumKernel: " << sumKernel << std::endl;
                    itreationKernel++;

                } 
            }

            if (sumKernel > maxNumber) {
                maxNumber = sumKernel;
            }
            if (sumKernel < minNumber) {
                minNumber = sumKernel;
            }
            output[x][y] = sumKernel;
            //std::cout << "output[x][y]: " << output[x][y] << std::endl;
            //std::cout << "itreationKernelRed: " << itreationKernel << std::endl;

        }


    }
    std::cout << "maxNumber: " << maxNumber << std::endl;
    std::cout << "minNumber: " << minNumber << std::endl;
    NormalizeImage(output, minNumber, maxNumber);
    /*
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "convX: " << convX << std::endl;
    std::cout << "convY: " << convY << std::endl;
    std::cout << "output.size(): " << output.size() << std::endl;
    std::cout << "output[0].size(): " << output[0].size() << std::endl; 
    */
    return output; 
}
std::vector<std::vector<int>> ConvolutionGreen(std::vector<std::vector<std::vector<unsigned char>>> image){
    //For normalization
    int minNumber = 0;
    int maxNumber = 0;

    int redSize = image[0].size();
    int greenSize = image[1].size();
    int blueSize = image[2].size();
    int padding = 0;
    int stride = 2;
    int filter = 5;
    int sizeX = image[0].size(); 
    int sizeY = image[0][0].size();
    //std::cout << "sizeX: " << sizeX << std::endl;
    //std::cout << "sizeY: " << sizeY << std::endl;
    std::vector<std::vector<int>> kernel{ 
        { 0, 0, 3, 0, 0 }, 
        { 0, 0, 3, 0, 0 }, 
        { 2, 3, 4, 2, 1 },
        { 0, 0, 3, 0, 0 }, 
        { 0, 0, 3, 0, 0 }, 
    };
    int convX = ((sizeX - filter + 2 * padding) / stride) + 1;
    int convY = ((sizeY - filter + 2 * padding) / stride) + 1;

    std::vector<std::vector<int>> output(convX, std::vector<int>(convY, 0)); 

    for(int x = 0; x != convX; ++x) {
       for(int y = 0; y != convY; ++y) {
            int sumKernel = 0;
            int itreationKernel = 0;
            //std::cout << "-------------------------------------------" << std::endl;
            //std::cout << "x: " << x << " y: " << y << std::endl;
            for(int yKernel = 0; yKernel != kernel.size(); ++yKernel ) {
                for(int xKernel = 0; xKernel != kernel[0].size(); ++xKernel ) {
                    //std::cout << "-------------------------------------------" << std::endl;
                    //std::cout << "x+xKernel: " << x+xKernel+stride << std::endl;
                    //std::cout << "y+yKernel: " << y+yKernel+stride << std::endl;
                    //std::cout << "image[0][x+xKernel][y+yKernel]: " << (int)image[0][x+xKernel][y+yKernel] << std::endl;
                    //std::cout << "kernel[xKernel][yKernel]: " << kernel[xKernel][yKernel] << std::endl;
                    sumKernel += image[1][x+xKernel+stride][y+yKernel+stride] * kernel[xKernel][yKernel];
                    //std::cout << "sumKernel: " << sumKernel << std::endl;
                    itreationKernel++;

                } 
            }
            output[x][y] = sumKernel;
            //std::cout << "output[x][y]: " << output[x][y] << std::endl;
            //std::cout << "itreationKernelGreen: " << itreationKernel << std::endl;

        } 
    }/*
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "convX: " << convX << std::endl;
    std::cout << "convY: " << convY << std::endl;
    std::cout << "output.size(): " << output.size() << std::endl;
    std::cout << "output[0].size(): " << output[0].size() << std::endl; 
    */
    return output; 
}
std::vector<std::vector<int>> ConvolutionBlue(std::vector<std::vector<std::vector<unsigned char>>> image){
    //For normalization
    int minNumber = 0;
    int maxNumber = 0;

    int redSize = image[0].size();
    int greenSize = image[1].size();
    int blueSize = image[2].size();
    int padding = 0;
    int stride = 2;
    int filter = 5;
    int sizeX = image[0].size(); 
    int sizeY = image[0][0].size();
    //std::cout << "sizeX: " << sizeX << std::endl;
    //std::cout << "sizeY: " << sizeY << std::endl;
    std::vector<std::vector<int>> kernel{ 
        { 0, 0, 3, 0, 0 }, 
        { 0, 0, 3, 0, 0 }, 
        { 2, 3, 4, 2, 1 },
        { 0, 0, 3, 0, 0 }, 
        { 0, 0, 3, 0, 0 }, 
    };
    int convX = ((sizeX - filter + 2 * padding) / stride) + 1;
    int convY = ((sizeY - filter + 2 * padding) / stride) + 1;

    std::vector<std::vector<int>> output(convX, std::vector<int>(convY, 0)); 

    for(int x = 0; x != convX; ++x) {
       for(int y = 0; y != convY; ++y) {
            int sumKernel = 0;
            int itreationKernel = 0;
            //std::cout << "-------------------------------------------" << std::endl;
            //std::cout << "x: " << x << " y: " << y << std::endl;
            for(int yKernel = 0; yKernel != kernel.size(); ++yKernel ) {
                for(int xKernel = 0; xKernel != kernel[0].size(); ++xKernel ) {
                    std::cout << "-------------------------------------------" << std::endl;
                    //std::cout << "x+xKernel: " << x+xKernel+stride << std::endl;
                    //std::cout << "y+yKernel: " << y+yKernel+stride << std::endl;
                    //std::cout << "image[0][x+xKernel][y+yKernel]: " << (int)image[0][x+xKernel][y+yKernel] << std::endl;
                    //std::cout << "kernel[xKernel][yKernel]: " << kernel[xKernel][yKernel] << std::endl;
                    sumKernel += image[2][x+xKernel+stride][y+yKernel+stride] * kernel[xKernel][yKernel];
                    //std::cout << "sumKernel: " << sumKernel << std::endl;
                    itreationKernel++;

                } 
            }
            output[x][y] = sumKernel;
            
            //std::cout << "output[x][y]: " << output[x][y] << std::endl;
            //std::cout << "itreationKernelBlue: " << itreationKernel << std::endl;
        
        } 
    }
    /*
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "convX: " << convX << std::endl;
    std::cout << "convY: " << convY << std::endl;
    std::cout << "output.size(): " << output.size() << std::endl;
    std::cout << "output[0].size(): " << output[0].size() << std::endl; 
    */
    return output; 
}
