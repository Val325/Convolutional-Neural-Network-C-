#include <iostream>
#include <cnn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include "functions.cpp"
#include <bits/stdc++.h>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebufalgo.h>

//using namespace OIIO;
class ConvolutionalNeuralNetwork {       
    private:
        std::string filename;   
        std::vector<std::pair<std::vector<std::vector<std::vector<int>>>, int>> dataset; 
        
        int amountEpoch;
        double learningRate;
        
        std::default_random_engine generator; 
        

        int sizeInputX; // for backpropogation
        int sizeInputY; // for backpropogation
        std::vector<std::vector<int>> convBeforePooling;

        std::vector<std::vector<std::vector<double>>> kernelsW1Red;
        std::vector<std::vector<std::vector<double>>> kernelsW1Green;
        std::vector<std::vector<std::vector<double>>> kernelsW1Blue;
        
        int numKernelsW1;
        int sizeKernelW1;

        std::vector<std::vector<std::vector<double>>> kernelsW2Red;
        std::vector<std::vector<std::vector<double>>> kernelsW2Green;
        std::vector<std::vector<std::vector<double>>> kernelsW2Blue;
        
        int numKernelsW2;
        int sizeKernelW2;
        std::vector<std::vector<std::vector<double>>> kernelsW3;
        std::vector<double> inputDesnse;
        int DenseLayerSize;
        int weightsHiddenOneSize;
        int weightsHiddenTwoSize;
        int weightsOutputSize;

        std::vector<std::vector<double>> weightsInput;
        std::vector<std::vector<double>> weightsHiddenOne;
        std::vector<double> biasHiddenOne;
        std::vector<std::vector<double>> weightsHiddenTwo;
        std::vector<double> biasHiddenTwo;
        std::vector<std::vector<double>> weightsOutput;

        std::vector<std::vector<double>> GradWeightsInput;
        std::vector<std::vector<double>> GradWeightsHiddenOne;
        std::vector<double> GradBiasHiddenOne;
        std::vector<std::vector<double>> GradWeightsHiddenTwo;
        std::vector<double> GradBiasHiddenTwo;
        std::vector<std::vector<double>> GradWeightsOutput;

        std::vector<std::pair<int, int>> firstMaxPoolBackpropIndex;
        int SizeXfirstMaxPool;
        int SizeYfirstMaxPool;
        std::vector<std::pair<int, int>> TwoMaxPoolBackpropIndex;
        int SizeXtwoMaxPool;
        int SizeYtwoMaxPool;
    public:
    std::vector<std::vector<int>> convolve2D(std::vector<std::vector<std::vector<int>>> image, int channel, std::vector<std::vector<double>> kernelConv) {
        //For normalization
        double minNumber = INT_MAX;
        double maxNumber = INT_MIN;
        int redSize = image[0].size();
        int greenSize = image[1].size();
        int blueSize = image[2].size();

        int padding = 0;
        int stride = 1;

        int sizeW = image[0].size(); 
        int sizeH = image[0][0].size();
        //std::cout << "sizeW: " << sizeW << std::endl;
        //std::cout << "sizeH: " << sizeH << std::endl;
        int convW = ((sizeW - kernelConv.size() + 2 * padding) / stride) + 1;
        int convH = ((sizeH - kernelConv.size() + 2 * padding) / stride) + 1;
        std::vector<std::vector<int>> output(convW, std::vector<int>(convH, 0));
        //std::cout << "convW: " << convW << std::endl;
        //std::cout << "convH: " << convH << std::endl;
        for (int h = 0; h < convH; h++){
            for(int w = 0; w < convW; w++){
                int sum = 0;

                  if (w-1 >= 0 && w < convW && h-1 >= 0 && h < convH){
                        sum = image[channel][w-1][h-1]*kernelConv[0][0] +
                        image[channel][w][h-1]*kernelConv[0][1] +
                        image[channel][w+1][h-1]*kernelConv[0][2] +
                        image[channel][w-1][h]*kernelConv[1][0] +
                        image[channel][w][h]*kernelConv[1][1] +
                        image[channel][w+1][h]*kernelConv[1][2] +
                        image[channel][w-1][h+1]*kernelConv[2][0] +
                        image[channel][w][h+1]*kernelConv[2][1] +
                        image[channel][w+1][h+1]*kernelConv[2][2];
                }
                output[w][h] = sum;
                if (output[w][h] > maxNumber) {
                    maxNumber = output[w][h];
                }
                if (output[w][h] < minNumber) {
                    minNumber = output[w][h];
                }   
            }
        }
        return NormalizeImage(output, minNumber, maxNumber); //output; 
    }
    std::vector<std::vector<int>> convolve2DSlow(std::vector<std::vector<std::vector<int>>> image, int channel, std::vector<std::vector<double>> kernelConv) {
        //For normalization
        double minNumber = INT_MAX;
        double maxNumber = INT_MIN;
        int redSize = image[0].size();
        int greenSize = image[1].size();
        int blueSize = image[2].size();

        int padding = 0;
        int stride = 1;

        int sizeW = image[0].size(); 
        int sizeH = image[0][0].size();
        //std::cout << "sizeW: " << sizeW << std::endl;
        //std::cout << "sizeH: " << sizeH << std::endl;
        //std::cout << "kernelConv[0].size(): " << kernelConv[0].size() << std::endl;
        //std::cout << "kernelConv.size(): " << kernelConv.size() << std::endl;
        int convW = ((sizeW - kernelConv[0].size() + 2 * padding) / stride) + 1;
        int convH = ((sizeH - kernelConv.size() + 2 * padding) / stride) + 1;

        std::vector<std::vector<int>> output(convW, std::vector<int>(convH, 0));
        //std::cout << "convW: " << convW << std::endl;
        //std::cout << "convH: " << convH << std::endl;    
        
        
        int kCenterX = kernelConv.size() / 2;
        int kCenterY = kernelConv.size() / 2;

        for (int h = 0; h < convH; h++){
            for(int w = 0; w < convW; w++){
                int sum = 0;
                for (int hk = 0; hk < kernelConv.size(); hk++){
                    int mm = kernelConv.size() - 1 - hk;      // row index of flipped kernel
                    for(int wk = 0; wk < kernelConv[0].size(); wk++){
                        int nn = kernelConv.size() - 1 - wk;  // column index of flipped kernel

                        // index of input signal, used for checking boundary
                        int hh = h + (kCenterY - mm);
                        int ww = w + (kCenterX - nn);

                        if (ww-1 >= 0 && ww < convW && hh-1 >= 0 && hh < convH){
                            output[h][w] += image[channel][hh][ww] * kernelConv[mm][nn];
                        }
                    }
                }
                //output[w][h] = sum;

                if (output[w][h] > maxNumber) {
                    maxNumber = output[w][h];
                }
                if (output[w][h] < minNumber) {
                    minNumber = output[w][h];
                }   
            }
        }
        return NormalizeImage(output, minNumber, maxNumber); //output; 
    }
    std::vector<std::vector<int>> convolve2DSlow(std::vector<std::vector<int>> image, std::vector<std::vector<double>> kernelConv) {
        //For normalization
        double minNumber = INT_MAX;
        double maxNumber = INT_MIN;
        int redSize = image[0].size();
        int greenSize = image[1].size();
        int blueSize = image[2].size();

        int padding = 0;
        int stride = 1;

        int sizeW = image.size(); 
        int sizeH = image[0].size();
        //std::cout << "sizeW: " << sizeW << std::endl;
        //std::cout << "sizeH: " << sizeH << std::endl;
        int convW = ((sizeW - kernelConv[0].size() + 2 * padding) / stride) + 1;
        int convH = ((sizeH - kernelConv.size() + 2 * padding) / stride) + 1;
        std::vector<std::vector<int>> output(convW, std::vector<int>(convH, 0));
        //std::cout << "convW: " << convW << std::endl;
        //std::cout << "convH: " << convH << std::endl;
        
        int kCenterX = kernelConv.size() / 2;
        int kCenterY = kernelConv.size() / 2;

        for (int h = 0; h < convH; h++){
            for(int w = 0; w < convW; w++){
                int sum = 0;
                for (int hk = 0; hk < kernelConv.size(); hk++){
                    int mm = kernelConv.size() - 1 - hk;      // row index of flipped kernel
                    for(int wk = 0; wk < kernelConv[0].size(); wk++){
                        int nn = kernelConv.size() - 1 - wk;  // column index of flipped kernel

                        // index of input signal, used for checking boundary
                        int hh = h + (kCenterY - mm);
                        int ww = w + (kCenterX - nn);
                        if (ww-1 >= 0 && ww < convW && hh-1 >= 0 && hh < convH){
                            output[h][w] += image[hh][ww] * kernelConv[mm][nn];
                        }
                    }
                }
                //output[w][h] = sum;
                if (output[w][h] > maxNumber) {
                    maxNumber = output[w][h];
                }
                if (output[w][h] < minNumber) {
                    minNumber = output[w][h];
                }   
            }
        }
        return NormalizeImage(output, minNumber, maxNumber); //output; 
    }
    /*std::vector<std::vector<int>> AveragePool(std::vector<std::vector<int>> image, std::vector<std::vector<double>> kernelConvPerv){
        int padding = 0;
        int stride = 2;
        int filter = 2;
        int sizeX = image.size(); 
        int sizeY = image[0].size();
        int sizePoolX = ((sizeX - kernelConvPerv[0][0].size() + 2 * padding) / stride) + 1;
        int sizePoolY = ((sizeY - kernelConvPerv[0][0].size() + 2 * padding) / stride) + 1;
        std::vector<std::vector<int>> output(sizePoolX, std::vector<int>(sizePoolY, 0));
        //std::cout << "sizePoolX: " << sizePoolX << std::endl;
        //std::cout << "sizePoolY: " << sizePoolY << std::endl;
        for (int i = 0; i < sizePoolX; ++i)              // rows
        {
            for (int j = 0; j < sizePoolY; ++j)          // columns
            {
                std::vector<std::vector<int>> pool(filter, std::vector<int>(filter, 0));
                if (i-1 >= 0 && i < sizeX && j-1 >= 0 && j < sizeY){
                    pool[0][0] = image[i][j]; 
                    pool[0][1] = image[i][j+1];
                    pool[1][0] = image[i+1][j];
                    pool[1][1] = image[i+1][j+1];
                    //std::cout << "--------------------------" << std::endl;
                    //std::cout << "pool[0][0]: " << pool[0][0] << std::endl;
                    //std::cout << "pool[0][1]: " << pool[0][1] << std::endl; 
                    //std::cout << "pool[1][0]: " << pool[1][0] << std::endl;
                    //std::cout << "pool[1][1]: " << pool[1][1] << std::endl;
                    //std::cout << "Avg: " << FindAverageElem(pool) << std::endl;
                    output[i][j] = FindAverageElem(pool);
                }
            }
        }
        return output;
    }*/
    MaxPoolingData MaxPool(std::vector<std::vector<int>> image){
        MaxPoolingData PoolingData;
        int padding = 0;
        int stride = 2;
        int filter = 2;
        
        MaxPoolingData dataLayer;
        int sizeX = image.size(); 
        int sizeY = image[0].size();
        int sizePoolX = ((sizeX - filter + 2 * padding) / stride) + 1;
        int sizePoolY = ((sizeY - filter + 2 * padding) / stride) + 1;
        std::vector<std::vector<int>> output(sizePoolX, std::vector<int>(sizePoolY, 0));
        //convBeforePooling = convBeforePooling(sizeInputX, std::vector<int>(sizeInputY, 0));
        //std::cout << "sizePoolX: " << sizePoolX << std::endl;
        //std::cout << "sizePoolY: " << sizePoolY << std::endl;
        for (int i = 0; i < sizePoolX; ++i)              // rows
        {
            for (int j = 0; j < sizePoolY; ++j)          // columns
            {
                std::vector<std::vector<int>> pool(filter, std::vector<int>(filter, 0));
                if (i-1 >= 0 && i < sizeX && j-1 >= 0 && j < sizeY){
                    pool[0][0] = image[i][j]; 
                    pool[0][1] = image[i][j+1];
                    pool[1][0] = image[i+1][j];
                    pool[1][1] = image[i+1][j+1];

                    std::vector<int> indexs = FindMaxIndex(pool);
                    std::pair<int, int> pair(indexs[0], indexs[1]);
                    std::pair<std::pair<int, int>, int> pairOut(pair, FindMaxElem(pool)); 
                    PoolingData.MaxPoolBackpropIndex.push_back(pairOut);
                    //std::cout << "--------------------------" << std::endl;
                    //std::cout << "i: " << indexs[0] << std::endl;
                    //std::cout << "j: " << indexs[1] << std::endl;

                    //std::cout << "pool[0][0]: " << pool[0][0] << std::endl;
                    //std::cout << "pool[0][1]: " << pool[0][1] << std::endl; 
                    //std::cout << "pool[1][0]: " << pool[1][0] << std::endl;
                    //std::cout << "pool[1][1]: " << pool[1][1] << std::endl;
                    //std::cout << "max: " << FindMaxElem(pool) << std::endl;
                    output[i][j] = FindMaxElem(pool);
                } 
            }
        }
        /*
struct MaxPoolingData
{
    std::vector<std::vector<int>> output;
    std::vector<std::pair<int, int>> MaxPoolBackpropIndex;
    int SizeXMaxPool;
    int SizeYMaxPool;
};
         */
        PoolingData.output = output;
        PoolingData.SizeXMaxPool = sizeX;
        PoolingData.SizeYMaxPool = sizeY;
        return PoolingData;
    }
    std::vector<std::vector<int>> MaxPoolBackprop(std::vector<double> data, MaxPoolingData MaxPoolDataLayer){
        int sizeX = MaxPoolDataLayer.SizeXMaxPool; 
        int sizeY = MaxPoolDataLayer.SizeYMaxPool;
        /*
struct MaxPoolingData
{
    std::vector<std::vector<int>> output;
    std::vector<std::pair<std::pair<int, int>, int>> MaxPoolBackpropIndex;
    int SizeXMaxPool;
    int SizeYMaxPool;
};
         */
        std::vector<std::vector<int>> output(sizeX, std::vector<int>(sizeY, 0));
        //std::cout << "sizePoolX: " << sizePoolX << std::endl;
        //std::cout << "sizePoolY: " << sizePoolY << std::endl;
        for (int i = 0; i < sizeX; ++i)              // rows
        {
            for (int j = 0; j < sizeY; ++j)          // columns
            {
                output[i][j] = 0;
                //int indexI = MaxPoolDataLayer
                //int indexJ = MaxPoolDataLayer
            }
        }
    
        for (int i = 0; i < MaxPoolDataLayer.MaxPoolBackpropIndex.size(); ++i){
            output[MaxPoolDataLayer.MaxPoolBackpropIndex[i].first.first][MaxPoolDataLayer.MaxPoolBackpropIndex[i].first.second] = MaxPoolDataLayer.MaxPoolBackpropIndex[i].second;
        }
        return output;
    }
    ConvolutionalNeuralNetwork() {
        learningRate = 0.01;
        amountEpoch = 10;
        weightsHiddenOneSize = 30;
        biasHiddenOne.resize(weightsHiddenOneSize);
        weightsHiddenTwoSize = 6;
        biasHiddenTwo.resize(weightsHiddenTwoSize);
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        weightsOutputSize = 2;

        numKernelsW1 = 1;
        sizeKernelW1 = 4;
        
        numKernelsW2 = 1;
        sizeKernelW2 = 4;
    
        for (int i=0; i<numKernelsW1; i++) {
            std::vector<std::vector<double>> kernRed; 
            std::vector<std::vector<double>> kernGreen;
            std::vector<std::vector<double>> kernBlue;
            for (int j=0; j<sizeKernelW1; j++) {
                std::vector<double> rowRed;
                std::vector<double> rowGreen;
                std::vector<double> rowBlue; 
                for (int k=0; k<sizeKernelW1; k++) {
                    
                    double red = distribution(generator);
                    double green = distribution(generator);
                    double blue = distribution(generator);
                    rowRed.push_back(red);
                    rowGreen.push_back(green);
                    rowBlue.push_back(blue);                    
                }
                kernRed.push_back(rowRed);
                kernGreen.push_back(rowGreen);
                kernBlue.push_back(rowBlue);
                rowRed.clear();
                rowGreen.clear();
                rowBlue.clear();
            }
            kernelsW1Red.push_back(kernRed);
            kernelsW1Green.push_back(kernGreen);
            kernelsW1Blue.push_back(kernBlue);
        }



        for (int i=0; i<numKernelsW2; i++) {
            std::vector<std::vector<double>> kernRed; 
            std::vector<std::vector<double>> kernGreen;
            std::vector<std::vector<double>> kernBlue;
            for (int j=0; j<sizeKernelW2; j++) {
                std::vector<double> rowRed;
                std::vector<double> rowGreen;
                std::vector<double> rowBlue; 
                for (int k=0; k<sizeKernelW2; k++) {
                    double red = distribution(generator);
                    double green = distribution(generator);
                    double blue = distribution(generator);
                    rowRed.push_back(red);
                    rowGreen.push_back(green);
                    rowBlue.push_back(blue);
                }
                kernRed.push_back(rowRed);
                kernGreen.push_back(rowGreen);
                kernBlue.push_back(rowBlue);
                rowRed.clear();
                rowGreen.clear();
                rowBlue.clear();
            }
            kernelsW2Red.push_back(kernRed);
            kernelsW2Green.push_back(kernGreen);
            kernelsW2Blue.push_back(kernBlue);
        }

  
        std::vector<std::vector<int>> convImageRed = convolve2DSlow(loadImage("datasetpreprocessing/test/cats/cat.1.png"), 0, kernelsW1Red[0]);
        std::vector<std::vector<int>> convImageGreen = convolve2DSlow(loadImage("datasetpreprocessing/test/cats/cat.1.png"), 1, kernelsW1Green[0]);
        std::vector<std::vector<int>> convImageBlue = convolve2DSlow(loadImage("datasetpreprocessing/test/cats/cat.1.png"), 2, kernelsW1Blue[0]);
               
        
        std::cout << "firts conv size: " << convImageRed.size() << " " << convImageRed[0].size() << std::endl;
        SizeXfirstMaxPool = convImageRed.size();
        SizeYfirstMaxPool = convImageRed[0].size(); 
        std::cout << "max pool 4x4" << std::endl; 
        MaxPoolingData MaxPoolRed = MaxPool(convImageRed);
        MaxPoolingData MaxPoolGreen = MaxPool(convImageGreen); 
        MaxPoolingData MaxPoolBlue = MaxPool(convImageBlue);

        std::cout << "MaxPool size: " << MaxPoolRed.output.size() << " " << MaxPoolRed.output[0].size() << std::endl; 
        std::cout << "max pool 4x4" << std::endl; 

        std::vector<std::vector<int>> convImageRedTwo = convolve2DSlow(MaxPoolRed.output, kernelsW2Red[0]);
        std::vector<std::vector<int>> convImageGreenTwo = convolve2DSlow(MaxPoolGreen.output, kernelsW2Green[0]);
        std::vector<std::vector<int>> convImageBlueTwo = convolve2DSlow(MaxPoolGreen.output, kernelsW2Blue[0]); 
        std::cout << "two conv size: " << convImageRedTwo.size() << " " << convImageRedTwo[0].size() << std::endl; 
        std::cout << "max pool 4x4" << std::endl; 

        SizeXtwoMaxPool = convImageRedTwo.size();
        SizeYtwoMaxPool = convImageRedTwo[0].size(); 
        MaxPoolingData MaxPoolRedTwo = MaxPool(convImageRedTwo);
        MaxPoolingData MaxPoolGreenTwo = MaxPool(convImageGreenTwo); 
        MaxPoolingData MaxPoolBlueTwo = MaxPool(convImageBlueTwo);
        
        std::cout << "MaxPool size: " << MaxPoolRedTwo.output.size() << " " << MaxPoolRedTwo.output[0].size() << std::endl; 
        std::cout << "dense" << std::endl; 

        const char* filename = "cat pool.1.png";
        const int xres = MaxPoolRedTwo.output.size(), yres = MaxPoolRedTwo.output[0].size(), channels = 3;
        std::cout << "max pool size x: " <<  MaxPoolRedTwo.output.size() << std::endl;
        std::cout << "max pool size y: " <<  MaxPoolRedTwo.output[0].size() << std::endl;

        std::cout << "Red channel input neuron: " <<  MaxPoolRedTwo.output.size() * MaxPoolRedTwo.output[0].size() << std::endl;
        std::cout << "Green channel input neuron: " <<  MaxPoolGreenTwo.output.size() * MaxPoolGreenTwo.output[0].size() << std::endl;
        std::cout << "Blue channel input neuron: " <<  MaxPoolBlueTwo.output.size() * MaxPoolBlueTwo.output[0].size() << std::endl;        

        DenseLayerSize = (MaxPoolRedTwo.output.size() * MaxPoolRedTwo.output[0].size()) + (MaxPoolGreenTwo.output.size() * MaxPoolGreenTwo.output[0].size()) + (MaxPoolBlueTwo.output.size() * MaxPoolBlueTwo.output[0].size()); 
        std::cout << "total dense layer: " << DenseLayerSize << std::endl; 
        
        //    
        // Normalization dense layer
        //
        for (int i=0; i<MaxPoolRedTwo.output.size(); i++) {
            for (int j=0; j<MaxPoolRedTwo.output[0].size(); j++) {
                inputDesnse.push_back(MaxPoolRedTwo.output[i][j]);
            } 
        } 
        for (int i=0; i<MaxPoolGreenTwo.output.size(); i++) {
            for (int j=0; j<MaxPoolGreenTwo.output[0].size(); j++) {
                inputDesnse.push_back(MaxPoolGreenTwo.output[i][j]);
            } 
        }
        for (int i=0; i<MaxPoolBlueTwo.output.size(); i++) {
            for (int j=0; j<MaxPoolBlueTwo.output[0].size(); j++) {
                inputDesnse.push_back(MaxPoolBlueTwo.output[i][j]);
            } 
        } 
        std::cout << "inputDesnse.size(): " << inputDesnse.size() << std::endl;
        int minInputDense = FindMin(inputDesnse);
        int maxInputDense = FindMax(inputDesnse);
        
          
        for (int j=0; j<inputDesnse.size(); j++) {
            //std::cout << "num: " << j << " inputDesnse: " << NormalizeImage(inputDesnse[j], 1.0d, (double)minInputDense, (double)maxInputDense) << std::endl;
            //std::cout << "min: " << (double)minInputDense << std::endl;
            //std::cout << "max: " << (double)maxInputDense << std::endl;

            inputDesnse[j] = NormalizeImage(inputDesnse[j], 1.0d, (double)minInputDense, (double)maxInputDense); 
        }

        // init input-hidden layer
        weightsInput.resize(inputDesnse.size());
        for (int i=0; i<inputDesnse.size(); i++) {
            weightsInput[i].resize(weightsHiddenOneSize); 
            for (int j=0; j<weightsHiddenOneSize; j++) {
                weightsInput[i][j] = distribution(generator); 
                //std::cout << "row: " << i << " columm: " << j << " weightsInput[i][j]: " << weightsInput[i][j] << std::endl;
            } 
        } 
        
        // init hidden1-hidden2 layer
        weightsHiddenOne.resize(weightsHiddenOneSize);
        for (int i=0; i<weightsHiddenOneSize; i++) {
            weightsHiddenOne[i].resize(weightsHiddenTwoSize); 
            for (int j=0; j<weightsHiddenTwoSize; j++) {
                weightsHiddenOne[i][j] = distribution(generator);
                //
            } 
        } 
        
        // init hidden2-output layer
        weightsHiddenTwo.resize(weightsHiddenTwoSize);
        for (int i=0; i<weightsHiddenTwoSize; i++) {
            
            weightsHiddenTwo[i].resize(weightsOutputSize); 
            for (int j=0; j<weightsHiddenTwo[i].size(); j++) {
                weightsHiddenTwo[i][j] = distribution(generator);
                //std::cout << "columm: " << j << std::endl;
                //std::cout << "row: " << i << " columm: " << j << " weightsHiddenTwo[i][j]: " << weightsHiddenTwo[i][j] << std::endl;
            }
        } 

        /*for (int i=0; i<biasInput.size(); i++) { 
            biasInput[i] = (double)((double)rand()) / RAND_MAX;
            //std::cout << "biasHiddenOne[i]: " << biasHiddenOne[i] << std::endl;
        } */        

        for (int i=0; i<biasHiddenOne.size(); i++) { 
            biasHiddenOne[i] = distribution(generator);
            //std::cout << "biasHiddenOne[i]: " << biasHiddenOne[i] << std::endl;
        }         

        for (int i=0; i<biasHiddenTwo.size(); i++) { 
            biasHiddenTwo[i] = distribution(generator);
            //std::cout << "biasHiddenTwo[i]: " << biasHiddenTwo[i] << std::endl;
        }   
        
        std::vector<double> HiddenFirst = dotNN(inputDesnse, weightsInput, biasHiddenOne);
        std::vector<double> HiddenTwo = dotNN(HiddenFirst, weightsHiddenOne, biasHiddenTwo);
        std::vector<double> OutputProb = dotNNSoftmax(HiddenTwo, weightsHiddenTwo);
        std::cout << "----------------------------------" << std::endl;
        std::cout << "OutputProb.size(): " << OutputProb.size() << std::endl;
        std::cout << "Cat: " << OutputProb[0] << std::endl;
        std::cout << "Dog: " << OutputProb[1] << std::endl;

        std::cout << "loss: " << MSEloss(OutputProb, 1) << std::endl;
        std::cout << "loss derivative 1: " << MSElossDerivative(OutputProb, 1)[0] << std::endl;
        std::cout << "loss derivative 2: " << MSElossDerivative(OutputProb, 1)[1] << std::endl;
        
        std::cout << "----------------------------------" << std::endl;
        std::cout << "inputDesnse min: " << minInputDense << std::endl;
        std::cout << "inputDesnse max: " << maxInputDense << std::endl;
        int pixels[xres * yres * channels];
        int totalPixel = 0;
        for (int i=0; i<xres; i++) {
            for (int j=0; j<yres; j++) {
                
                totalPixel++;
                pixels[totalPixel*channels] = MaxPoolRedTwo.output[i][j];
                pixels[totalPixel*channels + 1] = MaxPoolGreenTwo.output[i][j];
                pixels[totalPixel*channels + 2] = MaxPoolBlueTwo.output[i][j]; 
            }
        }

        std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(filename);
        OIIO::ImageSpec spec(xres, yres, channels, OIIO::TypeDesc::UINT8);
        out->open(filename, spec);
        out->write_image(OIIO::TypeDesc::UINT8, pixels);
        out->close(); 
    }
    std::vector<std::vector<std::vector<int>>> loadImage(std::string filepath){
        filename = filepath;
        auto inp = OIIO::ImageInput::open(filename);


        const OIIO::ImageSpec &spec = inp->spec();
        int xres = spec.width;
        int yres = spec.height;

        int nchannels = spec.nchannels;

        auto pixels = std::unique_ptr<int[]>(new int[xres * yres * nchannels]);
        inp->read_image(0, 0, 0, nchannels, OIIO::TypeDesc::UINT8, &pixels[0]);
        inp->close();

        std::vector<std::vector<std::vector<int>>> Image;

        std::vector<int> Rarray(xres*yres);
        std::vector<int> Garray(xres*yres);
        std::vector<int> Barray(xres*yres);
        
        for (int i=0; i<xres*yres; i++) { 
            Rarray[i] = pixels[i*nchannels];
            Garray[i] = pixels[i*nchannels + 1];
            Barray[i] = pixels[i*nchannels + 2];

        }

        std::vector<std::vector<int>> Rprocessed(xres, std::vector<int>(yres, 0));
        std::vector<std::vector<int>> Gprocessed(xres, std::vector<int>(yres, 0));
        std::vector<std::vector<int>> Bprocessed(xres, std::vector<int>(yres, 0));
        
        int pixelFlatToXY = 0; 
        for(unsigned int x = 0; x != xres; ++x ) {
            for(unsigned int y = 0; y != yres; ++y ) {
                Rprocessed[x][y] = (int)Rarray[pixelFlatToXY];
                Gprocessed[x][y] = (int)Garray[pixelFlatToXY];
                Bprocessed[x][y] = (int)Barray[pixelFlatToXY];
                pixelFlatToXY++;
            } 
        }

        Image.push_back(Rprocessed);
        Image.push_back(Gprocessed);
        Image.push_back(Bprocessed);
        return Image;
    }

    std::vector<std::string> getAllFiles(std::string pathfiles){
        std::string path = pathfiles;
        for (const auto & entry : std::filesystem::directory_iterator(path))
            std::cout << entry.path() << std::endl;
    }
    void loadDataset(){
        std::string pathCat = "datasetpreprocessing/train/cats";
        std::string pathDog = "datasetpreprocessing/train/dogs";
        int size = 100;
        int iterSize = 0;
        for (const auto & entry : std::filesystem::directory_iterator(pathCat)){
            if (iterSize >= size) break; 
            std::pair<std::vector<std::vector<std::vector<int>>>, int> catData;
            catData = std::make_pair(loadImage(entry.path().string()), 0);

            dataset.push_back(catData);
            std::cout << "amount dataset load: " << dataset.size() << std::endl;
            iterSize++;

        }
            //std::cout << entry.path() << std::endl;
        for (const auto & entry : std::filesystem::directory_iterator(pathDog)){
            if (iterSize >= size) break;  
                std::pair<std::vector<std::vector<std::vector<int>>>, int> dogData;
                dogData = std::make_pair(loadImage(entry.path().string()), 1);
                dataset.push_back(dogData);
                std::cout << "amount dataset load: " << dataset.size() << std::endl;
            iterSize++;

        }
        std::cout << "amount dataset: " << dataset.size() << std::endl;
    }

    void train(){
        int sizeDatasetLearning = 10;
        std::vector<double> inputNN;
        std::vector<double> HiddenFirst;
        std::vector<double> HiddenTwo;
        std::vector<double> OutputProb;
      

        for (int i=0; i<amountEpoch; i++) {
            for (int b=0; b<sizeDatasetLearning; b++){
                std::vector<std::vector<int>> convImageRed = convolve2DSlow(dataset[b].first, 0, kernelsW1Red[0]);
                std::vector<std::vector<int>> convImageGreen = convolve2DSlow(dataset[b].first, 1, kernelsW1Green[0]);
                std::vector<std::vector<int>> convImageBlue = convolve2DSlow(dataset[b].first, 2, kernelsW1Blue[0]);
                
                SizeXfirstMaxPool = convImageRed.size();
                SizeYfirstMaxPool = convImageRed[0].size(); 

                MaxPoolingData MaxPoolRed = MaxPool(convImageRed);
                MaxPoolingData MaxPoolGreen = MaxPool(convImageGreen); 
                MaxPoolingData MaxPoolBlue = MaxPool(convImageBlue);
                
                std::vector<std::vector<int>> convImageRedTwo = convolve2DSlow(MaxPoolRed.output, kernelsW2Red[0]);
                std::vector<std::vector<int>> convImageGreenTwo = convolve2DSlow(MaxPoolGreen.output, kernelsW2Green[0]);
                std::vector<std::vector<int>> convImageBlueTwo = convolve2DSlow(MaxPoolGreen.output, kernelsW2Blue[0]); 
                
                SizeXtwoMaxPool = convImageRedTwo.size();
                SizeYtwoMaxPool = convImageRedTwo[0].size(); 
                MaxPoolingData MaxPoolRedTwo = MaxPool(convImageRedTwo);
                MaxPoolingData MaxPoolGreenTwo = MaxPool(convImageGreenTwo); 
                MaxPoolingData MaxPoolBlueTwo = MaxPool(convImageBlueTwo);
       
                for (int i=0; i<MaxPoolRedTwo.output.size(); i++) {
                    for (int j=0; j<MaxPoolRedTwo.output[0].size(); j++) {
                        inputNN.push_back(MaxPoolRedTwo.output[i][j]);
                    } 
                } 
                for (int i=0; i<MaxPoolGreenTwo.output.size(); i++) {
                    for (int j=0; j<MaxPoolGreenTwo.output[0].size(); j++) {
                        inputNN.push_back(MaxPoolGreenTwo.output[i][j]);
                    } 
                }
                for (int i=0; i<MaxPoolBlueTwo.output.size(); i++) {
                    for (int j=0; j<MaxPoolBlueTwo.output[0].size(); j++) {
                        inputNN.push_back(MaxPoolBlueTwo.output[i][j]);
                    } 
                } 
        //std::cout << "inputDesnse.size(): " << inputDesnse.size() << std::endl;
        int minInputDense = FindMin(inputNN);
        int maxInputDense = FindMax(inputNN);

          
        for (int j=0; j<inputNN.size(); j++) {
            //std::cout << "num: " << j << " inputDesnse: " << NormalizeImage(inputDesnse[j], 1.0d, (double)minInputDense, (double)maxInputDense) << std::endl;
            inputNN[j] = NormalizeImage(inputNN[j], 1.0d, (double)minInputDense, (double)maxInputDense); 
        }

                HiddenFirst = dotNN(inputNN, weightsInput, biasHiddenOne);
                /* 
                for (int j=0;j < HiddenFirst.size();j++){
                    std::cout << "HiddenFirst: " << HiddenFirst[j] << std::endl;
                }*/ 
                HiddenTwo = dotNN(HiddenFirst, weightsHiddenOne, biasHiddenTwo);
                /*for (int j=0;j < HiddenTwo.size();j++){
                    std::cout << "HiddenTwo: " << HiddenTwo[j] << std::endl;
                } */
                OutputProb = dotNNSoftmax(HiddenTwo, weightsHiddenTwo);
                /*for (int j=0;j < OutputProb.size();j++){
                    std::cout << "OutputProb: " << OutputProb[j] << std::endl;
                }*/
                //std::vector<std::vector<double>> GradWeightsHiddenTwo; 
                std::vector<double> grad_temp_output;
                 
                std::cout << "OutputProb: " << OutputProb.size() << std::endl;

                std::cout << "------------------------------------------------" << std::endl;
                //std::cout << "Num train: " << b << std::endl;
                std::cout << "Epoch: " << i << std::endl;
                std::cout << "Loss: " << MSEloss(OutputProb, dataset[b].second) << std::endl;
                //std::cout << "softmaxDerivative(OutputProb).size(): " << softmaxDerivative(OutputProb).size() << std::endl;
                //std::cout << "softmaxDerivative(OutputProb)[0].size(): " << softmaxDerivative(OutputProb)[0].size() << std::endl;

                /*for (int j=0; j<MSElossDerivative(OutputProb, dataset[b].second).size(); j++) {
                    if(isnan(MSElossDerivative(OutputProb, dataset[b].second)[j])) break;
                    //std::cout << "Loss derivative: " << MSElossDerivative(OutputProb, dataset[b].second)[j] << std::endl; 

                }*/
                //std::cout << "Matrix.size(): " << multiplyMatrix(MSElossDerivative(OutputProb, dataset[b].second), softmaxDerivative(OutputProb)).size() << std::endl;
                //std::cout << "Matrix[0].size(): " <<multiplyMatrix(MSElossDerivative(OutputProb, dataset[b].second), softmaxDerivative(OutputProb))[0].size() << std::endl;
                for (int i=0; i<weightsHiddenTwo.size(); i++) {
                    for (int j=0; j<weightsHiddenTwo[i].size(); j++) {
                        //std::cout << "weightsHiddenTwo[i][j] before: " << weightsHiddenTwo[i][j] << std::endl; 
                        weightsHiddenTwo[i][j] -= HiddenTwo[i] * multiplyMatrix(MSElossDerivative(OutputProb, dataset[b].second), softmaxDerivative(OutputProb))[j][0];
                        //std::cout << "weightsHiddenTwo[i][j] after: " << weightsHiddenTwo[i][j] << std::endl; 
                    }
                }
                std::cout << "Matrix: " << multiplyMatrix(multiplyMatrix(MSElossDerivative(OutputProb, dataset[b].second), softmaxDerivative(OutputProb)), weightsHiddenTwo)[0].size() << std::endl;
                //std::cout << "Matrix: " << multiplyMatrix(multiplyMatrix(MSElossDerivative(OutputProb, dataset[b].second), softmaxDerivative(OutputProb)), weightsHiddenTwo)[0][0] << std::endl;
                std::cout << "weightsHiddenOne.size(): " << weightsHiddenOne.size() << std::endl;
                std::cout << "weightsHiddenOne[0].size(): " << weightsHiddenOne[0].size() << std::endl;
                
                for (int i=0; i<weightsHiddenOne.size(); i++) {
                    for (int j=0; j<weightsHiddenOne[i].size(); j++) {
                        weightsHiddenOne[i][j] -= multiplyMatrix(multiplyMatrix(MSElossDerivative(OutputProb, dataset[b].second), softmaxDerivative(OutputProb)), weightsHiddenTwo)[0][i] * sigmoid(HiddenTwo[i]) * HiddenFirst[j]; 
                    }
                }
                std::cout << "weights: " << multiplyMatrix(multiplyMatrix(multiplyMatrix(MSElossDerivative(OutputProb, dataset[b].second), softmaxDerivative(OutputProb)), weightsHiddenTwo)[0], weightsHiddenOne).size() << std::endl;
                /*
                for (int i=0; i<weightsInput.size(); i++) {
                    for (int j=0; j<weightsInput[i].size(); j++) {
                        weightsInput[i][j] -= 
                    }
                } */
                //std::cout << "weights.size(): " << weightsHiddenTwo.size() << std::endl;
                //std::cout << "weights[0].size(): " << weightsHiddenTwo[0].size() << std::endl;
                //std::cout << "------------------------------------------------" << std::endl;
                /*for (int i=0; i<softmaxDerivative(OutputProb).size(); i++) {
                    for (int k=0; k<softmaxDerivative(OutputProb)[0].size(); k++) {
                        if(isnan(softmaxDerivative(OutputProb)[i][k])) break;
                        //std::cout << "softmaxDerivative: " << softmaxDerivative(OutputProb)[i][k] << std::endl;
                    }
                } */
            }
        } 
    }
};

int main(){
   //srand(time(0));
   ConvolutionalNeuralNetwork cnn;
   cnn.loadDataset();
   cnn.train();
}
