#include <iostream>
#include <cnn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include "functions.cpp"
#include <bits/stdc++.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebufalgo.h>

//using namespace OIIO;
class ConvolutionalNeuralNetwork {       
    private:
        std::string filename;   
        std::vector<std::pair<std::vector<std::vector<std::vector<unsigned char>>>, int>> dataset; 
        
        int amountEpoch;

        int sizeInputX; // for backpropogation
        int sizeInputY; // for backpropogation
        std::vector<std::vector<int>> convBeforePooling;

        std::vector<std::vector<std::vector<float>>> kernelsW1Red;
        std::vector<std::vector<std::vector<float>>> kernelsW1Green;
        std::vector<std::vector<std::vector<float>>> kernelsW1Blue;
        
        int numKernelsW1;
        int sizeKernelW1;

        std::vector<std::vector<std::vector<float>>> kernelsW2Red;
        std::vector<std::vector<std::vector<float>>> kernelsW2Green;
        std::vector<std::vector<std::vector<float>>> kernelsW2Blue;
        
        int numKernelsW2;
        int sizeKernelW2;
        std::vector<std::vector<std::vector<float>>> kernelsW3;
        std::vector<float> inputDesnse;
        int DenseLayerSize;
        int weightsHiddenOneSize;
        int weightsHiddenTwoSize;
        int weightsOutputSize;

        std::vector<std::vector<float>> weightsInput;
        std::vector<std::vector<float>> weightsHiddenOne;
        std::vector<float> biasHiddenOne;
        std::vector<std::vector<float>> weightsHiddenTwo;
        std::vector<float> biasHiddenTwo;
        std::vector<std::vector<float>> weightsOutput;

        std::vector<std::vector<float>> GradWeightsInput;
        std::vector<std::vector<float>> GradWeightsHiddenOne;
        std::vector<float> GradBiasHiddenOne;
        std::vector<std::vector<float>> GradWeightsHiddenTwo;
        std::vector<float> GradBiasHiddenTwo;
        std::vector<std::vector<float>> GradWeightsOutput;

        std::vector<std::pair<int, int>> firstMaxPoolBackpropIndex;
        int SizeXfirstMaxPool;
        int SizeYfirstMaxPool;
        std::vector<std::pair<int, int>> TwoMaxPoolBackpropIndex;
        int SizeXtwoMaxPool;
        int SizeYtwoMaxPool;
    public:
    ConvolutionalNeuralNetwork() {
        amountEpoch = 3;
        weightsHiddenOneSize = 300;
        biasHiddenOne.resize(weightsHiddenOneSize);
        weightsHiddenTwoSize = 30;
        biasHiddenTwo.resize(weightsHiddenTwoSize);

        weightsOutputSize = 2;

        numKernelsW1 = 1;
        sizeKernelW1 = 4;
        
        numKernelsW2 = 1;
        sizeKernelW2 = 4;
    
        for (int i=0; i<numKernelsW1; i++) {
            std::vector<std::vector<float>> kernRed; 
            std::vector<std::vector<float>> kernGreen;
            std::vector<std::vector<float>> kernBlue;
            for (int j=0; j<sizeKernelW1; j++) {
                std::vector<float> rowRed;
                std::vector<float> rowGreen;
                std::vector<float> rowBlue; 
                for (int k=0; k<sizeKernelW1; k++) {
                    
                    float red = (float)((double)rand()) / RAND_MAX;
                    float green = (float)((double)rand()) / RAND_MAX;
                    float blue = (float)((double)rand()) / RAND_MAX;
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
            std::vector<std::vector<float>> kernRed; 
            std::vector<std::vector<float>> kernGreen;
            std::vector<std::vector<float>> kernBlue;
            for (int j=0; j<sizeKernelW2; j++) {
                std::vector<float> rowRed;
                std::vector<float> rowGreen;
                std::vector<float> rowBlue; 
                for (int k=0; k<sizeKernelW2; k++) {
                    float red = (float)((double)rand()) / RAND_MAX;
                    float green = (float)((double)rand()) / RAND_MAX;
                    float blue = (float)((double)rand()) / RAND_MAX;
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
        std::vector<std::vector<int>> MaxPoolRed = MaxPool(convImageRed);
        std::vector<std::vector<int>> MaxPoolGreen = MaxPool(convImageGreen); 
        std::vector<std::vector<int>> MaxPoolBlue = MaxPool(convImageBlue);

        std::cout << "MaxPool size: " << MaxPoolRed.size() << " " << MaxPoolRed[0].size() << std::endl; 
        std::cout << "max pool 4x4" << std::endl; 

        std::vector<std::vector<int>> convImageRedTwo = convolve2DSlow(MaxPoolRed, kernelsW2Red[0]);
        std::vector<std::vector<int>> convImageGreenTwo = convolve2DSlow(MaxPoolGreen, kernelsW2Green[0]);
        std::vector<std::vector<int>> convImageBlueTwo = convolve2DSlow(MaxPoolGreen, kernelsW2Blue[0]); 
        std::cout << "two conv size: " << convImageRedTwo.size() << " " << convImageRedTwo[0].size() << std::endl; 
        std::cout << "max pool 4x4" << std::endl; 

        SizeXtwoMaxPool = convImageRedTwo.size();
        SizeYtwoMaxPool = convImageRedTwo[0].size(); 
        std::vector<std::vector<int>> MaxPoolRedTwo = MaxPool(convImageRedTwo);
        std::vector<std::vector<int>> MaxPoolGreenTwo = MaxPool(convImageGreenTwo); 
        std::vector<std::vector<int>> MaxPoolBlueTwo = MaxPool(convImageBlueTwo);
        
        std::cout << "MaxPool size: " << MaxPoolRedTwo.size() << " " << MaxPoolRedTwo[0].size() << std::endl; 
        std::cout << "dense" << std::endl; 

        const char* filename = "cat pool.1.png";
        const int xres = MaxPoolRedTwo.size(), yres = MaxPoolRedTwo[0].size(), channels = 3;
        std::cout << "max pool size x: " <<  MaxPoolRedTwo.size() << std::endl;
        std::cout << "max pool size y: " <<  MaxPoolRedTwo[0].size() << std::endl;
        DenseLayerSize = (MaxPoolRedTwo.size() * MaxPoolRedTwo[0].size()) + (MaxPoolGreenTwo.size() * MaxPoolGreenTwo[0].size()) + (MaxPoolBlueTwo.size() * MaxPoolBlueTwo[0].size()); 
        std::cout << "total dense layer: " << DenseLayerSize << std::endl; 
        
        //    
        // Normalization dense layer
        //
        for (int i=0; i<MaxPoolRedTwo.size(); i++) {
            for (int j=0; j<MaxPoolRedTwo[0].size(); j++) {
                inputDesnse.push_back(MaxPoolRedTwo[i][j]);
            } 
        } 
        for (int i=0; i<MaxPoolGreenTwo.size(); i++) {
            for (int j=0; j<MaxPoolGreenTwo[0].size(); j++) {
                inputDesnse.push_back(MaxPoolGreenTwo[i][j]);
            } 
        }
        for (int i=0; i<MaxPoolBlueTwo.size(); i++) {
            for (int j=0; j<MaxPoolBlueTwo[0].size(); j++) {
                inputDesnse.push_back(MaxPoolBlueTwo[i][j]);
            } 
        } 
        std::cout << "inputDesnse.size(): " << inputDesnse.size() << std::endl;
        int minInputDense = FindMin(inputDesnse);
        int maxInputDense = FindMax(inputDesnse);

          
        for (int j=0; j<inputDesnse.size(); j++) {
            //std::cout << "num: " << j << " inputDesnse: " << NormalizeImage(inputDesnse[j], 1.0f, (float)minInputDense, (float)maxInputDense) << std::endl;
            inputDesnse[j] = NormalizeImage(inputDesnse[j], 1.0f, (float)minInputDense, (float)maxInputDense); 
        }

        // init input-hidden layer
        weightsInput.resize(inputDesnse.size());
        for (int i=0; i<inputDesnse.size(); i++) {
            weightsInput[i].resize(weightsHiddenOneSize); 
            for (int j=0; j<weightsHiddenOneSize; j++) {
                weightsInput[i][j] = (float)((double)rand()) / RAND_MAX; 
                //std::cout << "row: " << i << " columm: " << j << " weightsInput[i][j]: " << weightsInput[i][j] << std::endl;
            } 
        } 
        
        // init hidden1-hidden2 layer
        weightsHiddenOne.resize(weightsHiddenOneSize);
        for (int i=0; i<weightsHiddenOneSize; i++) {
            weightsHiddenOne[i].resize(weightsHiddenTwoSize); 
            for (int j=0; j<weightsHiddenTwoSize; j++) {
                weightsHiddenOne[i][j] = (float)((double)rand()) / RAND_MAX;
                //
            } 
        } 
        
        // init hidden2-output layer
        weightsHiddenTwo.resize(weightsHiddenTwoSize);
        for (int i=0; i<weightsHiddenTwoSize; i++) {
            
            weightsHiddenTwo[i].resize(weightsOutputSize); 
            for (int j=0; j<weightsHiddenTwo[i].size(); j++) {
                weightsHiddenTwo[i][j] = (float)((double)rand()) / RAND_MAX;
                //std::cout << "columm: " << j << std::endl;
                //std::cout << "row: " << i << " columm: " << j << " weightsHiddenTwo[i][j]: " << weightsHiddenTwo[i][j] << std::endl;
            }
        } 

        /*for (int i=0; i<biasInput.size(); i++) { 
            biasInput[i] = (float)((double)rand()) / RAND_MAX;
            //std::cout << "biasHiddenOne[i]: " << biasHiddenOne[i] << std::endl;
        } */        

        for (int i=0; i<biasHiddenOne.size(); i++) { 
            biasHiddenOne[i] = (float)((double)rand()) / RAND_MAX;
            //std::cout << "biasHiddenOne[i]: " << biasHiddenOne[i] << std::endl;
        }         

        for (int i=0; i<biasHiddenTwo.size(); i++) { 
            biasHiddenTwo[i] = (float)((double)rand()) / RAND_MAX;
            //std::cout << "biasHiddenTwo[i]: " << biasHiddenTwo[i] << std::endl;
        }   
        
        std::vector<float> HiddenFirst = dotNN(inputDesnse, weightsInput, biasHiddenOne);
        std::vector<float> HiddenTwo = dotNN(HiddenFirst, weightsHiddenOne, biasHiddenTwo);
        std::vector<float> OutputProb = dotNNSoftmax(HiddenTwo, weightsHiddenTwo);
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
        unsigned char pixels[xres * yres * channels];
        int totalPixel = 0;
        for (int i=0; i<xres; i++) {
            for (int j=0; j<yres; j++) {
                
                totalPixel++;
                pixels[totalPixel*channels] = MaxPoolRedTwo[i][j];
                pixels[totalPixel*channels + 1] = MaxPoolGreenTwo[i][j];
                pixels[totalPixel*channels + 2] = MaxPoolBlueTwo[i][j]; 
            }
        }

        std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(filename);
        OIIO::ImageSpec spec(xres, yres, channels, OIIO::TypeDesc::UINT8);
        out->open(filename, spec);
        out->write_image(OIIO::TypeDesc::UINT8, pixels);
        out->close(); 
    }
    std::vector<std::vector<std::vector<unsigned char>>> loadImage(std::string filepath){
        filename = filepath;
        auto inp = OIIO::ImageInput::open(filename);


        const OIIO::ImageSpec &spec = inp->spec();
        int xres = spec.width;
        int yres = spec.height;

        int nchannels = spec.nchannels;

        auto pixels = std::unique_ptr<unsigned char[]>(new unsigned char[xres * yres * nchannels]);
        inp->read_image(0, 0, 0, nchannels, OIIO::TypeDesc::UINT8, &pixels[0]);
        inp->close();

        std::vector<std::vector<std::vector<unsigned char>>> Image;

        std::vector<unsigned char> Rarray(xres*yres);
        std::vector<unsigned char> Garray(xres*yres);
        std::vector<unsigned char> Barray(xres*yres);
        
        for (int i=0; i<xres*yres; i++) { 
            Rarray[i] = pixels[i*nchannels];
            Garray[i] = pixels[i*nchannels + 1];
            Barray[i] = pixels[i*nchannels + 2];

        }

        std::vector<std::vector<unsigned char>> Rprocessed(xres, std::vector<unsigned char>(yres, 0));
        std::vector<std::vector<unsigned char>> Gprocessed(xres, std::vector<unsigned char>(yres, 0));
        std::vector<std::vector<unsigned char>> Bprocessed(xres, std::vector<unsigned char>(yres, 0));
        
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

        for (const auto & entry : std::filesystem::directory_iterator(pathCat)){
            std::pair<std::vector<std::vector<std::vector<unsigned char>>>, int> catData;
            catData = std::make_pair(loadImage(entry.path().string()), 0);

            dataset.push_back(catData);
            std::cout << "amount dataset load: " << dataset.size() << std::endl;

        }
            //std::cout << entry.path() << std::endl;
        for (const auto & entry : std::filesystem::directory_iterator(pathDog)){
            std::pair<std::vector<std::vector<std::vector<unsigned char>>>, int> dogData;
            dogData = std::make_pair(loadImage(entry.path().string()), 1);
            dataset.push_back(dogData);
            std::cout << "amount dataset load: " << dataset.size() << std::endl;

        }
        std::cout << "amount dataset: " << dataset.size() << std::endl;
    }

    void train(){
        std::vector<float> HiddenFirst;
        std::vector<float> HiddenTwo;
        std::vector<float> OutputProb;
        for (int i=0; i<amountEpoch; i++) { 
            HiddenFirst = dotNN(inputDesnse, weightsInput, biasHiddenOne);
            HiddenTwo = dotNN(HiddenFirst, weightsHiddenOne, biasHiddenTwo);
            OutputProb = dotNNSoftmax(HiddenTwo, weightsHiddenTwo);
            
            for (int j=0;j < weightsHiddenTwo.size();j++){
                for (int k=0;k < weightsHiddenTwo[j].size();k++){

                }
            }

            HiddenFirst.clear();
            HiddenTwo.clear();
            OutputProb.clear();
        }    
    }
    std::vector<std::vector<int>> convolve2D(std::vector<std::vector<std::vector<unsigned char>>> image, int channel, std::vector<std::vector<float>> kernelConv) {
        //For normalization
        float minNumber = INT_MAX;
        float maxNumber = INT_MIN;
        int redSize = image[0].size();
        int greenSize = image[1].size();
        int blueSize = image[2].size();

        int padding = 0;
        int stride = 1;

        int sizeW = image[0].size(); 
        int sizeH = image[0][0].size();
        std::cout << "sizeW: " << sizeW << std::endl;
        std::cout << "sizeH: " << sizeH << std::endl;
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
    std::vector<std::vector<int>> convolve2DSlow(std::vector<std::vector<std::vector<unsigned char>>> image, int channel, std::vector<std::vector<float>> kernelConv) {
        //For normalization
        float minNumber = INT_MAX;
        float maxNumber = INT_MIN;
        int redSize = image[0].size();
        int greenSize = image[1].size();
        int blueSize = image[2].size();

        int padding = 0;
        int stride = 1;

        int sizeW = image[0].size(); 
        int sizeH = image[0][0].size();
        std::cout << "sizeW: " << sizeW << std::endl;
        std::cout << "sizeH: " << sizeH << std::endl;
        std::cout << "kernelConv[0].size(): " << kernelConv[0].size() << std::endl;
        std::cout << "kernelConv.size(): " << kernelConv.size() << std::endl;
        int convW = ((sizeW - kernelConv[0].size() + 2 * padding) / stride) + 1;
        int convH = ((sizeH - kernelConv.size() + 2 * padding) / stride) + 1;

        std::vector<std::vector<int>> output(convW, std::vector<int>(convH, 0));
        std::cout << "convW: " << convW << std::endl;
        std::cout << "convH: " << convH << std::endl;    
        
        
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
    std::vector<std::vector<int>> convolve2DSlow(std::vector<std::vector<int>> image, std::vector<std::vector<float>> kernelConv) {
        //For normalization
        float minNumber = INT_MAX;
        float maxNumber = INT_MIN;
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
    /*std::vector<std::vector<int>> AveragePool(std::vector<std::vector<int>> image, std::vector<std::vector<float>> kernelConvPerv){
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
    std::vector<std::vector<int>> MaxPool(std::vector<std::vector<int>> image){
        int padding = 0;
        int stride = 2;
        int filter = 2;
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

                    //std::vector<int> indexs = FindMaxIndex(pool);
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
        return output;
    }
    std::vector<std::vector<int>> MaxPoolBackprop(std::vector<std::vector<int>> image){

        int sizeX = image.size(); 
        int sizeY = image[0].size();
        
        std::vector<std::vector<int>> output(sizeInputX, std::vector<int>(sizeInputY, 0));
        //std::cout << "sizePoolX: " << sizePoolX << std::endl;
        //std::cout << "sizePoolY: " << sizePoolY << std::endl;
        for (int i = 0; i < sizeInputX; ++i)              // rows
        {
            for (int j = 0; j < sizeInputY; ++j)          // columns
            {
 
            }
        }
        return output;
    }
};

int main() {
   srand(time(0));
   ConvolutionalNeuralNetwork cnn;
   //cnn.loadDataset();
}
