#include <iostream>
#include <cnn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include "functions.cpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebufalgo.h>

//using namespace OIIO;
class ConvolutionalNeuralNetwork {       
    private:
        std::string filename;    
        int sizeInputX; // for backpropogation
        int sizeInputY; // for backpropogation
        std::vector<std::vector<int>> convBeforePooling;
        std::vector<std::vector<std::vector<float>>> kernelsW1;
        std::vector<std::vector<std::vector<float>>> kernelsW2;
        std::vector<std::vector<std::vector<float>>> kernelsW3;
        int DenseLayerSize;
    public:
    ConvolutionalNeuralNetwork() {

        kernelsW1 = {
        { //Red
            {0, -1, 0, 0.1, 0.1}, 
            {-1, 4, 0.1, 0.1, 0.1}, 
            {0, -1, 0, 0.1, 0.1},
            {0, -1, 0, 0.1, 0.1},
            {-1, 4, 0.1, 0.1, 0.1}, 
        },
        { //Green
            {0, -1, 0, 0.1, 0.1}, 
            {-1, 4, 0.1, 0.1, 0.1}, 
            {0, -1, 0, 0.1, 0.1},
            {0, -1, 0, 0.1, 0.1},
            {-1, 4, 0.1, 0.1, 0.1}, 
        },
        { //Blue
            {0, -1, 0, 0.1, 0.1}, 
            {-1, 4, 0.1, 0.1, 0.1}, 
            {0, -1, 0, 0.1, 0.1},
            {0, -1, 0, 0.1, 0.1},
            {-1, 4, 0.1, 0.1, 0.1}, 
        },};
        
        kernelsW2 = {
        { //Red
            {0, -1, 0, 0.1, 0.1}, 
            {-1, 4, 0.1, 0.1, 0.1}, 
            {0, -1, 0, 0.1, 0.1},
            {0, -1, 0, 0.1, 0.1},
            {-1, 4, 0.1, 0.1, 0.1}, 
        },
        { //Green
            {0, -1, 0, 0.1, 0.1}, 
            {-1, 4, 0.1, 0.1, 0.1}, 
            {0, -1, 0, 0.1, 0.1},
            {0, -1, 0, 0.1, 0.1},
            {-1, 4, 0.1, 0.1, 0.1}, 
        },
        { //Blue
            {0, -1, 0, 0.1, 0.1}, 
            {-1, 4, 0.1, 0.1, 0.1}, 
            {0, -1, 0, 0.1, 0.1},
            {0, -1, 0, 0.1, 0.1},
            {-1, 4, 0.1, 0.1, 0.1}, 
        },};

        kernelsW3 = {
        { //Red
            {1.0, 1.0, 1.0, 1.0}, 
            {1.0, 1.0, 1.0, 1.0}, 
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},
        },
        { //Green
            {1.0, 1.0, 1.0, 1.0}, 
            {1.0, 1.0, 1.0, 1.0}, 
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},            
        },
        { //Blue
            {1.0, 1.0, 1.0, 1.0}, 
            {1.0, 1.0, 1.0, 1.0}, 
            {1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0},  
        },};

        std::vector<std::vector<int>> convImageRed = convolve2DSlow(loadImage("datasetpreprocessing/test/cats/cat.1.png"), 0, kernelsW1[0]);
        std::vector<std::vector<int>> convImageGreen = convolve2DSlow(loadImage("datasetpreprocessing/test/cats/cat.1.png"), 1, kernelsW1[1]);
        std::vector<std::vector<int>> convImageBlue = convolve2DSlow(loadImage("datasetpreprocessing/test/cats/cat.1.png"), 2, kernelsW1[2]);
        
        std::vector<std::vector<int>> MaxPoolRed = MaxPool(convImageRed);
        std::vector<std::vector<int>> MaxPoolGreen = MaxPool(convImageGreen); 
        std::vector<std::vector<int>> MaxPoolBlue = MaxPool(convImageBlue);

        std::vector<std::vector<int>> convImageRedTwo = convolve2DSlow(MaxPoolRed, kernelsW2[0]);
        std::vector<std::vector<int>> convImageGreenTwo = convolve2DSlow(MaxPoolGreen, kernelsW2[1]);
        std::vector<std::vector<int>> convImageBlueTwo = convolve2DSlow(MaxPoolGreen, kernelsW2[2]);

        std::vector<std::vector<int>> MaxPoolRedTwo = MaxPool(convImageRedTwo);
        std::vector<std::vector<int>> MaxPoolGreenTwo = MaxPool(convImageGreenTwo); 
        std::vector<std::vector<int>> MaxPoolBlueTwo = MaxPool(convImageBlueTwo);
        

        //std::vector<std::vector<int>> convImageRedThree = convolve2DSlow(MaxPoolRedTwo, kernelsW3[0]);
        //std::vector<std::vector<int>> convImageGreenThree = convolve2DSlow(MaxPoolGreenTwo, kernelsW3[1]);
        //std::vector<std::vector<int>> convImageBlueThree = convolve2DSlow(MaxPoolBlueTwo, kernelsW3[2]);        

        //std::vector<std::vector<int>> MaxPoolRedThree = MaxPool(convImageRedThree);
        //std::vector<std::vector<int>> MaxPoolGreenThree = MaxPool(convImageGreenThree); 
        //std::vector<std::vector<int>> MaxPoolBlueThree = MaxPool(convImageBlueThree);


        const char* filename = "cat pool.1.png";
        const int xres = MaxPoolRedTwo.size(), yres = MaxPoolRedTwo[0].size(), channels = 3;
        std::cout << "max pool size x: " <<  MaxPoolRedTwo.size() << std::endl;
        std::cout << "max pool size y: " <<  MaxPoolRedTwo[0].size() << std::endl;
        DenseLayerSize = (MaxPoolRedTwo.size() * MaxPoolRedTwo[0].size()) + (MaxPoolGreenTwo.size() * MaxPoolGreenTwo[0].size()) + (MaxPoolBlueTwo.size() * MaxPoolBlueTwo[0].size()); 
        std::cout << "total dense layer: " << DenseLayerSize << std::endl; 

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

    void getAllFiles(std::string pathfiles){
        std::string path = pathfiles;
        for (const auto & entry : std::filesystem::directory_iterator(path))
            std::cout << entry.path() << std::endl;
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
        //std::cout << "sizeW: " << sizeW << std::endl;
        //std::cout << "sizeH: " << sizeH << std::endl;
        int convW = ((sizeW - kernelConv[0].size() + 2 * padding) / stride) + 1;
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
    std::vector<std::vector<int>> AveragePool(std::vector<std::vector<int>> image){
        int padding = 0;
        int stride = 2;
        int filter = 2;
        int sizeX = image.size(); 
        int sizeY = image[0].size();
        int sizePoolX = ((sizeX - kernelsW1[0][0].size() + 2 * padding) / stride) + 1;
        int sizePoolY = ((sizeY - kernelsW1[0][0].size() + 2 * padding) / stride) + 1;
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
    }
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
   ConvolutionalNeuralNetwork cnn;
}
