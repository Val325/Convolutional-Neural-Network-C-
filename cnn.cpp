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
    
    int width, height; 
    int numberOfChannels;
    std::string filename;
    //uint8_t *imageData;
    
    public:
    ConvolutionalNeuralNetwork() {
        //"dataset/test_set/test_set/cats/cat.4001.jpg"
        
        //getAllFiles("dataset/test_set/test_set/cats");
        
        std::vector<float> input = {0.4, 0.7, 0.1};
        std::vector<float> output;
        output = sigmoid(input);
        for (int i=0; i < output.size(); i++){
            std::cout << "input sigmoid " << input[i] << " output sigmoid: " << output[i] << std::endl; 
        }
        std::vector<float> derivSig;
        derivSig = derivativeSigmoid(output);
        for (int i=0; i < derivSig.size(); i++){
            std::cout << "derivative sigmoid: " << derivSig[i] << std::endl; 
        }
        std::vector<float> softmaxData;
        softmaxData = softmax(input); 
        for (int i=0; i < input.size(); i++){
            std::cout << "Softmax: " << softmaxData[i] << std::endl; 
        }
        std::vector<std::vector<float>> softmaxDeriv;
        softmaxDeriv = softmaxDerivative(softmaxData);
        for(unsigned i = 0; i != softmaxDeriv[0].size(); ++i ) {
            for(unsigned j = 0; j != softmaxDeriv[1].size(); ++j ) {
                std::cout << "Softmax jacobean: " << softmaxDeriv[i][j] << std::endl; 
            } 
        }
        std::vector<std::vector<int>> redConv = ConvolutionRed(loadImage("dataset/test_set/test_set/cats/cat.4001.jpg"));
        //std::vector<std::vector<int>> greenConv = ConvolutionGreen(loadImage("dataset/test_set/test_set/cats/cat.4001.jpg"));
        //std::vector<std::vector<int>> blueConv = ConvolutionBlue(loadImage("dataset/test_set/test_set/cats/cat.4001.jpg"));

        //std::vector<std::vector<unsigned char>> img;
        //img = loadImage("dataset/test_set/test_set/cats/cat.4001.jpg");
        //std::cout << "Red: " << img[0].size() << " Green: " << img[1].size() << " Blue: " << img[2].size() << std::endl; 
        /*
        for (int r=0; r<img[0].size(); r++){
            for (int g=0; g<img[1].size(); g++){
                for(int b=0; b<img[2].size(); b++){
                    std::cout << "Red: " << img[0].size() << " Green: " << img[1].size() << " Blue: " << img[2].size() << std::endl; 
                }
            }
        }*/
    }
    std::vector<std::vector<std::vector<unsigned char>>> loadImage(std::string filepath){
        filename = filepath;
        auto inp = OIIO::ImageInput::open(filename);

        //if (! inp)
        //    return;
        const OIIO::ImageSpec &spec = inp->spec();
        int xres = spec.width;
        int yres = spec.height;
        std::cout << "xres: " << xres << std::endl;
        std::cout << "yrex: " << yres << std::endl;
        int nchannels = spec.nchannels;
        /*
        OIIO::ImageBuf image(filename);
        OIIO::ROI roi (0, 200, 0, 150, 0, 1, 0, nchannels);
        image = OIIO::ImageBufAlgo::resize(image, {}, roi);
        image.write("smallimage.jpg");
        */
        auto pixels = std::unique_ptr<unsigned char[]>(new unsigned char[xres * yres * nchannels]);
        inp->read_image(0, 0, 0, nchannels, OIIO::TypeDesc::UINT8, &pixels[0]);
        inp->close();
        
        //std::vector<std::vector<unsigned char>> Image;
        std::vector<std::vector<std::vector<unsigned char>>> Image;

        std::vector<unsigned char> Rarray(xres*yres);
        std::vector<unsigned char> Garray(xres*yres);
        std::vector<unsigned char> Barray(xres*yres);
        
        for (int i=0; i<xres*yres; i++) { 
            Rarray[i] = pixels[i*nchannels];
            //std::cout << "Red: " << (float)pixels[i*nchannels] << std::endl;
            Garray[i] = pixels[i*nchannels + 1];
            //std::cout << "Green: " << (float)pixels[i*nchannels + 1] << std::endl;
            Barray[i] = pixels[i*nchannels + 2];
            //std::cout << "Blue: " << (float)pixels[i*nchannels + 2] << std::endl;
            //here add arrays RGB
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
        std::cout << "Red: " << Rprocessed.size() << std::endl;
        std::cout << "Green: " << Gprocessed.size() << std::endl;
        std::cout << "Blue: " << Bprocessed.size() << std::endl;

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
};

int main() {
   ConvolutionalNeuralNetwork cnn;
}
