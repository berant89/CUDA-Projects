#ifndef HW1_H__
#define HW1_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

extern cv::Mat imageRGBA;
extern cv::Mat imageGrey;

extern uchar4        *d_rgbaImage__;
extern unsigned char *d_greyImage__;

size_t numRows();
size_t numCols();

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename);
void postProcess(const std::string& output_file, unsigned char* data_ptr);
void cleanup();
void generateReferenceImage(std::string input_filename, std::string output_filename);

#endif