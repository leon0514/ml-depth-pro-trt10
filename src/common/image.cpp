#include "image.hpp"

namespace TensorRT
{
    TensorRT::Image cvimg(const cv::Mat &image) { return TensorRT::Image(image.data, image.cols, image.rows); }
}
