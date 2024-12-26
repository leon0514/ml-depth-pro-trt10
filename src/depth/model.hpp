#ifndef MODEL_HPP__
#define MODEL_HPP__
#include <vector>
#include <memory>
#include <iostream>
#include "image.hpp"

namespace depth
{

struct DeepthMap
{
    int width = 0, height = 0;
    float fov_deg = 0.f;
    std::vector<float> depth_map;
    
    DeepthMap(int width, int height)
        : width(width), height(height), depth_map(width * height, 0.0f)
    { }


    DeepthMap(DeepthMap&& depth)
        : width(depth.width), height(depth.height), depth_map(std::move(depth.depth_map))
    { }


    bool valid() { return width != 0 && height != 0; }
};




class Infer {
public:
    virtual DeepthMap forward(const TensorRT::Image &image, void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, int gpu_id=0);

}



#endif