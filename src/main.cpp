#include "model.hpp"
#include "opencv2/opencv.hpp"
#include "timer.hpp"
#include "image.hpp"
#include <cmath>

float calculate_f_px(int w, float fov_deg) 
{
    float fov_rad = fov_deg * M_PI / 180.0;
    float model_f_px = 0.5 * w / tan(0.5 * fov_rad);
    return model_f_px;
}

cv::Mat compute_depth(cv::Mat depth_map, int w, int h, float fov_deg)
{
    float f_px = calculate_f_px(w, fov_deg);
    float factor = w / f_px;
    cv::Mat depth = depth_map * factor;

    float min_value = 1e-4;
    float max_value = 1e4;

    cv::Mat clipped_inverse_depth = depth.clone();
    cv::min(clipped_inverse_depth, max_value, clipped_inverse_depth);
    cv::max(clipped_inverse_depth, min_value, clipped_inverse_depth);

    cv::Mat inverse_depth = 1.0 / clipped_inverse_depth;
    return inverse_depth;
}

cv::Mat get_color_depth(cv::Mat inverse_depth)
{
    double min_val, max_val;
    cv::minMaxLoc(inverse_depth, &min_val, &max_val);

    float max_invdepth_vizu = std::min(static_cast<float>(max_val), 1.0f / 0.1f);
    float min_invdepth_vizu = std::max(static_cast<float>(1.0 / 250.0), static_cast<float>(min_val));

    cv::Mat inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu);
    
    cv::Mat inverse_depth_normalized_8u;
    inverse_depth_normalized.convertTo(inverse_depth_normalized_8u, CV_8UC1, 255);

    cv::Mat color_depth;
    cv::applyColorMap(inverse_depth_normalized_8u, color_depth, cv::COLORMAP_JET);

    cv::Mat color_depth_bgr;
    cv::cvtColor(color_depth, color_depth_bgr, cv::COLOR_BGR2RGB);
    return color_depth_bgr;
}


void depth_proInfer()
{
    auto depth_pro = depth::load("depth_pro.engine");
    if (depth_pro == nullptr) return;

    cv::Mat image = cv::imread("inference/girl.jpg");
    int width = image.cols;

    nv::EventTimer t;
    t.start();
    for (int i = 0; i < 100; i++)
    {
        int height = image.rows;
        auto depth_map = depth_pro->forward(TensorRT::cvimg(image));
        cv::Mat depth_mat(depth_map.depth_map);

        depth_mat = depth_mat.reshape(1, height);
        // cv::Mat inverse_depth = compute_depth(depth_mat, width, height, depth_map.fov_deg);
        
        // cv::Mat color_depth_bgr = get_color_depth(inverse_depth);
    }
    t.stop();
    // [⏰ timer]: 	19896.27734 ms
    
    int height = image.rows;
    auto depth_map = depth_pro->forward(TensorRT::cvimg(image));
    cv::Mat depth_mat(depth_map.depth_map);

    depth_mat = depth_mat.reshape(1, height);
    cv::Mat inverse_depth = compute_depth(depth_mat, width, height, depth_map.fov_deg);
    
    cv::Mat color_depth_bgr = get_color_depth(inverse_depth);
    cv::imwrite("color_map.jpg", color_depth_bgr);
    
}

int main()
{
    depth_proInfer();
    return 0;
}
