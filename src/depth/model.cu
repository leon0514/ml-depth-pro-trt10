#include "model.hpp"
#include "tensorrt.hpp"
#include "check.hpp"
#include "memory.hpp"
#include "affine.hpp"
#include "timer.hpp"

namespace depth
{

using namespace std;

class DepthProModelImpl : public Infer 
{
public:
    shared_ptr<TensorRT::Engine> trt_;
    string engine_file_;
    vector<shared_ptr<TensorRT::Memory<unsigned char>>> preprocess_buffers_;
    TensorRT::Memory<float> input_buffer_, output_buffer_, fov_deg_;
    TensorRT::Memory<float> depth_map_buffer_;
    int network_input_width_, network_input_height_;
    affine::Norm normalize_;
    bool isdynamic_model_ = false;

    virtual ~DepthProModelImpl() = default;

    void adjust_memory(int width, int height)
    {
        depth_map_buffer_.gpu(width * height);
        depth_map_buffer_.cpu(width * height);
    }

    void adjust_memory(int batch_size) 
    {
        // the inference batch_size
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);

        output_buffer_.gpu(batch_size * input_numel / 3);
        output_buffer_.cpu(batch_size * input_numel / 3);

        fov_deg_.gpu(batch_size * 1);
        fov_deg_.cpu(batch_size * 1);


        if ((int)preprocess_buffers_.size() < batch_size) 
        {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(make_shared<TensorRT::Memory<unsigned char>>());
        }
    }

    void preprocess(
        int ibatch, const TensorRT::Image &image,
        shared_ptr<TensorRT::Memory<unsigned char>> preprocess_buffer,
        void *stream = nullptr) 
    {
        affine::ResizeMatrix affine;
        affine.compute(make_tuple(image.width, image.height),
                    make_tuple(network_input_width_, network_input_height_));

        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image = image.width * image.height * 3;
        size_t size_matrix = sizeof(affine.d2i);
        uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
        float *affine_matrix_device = (float *)gpu_workspace;
        uint8_t *image_device = gpu_workspace + size_matrix;

        uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
        float *affine_matrix_host = (float *)cpu_workspace;
        uint8_t *image_host = cpu_workspace + size_matrix;

        // speed up
        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(
            cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
        checkRuntime(
            cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                    cudaMemcpyHostToDevice, stream_));

        affine::warp_affine_resize_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
                                                image.height, input_device, network_input_width_,
                                                network_input_height_, affine_matrix_device, 114,
                                                normalize_, stream_);
    }

    void postprocess( 
        int width,
        int height,
        void *stream = nullptr)
    {
        adjust_memory(width, height);
        affine::ResizeMatrix affine;
        affine.compute(make_tuple(network_input_width_, network_input_height_),make_tuple(width, height));

        size_t size_matrix = sizeof(affine.d2i);

        TensorRT::Memory<float> affine_matrix;

        float *affine_matrix_device = affine_matrix.gpu(size_matrix);

        float *affine_matrix_host = affine_matrix.cpu(size_matrix);
        
        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(
            cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                    cudaMemcpyHostToDevice, stream_));

        float *image_device = output_buffer_.gpu();
        float *dst_device = depth_map_buffer_.gpu();

        affine::warp_affine_bilinear_single_channel_plane(
            image_device, network_input_width_, network_input_width_, network_input_height_,
            dst_device, width, height, affine_matrix_device, 1000,
            stream_);
    }

    bool load(const string &engine_file) 
    {
        trt_ = TensorRT::load(engine_file);
        if (trt_ == nullptr) return false;

        trt_->print();

        auto input_dim = trt_->static_dims(0);

        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        float mean[3] = {0.5, 0.5, 0.5}; 
        float std[3] = {0.5, 0.5, 0.5};
        normalize_ = affine::Norm::mean_std(mean, std, 1/255.0,  affine::ChannelType::SwapRB);
        return true;
    }

    virtual DepthMap forward(const TensorRT::Image& image, void *stream = nullptr) override
    {
        auto input_dims = trt_->static_dims(0);
        int infer_batch_size = input_dims[0];
        assert (infer_batch_size == 1);
        // if (infer_batch_size != 1) 
        // {
        //     if (isdynamic_model_) 
        //     {
        //         infer_batch_size = 1;
        //         input_dims[0] = 1;
        //         if (!trt_->set_run_dims(0, input_dims)) return DeepthMap(0,0);
        //     } 
        //     else 
        //     {
        //         if (infer_batch_size < 1) 
        //         {
        //             printf(
        //                 "When using static shape model, number of images[%d] must be "
        //                 "less than or equal to the maximum batch[%d].",
        //                 1, infer_batch_size);
        //             return DeepthMap(0,0);
        //         }
        //     }
        // }
        adjust_memory(infer_batch_size);
    
        cudaStream_t stream_ = (cudaStream_t)stream;
        preprocess(0, image, preprocess_buffers_[0], stream_);
            

        vector<void *> bindings{input_buffer_.gpu(), output_buffer_.gpu(), fov_deg_.gpu()};

        if (!trt_->forward(std::unordered_map<std::string, const void *>{
                { "input", input_buffer_.gpu() }, 
                { "depth", output_buffer_.gpu() },
                { "fov",   fov_deg_.gpu() }
            }, stream_))
        {
            printf("Failed to tensorRT forward.");
            return DeepthMap(0,0);
        }

        postprocess(image.width, image.height, stream);

        checkRuntime(cudaMemcpyAsync(depth_map_buffer_.cpu(), depth_map_buffer_.gpu(),
                                    depth_map_buffer_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaMemcpyAsync(fov_deg_.cpu(), fov_deg_.gpu(),
                                    fov_deg_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        

        DepthMap arrout(image.width, image.height);
        std::memcpy(arrout.depth_map.data(), depth_map_buffer_.cpu(), image.width * image.height * sizeof(float));
        arrout.fov_deg = *(fov_deg_.cpu());
        return arrout;
    }
};

Infer *loadraw(const std::string &engine_file) 
{
    DepthProModelImpl *impl = new DepthProModelImpl();
    if (!impl->load(engine_file)) 
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

shared_ptr<Infer> load(const string &engine_file, int gpu_id) 
{
    checkRuntime(cudaSetDevice(gpu_id));
    return std::shared_ptr<DepthProModelImpl>((DepthProModelImpl *)loadraw(engine_file));
}


} // namespace depth

