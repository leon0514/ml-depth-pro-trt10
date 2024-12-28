#include <sstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "opencv2/opencv.hpp"
#include "model.hpp"

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace std;

namespace py=pybind11;

namespace pybind11 { namespace detail {
template<>
struct type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        //const int ndims = (int)info.ndim;
        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if(ndims == 2)
	{
            nh = info.shape[0];
            nw = info.shape[1];
        } 
	else if(ndims == 3)
	{
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }
	else
	{
            char msg[64];
            std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
            throw std::logic_error(msg);
            return false;
        }

        int dtype;
        if(info.format == format_descriptor<unsigned char>::format())
	{
            dtype = CV_8UC(nc);
        }
	else if (info.format == format_descriptor<int>::format())
	{
            dtype = CV_32SC(nc);
        }
	else if (info.format == format_descriptor<float>::format())
	{
            dtype = CV_32FC(nc);
        }
	else
	{
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }
        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! 2. cast cv::Mat to numpy.ndarray
    static handle cast(const cv::Mat& mat, return_value_policy, handle defval){
        UNUSED(defval);

        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;

        if(depth == CV_8U)
        {
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }
	else if(depth == CV_32S)
        {
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }
	else if(depth == CV_32F)
	{
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }
	else
	{
            throw std::logic_error("Unsupport type, only support uchar, int32, float");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) 
	{
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } 
	else if (dim == 3) 
	{
            bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
            strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
    }
};
}}//! end namespace pybind11::detail

class TrtDepthProInfer{
public:
    TrtDepthProInfer(std::string model_path, int gpu_id = 0)
    {
        instance_ = depth::load(model_path, gpu_id);
    }

    depth::DeepthMap forward(const cv::Mat& image)
    {
        return instance_->forward(TensorRT::cvimg(image));
    }

    depth::DeepthMap forward_path(const std::string& image_path)
    {
        cv::Mat image = cv::imread(image_path);
        return instance_->forward(TensorRT::cvimg(image));
    }

    bool valid()
    {
        return instance_ != nullptr;
    }

private:
    std::shared_ptr<depth::Infer> instance_;

};


PYBIND11_MODULE(trtdepthpro, m){
    py::class_<depth::DeepthMap>(m, "DeepthMap")
        .def_readwrite("width", &depth::DeepthMap::width)
        .def_readwrite("height", &depth::DeepthMap::height)
        .def_readwrite("fov_deg", &depth::DeepthMap::fov_deg)
        .def_readwrite("depth_map", &depth::DeepthMap::depth_map)
        .def("__repr__", [](const depth::DeepthMap &depth_map) {
            std::ostringstream oss;
            oss << "DeepthMap(width: " << depth_map.width
                << ", height: " << depth_map.height
                << ", fov_deg: " << depth_map.fov_deg
                << ")";
            return oss.str();
        });

    py::class_<TrtDepthProInfer>(m, "TrtDepthProInfer")
	.def(py::init<string, int>(), py::arg("model_path"), py::arg("gpu_id"))
	.def_property_readonly("valid", &TrtDepthProInfer::valid)
        .def("forward_path", &TrtDepthProInfer::forward_path, py::arg("image_path"))
	.def("forward", &TrtDepthProInfer::forward, py::arg("image"));
};
