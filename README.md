#  <center> ml-depth-pro C++ tensorrt ⚡

<div align="center">
<img height="500" src="https://github.com/leon0514/ml-depth-pro-trt10/blob/main/example.png" />
<br />
<br />
</div>


## 🛠️ 导出 onnx
`opset_version >= 17`

```python
import torch
from depth_pro import create_model_and_transforms

model, transform = create_model_and_transforms(
    device=torch.device('cuda:0')
)
model.eval()

x = torch.randn(1, 3, 1536, 1536, device='cuda:0')

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "model/depth_pro.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth', 'fov'],
        keep_initializers_as_inputs=None)
```

## 🛠️ 构建tensorrt引擎
```shell
trtexec --onnx=model/depth_pro.onnx --saveEngine=depth_pro.engine --fp16 --verbose
```

## ⚡ 推理
```shell
make runpro

Compile CXX src/main.cpp
Link workspace/pro
------------------------------------------------------
TensorRT-Engine 🌱 is Static Shape model
Inputs: 1
	0.input : {1 x 3 x 1536 x 1536} [float32]
Outputs: 2
	0.depth : {1 x 1 x 1536 x 1536} [float32]
	1.fov : {1 x 1 x 1 x 1} [float32]
------------------------------------------------------
[⏰ timer]: 	19896.27734 ms
```

## 🏃 速度
- 在`3090上`循环跑`100`次耗时`19896ms`
```c++
nv::EventTimer t;
t.start();
for (int i = 0; i < 100; i++)
{
    int height = image.rows;
    auto depth_map = depth_pro->forward(TensorRT::cvimg(image));
    cv::Mat depth_mat(depth_map.depth_map);

    depth_mat = depth_mat.reshape(1, height);
}
t.stop();
// [⏰ timer]: 	19896.27734 ms
```

## ✉️ pybind11 封装

- 编译
    ```shell
    make all
    ```

- 运行
    ```shell
    >>> import cv2
    >>> import trtdepthpro
    >>> engine_path = "depth_pro.engine"
    >>> image_path  = "images/bus.jpg"
    >>> image = cv2.imread(image_path)
    >>> model = trtdepthpro.TrtDepthProInfer(engine_path, 0)
    ------------------------------------------------------
    TensorRT-Engine 🌱 is Static Shape model
    Inputs: 1
        0.input : {1 x 3 x 1536 x 1536} [float32]
    Outputs: 2
        0.depth : {1 x 1 x 1536 x 1536} [float32]
        1.fov : {1 x 1 x 1 x 1} [float32]
    ------------------------------------------------------
    >>> res = model.forward(image)
    >>> res
    DeepthMap(width: 810, height: 1080, fov_deg: 37.1668)
    ```


## 不同分辨率模型测试
### 768 x 768
<img height="500" src="https://github.com/leon0514/ml-depth-pro-trt10/blob/main/768x768.jpg" />

### 1152 x 1152
<img height="500" src="https://github.com/leon0514/ml-depth-pro-trt10/blob/main/1152x1152.jpg" />
## 🤖 测试环境
- 系统  
Ubuntu 22.04 LTS 
- docker镜像    
nvcr.io/nvidia/tensorrt:24.10-py3
- gpu   
3090



## 👏 参考
- https://github.com/shouxieai/infer 
- https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution
- https://github.com/apple/ml-depth-pro
- https://github.com/yuvraj108c/ml-depth-pro-tensorrt/

