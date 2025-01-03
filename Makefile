cc        := g++
name      := trtdepthpro.so
workdir   := workspace
src_dir   := src
obj_dir   := objs
stdcpp    := c++17
cuda_home := /usr/local/cuda-12
cuda_arch := 8.6
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)


project_include_path := src src/common src/depth
opencv_include_path  := /workspace/__install/opencv490/include/opencv4
trt_include_path     := /usr/include/x86_64-linux-gnu/
cuda_include_path    := $(cuda_home)/include
ffmpeg_include_path  := 

python_include_path  := /usr/include/python3.10


include_paths        := $(project_include_path) \
						$(opencv_include_path) \
						$(trt_include_path) \
						$(cuda_include_path) \
						$(python_include_path)


opencv_library_path  := /workspace/__install/opencv490/lib
trt_library_path     := /usr/lib/x86_64-linux-gnu/
cuda_library_path    := $(cuda_home)/lib64/
python_library_path  := 

library_paths        := $(opencv_library_path) \
						$(trt_library_path) \
						$(cuda_library_path) \
						$(cuda_library_path) \
						$(python_library_path)

link_opencv       := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs
link_trt          := nvinfer nvinfer_plugin nvonnxparser
link_cuda         := cuda cublas cudart cudnn
link_sys          := stdc++ dl

link_libraries     := $(link_opencv) $(link_trt) $(link_cuda) $(link_sys)


empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

run_paths      := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths  := $(foreach item,$(include_paths),-I$(item))
library_paths  := $(foreach item,$(library_paths),-L$(item))
link_libraries := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread $(include_paths)
cu_compile_flags  := -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN' $(library_paths) $(link_librarys)

cpp_srcs := $(shell find $(src_dir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(src_dir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find $(src_dir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(src_dir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

pro_cpp_objs := $(filter-out objs/interface.cpp.o, $(cpp_objs))

ifneq ($(MAKECMDGOALS), clean)
include $(mks)
endif


$(name)   : $(workdir)/$(name)

all       : $(name)

pro       : $(workdir)/pro

run : pro
	@export LD_LIBRARY_PATH=$(library_path_export)
	@cd $(workdir) && ./pro

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(workdir)/pro : $(pro_cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(obj_dir)/%.cpp.o : $(src_dir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(obj_dir)/%.cu.o : $(src_dir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

$(obj_dir)/%.cpp.mk : $(src_dir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)

$(obj_dir)/%.cu.mk : $(src_dir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)


clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/pro $(workdir)/*.trtmodel $(workdir)/imgs

.PHONY : clean run $(name) pro