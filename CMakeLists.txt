cmake_minimum_required(VERSION 3.10)
project(Trt_Hrnet)

file(GLOB _onnx2trt ${CMAKE_CURRENT_SOURCE_DIR}/onnx2trt/*.*)
file(GLOB _trtinference ${CMAKE_CURRENT_SOURCE_DIR}/trtinference/*.*)

set(TRT_LIB -lrt
            -lnvrtc
	    -lcudart
	    -lcudnn
	    -lmyelin
	    -lnvinfer
	    -lnvparsers
	    -lnvonnxparser
	    -lnvinfer_plugin
	    -lopencv_core
	    -lopencv_imgcodecs
	    -lopencv_imgproc
	    -lopencv_highgui)

if(TRT_DIR)
    include_directories(${TRT_DIR}//include)
    link_directories(${TRT_DIR}//lib)
    message(${TRT_DIR})
endif()

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(onnx2trt ${_onnx2trt})
target_link_libraries(onnx2trt ${TRT_LIB})

add_executable(trtinference ${_trtinference})
target_link_libraries(trtinference ${TRT_LIB})
