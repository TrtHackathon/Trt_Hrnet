1.requirement
    1.下载cudnn、TensorRT
    2.安装opencv
    3.安装pytorch

2.download
    1.git clone https://github.com/TrtHackathon/Trt_Hrnet.git
    2.cd Trt_Hrnet && git submodule update

3.build
    mkdir build && cmake .. -DTRT_DIR xxx && make -j

3.usage
    1. python3 hrnet2onnx.py --cfg xx --modelDir xx
    2. 使用 ./onnx2trt 查看参数
    3. 使用./trtInference 查看参数
