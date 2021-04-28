
1.requirement
1.下载cudnn、TensorRT
2.安装opencv
3.安装pytorch
4.下载 hrnet https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git

2.build
1.编译Onnx2trt
  cd onnx2trt && make
2.编译TrtInference
  cd trtinference && make

3.usage
1.python3 hrnet2onnx.py
2.