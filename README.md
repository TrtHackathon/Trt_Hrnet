# TRT HRNET

TODO: sunli
配个图！

## 评估结果

| 实现    | 耗时(ms) | 吞吐(img/s) | 加速比 |
|---------|----------|-------------|--------|
| pytorch | X        | XX          | 1.0    |
| TRT     | Y        | YY          | ??     |

# Docker

国内网络由于众所周知的问题，`docker build`时容易抽风。

## 构建

```bash
docker build --pull -t trt_hrnet .
```

## 运行

```bash
docker run -it --rm -v $(realpath .):$(realpath .) trt_hrnet
```

# 裸机

用户 **自行解决** CUDNN、TensorRT的相关依赖。

```bash
apt install libopencv-dev
pip install pytorch
```

## 构建

```bash
mkdir build
cd build
# 如果自行安装TensorRT，请使用
# cmake .. -DTRT_DIR <path/to/trt/directory>
cmake ..
cmake --build .
```

# 使用

1. `python3 hrnet2onnx.py --cfg xx --modelDir xx`
2. `./onnx2trt` 查看相关参数
3. `./trtInference` 查看相关参数
