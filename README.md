# TRT HRNET

![Image text](https://github.com/TrtHackathon/Trt_Hrnet/tree/main/image/pose_hrnet_w48_384x288.png)

## 评估结果

| 实现      | 耗时(ms) | 吞吐(img/s) | 加速比 |
|-----------|----------|-------------|--------|
| B = 1     |----------|-------------|--------|
|-----------|----------|-------------|--------|
| pytorch   | 81       | 12.5        | 1.0    |
| TRT(float)| 19.7     | 50          | 4X     |
| TRT(half) | 7.9      | 125         | 10X    |
|-----------|----------|-------------|--------|
| B = 4     |----------|-------------|--------|
|-----------|----------|-------------|--------|
| pytorch   | 147      | 27          | 1.0    |
| TRT(float)| 19.9     | 200         | 7.5X   |
| TRT(half) | 8.2      | 500         | 18.5X  |

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
