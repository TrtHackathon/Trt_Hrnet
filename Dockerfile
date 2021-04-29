FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

# WIP, stuck on network issue

RUN echo 'APT::Get::AllowUnauthenticated "true";' > /etc/apt/apt.conf.d/99china_insecure && \
    echo 'Acquire::AllowInsecureRepositories "true";' >> /etc/apt/apt.conf.d/99china_insecure

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libopencv-dev \
        libnvinfer-dev \
        libnvonnxparsers-dev \
        libnvparsers-dev \
        libnvinfer-plugin-dev

COPY . /workspace
WORKDIR /workspace

RUN mkdir build && cd build \
    cmake .. && cmake --build .

WORKDIR /workspace/build
