#ifndef TRT_ENGINE_H

#define TRT_ENGINE_H

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>

#include <chrono>
#include <fstream>
#include <ios>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "logger.h"
#include "trt_naive_buf.h"

namespace TRT {

using namespace nvinfer1;

struct TrtEngine {
private:
    // device
    int gpu_id{0};
    int maximum_batch_size{0};
    size_t maximum_buffer_size{0};
    // trt engine
    int n_bindings{ 0 };
    ICudaEngine* engine{nullptr};
    // name index size dimension
    std::vector<std::tuple<std::string, int, size_t, DataType>> input_attr, output_attr;

    struct CpyContext {
        char* buf{nullptr};
        cudaStream_t stream;
    };

    TrtNaiveBuf<CpyContext, size_t, 4, CpyContextAlloc<CpyContext, size_t>> naive_cpy_context_buf;
    class RunContext {
    public:
        friend class TrtEngine;
        int batch_size;
        cudaStream_t stream;
        std::vector<void*> g_input_output;
        RunContext(TrtEngine* trt_engine, const int _batch_size, const std::unordered_map<std::string, char*>& _input,
                   std::unordered_map<std::string, std::vector<float>>& _output, CpyContext& cpy_ctx) {

            g_input_output.resize(trt_engine->input_attr.size() + trt_engine->output_attr.size());
            char* buf = cpy_ctx.buf;
            stream = cpy_ctx.stream;
            batch_size = _batch_size;

            for (auto& p : trt_engine->input_attr) {
                //cpy data
                auto buf_size = std::get<2>(p) * batch_size;
                auto data = _input.find(std::get<0>(p))->second;
                cudaMemcpyAsync(buf, data, buf_size, cudaMemcpyHostToDevice, stream);

                //set buf
                g_input_output[std::get<1>(p)] = buf;
                buf += buf_size;
            }

            for (auto& p : trt_engine->output_attr) {
                _output[std::get<0>(p)] = std::move(std::vector<float>(_batch_size * std::get<2>(p) / sizeof(float)));

                //set buf
                g_input_output[std::get<1>(p)] = buf;
                buf += std::get<2>(p) * _batch_size;
            }
        }

        RunContext(TrtEngine* trt_engine, const int _batch_size, CpyContext& cpy_ctx) {

            g_input_output.resize(trt_engine->input_attr.size() + trt_engine->output_attr.size());
            char* buf = cpy_ctx.buf;
            stream = cpy_ctx.stream;
            batch_size = _batch_size;

            cudaMemsetAsync(buf, 0, trt_engine->maximum_buffer_size, stream);

            for (auto& p : trt_engine->input_attr) {
                //set buf
                g_input_output[std::get<1>(p)] = buf;
                buf += std::get<2>(p) * _batch_size;
            }

            for (auto& p : trt_engine->output_attr) {
                //set buf
                g_input_output[std::get<1>(p)] = buf;
                buf += std::get<2>(p) * _batch_size;
            }
        }
    };


    TrtNaiveBuf<IExecutionContext*, ICudaEngine*, 2, ExecContextAlloc<IExecutionContext*, ICudaEngine*>>
        naive_exec_context_buf;

    TrtNaiveBuf<char*, size_t, 16, LockedAlloc<char*, size_t>, std::unique_ptr<char, std::function<void(char*)>>>
        trt_naive_locked_buf;

    bool _predict(RunContext&, IExecutionContext*);

public:
    TrtEngine();
    ~TrtEngine();
    bool init(const uint32_t device_id, const std::string& trt_file);

    uint32_t get_maximum_batch_size() {
        return maximum_batch_size;
    }

    size_t get_maximum_buffer_size() {
        return maximum_buffer_size;
    }

    std::unique_ptr<char, std::function<void(char*)>> get_buf() { return trt_naive_locked_buf.getBuf(); }

    bool warmup(const int batch_size) {
        __CHECKCUDA__(cudaSetDevice(gpu_id));
        auto cpy_ctx = naive_cpy_context_buf.getBuf();
        RunContext run_ctx(this, batch_size, cpy_ctx.get());

        auto exec_ctx = naive_exec_context_buf.getBuf();
        __CHECK__(_predict(run_ctx, exec_ctx.get()));
        return true;
    }

    bool predict(const int batch_size, const std::unordered_map<std::string, char*>& input,
                 std::unordered_map<std::string, std::vector<float>>& output,
                 std::chrono::steady_clock::time_point* spot = nullptr) {
        __CHECKCUDA__(cudaSetDevice(gpu_id));
        auto cpy_ctx = naive_cpy_context_buf.getBuf();
        RunContext run_ctx(this, batch_size, input, output, cpy_ctx.get());

        //__CHECKCUDA__(cudaStreamSynchronize(run_ctx.stream));
        if (spot) {
            *spot = std::chrono::steady_clock::now();
        }
        {
            auto exec_ctx = naive_exec_context_buf.getBuf();
            __CHECK__(_predict(run_ctx, exec_ctx.get()));
        }
        // sync copy
        for (auto& p : output_attr) {
            auto dst = output.find(std::get<0>(p))->second.data();
            __CHECKCUDA__(cudaMemcpy(dst, run_ctx.g_input_output[std::get<1>(p)], std::get<2>(p) * batch_size,
                       cudaMemcpyDeviceToHost));
        }
        return true;
    }
};
}  // namespace TRT

#endif
