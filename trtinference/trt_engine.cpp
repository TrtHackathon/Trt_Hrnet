
#include <mutex>
#include "trt_engine.h"

namespace TRT {

static Logger logger;
static std::mutex trt_mtx;
static IRuntime* runtime = NULL;

TrtEngine::TrtEngine() {
    std::unique_lock<std::mutex> lock(trt_mtx);
    if(!runtime) {
        runtime = createInferRuntime(logger.getTRTLogger());
        initLibNvInferPlugins((void*)&logger.getTRTLogger(), "");
    }
}

TrtEngine::~TrtEngine() {
    if (engine) engine->destroy();
}

bool TrtEngine::init(const uint32_t device_id, const std::string& trt_file) {
    // set device
    gpu_id = device_id;
    __CHECKCUDA__(cudaSetDevice(gpu_id));

    // trt model file
    auto exist = (trt_file.find(".trt") != std::string::npos);
    __CHECK__(exist);

    std::ifstream ifs(trt_file, std::ios::binary);
    __CHECK__(!ifs.fail());

    ifs.seekg(0, std::ios::end);
    uint64_t len = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::vector<char> serial(len);
    ifs.read(serial.data(), len);
    ifs.close();

    // deserialize engine
    __CHECK__(runtime);
    try {
        engine = runtime->deserializeCudaEngine(serial.data(), len, NULL);
    } catch (...) {
        return false;
    }
    __CHECK__(engine);

    // profile bindings
    n_bindings = engine->getNbBindings();
    maximum_batch_size = engine->getMaxBatchSize();

    maximum_buffer_size = 0;
    for (int index = 0; index < n_bindings ; ++index) {
        auto dtype = engine->getBindingDataType(index);
        std::string name = engine->getBindingName(index);
        auto dim = std::move(engine->getBindingDimensions(index));

        size_t size = sizeof(float);
        for (int idx = 0; idx < dim.nbDims; ++idx) {
            if (dim.d[idx] > 0) size *= dim.d[idx];
        }

        if(engine->bindingIsInput(index)) {
            input_attr.emplace_back(name, index, size, dtype);
        } else {
            output_attr.emplace_back(name, index, size, dtype);
        }
        maximum_buffer_size += size * maximum_batch_size;
    }


    __CHECK__(naive_exec_context_buf.initialize(engine));
    __CHECK__(trt_naive_locked_buf.initialize(maximum_buffer_size * 1.1));
    __CHECK__(naive_cpy_context_buf.initialize(maximum_buffer_size * 1.1));
    return true;
}

bool TrtEngine::_predict(RunContext& run_ctx, IExecutionContext* exec_ctx) {
    std::vector<void*> input_output(n_bindings, nullptr);
    exec_ctx->enqueue(run_ctx.batch_size, run_ctx.g_input_output.data(), run_ctx.stream, nullptr);

    //exec_ctx.context->enqueueV2(input_output.data(), run_ctx.stream, nullptr);
    __CHECKCUDA__(cudaStreamSynchronize(run_ctx.stream));

    return true;
}

}  // namespace TRT
