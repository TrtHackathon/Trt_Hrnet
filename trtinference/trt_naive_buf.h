#ifndef TRT_NAIVE_BUF_H

#define TRT_NAIVE_BUF_H

#include <NvInfer.h>
#include <cuda_runtime.h>
//#include <tbb/concurrent_queue.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <queue>

namespace TRT {

#define __CHECK__(status)                                                               \
    if (!status) {                                                                      \
        std::cerr << "Trt Error Occurs : " << __FILE__ << " " << __LINE__ << std::endl; \
        return false;                                                                   \
    }

#define __CHECKCUDA__(status)                                                                \
    if (status != 0) {                                                                       \
        std::cerr << "Trt Cuda Error Occurs : " << __FILE__ << " " << __LINE__ << std::endl; \
        return false;                                                                        \
    }

#define __LOG__(info) std::cout << __FILE__ << " " << __LINE__ << " -> " << info << std::endl;

template <class T, class D>
class BufWrapper {
public:
    BufWrapper(T value, D dealloc) : _value(value), _dealloc(dealloc){};
    ~BufWrapper() { _dealloc(_value); }
    T& get() { return _value; }

private:
    T _value;
    D _dealloc;
};

// template <typename Alloc, typename Wrapper, typename T, typename P = int, int MAX = 8>
template <typename T, typename P, int N, typename Alloc, typename Wrapper = BufWrapper<T, std::function<void(T)>>>
class TrtNaiveBuf {
public:
    bool initialize(P args) {
        for (int idx = 0; idx < N; ++idx) {
            T buf;
            __CHECK__(_alloc.alloc(buf, args));
            _queue.push(buf);
        }
        return true;
    }

    Wrapper getBuf() {
        T buf;
        std::unique_lock<std::mutex> lock(_mtx);
        this->_cv.wait(lock, [this, &buf] {
            if (this->_queue.empty()) {
                return false;
            } else {
                buf = this->_queue.front();
                this->_queue.pop();
                return true;
            }
            /*return this->_queue.try_pop(buf);*/
        });

        auto del = [this](T buf) {
            {
                std::unique_lock<std::mutex> lock(this->_mtx);
                this->_queue.push(buf);
            }
            this->_cv.notify_one();
        };
        return Wrapper(buf, del);
    }

    ~TrtNaiveBuf() {
        while (!_queue.empty()) {
            _alloc.dealloc(_queue.front());
            _queue.pop();
        }
    }

private:
    Alloc _alloc;
    std::mutex _mtx;
    std::queue<T> _queue;
    std::condition_variable _cv;
    //tbb::concurrent_queue<T> _queue;
};

template <typename T, typename P = size_t>
class LockedAlloc {
public:
    bool alloc(T& value, P& size) {
        __CHECKCUDA__(cudaHostAlloc(&value, size, 0));
        return true;
    }

    void dealloc(T value) { cudaFreeHost(value); }
};

template <typename T, typename P = size_t>
class CpyContextAlloc {
public:
    bool alloc(T& value, P& size) {
        __CHECKCUDA__(cudaMalloc(&value.buf, size));
        __CHECKCUDA__(cudaStreamCreate(&value.stream));
        return true;
    }

    void dealloc(T value) {
        cudaFree(value.buf);
        cudaStreamDestroy(value.stream);
    }
};

template <typename T, typename P>
class ExecContextAlloc {
public:
    bool alloc(T& value, P& args) {
        value = args->createExecutionContext();
        return true;
    }

    void dealloc(T value) {}
};
}  // namespace TRT
#endif
