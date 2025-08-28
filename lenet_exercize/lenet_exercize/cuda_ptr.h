#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <type_traits>

#include <cuda_runtime.h>

#include <lenet_exercize/common.h>

template <typename T> static inline
T* alloc_cuda_ptr(std::size_t _size)
{
    T *p = nullptr;
    checkCudaErrors(cudaMalloc(&p, sizeof(T) * _size));
    return p;
}

template <typename T> static inline void free_cuda_ptr(T * p)
{
    checkCudaErrors(cudaFree(p));
}

template <typename T> class cuda_ptr
{
    std::unique_ptr<T, std::remove_reference_t<decltype(free_cuda_ptr<T>)> *> holder;
    std::size_t __size;

public:
    cuda_ptr() : __size(0), holder(nullptr, free_cuda_ptr<T>) {}

    cuda_ptr(std::size_t _size) : __size(_size), holder(alloc_cuda_ptr<T>(_size), free_cuda_ptr<T>)
    {
        if (not bool(holder)) {
            std::stringstream ss("cannot allocate CUDA memory for ");
            ss << _size << " bytes";
            throw std::runtime_error(ss.str());
        }
    }

    cuda_ptr(const cuda_ptr<T> &) = delete;
    cuda_ptr(cuda_ptr<T> &&) = default;

    cuda_ptr<T> operator=(const cuda_ptr<T> &) = delete;
    cuda_ptr<T> operator=(cuda_ptr<T> &&) = delete;

    T * operator()()
    {
        return holder.get();
    }
    const T *operator()() const { return holder.get(); }

    // implicit conversion to contained type
    operator T*() {
        return holder.get();
    }

    void reset(std::size_t _size) {
        holder.reset(alloc_cuda_ptr<T>(_size));
        __size = _size;
    }

    std::size_t size() const {
        return __size;
    }
};
