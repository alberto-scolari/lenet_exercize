#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <sstream>

#include <cuda_runtime.h>

#include "common.h"

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
    std::unique_ptr<T, decltype(free_cuda_ptr<T>)> holder;

public:
    cuda_ptr(std::size_t _size) : holder(alloc_cuda_ptr(_size), free_cuda_ptr<T>) {
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
};