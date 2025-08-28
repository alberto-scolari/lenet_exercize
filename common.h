#pragma once

#include <sstream>
#include <iostream>
#include <string_view>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
constexpr inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

void FatalError(std::string_view s, std::decay_t<decltype(__FILE__)> file = __FILE__, decltype(__LINE__) line = __LINE__);

void checkCUDNN(cudnnStatus_t status, std::decay_t<decltype(__FILE__)> file = __FILE__, decltype(__LINE__) line = __LINE__);

void checkCuBLAS(cublasStatus_t status, std::decay_t<decltype(__FILE__)> file = __FILE__, decltype(__LINE__) line = __LINE__);

void checkCudaErrors(cudaError_t status, std::decay_t<decltype(__FILE__)> file = __FILE__, decltype(__LINE__) line = __LINE__);
