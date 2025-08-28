#include <lenet_exercize/common.h>

#include <iostream>
#include <sstream>

void FatalError(std::string_view s, std::decay_t<decltype(__FILE__)> file,
                decltype(__LINE__) line) {
  std::cerr << s << "\n" << file << ':' << line << "\nAborting...\n";
  cudaDeviceReset();
  exit(1);
}

void checkCUDNN(cudnnStatus_t status, std::decay_t<decltype(__FILE__)> file,
                decltype(__LINE__) line) {
  if (status == CUDNN_STATUS_SUCCESS) {
    return;
  }
  std::stringstream _error;
  _error << "CUDNN failure: " << cudnnGetErrorString(status);
  FatalError(_error.str(), file, line);
}

void checkCuBLAS(cublasStatus_t status, std::decay_t<decltype(__FILE__)> file,
                 decltype(__LINE__) line) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::stringstream _error;
  _error << "cuBLAS failure: " << status;
  FatalError(_error.str(), file, line);
}

void checkCudaErrors(cudaError_t status, std::decay_t<decltype(__FILE__)> file,
                     decltype(__LINE__) line) {
  if (status == cudaSuccess) {
    return;
  }
  std::stringstream _error;
  _error << "Cuda failure: " << status << ' ' << cudaGetErrorName(status)
         << " - " << cudaGetErrorString(status);
  FatalError(_error.str(), file, line);
}
