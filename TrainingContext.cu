#include <vector>
#include <tuple>

#include <cublas_v2.h>

#include "TrainingContext.h"
#include "common.h"

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */
__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

TrainingContext::TrainingContext(int gpuid, int batch_size, unsigned _blockWidth,
                                 ConvBiasLayer &conv1, MaxPoolLayer &pool1, ConvBiasLayer &conv2, MaxPoolLayer &pool2,
                                 FullyConnectedLayer &fc1, FullyConnectedLayer &fc2) :
                                 gpuid(gpuid), batchSize(batch_size), blockWidth(_blockWidth), ref_fc1(fc1), ref_fc2(fc2)
{
    // Create CUBLAS and CUDNN handles
    checkCudaErrors(cudaSetDevice(gpuid));
    checkCuBLAS(cublasCreate(&cublasHandle));
    checkCUDNN(cudnnCreate(&cudnnHandle));

    // Create tensor descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

    checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));

    checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));

    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

    // Set tensor descriptor sizes
    checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasTensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            1, conv1.out_channels,
                                            1, 1));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasTensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            1, conv2.out_channels,
                                            1, 1));

    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_PROPAGATE_NAN,
                                            pool1.size, pool1.size,
                                            0, 0,
                                            pool1.stride, pool1.stride));
    checkCUDNN(cudnnSetTensor4dDescriptor(pool2Tensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            batch_size, conv2.out_channels,
                                            conv2.out_height / pool2.stride,
                                            conv2.out_width / pool2.stride));

    checkCUDNN(cudnnSetTensor4dDescriptor(fc1Tensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            batch_size, fc1.outputs, 1, 1));

    checkCUDNN(cudnnSetTensor4dDescriptor(fc2Tensor,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            batch_size, fc2.outputs, 1, 1));

    checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));

    // Set convolution tensor sizes and compute workspace size
    size_t workspace = 0;
    workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

    workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

    // The workspace is allocated later (if necessary)
    workspaceSize = workspace;
}

TrainingContext::~TrainingContext()
{
    checkCudaErrors(cudaSetDevice(gpuid));

    checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(pool1Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
    checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
    checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    checkCuBLAS(cublasDestroy(cublasHandle));
    checkCUDNN(cudnnDestroy(cudnnHandle));
}

size_t TrainingContext::SetFwdConvolutionTensors(ConvBiasLayer &conv, cudnnTensorDescriptor_t &srcTensorDesc, cudnnTensorDescriptor_t &dstTensorDesc,
                                                    cudnnFilterDescriptor_t &filterDesc, cudnnConvolutionDescriptor_t &convDesc,
                                                    cudnnConvolutionFwdAlgo_t &algo)
{
    int n = batchSize;
    int c = conv.in_channels;
    int h = conv.in_height;
    int w = conv.in_width;

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            n, c,
                                            h, w));

    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                            CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW,
                                            conv.out_channels,
                                            conv.in_channels,
                                            conv.kernel_size,
                                            conv.kernel_size));

#if CUDNN_MAJOR > 5
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                0, 0,
                                                1, 1,
                                                1, 1,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
#else
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                0, 0,
                                                1, 1,
                                                1, 1,
                                                CUDNN_CROSS_CORRELATION));
#endif

    // Find dimension of convolution output
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                        srcTensorDesc,
                                                        filterDesc,
                                                        &n, &c, &h, &w));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                            CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            n, c,
                                            h, w));
    int count = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &count));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_info(count);
    int returned_count;
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnnHandle,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    dstTensorDesc,
                                                    count,
                                                    &returned_count,
                                                    perf_info.data()));
    algo = perf_info[0].algo;
    size_t sizeInBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       algo,
                                                       &sizeInBytes));

    return sizeInBytes;
}

void TrainingContext::ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                                            float *fc2, float *result,
                                            float *pconv1, float *pconv1bias,
                                            float *pconv2, float *pconv2bias,
                                            float *pfc1, float *pfc1bias,
                                            float *pfc2, float *pfc2bias, void *workspace, float *onevec)
{
    // LeNet-4 structure
    // https://sh-tsang.medium.com/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17
    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cudaSetDevice(gpuid));

    // Conv1 layer
    checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                                        data, conv1filterDesc, pconv1, conv1Desc,
                                        conv1algo, workspace, workspaceSize, &beta,
                                        conv1Tensor, conv1));
    checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
                                pconv1bias, &alpha, conv1Tensor, conv1));

    // Pool1 layer
    checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                                    conv1, &beta, pool1Tensor, pool1));

    // Conv2 layer
    checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                                        pool1, conv2filterDesc, pconv2, conv2Desc,
                                        conv2algo, workspace, workspaceSize, &beta,
                                        conv2Tensor, conv2));
    checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                                pconv2bias, &alpha, conv2Tensor, conv2));

    // Pool2 layer
    checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                                    conv2, &beta, pool2Tensor, pool2));

    // FC1 layer
    // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                            ref_fc1.outputs, batchSize, ref_fc1.inputs,
                            &alpha,
                            pfc1, ref_fc1.inputs,
                            pool2, ref_fc1.inputs,
                            &beta,
                            fc1, ref_fc1.outputs));
    // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ref_fc1.outputs, batchSize, 1,
                            &alpha,
                            pfc1bias, ref_fc1.outputs,
                            onevec, 1,
                            &alpha,
                            fc1, ref_fc1.outputs));

    // ReLU activation
    checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                                        fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));

    // FC2 layer
    // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                            ref_fc2.outputs, batchSize, ref_fc2.inputs,
                            &alpha,
                            pfc2, ref_fc2.inputs,
                            fc1relu, ref_fc2.inputs,
                            &beta,
                            fc2, ref_fc2.outputs));
    // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            ref_fc2.outputs, batchSize, 1,
                            &alpha,
                            pfc2bias, ref_fc2.outputs,
                            onevec, 1,
                            &alpha,
                            fc2, ref_fc2.outputs));

    // Softmax loss
    checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
}

size_t TrainingContext::SetBwdConvolutionTensors(cudnnTensorDescriptor_t &srcTensorDesc, cudnnTensorDescriptor_t &dstTensorDesc,
                                                    cudnnFilterDescriptor_t &filterDesc, cudnnConvolutionDescriptor_t &convDesc,
                                                    cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
{
    size_t sizeInBytes = 0, tmpsize = 0;

    // If backprop filter algorithm was requested
    if (falgo) {
        int count = 0;
        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle, &count));
        std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_info(count);
        int returned_count = 0;
        checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
            cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
            count, &returned_count, perf_info.data()));
        *falgo = perf_info[0].algo;

        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
            *falgo, &tmpsize));

        sizeInBytes = std::max(sizeInBytes, tmpsize);
    }

    // If backprop data algorithm was requested
    if (dalgo) {
        int count = 0;
        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle, &count));
        std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_info(count);
        int returned_count = 0;

        checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
            cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
            count, &returned_count, perf_info.data()));
        *dalgo = perf_info[0].algo;

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
            *dalgo, &tmpsize));

        sizeInBytes = std::max(sizeInBytes, tmpsize);
    }

    return sizeInBytes;
}

void TrainingContext::Backpropagation(ConvBiasLayer &layer_conv1, MaxPoolLayer &layer_pool1, ConvBiasLayer &layer_conv2, MaxPoolLayer &layer_pool2,
                                      float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                                      float *fc2, float *fc2smax, float *dloss_data,
                                      float *pconv1, float *pconv1bias,
                                      float *pconv2, float *pconv2bias,
                                      float *pfc1, float *pfc1bias,
                                      float *pfc2, float *pfc2bias,
                                      float *gconv1, float *gconv1bias, float *dpool1,
                                      float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                                      float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                                      float *gfc2, float *gfc2bias, float *dfc2,
                                      void *workspace, float *onevec)
{
    // disable warnings by ognoring variables
    std::ignore = std::make_tuple(layer_conv1, layer_pool1, layer_conv2, layer_conv2, layer_pool2);
    std::ignore = std::make_tuple(fc2, pconv1, pconv1bias, pconv2bias, pfc1bias, pfc2bias);

    float alpha = 1.0f, beta = 0.0f;

    float scalVal = 1.0f / static_cast<float>(batchSize);

    checkCudaErrors(cudaSetDevice(gpuid));

    // Initialization (using the training error function)
    checkCudaErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));

    // Softmax layer
    SoftmaxLossBackprop<<<RoundUp(batchSize, blockWidth), blockWidth>>>(labels, ref_fc2.outputs, batchSize, dloss_data);

    // Accounting for batch size in SGD
    checkCuBLAS(cublasSscal(cublasHandle, ref_fc2.outputs * batchSize, &scalVal, dloss_data, 1));

    // FC2 layer
    // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, batchSize,
                            &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
    // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
    checkCuBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, batchSize,
                            &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
    // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, batchSize, ref_fc2.outputs,
                            &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));

    // ReLU activation
    checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                                        fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                        fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));

    // FC1 layer
    // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, batchSize,
                            &alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
    // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
    checkCuBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, batchSize,
                            &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
    // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
    checkCuBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, batchSize, ref_fc1.outputs,
                            &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));

    // Pool2 layer
    checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                                    pool2Tensor, pool2, pool2Tensor, dfc1,
                                    conv2Tensor, conv2, &beta, conv2Tensor, dpool2));

    // Conv2 layer
    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                            dpool2, &beta, conv2BiasTensor, gconv2bias));

    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
                                                pool1, conv2Tensor, dpool2, conv2Desc,
                                                conv2bwfalgo, workspace, workspaceSize,
                                                &beta, conv2filterDesc, gconv2));

    checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
                                            pconv2, conv2Tensor, dpool2, conv2Desc,
                                            conv2bwdalgo, workspace, workspaceSize,
                                            &beta, pool1Tensor, dconv2));

    // Pool1 layer
    checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                                    pool1Tensor, pool1, pool1Tensor, dconv2,
                                    conv1Tensor, conv1, &beta, conv1Tensor, dpool1));

    // Conv1 layer
    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                            dpool1, &beta, conv1BiasTensor, gconv1bias));

    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                                data, conv1Tensor, dpool1, conv1Desc,
                                                conv1bwfalgo, workspace, workspaceSize,
                                                &beta, conv1filterDesc, gconv1));

    // No need for convBackwardData because there are no more layers below
}

void TrainingContext::UpdateWeights(float learning_rate,
                                    ConvBiasLayer &conv1, ConvBiasLayer &conv2,
                                    float *pconv1, float *pconv1bias,
                                    float *pconv2, float *pconv2bias,
                                    float *pfc1, float *pfc1bias,
                                    float *pfc2, float *pfc2bias,
                                    float *gconv1, float *gconv1bias,
                                    float *gconv2, float *gconv2bias,
                                    float *gfc1, float *gfc1bias,
                                    float *gfc2, float *gfc2bias)
{
    float alpha = -learning_rate;

    checkCudaErrors(cudaSetDevice(gpuid));

    // Conv1
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
                            &alpha, gconv1, 1, pconv1, 1));
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
                            &alpha, gconv1bias, 1, pconv1bias, 1));

    // Conv2
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
                            &alpha, gconv2, 1, pconv2, 1));
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
                            &alpha, gconv2bias, 1, pconv2bias, 1));

    // Fully connected 1
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
                            &alpha, gfc1, 1, pfc1, 1));
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
                            &alpha, gfc1bias, 1, pfc1bias, 1));

    // Fully connected 2
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                            &alpha, gfc2, 1, pfc2, 1));
    checkCuBLAS(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
                            &alpha, gfc2bias, 1, pfc2bias, 1));
}
