#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <tuple>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <argparse/argparse.hpp>

#include "common.h"
#include "readubyte.h"
#include "ConvBiasLayer.h"
#include "MaxPoolLayer.h"
#include "FullyConnectedLayer.h"
#include "TrainingContext.h"
#include "cuda_ptr.h"

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////
// Main function

int main(int argc, char **argv)
{
    int gpu = 0;
    int iterations = 1000;
    int random_seed = -1;
    int classify = -1;
    std::size_t batch_size = 64;
    bool pretrained = false;
    bool save_data = false;
    const std::string base_path("deps/mnist/");
    std::string FLAGS_train_images = base_path + "train-images-idx3-ubyte";
    std::string FLAGS_train_labels = base_path + "train-labels-idx1-ubyte";
    std::string FLAGS_test_images = base_path + "t10k-images-idx3-ubyte";
    std::string FLAGS_test_labels = base_path + "t10k-labels-idx1-ubyte";
    double learning_rate = 0.01;
    double lr_gamma = 0.01;
    double lr_power = 0.75;

    try
    {
        // first parse arguments from command line
        argparse::ArgumentParser program("lenet_exercize", "1.0", argparse::default_arguments::help);

        program.add_argument("--gpu").scan<'i', int>().store_into(gpu).help("GPU to run on");
        program.add_argument("--iterations").scan<'i', int>().store_into(iterations).help("Number of iterations for training");
        program.add_argument("--random_seed").scan<'i', int>().store_into(random_seed).help("Override random seed (default uses std::random_device)");
        program.add_argument("--classify").scan<'i', int>().store_into(classify).help("Number of images to classify to compute error rate (default uses entire test set)");
        program.add_argument("--batch_size").scan<'u', std::size_t>().store_into(classify).help("Batch size for training");

        program.add_argument("--pretrained").store_into(pretrained).help("Use the pretrained CUDNN model as input");
        program.add_argument("--save_data").store_into(save_data).help("Save pretrained weights to file");

        program.add_argument("--train_images").store_into(FLAGS_train_images).help("Training images filename");
        program.add_argument("--train_labels").store_into(FLAGS_train_labels).help("Training labels filename");
        program.add_argument("--test_images").store_into(FLAGS_test_images).help("Test images filename");
        program.add_argument("--test_labels").store_into(FLAGS_test_labels).help("Test labels filename");

        program.add_argument("--learning_rate").store_into(learning_rate).help("Base learning rate");
        program.add_argument("--lr_gamma").store_into(lr_gamma).help("Learning rate policy gamma");
        program.add_argument("--lr_power").store_into(lr_power).help("Learning rate policy power");

        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << "Program error! An exception occurred: \n" << err.what();
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred.\nAborting.";
    }

    size_t width, height, channels = 1;

    // Open input data
    std::cout << "Reading input data" << std::endl;

    // Read dataset sizes
    const size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), nullptr, nullptr, width, height);
    const size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), nullptr, nullptr, width, height);
    if (train_size == 0) {
        return 1;
    }

    std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
    std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);

    // Read data from datasets
    if (ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height) != train_size) {
        return 2;
    }
    if (ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height) != test_size) {
        return 3;
    }

    std::cout << "Done. Training dataset size: " << train_size << ", Test dataset size: " << test_size << std::endl;
    std::cout << "Batch size: " << batch_size << ", iterations : " << iterations << std::endl;

    // Choose GPU
    int num_gpus;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    if (gpu < 0 || gpu >= num_gpus) {
        std::cerr << "ERROR: Invalid GPU ID " << gpu << " (There are " << num_gpus << " GPUs on this machine)" << std::endl;
        return 4;
    }

    // Create the LeNet network architecture
    ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
    MaxPoolLayer pool1(2, 2);
    ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
    MaxPoolLayer pool2(2, 2);
    FullyConnectedLayer fc1((conv2.out_channels * conv2.out_width * conv2.out_height) / (pool2.stride * pool2.stride), 500);
    FullyConnectedLayer fc2(fc1.outputs, 10);

    // Initialize CUDNN/CUBLAS training context
    TrainingContext context(gpu, batch_size, conv1, pool1, conv2, pool2, fc1, fc2);

    // Determine initial network structure
    bool bRet = true;
    if (pretrained)
    {
        bRet = conv1.FromFile("conv1");
        bRet &= conv2.FromFile("conv2");
        bRet &= fc1.FromFile("ip1");
        bRet &= fc2.FromFile("ip2");
    }
    if (!bRet || !pretrained)
    {
        // Create random network
        std::random_device rd;
        std::mt19937 gen(random_seed < 0 ? rd() : static_cast<unsigned int>(random_seed));

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
        float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        // Randomize network
        for (auto &&iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto &&iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto &&iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto &&iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));
        for (auto &&iter : fc1.pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto &&iter : fc1.pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto &&iter : fc2.pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto &&iter : fc2.pbias)
            iter = static_cast<float>(dfc2(gen));
    }

    /////////////////////////////////////////////////////////////////////////////
    // Create GPU data structures

    // Forward propagation data
    //                         Buffer    | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    // checkCudaErrors(cudaMalloc(&d_data, sizeof(float) * context.m_batchSize * channels * height * width));
    cuda_ptr<float> d_data(context.m_batchSize * channels * height * width);
    cuda_ptr<float> d_labels(context.m_batchSize * 1 * 1 * 1);
    cuda_ptr<float> d_conv1(context.m_batchSize * conv1.out_channels * conv1.out_height * conv1.out_width);
    cuda_ptr<float> d_pool1(context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
    cuda_ptr<float> d_conv2(context.m_batchSize * conv2.out_channels * conv2.out_height * conv2.out_width);
    cuda_ptr<float> d_pool2(context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride));
    cuda_ptr<float> d_fc1(context.m_batchSize * fc1.outputs);
    cuda_ptr<float> d_fc1relu(context.m_batchSize * fc1.outputs);
    cuda_ptr<float> d_fc2(context.m_batchSize * fc2.outputs);
    cuda_ptr<float> d_fc2smax(context.m_batchSize * fc2.outputs);

    // Network parameters
    cuda_ptr<float> d_pconv1(conv1.pconv.size());
    cuda_ptr<float> d_pconv1bias(conv1.pbias.size());
    cuda_ptr<float> d_pconv2(conv2.pconv.size());
    cuda_ptr<float> d_pconv2bias(conv2.pbias.size());
    cuda_ptr<float> d_pfc1(fc1.pneurons.size());
    cuda_ptr<float> d_pfc1bias(fc1.pbias.size());
    cuda_ptr<float> d_pfc2(fc2.pneurons.size());
    cuda_ptr<float> d_pfc2bias(fc2.pbias.size());

    // Network parameter gradients
    cuda_ptr<float> d_gconv1(conv1.pconv.size());
    cuda_ptr<float> d_gconv1bias(conv1.pbias.size());
    cuda_ptr<float> d_gconv2(conv2.pconv.size());
    cuda_ptr<float> d_gconv2bias(conv2.pbias.size());
    cuda_ptr<float> d_gfc1(fc1.pneurons.size());
    cuda_ptr<float> d_gfc1bias(fc1.pbias.size());
    cuda_ptr<float> d_gfc2(fc2.pneurons.size());
    cuda_ptr<float> d_gfc2bias(fc2.pbias.size());

    // Differentials w.r.t. data
    //                         Buffer     | Element       | N                   | C                  | H                                 | W
    //-----------------------------------------------------------------------------------------------------------------------------------------
    cuda_ptr<float> d_dpool1(context.m_batchSize * conv1.out_channels * conv1.out_height * conv1.out_width);
    cuda_ptr<float> d_dpool2(context.m_batchSize * conv2.out_channels * conv2.out_height * conv2.out_width);
    cuda_ptr<float> d_dconv2(context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
    cuda_ptr<float> d_dfc1(context.m_batchSize * fc1.inputs);
    cuda_ptr<float> d_dfc1relu(context.m_batchSize * fc1.outputs);
    cuda_ptr<float> d_dfc2(context.m_batchSize * fc2.inputs);
    cuda_ptr<float> d_dfc2smax(context.m_batchSize * fc2.outputs);
    cuda_ptr<float> d_dlossdata(context.m_batchSize * fc2.outputs);

    // Temporary buffers and workspaces
    void *d_cudnn_workspace = nullptr;
    cuda_ptr<float> d_onevec(context.m_batchSize);
    if (context.m_workspaceSize > 0)
        checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));

    /////////////////////////////////////////////////////////////////////////////

    // Copy initial network to device
    checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0], sizeof(float) * conv1.pconv.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0], sizeof(float) * conv2.pconv.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0], sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0], sizeof(float) * fc1.pbias.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0], sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0], sizeof(float) * fc2.pbias.size(), cudaMemcpyHostToDevice));

    // Fill one-vector with ones
    FillOnes<<<RoundUp(context.m_batchSize, BW), BW>>>(d_onevec, context.m_batchSize);

    std::cout << "Preparing dataset" << std::endl;

    // Normalize training set to be in [0,1]
    std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
    for (size_t i = 0; i < train_size * channels * width * height; ++i)
        train_images_float[i] = (float)train_images[i] / 255.0f;

    for (size_t i = 0; i < train_size; ++i)
        train_labels_float[i] = (float)train_labels[i];

    std::cout << "Training..." << std::endl;

    // Use SGD to train the network
    checkCudaErrors(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        // Train
        int imageid = iter % (train_size / context.m_batchSize);

        // Prepare current batch on device
        checkCudaErrors(cudaMemcpyAsync(d_data(), &train_images_float[imageid * context.m_batchSize * width * height * channels],
                                        sizeof(float) * context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize],
                                        sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice));

        // Forward propagation
        context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                    d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                    d_cudnn_workspace, d_onevec);

        // Backward propagation
        context.Backpropagation(conv1, pool1, conv2, pool2,
                                d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias,
                                d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);

        // Compute learning rate
        float learningRate = static_cast<float>(learning_rate * pow((1.0 + lr_gamma * iter), (-lr_power)));

        // Update weights
        context.UpdateWeights(learningRate, conv1, conv2,
                                d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                                d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Iteration time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iterations)
        << " ms" << std::endl;

    if (save_data)
    {
        // Copy trained weights from GPU to CPU
        checkCudaErrors(cudaMemcpy(&conv1.pconv[0], d_pconv1, sizeof(float) * conv1.pconv.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv1.pbias[0], d_pconv1bias, sizeof(float) * conv1.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv2.pconv[0], d_pconv2, sizeof(float) * conv2.pconv.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&conv2.pbias[0], d_pconv2bias, sizeof(float) * conv2.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc1.pneurons[0], d_pfc1, sizeof(float) * fc1.pneurons.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc1.pbias[0], d_pfc1bias, sizeof(float) * fc1.pbias.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc2.pneurons[0], d_pfc2, sizeof(float) * fc2.pneurons.size(), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&fc2.pbias[0], d_pfc2bias, sizeof(float) * fc2.pbias.size(), cudaMemcpyDeviceToHost));

        // Now save data
        std::cout << "Saving data to file" << std::endl;
        conv1.ToFile("conv1");
        conv2.ToFile("conv2");
        fc1.ToFile("ip1");
        fc2.ToFile("ip2");
    }

    float classification_error = 1.0f;

    int classifications = classify;
    if (classifications < 0)
        classifications = (int)test_size;

    // Test the resulting neural network's classification
    if (classifications > 0)
    {
        // Initialize a TrainingContext structure for testing (different batch size)
        TrainingContext test_context(gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);

        // Ensure correct workspaceSize is allocated for testing
        if (context.m_workspaceSize < test_context.m_workspaceSize)
        {
            checkCudaErrors(cudaFree(d_cudnn_workspace));
            checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
        }

        int num_errors = 0;
        for (int i = 0; i < classifications; ++i)
        {
            std::vector<float> data(width * height);
            // Normalize image to be in [0,1]
            for (unsigned j = 0; j < width * height; ++j)
                data[j] = (float)test_images[i * width * height * channels + j] / 255.0f;

            checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));

            // Forward propagate test image
            test_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
                                            d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);

            // Perform classification
            std::vector<float> class_vec(10);

            // Copy back result
            checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost));

            // Determine classification according to maximal response
            int chosen = 0;
            for (int id = 1; id < 10; ++id)
            {
                if (class_vec[chosen] < class_vec[id])
                    chosen = id;
            }

            if (chosen != test_labels[i])
                ++num_errors;
        }
        classification_error = (float)num_errors / (float)classifications;

        std::cout << "Classification result: " << (classification_error * 100.0f) << "%% error (used "
            << classifications << " images)" << std::endl;
    }

    if (d_cudnn_workspace != nullptr)
        checkCudaErrors(cudaFree(d_cudnn_workspace));

    return 0;
}
