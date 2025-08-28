# LeNet in CUDNN

This is an implementation of a classifier very similar to [LeNet](https://en.wikipedia.org/wiki/LeNet), the groundbraking classifier for digits designed by LeCun and others. It starts from the code in https://github.com/tbennun/cudnn-training and implements several improvements:

1. the original version runs on CUDNN 6 and does not compile on recent (8+) CUDNN versions; this one does (tested with CUDNN 8.2)
2. modular code design, with layers and functionalities clearly split into different files
3. polished code, using C++ features and algorithms and several convenience classes to abstract functionalities
4. compilation infrastructure that build, download the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) for you and runs a tets version

You can read [here](https://sh-tsang.medium.com/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17) for a list of LeNet versions with their architectures and [the proceedings paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) for the full descrition of LeNet 5 from the authors.

## Quickstart

Relevant make targets are

1. `build/lenet_example.exe` to build the executable
2. `mnist` to download the MNIST dataset
3. `run` to run the executable (also builds and downloads mnist)
4. `help` to see the help from the executable with command line options and their default parameters
5. `clean` to remove the built binaries
6. `distclean` to remove the built binaries and the mnist files

To build this code, you need a recent enough (8+) version of CUDA and the `nvcc` compiler.

`Makefile` variables of interest

1. `CXX` is currently `nvcc`
2. `CUDA_DIR`, currently `/usr/local/cuda` is the location of CUDA you are using

## Dependencies
A slightly modified version of [argparse](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) is included, with fixes to compile with `nvcc` version 11; this compiler is known to have a bug in inferring `std::tuple` template parameters leading to sudden crashes.
