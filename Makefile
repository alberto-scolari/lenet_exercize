CXX = nvcc
CLANG_FORMAT_EXE ?= clang-format

.SUFFIXES: .o .cpp .h .cu .cu.o .cpp.o .d

CPPFLAGS := --std c++17 -Wno-deprecated-gpu-targets -I/usr/local/cuda/include \
	-I/usr/local/cuda/targets/x86_64-linux/include -isystem deps/argparse/ \
	-MMD -MP -Xcompiler -Wall -Xcompiler -Wextra
LDLIBS := -lcudnn -lcublas -lcuda

sources := lenet.cu ConvBiasLayer.cpp FullyConnectedLayer.cpp readubyte.cpp TrainingContext.cu common.cpp
objs := $(addprefix build/, $(addsuffix .o, $(sources)))
deps = $(objs:.o=.d)

exe_name := build/lenet_example.exe

mnist_bins := t10k-images-idx3-ubyte t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte
mnist_paths := $(addprefix deps/mnist/, $(mnist_bins))

.PHONY: all clean run clang-format

all: $(exe_name)

clang-format:
	@$(CLANG_FORMAT_EXE) -i $$(find . -iname '*.h' -o -iname '*.cpp' -o -iname '*.cu')

deps/mnist:
	mkdir -p deps/mnist

$(mnist_paths): | deps/mnist

deps/mnist/%: 
	wget https://github.com/harrypnh/lenet5-from-scratch/blob/main/dataset/MNIST/$%

mnist : $(mnist_paths)

-include $(deps)

build:
	mkdir -p build

build/%.cu.o : %.cu
	$(CXX) $< -c -o $@ $(CPPFLAGS)

build/%.cpp.o : %.cpp
	$(CXX) $< -c -o $@ $(CPPFLAGS)

$(objs): | build

$(exe_name): $(objs)
	$(CXX) $^ -o $@ $(LDLIBS)

run: mnist build
	./$(exe_name) $(ARGS)

help:
	./$(exe_name) --help

clean:
	rm -rf build
