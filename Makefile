CXX = nvcc

.SUFFIXES: .o .cpp .h .cu .cu.o .cpp.o .d

CPPFLAGS := --std c++17 -Wno-deprecated-gpu-targets -I/usr/local/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include -MMD -MP
LDLIBS := -lcudnn -lcublas -lcuda

sources := ConvBiasLayer.cpp FullyConnectedLayer.cpp readubyte.cpp lenet.cu TrainingContext.cu
objs := $(addprefix build/, $(addsuffix .o, $(sources)))
deps = $(objs:%.o=%.d)

exe_name := build/lenet_example.exe

all: $(exe_name)

-include $(deps)

build:
	mkdir -p build

build/%.cu.o : %.cu | build
	$(CXX) $< -c -o $@ $(CPPFLAGS)

build/%.cpp.o : %.cpp | build
	$(CXX) $< -c -o $@ $(CPPFLAGS)

$(exe_name): $(objs) | build
	$(CXX) $^  -o $@ $(LDLIBS)

run:
	./$(exe_name) $(ARGS)

clean:
	rm -rf build
