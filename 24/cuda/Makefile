DEPDIR := .deps

all: main

NVCC=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvcc

CXX=g++-11

NVCCFLAGS=-ccbin=$(CXX) --gpu-code sm_75 --gpu-architecture compute_75 -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING -rdc=true --expt-relaxed-constexpr -O4 --extended-lambda

LDLIBS=-lcudadevrt

NVPPFLAGS=-I/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/include -I/home/pseyfert/coding/explore-rangev3/install-range-v3/include/

CPPFLAGS=-I/home/pseyfert/coding/explore-rangev3/install-range-v3/include/ -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include

CXXFLAGS=-Wall -Wextra -std=c++20

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

# round_test: LDLIBS+=-lgtest_main -lgtest
# round_test: round_test.cu.o workings.cu.o parse.cpp.o
# 	echo $(DEPFILES)
# 	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

main: walk.cu.o parse.cpp.o
	echo $(DEPFILES)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

%.cu.o : %.cu Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(NVCC) $(NVCCFLAGS) $(NVPPFLAGS) $< -c -o $@ $(CPPFLAGS) $(DEPFLAGS)

%.cpp.o : %.cpp Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ -c $(DEPFLAGS)


$(DEPDIR): ; @mkdir -p $@

CPPSRCS := $(wildcard *.cpp)
DEPFILES := $(CPPSRCS:%.cpp=$(DEPDIR)/%.d)
CUSRCS := $(wildcard *.cu)
DEPFILES += $(CUSRCS:%.cu=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))

clean:
	rm -f *_test main *.o

.PHONY: clean all
