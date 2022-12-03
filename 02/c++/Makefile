DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

CPPFLAGS=-I/home/pseyfert/coding/explore-SOAContainer/install/include -I/home/pseyfert/coding/explore-rangev3/install-range-v3/include

# CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvc++
# CXXFLAGS=-gpu=ccnative --c++20 -acc=gpu -stdpar=gpu -Minfo=stdpar

CXXFLAGS=-Wall -Wextra --std=c++20 -O3 -march=native -g
LDLIBS=-L/home/pseyfert/coding/explore-rangev3/install-range-v3/lib -ltbb

all: main

main: main.o parts.o

main : CC = ${CXX}

%.o : %.cpp
%.o : %.cpp Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<


parts.o : CXXFLAGS += -fopt-info-vec-missed

# copied from https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/

SRCS  := $(wildcard *.cc *.cpp)

# copied from default `make -p`
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))
