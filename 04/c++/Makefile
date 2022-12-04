DEPDIR := .deps

CPPFLAGS=-isystem /home/pseyfert/coding/explore-SOAContainer/install/include -isystem /home/pseyfert/coding/explore-rangev3/install-range-v3/include

all: main
main: main.o input.o part1.o part2.o

### NVC++
CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvc++
TARGET_ARCH=-gpu=ccnative --c++20 -acc=gpu -stdpar=gpu -Minfo=stdpar
# TARGET_ARCH=--c++20 -acc=multicore -O4 -Mvect=simd:256 -stdpar=multicore -Minfo=stdpar
DEPFLAGS = -MMD$(DEPDIR)/$*.d

### G++
# CXXFLAGS=-Wall -Wextra --std=c++20 -O3 -g -march=native
# LDLIBS=-L/home/pseyfert/coding/explore-rangev3/install-range-v3/lib -ltbb -L/home/pseyfert/coding/vcdevel/install/lib/ -lVc
# DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d


# common

main : CC = ${CXX}

%.o : %.cpp
%.o : %.cpp Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

# copied from https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/

SRCS  := $(wildcard *.cc *.cpp)

# copied from default `make -p`
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))
