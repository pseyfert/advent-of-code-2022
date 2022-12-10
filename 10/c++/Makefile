DEPDIR := .deps

CPPFLAGS=-isystem /home/pseyfert/coding/explore-rangev3/install-range-v3/include -I../../utils/include

all: main

test: LDLIBS+=-lgtest_main -lgtest -pthread

### NVC++
# CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvc++
# TARGET_ARCH=-gpu=ccnative --c++20 -acc=gpu -stdpar=gpu -Minfo=stdpar
# # TARGET_ARCH=--c++20 -acc=multicore -O4 -Mvect=simd:256 -stdpar=multicore -Minfo=stdpar
# DEPFLAGS = -MMD$(DEPDIR)/$*.d

### G++
CXXFLAGS=-Wall -Wextra --std=c++20 -O3 -g -march=native -fconcepts-diagnostics-depth=3 -fdiagnostics-color=always
LDLIBS=-L/home/pseyfert/coding/explore-rangev3/install-range-v3/lib
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d


# common

%: CC=${CXX}

# delete builtin. Otherwise %: $.cpp with (LINK.cpp) kicks in
% : %.cpp

%.o : %.cpp
%.o : %.cpp Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

# copied from https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/

SRCS := $(wildcard *.cc *.cpp)

# copied from default `make -p`
COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))

clean:
	rm -f main test *.o

.PHONY: clean all
