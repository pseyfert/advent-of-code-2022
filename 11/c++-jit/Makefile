DEPDIR := .deps

CPPFLAGS=-isystem /home/pseyfert/coding/explore-rangev3/install-range-v3/include -I../../utils/include -isystem /home/pseyfert/coding/vcdevel/install/include/ -isystem /home/pseyfert/coding/explore-fmt/install/include

all: main

test: LDLIBS+=-lgtest_main -lgtest -pthread

### NVC++
# CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvc++
# TARGET_ARCH=-gpu=ccnative --c++20 -acc=gpu -stdpar=gpu -Minfo=stdpar
# # TARGET_ARCH=--c++20 -acc=multicore -O4 -Mvect=simd:256 -stdpar=multicore -Minfo=stdpar
# DEPFLAGS = -MMD$(DEPDIR)/$*.d

### G++
CXXFLAGS=-Wshadow -Wall -Wextra --std=c++20 -O3 -ggdb -march=native -fconcepts-diagnostics-depth=3 -fdiagnostics-color=always -fsanitize=address
LDLIBS=-L/home/pseyfert/coding/explore-rangev3/install-range-v3/lib -lfmt -L/home/pseyfert/coding/vcdevel/install/lib/ -lVc -L/home/pseyfert/coding/explore-fmt/install/lib -lfmt -fsanitize=address
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d


# common

%: CC=${CXX}

# delete builtin. Otherwise %: $.cpp with (LINK.cpp) kicks in
% : %.cpp

%.o : %.cpp
%.o : %.cpp Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

monkeys/%.o: CXX=ccache g++
monkeys/%.o: monkeys/%.cpp Makefile main $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) -fpic $<

# copied from https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/

SRCS := $(wildcard *.cc *.cpp)

# copied from default `make -p`
COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))

clean:
	rm -f main test *.o libmonkeys.so monkeys/*

.PHONY: clean all
