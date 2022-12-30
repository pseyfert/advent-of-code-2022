DEPDIR := .deps

CPPFLAGS=-isystem /home/pseyfert/coding/explore-SOAContainer/install/include -isystem /home/pseyfert/coding/explore-rangev3/install-range-v3/include -I../../utils/include -isystem /home/pseyfert/coding/explore-fmt/install/include

all: main
main: main.o read_input.o

### G++
CXXFLAGS=-Wall -Wextra --std=c++20 -O3 -g -march=native
LDLIBS=-L/home/pseyfert/coding/explore-rangev3/install-range-v3/lib -ltbb -L/home/pseyfert/coding/explore-fmt/install/lib -lfmt
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d


# common

main : CC = ${CXX}

# delete builtin. Otherwise %: $.cpp with (LINK.cpp) kicks in
% : %.cpp

%.o : %.cpp
%.o : %.cpp Makefile $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

# copied from https://make.mad-scientist.net/papers/advanced-auto-dependency-generation/

SRCS  := $(wildcard *.cc *.cpp)

# copied from default `make -p`
COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))

clean:
	rm -f main *.o

.PHONY: clean all
