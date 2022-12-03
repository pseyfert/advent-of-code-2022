#include "shared.h"

template <typename F>
auto compute_score(
    const SOA::Container<std::vector, unprocessed::skin>& input, F&& f) {
  using namespace unprocessed;
  return std::transform_reduce(
      // std::execution::par_unseq,
      std::execution::unseq,
#ifndef __NVCOMPILER
      // first1
      input.begin(),
      // last1
      input.end(),
#else
      // first1
      input.template begin<opponent>(),
      // last1
      input.template end<opponent>(),
      // first2
      input.template begin<self>(),
#endif
      // init
      0,
      // reduce
      std::plus(),
      // transform
      f);
}

int part1(const SOA::Container<std::vector, unprocessed::skin>& input) {
  return compute_score(
      input,
#ifndef __NVCOMPILER
      [](auto proxy) {
        auto o = proxy.opponent();
        auto s = proxy.self();
        if (o == s) {
          return 3 + s;
        } else if ((o % 3) == ((s + 1) % 3)) {
          return 0 + s;
        } else if (((o + 1) % 3) == (s % 3)) {
          return 6 + s;
#else
      [](auto other, auto self) {
        if (other == self) {
          // printf("tie + %d\n", self);
          return 3 + self;
        } else if ((other % 3) == ((self + 1) % 3)) {
          // printf("loss against %d + %d\n", other, self);
          return 0 + self;
        } else if (((other + 1) % 3) == (self % 3)) {
          // printf("win + %d\n", self);
          return 6 + self;
#endif
        } else {
          // printf("ERROR\n");
          return -1000;
        }
      });
}

int part2(const SOA::Container<std::vector, unprocessed::skin>& input) {
  return compute_score(
      input,
#ifndef __NVCOMPILER
      [](auto proxy) {
        if (proxy.self() == 2) {
          return 3 + proxy.opponent();
        } else if (proxy.self() == 1) {
          return (proxy.opponent() + 2 - 1) % 3 + 1;
        } else if (proxy.self() == 3) {
          return 6 + (proxy.opponent() + 1 - 1) % 3 + 1;
#else
      [](auto other, auto self) {
        if (self == 2) {
          // draw. own shape = other shape
          return 3 + other;
        } else if (self == 1) {
          // loos. self = other - 1
          return (other + 2 - 1) % 3 + 1;
        } else if (self == 3) {
          return 6 + (other + 1 - 1) % 3 + 1;
#endif
        } else {
          // printf("ERROR\n");
          return -1000;
        }
      });
}
