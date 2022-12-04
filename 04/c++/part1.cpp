#include <execution>
#include <numeric>
#include "shared.h"

template <typename T>
score_t score1(T t) {
  if (((t.start_a() <= t.start_b()) && (t.end_a() >= t.end_b())) ||
      ((t.start_a() >= t.start_b()) && (t.end_a() <= t.end_b()))) {
    return 1;
  } else {
    return 0;
  }
}

score_t part1(const container_t& data) {
  return std::transform_reduce(
      std::execution::par_unseq, data.begin(), data.end(),
      static_cast<score_t>(0), std::plus{}, [](auto p) { return score1(p); });
}
