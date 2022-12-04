#include <execution>
#include <numeric>
#include "shared.h"

#ifndef __NVCOMPILER

template <typename T>
score_t score2(T t) {
  if (((t.start_a() < t.start_b()) && (t.end_a() < t.start_b())) ||
      ((t.start_b() < t.start_a()) && (t.end_b() < t.start_a()))) {
    return 0;
  } else {
    return 1;
  }
}

score_t part2(const container_t& data) {
  return std::transform_reduce(
      std::execution::par_unseq, data.begin(), data.end(),
      static_cast<score_t>(0), std::plus{}, [](auto p) { return score2(p); });
}

#else

score_t part2(const container_t& data) {
  using namespace unprocessed;

  auto and_ = [](auto a, auto b) {
    if (a && b)
      return 1;
    else
      return 0;
  };
  auto nor_ = [](auto a, auto b) {
    if (a || b)
      return 0;
    else
      return 1;
  };

  std::vector<data_t> A(data.size());
  std::vector<data_t> B(data.size());
  std::vector<data_t> C(data.size());

  auto le = [](const auto lhs, const auto rhs) {
    if (lhs < rhs)
      return 1;
    else
      return 0;
  };
  std::transform(
      std::execution::par_unseq, data.template begin<start_a>(),
      data.template end<start_a>(), data.template begin<start_b>(), A.begin(),
      le);
  std::transform(
      std::execution::par_unseq, data.template begin<end_a>(),
      data.template end<end_a>(), data.template begin<start_b>(), B.begin(),
      le);

  std::transform(
      std::execution::par_unseq, A.begin(), A.end(), B.begin(), A.begin(),
      and_);

  std::transform(
      std::execution::par_unseq, data.template begin<start_b>(),
      data.template end<start_b>(), data.template begin<start_a>(), B.begin(),
      le);
  std::transform(
      std::execution::par_unseq, data.template begin<end_b>(),
      data.template end<end_b>(), data.template begin<start_a>(), C.begin(),
      le);

  std::transform(
      std::execution::par_unseq, B.begin(), B.end(), C.begin(), B.begin(),
      and_);

  return std::transform_reduce(
      std::execution::par_unseq, A.begin(), A.end(), B.begin(),
      static_cast<score_t>(0), std::plus{}, nor_);
}

#endif
