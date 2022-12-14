/*
 * Copyright (C) 2022  Paul Seyfert
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#include <execution>
#include <numeric>
#include "shared.h"

#ifndef __NVCOMPILER

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

#else

score_t part1(const container_t& data) {
  using namespace unprocessed;

  std::vector<data_t> A(data.size());
  std::vector<data_t> B(data.size());
  std::vector<data_t> C(data.size());

  auto leq = [](const auto lhs, const auto rhs) {
    if (lhs <= rhs)
      return 1;
    else
      return 0;
  };
  auto and_ = [](auto a, auto b) {
    if (a && b)
      return 1;
    else
      return 0;
  };
  auto or_ = [](auto a, auto b) {
    if (a || b)
      return 1;
    else
      return 0;
  };

  std::transform(
      std::execution::par_unseq, data.template begin<start_a>(),
      data.template end<start_a>(), data.template begin<start_b>(), A.begin(),
      leq);
  std::transform(
      std::execution::par_unseq, data.template begin<end_b>(),
      data.template end<end_b>(), data.template begin<end_a>(), B.begin(), leq);

  std::transform(
      std::execution::par_unseq, A.begin(), A.end(), B.begin(), A.begin(),
      and_);

  std::transform(
      std::execution::par_unseq, data.template begin<start_b>(),
      data.template end<start_b>(), data.template begin<start_a>(), B.begin(),
      leq);
  std::transform(
      std::execution::par_unseq, data.template begin<end_a>(),
      data.template end<end_a>(), data.template begin<end_b>(), C.begin(), leq);

  std::transform(
      std::execution::par_unseq, B.begin(), B.end(), C.begin(), B.begin(),
      and_);

  return std::transform_reduce(
      std::execution::par_unseq, A.begin(), A.end(), B.begin(),
      static_cast<score_t>(0), std::plus{}, or_);
}

#endif
