#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <meta/meta.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range_access.hpp>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/view.hpp>
#include <range/v3/view_facade.hpp>

// Copied from the example/calendar.cpp from range-v3
//
//  Copyright Eric Niebler 2013-present
//
//  Use, modification and distribution is subject to the
//  Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
// Project home: https://github.com/ericniebler/range-v3

template <class Rngs>
class interleave_view : public ranges::view_facade<interleave_view<Rngs>> {
  friend ranges::range_access;
  std::vector<ranges::range_value_t<Rngs>> rngs_;
  struct cursor;
  cursor begin_cursor() {
    return {
        0, &rngs_,
        ranges::views::transform(rngs_, ranges::begin) |
            ranges::to<std::vector>};
  }

 public:
  interleave_view() = default;
  explicit interleave_view(Rngs rngs)
      : rngs_(std::move(rngs) | ranges::to<std::vector>) {}
};

template <class Rngs>
struct interleave_view<Rngs>::cursor {
  std::size_t n_;
  std::vector<ranges::range_value_t<Rngs>>* rngs_;
  std::vector<ranges::iterator_t<ranges::range_value_t<Rngs>>> its_;
  decltype(auto) read() const {
    return *its_[n_];
  }
  void next() {
    if (0 == ((++n_) %= its_.size()))
      ranges::for_each(its_, [](auto& it) { ++it; });
  }
  bool equal(ranges::default_sentinel_t) const {
    if (n_ != 0)
      return false;
    auto ends = *rngs_ | ranges::views::transform(ranges::end);
    return its_.end() !=
           std::mismatch(
               its_.begin(), its_.end(), ends.begin(), std::not_equal_to<>{})
               .first;
  }
  CPP_member auto equal(cursor const& that) const -> CPP_ret(bool)(
      requires ranges::forward_range<ranges::range_value_t<Rngs>>) {
    return n_ == that.n_ && its_ == that.its_;
  }
};

auto interleave() {
  return ranges::make_view_closure([](auto&& rngs) {
    using Rngs = decltype(rngs);
    return interleave_view<ranges::views::all_t<Rngs>>(
        ranges::views::all(std::forward<Rngs>(rngs)));
  });
}

auto transpose() {
  return ranges::make_view_closure([](auto&& rngs) {
    using Rngs = decltype(rngs);
    CPP_assert(ranges::forward_range<Rngs>);
    return std::forward<Rngs>(rngs) | interleave() |
           ranges::views::chunk(
               static_cast<std::size_t>(ranges::distance(rngs)));
  });
}
