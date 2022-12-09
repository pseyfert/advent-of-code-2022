#pragma once
#include <concepts>
#include <cstddef>
#include <experimental/mdspan>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/algorithm/count_if.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range/primitives.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/reverse.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include <utility>

using storage_t = std::vector<short>;

struct x_tag {};
struct y_tag {};

template <typename T>
concept rank = requires(T t) {
  std::is_same_v<T, x_tag> || std::is_same_v<T, y_tag>;
};


template <rank T>
struct idx_t {
  std::size_t data;
  template <std::integral I>
  explicit idx_t(I o) : data{o} {}
  explicit operator std::size_t() {
    return data;
  }
  idx_t& operator++() {
    ++data;
    return *this;
  }
};

template <rank T, std::integral I>
idx_t<T> operator+(idx_t<T> lhs, I rhs) {
  return idx_t<T>(lhs.data + rhs);
}

using underlying_span = std::experimental::mdspan<
    short, std::experimental::extents<
               size_t, std::experimental::dynamic_extent,
               std::experimental::dynamic_extent>>;

struct myspan {
  underlying_span m_data;
  myspan(underlying_span o) : m_data{o} {}

  auto operator()(idx_t<y_tag> idy, idx_t<x_tag> idx) {
    return m_data(static_cast<std::size_t>(idy), static_cast<std::size_t>(idx));
  }
  auto extent(std::size_t rank) {
    return m_data.extent(rank);
  }
  template <typename Arg1, typename Arg2, typename Arg3>
  myspan(Arg1&& arg1, Arg2&& arg2, Arg3&& arg3)
      : m_data{
            std::forward<Arg1>(arg1), std::forward<Arg2>(arg2),
            std::forward<Arg3>(arg3)} {}
};

using data_t = std::pair<storage_t, myspan>;

inline data_t input(const std::filesystem::path& infile) {
  std::ifstream in_stream{infile};

  std::optional<std::size_t> n_cols{std::nullopt};

  auto test_data =
      ranges::getlines_view(in_stream) |
      ranges::view::transform([&n_cols](const auto& one_row) {
        n_cols = ranges::size(one_row);
        return one_row | ranges::view::transform([](const auto& one_char) {
                 // probably horribly inefficient
                 return static_cast<short>(
                     std::atoi(std::string{one_char}.c_str()));
               });
      }) |
      ranges::view::join | ranges::to_vector;
  static_assert(std::is_same_v<decltype(test_data), storage_t>);

  return {
      std::move(test_data),
      myspan{test_data.data(), test_data.size() / (*n_cols), *n_cols}};
}

auto xrange_up(idx_t<x_tag> idx, myspan& forest) {
  return ranges::view::indices(idx.data + 1, forest.extent(1)) |
         ranges::view::transform(
             [](std::size_t untagged) { return idx_t<x_tag>(untagged); });
}
auto xrange_down(idx_t<x_tag> idx) {
  return ranges::view::indices(idx.data) |
         ranges::view::transform(
             [](std::size_t untagged) { return idx_t<x_tag>(untagged); });
}
auto yrange_up(idx_t<y_tag> idy, myspan& forest) {
  return ranges::view::indices(idy.data + 1, forest.extent(0)) |
         ranges::view::transform(
             [](std::size_t untagged) { return idx_t<y_tag>(untagged); });
}
auto yrange_down(idx_t<y_tag> idy) {
  return ranges::view::indices(idy.data) |
         ranges::view::transform(
             [](std::size_t untagged) { return idx_t<y_tag>(untagged); });
}

inline bool visible_from_east(
    myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_right = xrange_up(idx, forest) |
                     ranges::view::transform(
                         [&forest, idy](auto idx) { return forest(idy, idx); });
  return ranges::all_of(
      trees_right, [tree](auto other) { return other < tree; });
}

inline bool visible_from_west(
    myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_left =
      xrange_down(idx) | ranges::view::transform([&forest, idy](auto idx) {
        return forest(idy, idx);
      });
  return ranges::all_of(
      trees_left, [tree](auto other) { return other < tree; });
}

inline bool visible_from_north(
    myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_above =
      yrange_down(idy) | ranges::view::transform([&forest, idx](auto idy) {
        return forest(idy, idx);
      });
  return ranges::all_of(
      trees_above, [tree](auto other) { return other < tree; });
}

inline bool visible_from_south(
    myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_below = yrange_up(idy, forest) |
                     ranges::view::transform(
                         [&forest, idx](auto idy) { return forest(idy, idx); });
  return ranges::all_of(
      trees_below, [tree](auto other) { return other < tree; });
}

inline bool visible(myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  return visible_from_north(forest, idx, idy) ||
         visible_from_south(forest, idx, idy) ||
         visible_from_east(forest, idx, idy) ||
         visible_from_west(forest, idx, idy);
}

inline auto view_southwards(myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_going_south = yrange_up(idy, forest);

  auto view_blocker = ranges::find_if(
      trees_going_south,
      [&forest, idx, tree](auto idy) { return forest(idy, idx) >= tree; });

  decltype(ranges::distance(
      trees_going_south.begin(), view_blocker)) off_by_one;
  if (view_blocker != trees_going_south.end()) {
    off_by_one = 1;
  } else {
    off_by_one = 0;
  }

  return off_by_one + ranges::distance(trees_going_south.begin(), view_blocker);
}

inline auto view_northwards(myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_going_north = yrange_down(idy) | ranges::view::reverse;
  // std::cout << "investigating " <<  idy << ',' << idx << '\n';

  auto view_blocker = ranges::find_if(
      trees_going_north,
      [&forest, idx, tree](auto idy) { return forest(idy, idx) >= tree; });
  decltype(ranges::distance(
      trees_going_north.begin(), view_blocker)) off_by_one;
  if (view_blocker != trees_going_north.end()) {
    off_by_one = 1;
  } else {
    off_by_one = 0;
  }

  return off_by_one + ranges::distance(trees_going_north.begin(), view_blocker);
}

inline auto view_westwards(myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_going_west = xrange_down(idx) | ranges::view::reverse;

  auto view_blocker = ranges::find_if(
      trees_going_west,
      [&forest, idy, tree](auto idx) { return forest(idy, idx) >= tree; });
  decltype(ranges::distance(trees_going_west.begin(), view_blocker)) off_by_one;
  if (view_blocker != trees_going_west.end()) {
    off_by_one = 1;
  } else {
    off_by_one = 0;
  }

  return off_by_one + ranges::distance(trees_going_west.begin(), view_blocker);
}

inline auto view_eastwards(myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  short tree = forest(idy, idx);
  auto trees_going_east = xrange_up(idx, forest);

  auto view_blocker = ranges::find_if(
      trees_going_east,
      [&forest, idy, tree](auto idx) { return forest(idy, idx) >= tree; });
  decltype(ranges::distance(trees_going_east.begin(), view_blocker)) off_by_one;
  if (view_blocker != trees_going_east.end()) {
    off_by_one = 1;
  } else {
    off_by_one = 0;
  }

  return off_by_one + ranges::distance(trees_going_east.begin(), view_blocker);
}

inline auto score(myspan forest, idx_t<x_tag> idx, idx_t<y_tag> idy) {
  // std::cout << "north score\n" << view_northwards(forest, idx, idy) << '\n';
  // std::cout << "west score \n" << view_westwards(forest, idx, idy) << '\n';
  // std::cout << "south score\n" << view_southwards(forest, idx, idy) << '\n';
  // std::cout << "east score \n" << view_eastwards(forest, idx, idy) << '\n';
  return view_eastwards(forest, idx, idy) * view_westwards(forest, idx, idy) *
         view_northwards(forest, idx, idy) * view_southwards(forest, idx, idy);
}

auto xrange(myspan& forest) {
  return ranges::view::indices(forest.extent(1)) |
         ranges::view::transform(
             [](std::size_t untagged) { return idx_t<x_tag>(untagged); });
}
auto yrange(myspan& forest) {
  return ranges::view::indices(forest.extent(0)) |
         ranges::view::transform(
             [](std::size_t untagged) { return idx_t<y_tag>(untagged); });
}

// TODO: some tagged types to leaverage the typesystem to avoid x/y mixups would
// be nice.
inline auto part1(myspan forest) {
  auto x_range = xrange(forest);
  auto y_range = yrange(forest);
  auto index_pairs = ranges::view::cartesian_product(y_range, x_range);

  return ranges::count_if(index_pairs, [&forest](const auto& pair) {
    auto idx = std::get<1>(pair);
    auto idy = std::get<0>(pair);
    return visible(forest, idx, idy);
  });
}

inline auto part2(myspan forest) {
  auto x_range = xrange(forest);
  auto y_range = yrange(forest);
  auto index_pairs = ranges::view::cartesian_product(y_range, x_range);

  auto scores =
      index_pairs | ranges::view::transform([&forest](const auto& pair) {
        auto idx = std::get<1>(pair);
        auto idy = std::get<0>(pair);
        return score(forest, idx, idy);
      });
  return ranges::max(scores);
}
