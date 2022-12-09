#pragma once
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

using myspan = std::experimental::mdspan<
    short, std::experimental::extents<
               size_t, std::experimental::dynamic_extent,
               std::experimental::dynamic_extent>>;

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

inline bool visible_from_east(myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_right = ranges::view::indices(idx + 1, forest.extent(1)) |
                     ranges::view::transform(
                         [&forest, idy](auto idx) { return forest(idy, idx); });
  return ranges::all_of(
      trees_right, [tree](auto other) { return other < tree; });
}

inline bool visible_from_west(myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_left = ranges::view::indices(idx) |
                    ranges::view::transform(
                        [&forest, idy](auto idx) { return forest(idy, idx); });
  return ranges::all_of(
      trees_left, [tree](auto other) { return other < tree; });
}

inline bool visible_from_north(
    myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_above = ranges::view::indices(idy) |
                     ranges::view::transform(
                         [&forest, idx](auto idy) { return forest(idy, idx); });
  return ranges::all_of(
      trees_above, [tree](auto other) { return other < tree; });
}

inline bool visible_from_south(
    myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_below = ranges::view::indices(idy + 1, forest.extent(0)) |
                     ranges::view::transform(
                         [&forest, idx](auto idy) { return forest(idy, idx); });
  return ranges::all_of(
      trees_below, [tree](auto other) { return other < tree; });
}

inline bool visible(myspan forest, std::size_t idx, std::size_t idy) {
  return visible_from_north(forest, idx, idy) ||
         visible_from_south(forest, idx, idy) ||
         visible_from_east(forest, idx, idy) ||
         visible_from_west(forest, idx, idy);
}

inline auto view_southwards(myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_going_south = ranges::view::indices(idy + 1, forest.extent(0));

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

inline auto view_northwards(myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_going_north = ranges::view::indices(idy) | ranges::view::reverse;
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

inline auto view_westwards(myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_going_west = ranges::view::indices(idx) | ranges::view::reverse;

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

inline auto view_eastwards(myspan forest, std::size_t idx, std::size_t idy) {
  short tree = forest(idy, idx);
  auto trees_going_east = ranges::view::indices(idx + 1, forest.extent(1));

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

inline auto score(myspan forest, std::size_t idx, std::size_t idy) {
  // std::cout << "north score\n" << view_northwards(forest, idx, idy) << '\n';
  // std::cout << "west score \n" << view_westwards(forest, idx, idy) << '\n';
  // std::cout << "south score\n" << view_southwards(forest, idx, idy) << '\n';
  // std::cout << "east score \n" << view_eastwards(forest, idx, idy) << '\n';
  return view_eastwards(forest, idx, idy) * view_westwards(forest, idx, idy) *
         view_northwards(forest, idx, idy) * view_southwards(forest, idx, idy);
}

// TODO: some tagged types to leaverage the typesystem to avoid x/y mixups would
// be nice.
inline auto part1(myspan forest) {
  auto x_range = ranges::view::indices(forest.extent(1));
  auto y_range = ranges::view::indices(forest.extent(0));
  auto index_pairs = ranges::view::cartesian_product(y_range, x_range);

  return ranges::count_if(index_pairs, [&forest](const auto& pair) {
    auto idx = std::get<1>(pair);
    auto idy = std::get<0>(pair);
    return visible(forest, idx, idy);
  });
}

inline auto part2(myspan forest) {
  auto x_range = ranges::view::indices(forest.extent(1));
  auto y_range = ranges::view::indices(forest.extent(0));
  auto index_pairs = ranges::view::cartesian_product(y_range, x_range);

  auto scores =
      index_pairs | ranges::view::transform([&forest](const auto& pair) {
        auto idx = std::get<1>(pair);
        auto idy = std::get<0>(pair);
        return score(forest, idx, idy);
      });
  return ranges::max(scores);
}
