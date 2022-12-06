#pragma once
#include <filesystem>
#include <fstream>
#include <iostream>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/distance.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range/primitives.hpp>  // empty
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/sliding.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <string>

#ifdef PART1
constexpr std::size_t marker_size = 4;
#else
constexpr std::size_t marker_size = 14;
#endif

template <typename T>
struct warner;

int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }
  std::filesystem::path in_path(argv[1]);
  std::ifstream instream(in_path);

  // TODO:
  // can probably do fancier parallel algorithms by splitting into multiple
  // buffered subranges [begin, begin+chunksize+stride) [begin+chunksize,
  // begin+2chunksize+stride) but don't want to put in more time now.
  auto the_line =
      *(ranges::getlines_view(instream) | ranges::view::take(1)).begin() |
      ranges::to<std::string>();

  auto indexed_marker =
      *(ranges::view::enumerate(the_line | ranges::view::sliding(marker_size)) |
        ranges::view::filter([](auto indexed_quartet) {
          auto& quartet = std::get<1>(indexed_quartet);

          bool found_offender = !(ranges::empty(
              quartet | ranges::view::transform([&quartet](auto letter) {
                return ranges::count(quartet, letter);
              }) |
              ranges::view::filter([](auto count) { return count != 1; })));

          return !found_offender;
        }) |
        ranges::view::take(1))
           .begin();

  std::cout << std::get<0>(indexed_marker) + marker_size << '\n';

  return 0;
}
