#include <filesystem>
#include <fstream>
#include <iostream>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/distance.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/sliding.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <string>

template <typename T>
struct warner;

int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }
  std::filesystem::path in_path(argv[1]);
  std::ifstream instream(in_path);

  auto the_line =
      *(ranges::getlines_view(instream) | ranges::view::take(1)).begin() |
      ranges::to<std::string>();

  auto indexed_marker =
      *(ranges::view::enumerate(the_line | ranges::view::sliding(4)) |
        ranges::view::filter([](auto indexed_quartet) {
          auto& quartet = std::get<1>(indexed_quartet);

          // gets extremely lazy towards the end, count for every letter how
          // often it occurs, but take only the first count that isn't 1. If
          // there is such a count, we don't have the marker.
          bool found_offender =
              (1 ==
               ranges::distance(
                   quartet | ranges::view::transform([&quartet](auto letter) {
                     return ranges::count(quartet, letter);
                   }) |
                   ranges::view::filter([](auto count) { return count != 1; }) |
                   ranges::view::take(1)));

          return !found_offender;
        }) |
        ranges::view::take(1))
           .begin();

  std::cout << std::get<0>(indexed_marker) + 4 << '\n';

  return 0;
}
