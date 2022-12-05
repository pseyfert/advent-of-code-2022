#include <array>
#include <boost/container/static_vector.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "calendar_copy_and_paste.h"

#include <meta/meta.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/drop_last.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/subrange.hpp>
#include <range/v3/view/transform.hpp>

int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }
  std::filesystem::path in_path(argv[1]);
  std::ifstream instream(in_path);

  auto buffer = ranges::getlines_view(instream) | ranges::to_vector;

  auto split_point =
      ranges::find_if(buffer, [](const auto line) { return line.empty(); });

  // there are less than 60 crates in total.
  std::array<boost::container::static_vector<char, 60>, 9> arena;

  for (auto p : ranges::view::enumerate(
           ranges::subrange(buffer.begin(), split_point) | transpose() |
           ranges::view::drop(1) | ranges::view::stride(4) |
           ranges::view::transform([](auto stack) {
             return stack |
                    ranges::view::filter([](auto c) { return c != ' '; }) |
                    ranges::view::drop_last(1);
           }))) {
    for (const auto c : std::get<1>(p)) {
      arena[std::get<0>(p)].push_back(c);
    }
  }

  return 0;
}
