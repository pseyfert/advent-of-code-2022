#include <execution>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <numeric>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/transform.hpp>
#include "aoc_utils/to.hpp"
#include "conv.h"

int main(int, char** argv) {
  std::ifstream in_stream{std::filesystem::path(argv[1])};
  auto input = aoc_utils::to<std::vector<unsigned long>>(
      ranges::getlines_view(in_stream) |
      ranges::view::transform([](const auto line) { return fromSNAFU(line); }));

  auto part1 =
      std::reduce(std::execution::par_unseq, input.begin(), input.end());

  fmt::print("{}\n", toSNAFU(part1));
}
