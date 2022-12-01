#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <string_view>
#include <vector>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/chunk_by.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/transform.hpp>
#include "range/v3/range/conversion.hpp"

void all(const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);

  // would prefer not to dump this into a vector.
  auto data = ranges::getlines_view(instream) | ranges::to_vector;

  std::array<int, 3> dest;
  std::ranges::partial_sort_copy(
      data | ranges::views::chunk_by([](auto, auto s) { return !s.empty(); }) |
          ranges::views::transform([](auto inner) {
            return ranges::accumulate(
                inner | ranges::views::filter([](auto str) {
                  return !str.empty();
                }) | ranges::views::transform([](auto str) {
                  return std::atoi(str.c_str());
                }),
                0, std::plus());
          }),
      dest, std::greater{});

  printf("part 1 %d\n", dest[0]);
  printf("part 2 %d\n", ranges::accumulate(dest, 0, std::plus()));
  return;
}

int main(int argc, char** argv) {
  all(argv[1]);
  return 0;
}
