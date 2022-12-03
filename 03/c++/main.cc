#include <algorithm>
#include <array>
#include <cctype>
#include <compare>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <string_view>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/distance.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/slice.hpp>
#include <range/v3/view/subrange.hpp>
#include <range/v3/view/transform.hpp>

int letter_priority(char c) {
  if (islower(c)) {
    return c - '`';
  } else {
    return 27 + (c - 'A');
  }
}

auto part1(const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);

  auto data = ranges::getlines_view(instream);

  return ranges::accumulate(data, 0, std::plus(), [](auto line) {
    auto line_length = ranges::distance(line);
    auto compartment_1 =
        line | ranges::views::slice(
                   static_cast<decltype(line_length)>(0), line_length / 2);
    auto compartment_2 =
        line | ranges::views::slice(line_length / 2, line_length);
    std::ranges::sort(compartment_1);
    std::ranges::sort(compartment_2);
    // std::cout << compartment_1 << compartment_2 << '\n';
    auto it1 = compartment_1.begin();
    auto it2 = compartment_2.begin();
    while (true) {
      auto threeway = (*it1 <=> *it2);
      if (threeway == std::strong_ordering::equal) {
        auto score = letter_priority(*it1);
        // printf("scored = %d\n", score);
        return score;
      } else if (threeway == std::strong_ordering::less) {
        ++it1;
      } else if (threeway == std::strong_ordering::greater) {
        ++it2;
      } else {
        __builtin_unreachable();
      }

      if (it1 == compartment_1.end() || it2 == compartment_2.end()) {
        printf("ALARM\n");
        return -100;
      }
    }
  });
}

auto part2(const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);

  auto data = ranges::getlines_view(instream);

  constexpr auto group_size = 3;

  return ranges::accumulate(
      data | ranges::views::chunk(group_size), 0, std::plus(), [](auto group) {
        // for (auto group : data | ranges::views::chunk(group_size)) {
        auto g_iter = group.begin();

        // I *guess* I need to buffer the elves myself.
        auto elves = std::array{
            *(g_iter++),
            *(g_iter++),
            *(g_iter++),
        };

        ranges::for_each(elves, [](auto& elve) { std::ranges::sort(elve); });
        using iter_t = decltype(elves[0].begin());
        using array_t = typename std::array<iter_t, group_size>;

        array_t iters;

        // There has to be a better way for this.
        for (std::size_t i = 0; i < group_size; ++i) {
          iters[i] = elves[i].begin();
        }

        while (true) {
          if (*iters[0] == *iters[1] && *iters[1] == *iters[2]) {
            // std::cout << "common letter is " << *iters[0] << '\n';
            return letter_priority(*iters[0]);
          } else {
            (*std::ranges::min_element(
                iters, ranges::less{}, [](auto iter) { return *iter; }))++;
                // iters,
                // [](auto iter1, auto iter2) { return *iter1 < *iter2; }))++;
          }
          // Output during development. NB: << works better with
          // ranges::subrange than with std::subrange.

          // for (std::size_t i = 0; i < group_size; ++i) {
          //   std::cout << "elve " << i << ":\n";
          //   std::cout << elves[i] << '\n';
          //   std::cout << ranges::subrange(elves[i].begin(), iters[i]) << '\t'
          //             << ranges::subrange(iters[i], elves[i].end()) << '\n';
          // }
        }
      });
}

int main(int argc, char** argv) {
  printf("part1 = %d\n", part1(argv[1]));
  printf("part2 = %d\n", part2(argv[1]));
  // for (auto l : {'p', 'L', 'P', 'v', 't', 's'}) {
  //   std::cout << l << " evaluates to " << letter_priority(l) << '\n';
  // }
  return 0;
}
