#pragma once
#include <array>
#include <boost/container/static_vector.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "calendar_copy_and_paste.h"
#include "parse_int.h"

#include <meta/meta.hpp>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/drop_last.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/split.hpp>
#include <range/v3/view/stride.hpp>
#include <range/v3/view/subrange.hpp>
#include <range/v3/view/transform.hpp>

template <typename T>
void print_arena(const T& arena) {
  std::cout << "==========================\n";
  for (const auto& stack : arena) {
    std::cout << ranges::view::transform(stack, [](auto x) { return x; })
              << '\n';
  }
  std::cout << "--------------------------\n";
}

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

  for (auto cmd : ranges::subrange(split_point + 1, buffer.end())) {
    auto numit = (cmd | ranges::view::split(' ') | ranges::view::drop(1) |
                  ranges::view::stride(2))
                     .begin();
    // print_arena(arena);

    auto amount = parse_int(*(numit++));
    auto from = parse_int(*(numit++)) - 1;
    auto to = parse_int(*(numit++)) - 1;

#ifdef PART1
    arena[to].insert(
        arena[to].begin(), arena[from].rend() - amount, arena[from].rend());
#elif defined(PART2)
    arena[to].insert(
        arena[to].begin(), arena[from].begin(), arena[from].begin() + amount);
#else
#error "must be either part 1 or part 2"
#endif
    arena[from].erase(arena[from].begin(), arena[from].begin() + amount);
  }

  // print_arena(arena);

  auto tops =
      arena | ranges::view::filter([](auto stack) { return !stack.empty(); }) |
      ranges::view::transform([](auto stack) { return stack.front(); }) |
      ranges::to<std::string>();
  std::cout << tops << '\n';

  return 0;
}
