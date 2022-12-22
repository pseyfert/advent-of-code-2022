#include <filesystem>
#include <fstream>
#include <iostream>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "try.h"

input_data read() {
  std::ifstream in_stream("../example.txt");
  auto input =
      ranges::getlines_view(in_stream) | ranges::to<std::vector<std::string>>();
  int COLS;
  int ROWS;
  ROWS = input.size();
  COLS = input[0].size();
  auto numberized =
      input | ranges::view::transform([](const std::string& line) {
        return line | ranges::view::transform([](const auto& letter) -> int {
                 if (letter == 'S')
                   return -10;
                 if (letter == 'E')
                   return 'z' - 'a';
                 return letter - 'a';
               });
      }) |
      ranges::view::join | ranges::to<std::vector<int>>();

  auto scores = ranges::view::transform(
                    numberized,
                    [](const int height) {
                      if (height == -10)
                        return 0;
                      else
                        return std::numeric_limits<int>::max();
                    }) |
                ranges::to<std::vector<int>>();

  for (auto& n : numberized) {
    if (n == -10)
      n = 0;
  }

  auto E_range = ranges::view::filter(
      ranges::view::transform(
          ranges::view::enumerate(input),
          [](auto arg) {
            auto& [row_idx, row] = arg;
            return std::make_tuple(
                row_idx, ranges::view::filter(
                             ranges::view::enumerate(row), [](auto arg) {
                               auto& [col_idx, character] = arg;
                               return character == 'E';
                             }));
          }),
      [](auto arg) {
        auto& [row_ids, filtered_enumerated_row] = arg;
        return ranges::distance(filtered_enumerated_row) == 1;
      });

  int goal_x, goal_y;
  for (auto arg : E_range) {
    auto& [row_idx, enumerated_row] = arg;
    for (auto arg : enumerated_row) {
      auto& [col_idx, character] = arg;
      goal_x = col_idx;
      goal_y = row_idx;
    }
  }

  return input_data{
      .heights = std::move(numberized),
      .scores = std::move(scores),
      .rows = ROWS,
      .cols = COLS,
      .goal_x = goal_x,
      .goal_y = goal_y};
}
