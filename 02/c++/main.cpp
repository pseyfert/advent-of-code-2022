/*
 * Copyright (C) 2022  Paul Seyfert
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#include "parts.h"
#include "shared.h"

#include <filesystem>
#include <fstream>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/getlines.hpp>
#include <ranges>
#include <stdio.h>

auto input(const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);

  SOA::Container<std::vector, unprocessed::skin> retval;

  ranges::for_each(ranges::getlines_view(instream), [&retval](const auto line) {
    auto opponent_move = [](const auto& letter) -> data_t {
      if (letter == 'A') {
        return 1;
      } else if (letter == 'B') {
        return 2;
      } else if (letter == 'C') {
        return 3;
      } else {
        printf("Error parsing other char: %s\n", letter);
        return -100;
      }
    };
    auto self_move = [](const auto& letter) -> data_t {
      if (letter == 'X') {
        return 1;
      } else if (letter == 'Y') {
        return 2;
      } else if (letter == 'Z') {
        return 3;
      } else {
        printf("Error parsing self char: %s\n", letter);
        return -100;
      }
    };
    retval.emplace_back(opponent_move(line[0]), self_move(line[2]));
  });

  return retval;
}

int main(int argc, char** argv) {
  auto d = input(argv[1]);
  auto part1_ = part1(d);

  printf("part 1: %d\n", part1_);

  auto part2_ = part2(d);

  printf("part 2: %d\n", part2_);

  return 0;
}
