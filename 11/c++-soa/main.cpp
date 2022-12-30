/*
 * Copyright (C) 2022  <name of copyright holder>
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#include <algorithm>
#include <execution>
#include <fmt/core.h>
#include <range/v3/view/indices.hpp>
#include <vector>
#include "data_format.h"

int main(int argc, char** argv) {
  auto data = input(argv[1]);
  std::vector<Monkey> monkeys = std::get<1>(data);
  container_t items = std::get<0>(data);

  for ([[maybe_unused]] auto _ : ranges::view::indices(20)) {
    for (auto& monkey : monkeys) {
      for (auto item : items) {
        if (item.owner() == monkey.self) {
          monkey.inspections++;
          item.worries() = std::visit(
              [worry = item.worries(), operand = monkey.operand](
                  const auto& op) { return op(worry, operand); },
              monkey.operation);
          item.worries() /= 3;
          if (item.worries() % monkey.test_divisor) {
            item.owner() = monkey.true_receiver;
          } else {
            item.owner() = monkey.true_receiver;
          }
        }
      }
    }
  }

  std::nth_element(
      std::execution::par_unseq, monkeys.begin(), monkeys.begin() + 2,
      monkeys.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.inspections > rhs.inspections;
      });
  auto p1 = std::transform_reduce(
      std::execution::par_unseq, monkeys.begin(), monkeys.begin() + 2,
      static_cast<worry_t>(1), std::multiplies{},
      [](const auto& m) { return m.inspections; });

  fmt::print("part 1: {}\n", p1);
}
