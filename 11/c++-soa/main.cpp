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
  std::vector<Monkey> monkeys_p1 = std::get<1>(data);
  std::vector<Monkey> monkeys_p2 = std::get<1>(data);
  container_t items_p1 = std::get<0>(data);
  container_t items_p2 = std::get<0>(data);

  // fmt::print("{}\n", items_p1);

  for ([[maybe_unused]] auto round : ranges::view::indices(20)) {
    for (auto& monkey : monkeys_p1) {
      for (auto item : items_p1) {
        if (item.owner() == monkey.self) {
          monkey.inspections++;
          item.worries() = std::visit(
              [worry = item.worries(), operand = monkey.operand](
                  const auto& op) { return op(worry, operand); },
              monkey.operation);
          item.worries() /= 3;
          if (0 == (item.worries() % monkey.test_divisor)) {
            // fmt::print(
            //     "Monkey {} gives {} to {}\n", monkey.self, item.worries(),
            //     monkey.true_receiver);
            item.owner() = monkey.true_receiver;
          } else {
            // fmt::print(
            //     "Monkey {} gives {} to {}\n", monkey.self, item.worries(),
            //     monkey.false_receiver);
            item.owner() = monkey.false_receiver;
          }
        }
      }
      // fmt::print("{}\n", items_p1);
    }
    // fmt::print("After round {}:\n{}\n", round + 1, items_p1);
  }

  std::nth_element(
      std::execution::par_unseq, monkeys_p1.begin(), monkeys_p1.begin() + 2,
      monkeys_p1.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.inspections > rhs.inspections;
      });
  auto p1 = std::transform_reduce(
      std::execution::par_unseq, monkeys_p1.begin(), monkeys_p1.begin() + 2,
      static_cast<worry_t>(1), std::multiplies{},
      [](const auto& m) { return m.inspections; });

  fmt::print("part 1: {}\n", p1);

  auto global_divisor = std::transform_reduce(
      std::execution::par_unseq, monkeys_p2.begin(), monkeys_p2.end(),
      static_cast<worry_t>(1),
      [](const auto lhs, const auto rhs) { return std::lcm(lhs, rhs); },
      [](const auto& m) { return m.test_divisor; });

  for ([[maybe_unused]] auto round : ranges::view::indices(10000)) {
    for (auto& monkey : monkeys_p2) {
      for (auto item : items_p2) {
        if (item.owner() == monkey.self) {
          monkey.inspections++;
          item.worries() = std::visit(
              [worry = item.worries(), operand = monkey.operand](
                  const auto& op) { return op(worry, operand); },
              monkey.operation);
          item.worries() %= global_divisor;
          if (0 == (item.worries() % monkey.test_divisor)) {
            // fmt::print(
            //     "Monkey {} gives {} to {}\n", monkey.self, item.worries(),
            //     monkey.true_receiver);
            item.owner() = monkey.true_receiver;
          } else {
            // fmt::print(
            //     "Monkey {} gives {} to {}\n", monkey.self, item.worries(),
            //     monkey.false_receiver);
            item.owner() = monkey.false_receiver;
          }
        }
      }
      // fmt::print("{}\n", items_p1);
    }
    // fmt::print("After round {}:\n{}\n", round + 1, items_p1);
  }

  for (const auto& m: monkeys_p2) {
    fmt::print("Monkey {} inspected items {} times.\n", m.self, m.inspections);
  }

  std::nth_element(
      std::execution::par_unseq, monkeys_p2.begin(), monkeys_p2.begin() + 2,
      monkeys_p2.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.inspections > rhs.inspections;
      });
  auto p2 = std::transform_reduce(
      std::execution::par_unseq, monkeys_p2.begin(), monkeys_p2.begin() + 2,
      static_cast<worry_t>(1), std::multiplies{},
      [](const auto& m) { return m.inspections; });

  fmt::print("part 2: {}\n", p2);
}
