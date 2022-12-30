/*
 * Copyright (C) 2022  <name of copyright holder>
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#include <fstream>
#include <iostream>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/split.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include "data_format.h"

std::tuple<container_t, std::vector<Monkey>> input(
    const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);

  container_t retval;
  std::vector<Monkey> monkeys;

  ranges::for_each(
      ranges::getlines_view(instream) | ranges::view::chunk(7),
      [&monkeys, &retval](const auto monkey_data) {
        std::size_t this_monkey_id = monkeys.size();
        auto iter = monkey_data.begin();
        const auto header_line = *iter++;
        ranges::for_each(
            (*iter++) | ranges::view::drop(18) |
                ranges::view::filter([](auto c) { return c != ' '; }) |
                ranges::view::split(',') |
                ranges::view::transform([](auto number) {
                  return std::stoi(ranges::to<std::string>(number));
                }),
            [this_monkey_id, &retval](const auto initial_worry) {
              retval.emplace_back(initial_worry, this_monkey_id);
            });
        const auto operator_ =
            (*iter++) | ranges::view::drop(23) | ranges::to<std::string>();

        auto oper = [&operator_, &monkey_data]() {
          using ret_type = std::tuple<operation_t, worry_t>;
          auto just_the_arg = operator_ | ranges::view::drop(2);
              

          switch (*(operator_.begin())) {
            case '+':
              return ret_type(std::plus<worry_t>{}, std::stoi(just_the_arg | ranges::to<std::string>()));
            case '*': {
              auto it = operator_.begin();
              it++;
              it++;
              if (*it == 'o') {
                return ret_type(power<worry_t>{}, 2);
              } else {
                return ret_type(std::multiplies<worry_t>{}, std::stoi(just_the_arg | ranges::to<std::string>()));
              }
            }
          }
        }();

        const auto test = std::stoi(
            (*iter++) | ranges::view::drop(21) | ranges::to<std::string>());
        const auto true_branch = std::stoi(
            (*iter++) | ranges::view::drop(29) | ranges::to<std::string>());
        const auto false_branch = std::stoi(
            (*iter++) | ranges::view::drop(30) | ranges::to<std::string>());

        monkeys.push_back(Monkey{
            .self = this_monkey_id,
            .true_receiver = true_branch,
            .false_receiver = false_branch,
            .test_divisor = test,
            .operation = std::get<0>(oper),
            .operand = std::get<1>(oper),
            .inspections = 0ul
        });
      });
  return std::make_tuple(std::move(retval), std::move(monkeys));
}
