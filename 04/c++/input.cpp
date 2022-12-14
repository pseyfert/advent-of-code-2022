/*
 * Copyright (C) 2022  Paul Seyfert
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#include <charconv>
#include <fstream>
#include <range/v3/algorithm/for_each.hpp>
// #include <range/v3/range/conversion.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/split.hpp>
#include <range/v3/view/transform.hpp>
#include <stdexcept>
#include <string_view>
#include "aoc_utils/soa_support.hpp"
#include "aoc_utils/to.hpp"
#include "shared.h"

// NB: ranges aren't null-terminated.
auto parse_int(std::string_view s) {
  data_t retval;
  const auto parse_code = std::from_chars(s.begin(), s.end(), retval);
  if (parse_code.ec == std::errc::invalid_argument) {
    throw std::invalid_argument{"invalid_argument"};
  } else if (parse_code.ec == std::errc::result_out_of_range) {
    throw std::out_of_range{"out_of_range"};
  }
  return retval;
}

container_t input(const std::filesystem::path& in_path) {
  std::ifstream instream(in_path);

  // NB: since moving to aoc_utils, filling the container with nvc++ is broken.
  // NB2: fixed with
  // https://forums.developer.nvidia.com/t/compilation-failure-with-piped-ranges-in-range-v3-for-each-and-nvc-22-11/236350/2
  return aoc_utils::to<container_t>(
      ranges::getlines_view(instream) |
      ranges::view::transform([](const auto line) {
        // auto integered_team =
        auto rng = line | ranges::views::split(',') |
                   ranges::views::transform([](auto one_elf) {
                     auto rng = one_elf | ranges::views::split('-') |
                                ranges::views::transform([](auto section_str) {
                                  std::string_view section_str_view{
                                      &*section_str.begin(),
                                      static_cast<std::size_t>(
                                          ranges::distance(section_str))};

                                  return parse_int(section_str_view);
                                });  // range of two integers
                     auto it = rng.begin();
                     data_t s = *it++;
                     data_t e = *it;
                     return std::make_pair(s, e);
                   });  // range of two pairs
        auto it = rng.begin();
        std::pair<data_t, data_t> e1 = *it++;
        std::pair<data_t, data_t> e2 = *it;

        return std::make_tuple(e1.first, e1.second, e2.first, e2.second);
      }));
}
