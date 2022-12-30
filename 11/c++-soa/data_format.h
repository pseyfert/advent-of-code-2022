#pragma once
/*
 * Copyright (C) 2022  Paul Seyfert
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <tuple>
#include <variant>
#include <vector>
#include "SOAContainer.h"

using worry_t = std::uint64_t;

struct Monkey;

namespace items {
SOAFIELD_TRIVIAL(worries, worries, worry_t);
SOAFIELD_TRIVIAL(owner, owner, std::size_t);
SOASKIN_TRIVIAL(items, worries, owner);
}  // namespace items

using container_t = typename SOA::Container<std::vector, items::items>;

std::tuple<container_t, std::vector<Monkey>> input(
    const std::filesystem::path& inpath);

template <typename T>
struct power {
  T operator()(const T& base, const T& exponent) const {
    return std::pow(base, exponent);
  }
};

using operation_t =
    std::variant<std::plus<worry_t>, std::multiplies<worry_t>, power<worry_t>>;

struct Monkey {
  std::size_t self;
  std::size_t true_receiver;
  std::size_t false_receiver;
  worry_t test_divisor;
  operation_t operation;
  worry_t operand;
  std::size_t inspections;
};
