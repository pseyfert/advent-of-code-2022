/*
 * Copyright (C) 2022  Paul Seyfert
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#pragma once
#define HAS_SOA
#include "aoc_utils/to.hpp"
#include "SOAContainer.h"

namespace aoc_utils::detail {
template <
    template <typename...> class C, template <typename> class S, typename... F>
struct is_soa<SOA::Container<C, S, F...>> : std::true_type {};
template <typename T>
concept SOA = requires {
  is_soa<T>::value;
};
}  // namespace aoc_utils::detail
