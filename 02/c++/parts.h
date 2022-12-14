/*
 * Copyright (C) 2022  <name of copyright holder>
 * Author: Paul Seyfert <pseyfert.mathphys@gmail.com>
 *
 * This software is distributed under the terms of the GNU General Public
 * Licence version 3 (GPL Version 3), copied verbatim in the file "LICENSE".
 */

#pragma once

#include "shared.h"

#include "SOAContainer.h"
#include <vector>

namespace unprocessed {
SOAFIELD_TRIVIAL(opponent, opponent, data_t);
SOAFIELD_TRIVIAL(self, self, data_t);
SOASKIN_TRIVIAL(skin, opponent, self);
}  // namespace unprocessed

int part1(const SOA::Container<std::vector, unprocessed::skin>& input);
int part2(const SOA::Container<std::vector, unprocessed::skin>& input);

