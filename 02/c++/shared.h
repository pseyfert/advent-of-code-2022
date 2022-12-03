#include <algorithm>
#include <execution>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <ranges>
#include <stdio.h>
#include <vector>

#include "SOAContainer.h"

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/getlines.hpp>

#include "range/v3/range/conversion.hpp"

using data_t = uint16_t;

namespace unprocessed {
SOAFIELD_TRIVIAL(opponent, opponent, data_t);
SOAFIELD_TRIVIAL(self, self, data_t);
SOASKIN_TRIVIAL(skin, opponent, self);
}  // namespace unprocessed

int part1(const SOA::Container<std::vector, unprocessed::skin>& input);
int part2(const SOA::Container<std::vector, unprocessed::skin>& input);
