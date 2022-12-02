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

namespace unprocessed {
SOAFIELD_TRIVIAL(opponent, opponent, short);
SOAFIELD_TRIVIAL(self, self, short);
SOASKIN_TRIVIAL(skin, opponent, self);
}  // namespace unprocessed

auto input(const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);

  SOA::Container<std::vector, unprocessed::skin> retval;

  ranges::for_each(ranges::getlines_view(instream), [&retval](const auto line) {
    auto opponent_move = [](const auto& letter) -> short {
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
    auto self_move = [](const auto& letter) -> short {
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

template <typename T>
auto compute_score(T& input) {
  // NB: input.data() doesn't exist upstream. I just added `auto& data() {
  // return this->m_storage; } to SOAContainer.h.
  using soa_t = std::decay_t<T>;
  return std::transform_reduce(
      std::execution::par_unseq,
      // std::execution::seq,
      // first1
      std::get<(soa_t::template memberno<unprocessed::opponent>())>(
          input.data())
          .begin(),
      // last1
      std::get<(soa_t::template memberno<unprocessed::opponent>())>(
          input.data())
          .end(),
      // first2
      std::get<(soa_t::template memberno<unprocessed::self>())>(input.data())
          .begin(),
      // init
      0,
      //
      // reduce
      std::plus(),

      //
      // transform
      [](auto other, auto self) {
        if (other == self) {
          // printf("tie + %d\n", self);
          return 3 + self;
        } else if ((other % 3) == ((self + 1) % 3)) {
          // printf("loss against %d + %d\n", other, self);
          return 0 + self;
        } else if (((other + 1) % 3) == (self % 3)) {
          // printf("win + %d\n", self);
          return 6 + self;
        } else {
          printf("ERROR\n");
          return -1000;
        }
      });
}

int main(int argc, char** argv) {
  auto d = input(argv[1]);
  auto part1 = compute_score(d);

  printf("part 1: %d\n", part1);

  return 0;
}
