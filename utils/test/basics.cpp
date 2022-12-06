#include <tuple>
#include "SOAContainer.h"
#include "aoc_utils/to.hpp"
#include "gtest/gtest.h"

struct Unconvertible {};

struct IFSConstructible {
  using tuple = std::tuple<int, float, Unconvertible>;
  IFSConstructible(int, float, Unconvertible) {}
};

namespace soa {
SOAFIELD_TRIVIAL(i, i, int);
SOAFIELD_TRIVIAL(f, f, float);
SOAFIELD_TRIVIAL(u, u, Unconvertible);
SOASKIN_TRIVIAL(skin, i, f, u);
}  // namespace soa

TEST(std_tests, emplace_vector) {
  IFSConstructible::tuple t;
  auto feed = std::vector{t};

  auto v = aoc_utils::to<std::vector<IFSConstructible>>(feed);
  auto w = aoc_utils::to<std::vector<IFSConstructible::tuple>>(feed);
  auto x = aoc_utils::to<std::vector<IFSConstructible>>(v);
}

TEST(soa_test, emplace_from_tuple) {
  IFSConstructible::tuple t;
  auto feed = std::vector{t};

  auto soa_vector = aoc_utils::to<SOA::Container<std::vector, soa::skin>>(feed);

  // SOA::Container<std::vector, soa::skin> v;
  // for (auto& tt : feed) {
  //   v.emplace_back(tt);
  // }
}
