#include <tuple>
#include "SOAContainer.h"
#include "aoc_utils/to.hpp"
#include "gtest/gtest.h"

struct Unconvertible {};

struct IFSConstructible {
  using tuple = std::tuple<int, float, Unconvertible>;
  IFSConstructible(int, float, Unconvertible) {}
};

TEST(soa, emplace) {
  IFSConstructible::tuple t;
  auto feed = std::vector{t};

  auto v = aoc_utils::to<std::vector<IFSConstructible>>(feed);
  auto w = aoc_utils::to<std::vector<IFSConstructible::tuple>>(feed);
  auto x = aoc_utils::to<std::vector<IFSConstructible>>(v);
}
