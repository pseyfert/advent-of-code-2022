#include <gtest/gtest.h>
#include "main.h"

TEST(example, example) {
  auto in = input("../example.txt");

  auto& forest = in.second;

  EXPECT_EQ(in.first.size(), 25);
  ASSERT_TRUE(in.first.size() == 25);

  // auto x_range = ranges::view::indices(forest.extent(1));
  // auto y_range = ranges::view::indices(forest.extent(0));
  // auto index_pairs = ranges::view::cartesian_product(y_range, x_range);
  // for (const auto& idx_pair : index_pairs) {
  //   std::cout << '(' << std::get<0>(idx_pair) << ',' << std::get<1>(idx_pair)
  //             << ")\t"
  //             << visible(
  //                    forest, std::get<1>(idx_pair), std::get<0>(idx_pair))
  //             << '\n';
  // }

  EXPECT_EQ(part1(forest), 21);
}

TEST(example, part2_tests) {
  auto in = input("../example.txt");

  auto& forest = in.second;

  // 30373
  // 25512
  // 65332
  // 33549
  // 35390

  for (idx_t<x_tag> i{0}; i.data < 5; ++i) {
    EXPECT_EQ(view_northwards(forest, i, idx_t<y_tag>(0)), 0);
  }
  EXPECT_EQ(view_northwards(forest, idx_t<x_tag>(0), idx_t<y_tag>(1)), 1);
  EXPECT_EQ(view_northwards(forest, idx_t<x_tag>(1), idx_t<y_tag>(1)), 1);
  EXPECT_EQ(view_northwards(forest, idx_t<x_tag>(2), idx_t<y_tag>(1)), 1);
  EXPECT_EQ(view_northwards(forest, idx_t<x_tag>(3), idx_t<y_tag>(1)), 1);
  EXPECT_EQ(view_northwards(forest, idx_t<x_tag>(4), idx_t<y_tag>(1)), 1);
  EXPECT_EQ(view_northwards(forest, idx_t<x_tag>(3), idx_t<y_tag>(4)), 4);
  EXPECT_EQ(view_northwards(forest, idx_t<x_tag>(4), idx_t<y_tag>(3)), 3);

  EXPECT_EQ(score(forest, idx_t<x_tag>(2), idx_t<y_tag>(1)), 4);
  EXPECT_EQ(score(forest, idx_t<x_tag>(2), idx_t<y_tag>(3)), 8);

  EXPECT_EQ(part2(forest), 8);
}

TEST(example, flat_example) {
  auto in = input("../flat_example.txt");

  auto& thespan = in.second;

  EXPECT_EQ(in.first.size(), 10);
  ASSERT_TRUE(in.first.size() == 10);
  EXPECT_EQ(thespan.extent(0), 2);
  ASSERT_TRUE(thespan.extent(0) == 2);
  EXPECT_EQ(thespan.extent(1), 5);
  ASSERT_TRUE(thespan.extent(1) == 5);

  // codegenerated
  EXPECT_EQ(thespan(idx_t<y_tag>(0), idx_t<x_tag>(0)), 3);
  EXPECT_EQ(thespan(idx_t<y_tag>(0), idx_t<x_tag>(1)), 0);
  EXPECT_EQ(thespan(idx_t<y_tag>(0), idx_t<x_tag>(2)), 3);
  EXPECT_EQ(thespan(idx_t<y_tag>(0), idx_t<x_tag>(3)), 7);
  EXPECT_EQ(thespan(idx_t<y_tag>(0), idx_t<x_tag>(4)), 3);
  EXPECT_EQ(thespan(idx_t<y_tag>(1), idx_t<x_tag>(0)), 2);
  EXPECT_EQ(thespan(idx_t<y_tag>(1), idx_t<x_tag>(1)), 5);
  EXPECT_EQ(thespan(idx_t<y_tag>(1), idx_t<x_tag>(2)), 5);
  EXPECT_EQ(thespan(idx_t<y_tag>(1), idx_t<x_tag>(3)), 1);
  EXPECT_EQ(thespan(idx_t<y_tag>(1), idx_t<x_tag>(4)), 2);
}
