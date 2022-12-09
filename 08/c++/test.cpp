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

  for (auto i = 0; i < 5; ++i) {
    EXPECT_EQ(view_northwards(forest, i, 0), 0);
  }
  EXPECT_EQ(view_northwards(forest, 0, 1), 1);
  EXPECT_EQ(view_northwards(forest, 1, 1), 1);
  EXPECT_EQ(view_northwards(forest, 2, 1), 1);
  EXPECT_EQ(view_northwards(forest, 3, 1), 1);
  EXPECT_EQ(view_northwards(forest, 4, 1), 1);
  EXPECT_EQ(view_northwards(forest, 3, 4), 4);
  EXPECT_EQ(view_northwards(forest, 4, 3), 3);

  EXPECT_EQ(score(forest, 2, 1), 4);
  EXPECT_EQ(score(forest, 2, 3), 8);

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
  EXPECT_EQ(thespan(0, 0), 3);
  EXPECT_EQ(thespan(0, 1), 0);
  EXPECT_EQ(thespan(0, 2), 3);
  EXPECT_EQ(thespan(0, 3), 7);
  EXPECT_EQ(thespan(0, 4), 3);
  EXPECT_EQ(thespan(1, 0), 2);
  EXPECT_EQ(thespan(1, 1), 5);
  EXPECT_EQ(thespan(1, 2), 5);
  EXPECT_EQ(thespan(1, 3), 1);
  EXPECT_EQ(thespan(1, 4), 2);
}
