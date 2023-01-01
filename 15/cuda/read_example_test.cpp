#include <gtest/gtest.h>
#include "read.h"

TEST(Read, Example) {
  auto data = read("../example.txt");

  ASSERT_EQ(data.size(), 14);
  EXPECT_EQ(std::int32_t(data[13].S.x), 20);
  EXPECT_EQ(std::int32_t(data[13].S.y), 1);
  EXPECT_EQ(std::int32_t(data[13].B.x), 15);
  EXPECT_EQ(std::int32_t(data[13].B.y), 3);
  EXPECT_EQ(std::int32_t(data[0].S.x), 2);
  EXPECT_EQ(std::int32_t(data[0].S.y), 18);
  EXPECT_EQ(std::int32_t(data[0].B.x), -2);
  EXPECT_EQ(std::int32_t(data[0].B.y), 15);
}
