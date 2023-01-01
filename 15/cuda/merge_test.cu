#include <gtest/gtest.h>
#include "helpers.cuh"

TEST(Interval, Simple) {
  {
    std::vector<std::optional<x_interval>> v{
        std::nullopt, std::nullopt, std::nullopt};
    EXPECT_TRUE(simplify(v).empty());
  }
  {
    std::vector<std::optional<x_interval>> v{
        std::make_optional(x_interval{
            tagged_int<x_tag>(9), tagged_int<x_tag>(10)
        }),
        std::make_optional(x_interval{
            tagged_int<x_tag>(10), tagged_int<x_tag>(14)
        })};

    auto m = simplify(v);
    ASSERT_EQ(m.size(), 1);
    EXPECT_EQ(std::int32_t(m.front().first), 9);
    EXPECT_EQ(std::int32_t(m.front().second), 14);
  }
  {
    std::vector<std::optional<x_interval>> v{
        std::make_optional(x_interval{
            tagged_int<x_tag>(9), tagged_int<x_tag>(10)
        }),
        std::make_optional(x_interval{
            tagged_int<x_tag>(1), tagged_int<x_tag>(4)
        })};

    auto m = simplify(v);
    ASSERT_EQ(m.size(), 2);
    EXPECT_EQ(std::int32_t(m.front().first), 1);
    EXPECT_EQ(std::int32_t(m.front().second), 4);
    EXPECT_EQ(std::int32_t(m.back().first), 9);
    EXPECT_EQ(std::int32_t(m.back().second), 10);
  }
  {
    std::vector<std::optional<x_interval>> v{
        std::make_optional(x_interval{
            tagged_int<x_tag>(1), tagged_int<x_tag>(49)
        }),
        std::nullopt,
        std::make_optional(x_interval{
            tagged_int<x_tag>(3), tagged_int<x_tag>(19)
        }),
        std::nullopt};

    auto m = simplify(v);
    ASSERT_EQ(m.size(), 1);
    EXPECT_EQ(std::int32_t(m.front().first), 1);
    EXPECT_EQ(std::int32_t(m.front().second), 49);
  }
  {
    std::vector<std::optional<x_interval>> v{
        std::nullopt,
        std::make_optional(x_interval{
            tagged_int<x_tag>(3), tagged_int<x_tag>(19)
        }),
        std::nullopt};

    auto m = simplify(v);
    ASSERT_EQ(m.size(), 1);
    EXPECT_EQ(std::int32_t(m.front().first), 3);
    EXPECT_EQ(std::int32_t(m.front().second), 19);
  }

}
