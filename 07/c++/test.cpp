#include <range/v3/range/primitives.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "reinvent_dir.hpp"

TEST(somestuff, something) {
  std::vector<int> foo;

  static_assert(!back_aware_tag<std::vector<int>>::value);
  static_assert(std::is_same_v<

                std::conditional_t<
                    back_aware_tag<std::vector<int>>::value, double,
                    typename std::vector<int>::iterator>,
                std::vector<int>::iterator>);

  back_aware_iter<std::vector<int>> it{&foo, foo.begin()};

  EXPECT_TRUE(it == end_reached);
  foo.push_back(42);

  it = back_aware_iter{&foo, foo.begin()};

  auto std_it = foo.begin();
  for (;;) {
    static_assert(std::is_same_v<decltype(*it), decltype(*std_it)>);
    static_assert(std::is_same_v<decltype(it.m_iter), decltype(std_it)>);
    static_assert(std::is_same_v<int, std::decay_t<decltype(*std_it)>>);
    EXPECT_EQ(it.m_iter, std_it);
    EXPECT_TRUE(&*(it.m_iter) == &*(std_it));
    EXPECT_TRUE(it.m_iter == std_it);
    EXPECT_EQ(*it, *std_it);
    it++;
    std_it++;

    EXPECT_TRUE(it == end_reached);
    if (it == end_reached) {
      break;
    }
  }
}

TEST(more, stuff) {
  back_aware_vector<std::string> stuff;
  EXPECT_TRUE(std::ranges::range<back_aware_vector<std::string>>);
  EXPECT_TRUE(std::ranges::sized_range<back_aware_vector<std::string>>);
  EXPECT_TRUE(std::ranges::forward_range<back_aware_vector<std::string>>);
  EXPECT_TRUE(std::ranges::random_access_range<back_aware_vector<std::string>>);
  EXPECT_TRUE(std::ranges::contiguous_range<back_aware_vector<std::string>>);

  stuff.push_back("foo");
  stuff.push_back("bar");

  for (auto x : stuff) {
    static_assert(std::is_same_v<std::decay_t<decltype(x)>, std::string>);
    EXPECT_TRUE(x.size() == 3);
  }
  auto it = stuff.begin();
  static_assert(std::is_same_v<
                decltype(it), back_aware_iter<back_aware_vector<std::string>>>);
  EXPECT_EQ(*it++, std::string{"foo"});
  EXPECT_EQ(*it++, std::string{"bar"});
  EXPECT_TRUE(it == stuff.end());
  EXPECT_TRUE(it == end_reached);

  it = stuff.begin();
  EXPECT_TRUE(it + 2 == end_reached);
  it += 1;
  EXPECT_EQ(it - stuff.begin(), 1);
}

TEST(range, back_aware) {
  back_aware_vector<double> d;
  d.push_back(3.45);
  d.push_back(0.05);
  d.push_back(3.45);

  auto dd = d | ranges::view::transform([](auto x) { return x * 10; }) |
            ranges::view::filter([](auto x) { return x > 10; });

  EXPECT_EQ(ranges::distance(dd), 2);
  for (auto x : dd) {
    EXPECT_EQ(x, 3.45 * 10);
  }
}

TEST(paths, basics) {
  dir root("/");
  root.m_subdirs.push_back(dir{"A"});
  root.m_subdirs.back().m_subdirs.push_back(dir{"a"});
  root.m_subdirs.push_back(dir{"B"});

  path pit{root.m_subdirs.begin()};

  EXPECT_EQ((std::string)pit++, std::string{"A"});
  EXPECT_EQ((std::string)pit++, std::string{"A/a"});
  EXPECT_EQ((std::string)pit++, std::string{"B"});
  EXPECT_TRUE(pit == path_end);
}
