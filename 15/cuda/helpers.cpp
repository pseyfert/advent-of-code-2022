#include "helpers.cuh"
#include <aoc_utils/to.hpp>
#include <execution>
#include <optional>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>
#include <thrust/sort.h>

std::vector<x_interval> simplify(
    const std::vector<std::optional<x_interval>>& cross_sections) {
  auto good_cross_sections = aoc_utils::to<std::vector<x_interval>>(
      cross_sections |
      ranges::view::filter([](const auto& o) { return o.has_value(); }) |
      ranges::view::transform([](const auto& o) -> x_interval { return *o; }));

  std::vector<x_interval> merged_cross_sections;

  if (good_cross_sections.empty())
    return merged_cross_sections;

  std::sort(
      std::execution::par_unseq, good_cross_sections.begin(),
      good_cross_sections.end(), first_cmp);

  auto iter = good_cross_sections.begin() + 1;
  merged_cross_sections.push_back(*good_cross_sections.cbegin());

  for (; iter != good_cross_sections.cend(); ++iter) {
    if (iter->first > merged_cross_sections.back().second) {
      merged_cross_sections.push_back(*iter);
    } else {
      merged_cross_sections.back().second =
          std::max(iter->second, merged_cross_sections.back().second);
    }
  }

  return merged_cross_sections;
}
