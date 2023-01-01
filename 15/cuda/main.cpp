#include "helpers.cuh"
#include "read.h"

#include <execution>
#include <numeric>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

int main() {
  const auto data = read("../example.txt");

  auto coverages = data |
                   ranges::view::transform([](const SensorBeaconPair& sbl) {
                     return toCircle(sbl);
                   }) |
                   ranges::view::transform([](const Circle& c) {
                     return c.project(tagged_int<y_tag>(10));
                   }) |
                   ranges::to_vector;

  auto m = simplify(coverages);

  auto excluded = std::transform_reduce(
      std::execution::par_unseq, m.begin(), m.end(), std::int32_t(0),
      std::plus{},
      [](const x_interval inter) { return inter.second - inter.first; });

  std::cout << excluded << " positions are ruled out.\n";
  return 0;
}
