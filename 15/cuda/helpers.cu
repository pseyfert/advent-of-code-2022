#include <thrust/sort.h>
#include "helpers.cuh"

__host__ __device__ bool first_cmp(
    const std::optional<x_interval>& lhs,
    const std::optional<x_interval>& rhs) {
  if (lhs.has_value() && rhs.has_value())
    return lhs->first < rhs->first;
  if (lhs.has_value() && !rhs.has_value())
    return true;
  return false;
}

__host__ __device__ int32_t manhattan(const Location& a, const Location& b) {
  return std::abs(a.x - b.x) + std::abs(a.y - b.y);
}

__host__ __device__ Circle toCircle(const SensorBeaconPair& sb) {
  return Circle{.center = sb.S, .radius = manhattan(sb.S, sb.B)};
}
