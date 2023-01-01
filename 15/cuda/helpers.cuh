#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <thrust/universal_vector.h>
#include <utility>
#include <vector>

struct x_tag {};
struct y_tag {};

template <typename T>
struct tagged_int {
  tagged_int() = delete;
  tagged_int(const tagged_int<T>&) = default;
  tagged_int(tagged_int<T>&&) = default;
  tagged_int& operator=(const tagged_int&) = default;
  tagged_int& operator=(tagged_int&&) = default;

  __host__ __device__ auto operator==(const tagged_int<T>& o) const {
    return m_data == o.m_data;
  }
  __host__ __device__ auto operator!=(const tagged_int<T>& o) const {
    return m_data != o.m_data;
  }
  __host__ __device__ auto operator<=(const tagged_int<T>& o) const {
    return m_data <= o.m_data;
  }
  __host__ __device__ auto operator<(const tagged_int<T>& o) const {
    return m_data < o.m_data;
  }
  __host__ __device__ auto operator>=(const tagged_int<T>& o) const {
    return m_data >= o.m_data;
  }
  __host__ __device__ auto operator>(const tagged_int<T>& o) const {
    return m_data > o.m_data;
  }

  __host__ __device__ explicit tagged_int<T>(std::int32_t i) : m_data{i} {}
  __host__ __device__ explicit operator std::int32_t() const {
    return m_data;
  }
  __host__ __device__ tagged_int<T> operator+(std::int32_t o) const {
    return tagged_int<T>(m_data + o);
  }
  __host__ __device__ tagged_int<T> operator-(std::int32_t o) const {
    return tagged_int<T>(m_data - o);
  }
  __host__ __device__ std::int32_t operator-(tagged_int<T> const& o) const {
    return m_data - o.m_data;
  }

 private:
  std::int32_t m_data;
};

namespace std {
__host__ __device__ inline void swap(
    tagged_int<x_tag>& a, tagged_int<x_tag>& b) {
  tagged_int<x_tag> c{a};
  a = b;
  b = c;
}
__host__ __device__ inline void swap(
    tagged_int<y_tag>& a, tagged_int<y_tag>& b) {
  tagged_int<y_tag> c{a};
  a = b;
  b = c;
}
}  // namespace std

struct Location {
  tagged_int<x_tag> x;
  tagged_int<y_tag> y;
};

struct SensorBeaconPair {
  Location S;
  Location B;
};

__host__ __device__ int32_t manhattan(const Location& a, const Location& b);

using x_interval = std::pair<tagged_int<x_tag>, tagged_int<x_tag>>;

struct Circle {
  Location center;
  std::int32_t radius;
  __host__ __device__ bool is_inside(const Location& point) const {
    return manhattan(center, point) <= radius;
  }
  __host__ __device__ std::optional<x_interval> project(
      tagged_int<y_tag> height) const {
    if (std::int32_t cross_section =
            radius - std::int32_t(std::abs(center.y - height));
        cross_section < 0) {
      return std::nullopt;
    } else {
      return x_interval(center.x - cross_section, center.x + cross_section);
    }
  }
};

__host__ std::vector<x_interval> simplify(
    const std::vector<std::optional<x_interval>>& cross_sections);

__host__ __device__ bool first_cmp(
    const std::optional<x_interval>&, const std::optional<x_interval>&);

__host__ __device__ inline Circle toCircle(const SensorBeaconPair& sb);
