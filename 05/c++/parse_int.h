#pragma once
#include <string_view>
#include <stdexcept>
#include <charconv>

template <typename T>
auto parse_int(T&& rng) {
  auto sv = std::string_view(&*rng.begin(), ranges::distance(rng));
  std::size_t retval;
  const auto res = std::from_chars(sv.begin(), sv.begin() + sv.size(), retval);
  if (res.ec == std::errc::invalid_argument) {
    throw std::invalid_argument{"invalid_argument"};
  } else if (res.ec == std::errc::result_out_of_range) {
    throw std::out_of_range{"out_of_range"};
  }
  return retval;
}
