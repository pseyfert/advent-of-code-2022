#pragma once
#include <experimental/type_traits>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/concepts.hpp>
#include <range/v3/range/primitives.hpp>
#include <range/v3/range/traits.hpp>
#include <tuple>
#include <type_traits>
#include <utility>

namespace aoc_utils {

namespace detail {
template <typename T>
struct warner;

template <typename R, typename E>
using detect_push_back_t =
    typename std::void_t<decltype(std::declval<R>().push_back(
        std::declval<E>()))>;

template <typename R, typename... E>
using detect_emplace_back_t =
    typename std::void_t<decltype(std::declval<R>().emplace_back(
        std::declval<E>()...))>;

template <typename C, typename... T>
struct to_write_two {
  using emplaced_type =
      decltype(std::declval<C>().emplace_back(std::declval<T>()...));
  constexpr static bool value =
      std::is_constructible_v<std::decay_t<emplaced_type>, T...>;
};

template <typename, typename>
struct to_write : std::false_type {};
template <typename C, typename... TUPLE>
struct to_write<C, std::tuple<TUPLE...>> : to_write_two<C, TUPLE...> {};

template <typename>
struct is_tuple : std::false_type {};
template <typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template <typename T>
concept Pushable = requires(T m) {
  m.push_back(*m.begin());
};
}  // namespace detail

// About the requires here:
// I think I'm generally fine with value type conversions,
// as long as push_back works.
// Except, for tuples there is special treatment (motivated by SOAContainer).
// Therefore, I only accept tuples if all goes through.
//
// The forwarding is somewhat guess work.
template <detail::Pushable OUT, typename IN>
requires(
    (!(detail::is_tuple<
         typename ranges::range_value_t<std::decay_t<IN>>>::value) ||
     (std::is_same_v<
         typename ranges::range_value_t<std::decay_t<OUT>>,
         typename ranges::range_value_t<
             std::decay_t<IN>>>)&&(requires(OUT & c1, IN in) {
       c1.push_back(*in.begin());
     }))) auto to(IN&& in) {
  OUT retval;
  if constexpr (ranges::sized_range<std::decay_t<IN>>) {
    retval.reserve(ranges::size(in));
  }
  ranges::for_each(std::forward<IN>(in), [&retval](auto&& el) {
    retval.push_back(std::forward<decltype(el)>(el));
  });
  return retval;
}

template <typename OUT, typename IN>
requires(
    ((detail::is_tuple<
         typename ranges::range_value_t<std::decay_t<IN>>>::value) &&
     !(std::is_same_v<
         typename ranges::range_value_t<std::decay_t<OUT>>,
         typename ranges::range_value_t<std::decay_t<IN>>>)&&detail::
         to_write<
             std::decay_t<OUT>,
             std::decay_t<ranges::range_value_t<std::decay_t<IN>>>>::
             value))  // avoid line join
    auto to(IN&& in) {
  // didn't manage the part below as requires :(
  OUT retval;
  if constexpr (ranges::sized_range<std::decay_t<IN>>) {
    retval.reserve(ranges::size(in));
  }
  ranges::for_each(std::forward<IN>(in), [&retval](auto&& el) {
    std::apply(
        [&retval](auto... x) { retval.emplace_back(x...); },
        std::forward<decltype(el)>(el));
  });
  return retval;
}
}  // namespace aoc_utils
