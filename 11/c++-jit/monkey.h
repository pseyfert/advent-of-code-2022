#pragma once
#include <Vc/Vc>
#include <bit>
#include <cstddef>
#include <fmt/core.h>
#include <fmt/format.h>
#include <immintrin.h>
#include <range/v3/view/enumerate.hpp>
#include <sstream>
#include <vector>
#include "blend1.h"
#include "blend2.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#include "/home/pseyfert/coding/simdprune/simdprune_tables.h"
#pragma GCC diagnostic pop

using monkey_ptr =
    std::add_pointer<std::tuple<__m256i, __m256i, std::size_t, std::size_t>(
        __m256i)>::type;

bool dbg{false};

struct Monkey {
  std::vector<__m256i> items;
  int actual_size{0};

  monkey_ptr oper{nullptr};
  int inspections{0};

  void perform_oper(std::vector<Monkey>& monkeys) {
    for (auto [i, item] : ranges::view::enumerate(items)) {
      /*std::tuple<__m256i, __m256i, std::size_t, std::size_t>*/ auto
          [vals, select, true_target, false_target] = oper(item);
      Vc::AVX2::int_v select_true =
          (Vc::AVX2::int_v{select} && Vc::AVX2::int_v(loop_mask(i))).dataI();
      // TODO: there's gotta be a better cast than !!
      Vc::AVX2::int_v select_false =
          ((!Vc::AVX2::int_v{select}) && (!!Vc::AVX2::int_v(loop_mask(i))))
              .dataI();
      monkeys[true_target].my_compress_store(vals, select_true.data());
      monkeys[false_target].my_compress_store(vals, select_false.data());
    }
    items.clear();
    inspections += actual_size;
    actual_size = 0;
  }

  void my_compress_store(const __m256i& other, const __m256i& mask) {
    uint8_t mask_i = _mm256_movemask_ps(_mm256_castsi256_ps(mask));

    auto to_insert = std::popcount(mask_i);

    // TODO: shouldn't the code be okay with that?
    // segfaulted, and debugger said that was the value of to_insert.
    if (to_insert == 0)
      return;
    int space_left;
    if (actual_size == 0) {
      space_left = 0;
    } else {
      space_left = 8 - (1 + ((actual_size - 1) % 8));
    }
    auto offset = (8 * (mask_i ^ 0xFF) - (actual_size % 8));
    const __m256i* lookups;
    if (offset < 0) {
      std::cout << "WATCH OUT!\n";
      // can use mask_i ^ 0x80 instead if space_left != 0
      // intentional shadowing
      auto offset = (8 * (mask_i ^ 0x7F) - (actual_size % 8));
      lookups = reinterpret_cast<const __m256i*>(mask256_epi32 + offset);
    } else {
      lookups = reinterpret_cast<const __m256i*>(mask256_epi32 + offset);
    }
    auto idxs = _mm256_loadu_si256(lookups);

    if (dbg) {
      fmt::print("offset = {}, mask_i = {}", offset, mask_i);
      std::cout << "\nidxs = " << Vc::AVX2::int_v(idxs);
    }

    __m256i compressed_buffer;
    if (space_left > 0) {
      compressed_buffer = _mm256_permutevar8x32_epi32(other, idxs);
      if (dbg) {
        std::cout << "\ncmpb = " << Vc::AVX2::int_v(compressed_buffer) << '\n';
        std::cout << "other = " << Vc::AVX2::int_v(other)
                  << "\nllm = " << Vc::AVX2::int_v(last_loop_mask()) << '\n';
        std::cout << "back(o) = " << Vc::AVX2::int_v(items.back());
      }
      items.back() =
          blendv_epi32_pp(compressed_buffer, items.back(), last_loop_mask());
      if (dbg) {
        std::cout << "\nback(n) = " << Vc::AVX2::int_v(items.back()) << '\n';
      }
    }

    if (to_insert > space_left) {
      // Restore the stupid offset < 0 handling above
      lookups = reinterpret_cast<const __m256i*>(mask256_epi32 + offset);
      __m256i idxs2;
      if (space_left > 0) {
        idxs2 = _mm256_loadu_si256(lookups + 1);
      } else {
        idxs2 = _mm256_loadu_si256(lookups);
      }

      __m256i remainders = _mm256_permutevar8x32_epi32(other, idxs2);
      if (dbg) {
        std::cout << "idxs2 = " << Vc::AVX2::int_v(idxs2)
                  << "\nother = " << Vc::AVX2::int_v(other)
                  << "\nrem = " << Vc::AVX2::int_v(remainders) << '\n';
      }
      items.push_back(remainders);
      if (dbg) {
        std::cout << "back(n) = " << Vc::AVX2::int_v(items.back()) << '\n';
      }
    }
    actual_size += to_insert;
  }

  __m256i loop_mask(int i) {
    if (i == items.size() - 1) {
      return last_loop_mask();
    } else {
      return Vc::AVX2::int_v(0xffffffff).data();
    }
  }

  __m256i last_loop_mask() {
    std::array<int, 8> a{0, 1, 2, 3, 4, 5, 6, 7};
    return (Vc::AVX2::int_v{a.data()} < (1 + ((actual_size - 1) % 8))).data();
  }
};

template <>
struct fmt::formatter<Monkey> {
  // inspired by https://fmt.dev/latest/api.html#formatting-user-defined-types
  constexpr auto parse(fmt::format_parse_context& ctx) {
    auto it = ctx.begin();
    auto end = ctx.end();
    if (it != end) {
      it++;
      mid = *it++;
    }

    // Check if reached the end of the range:
    if (it != end && *it != '}')
      throw fmt::format_error("invalid format");

    // Return an iterator past the end of the parsed range:
    return it;
  }

  char mid = '_';

  template <typename FormatContext>
  constexpr auto format(Monkey const& m, FormatContext& ctx) const {
    std::ostringstream ss;
    for (auto& item : m.items) {
      ss << Vc::AVX2::int_v(item) << ',';
    }
    ss << "\t(" << m.actual_size << ')';
    ss << "\t activity: " << m.inspections;
    if (mid == '_')
      return fmt::format_to(ctx.out(), "Monkey: {}", ss.str());
    else
      return fmt::format_to(ctx.out(), "Monkey {}: {}", mid, ss.str());
  }
};

void print_monkeys(std::vector<Monkey> const& monkeys) {
  for (auto [i, m] : ranges::view::enumerate(monkeys)) {
    if (m.oper == nullptr)
      std::cout << "PANIC\n";
    if (i == 0) {
      fmt::print("{:m0}\n", m);
    } else if (i == 1) {
      fmt::print("{:m1}\n", m);
    } else if (i == 2) {
      fmt::print("{:m2}\n", m);
    } else if (i == 3) {
      fmt::print("{:m3}\n", m);
    } else if (i == 4) {
      fmt::print("{:m4}\n", m);
    } else if (i == 5) {
      fmt::print("{:m5}\n", m);
    } else if (i == 6) {
      fmt::print("{:m6}\n", m);
    } else if (i == 7) {
      fmt::print("{:m7}\n", m);
    } else {
      fmt::print("{}\n", m);
    }
  }
}
