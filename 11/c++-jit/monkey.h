#pragma once
#include <Vc/Vc>
#include <bit>
#include <cstddef>
#include <immintrin.h>
#include <vector>
#include "blend1.h"
#include "blend2.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
#include "/home/pseyfert/coding/simdprune/simdprune_tables.h"
#pragma GCC diagnostic pop

struct Monkey {
  std::vector<__m256i> items;
  int actual_size;

  void my_compress_store(const __m256i& other, const __m256i& mask) {
    uint8_t mask_i = _mm256_movemask_ps(_mm256_castsi256_ps(mask));

    auto to_insert = std::popcount(mask_i);
    int space_left;
    if (actual_size == 0) {
      space_left = 0;
    } else {
      space_left = 8 - (1 + ((actual_size - 1) % 8));
    }
    auto offset = (8 * (mask_i ^ 0xFF) - (actual_size % 8));
    const __m256i* lookups;
    if (offset < 0) {
      // can use mask_i ^ 0x80 instead if space_left != 0
      std::cout << "PANIC\n";
      reinterpret_cast<const __m256i*>(mask256_epi32);
    } else {
      lookups = reinterpret_cast<const __m256i*>(mask256_epi32 + offset);
    }
    auto idxs = _mm256_load_si256(lookups);

    __m256i compressed_buffer;
    if (space_left > 0) {
      compressed_buffer = _mm256_permutevar8x32_epi32(other, idxs);
      items.back() =
          blendv_epi32_f(compressed_buffer, items.back(), last_loop_mask());
    }

    if (to_insert > space_left) {
      auto idxs2 = _mm256_load_si256(lookups + 1);

      __m256i remainders = _mm256_permutevar8x32_epi32(other, idxs2);
      items.push_back(remainders);
    }
    actual_size += to_insert;
  }

  __m256i last_loop_mask() {
    std::array<int, 8> a{0, 1, 2, 3, 4, 5, 6, 7};
    return (Vc::AVX2::int_v{a.data()} < (actual_size % 8)).data();
  }
};
