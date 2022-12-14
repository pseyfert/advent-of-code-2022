#define SCORES_IMPL
#include <immintrin.h>
#include "Vc/Vc"
#include "shared.h"
#include "scores.h"

__attribute__((const)) __m256i score(__m256i input1, __m256i input2) noexcept
    asm("_ZGVdN8vv__Z5scoreii");
__attribute__((const)) __m256i score(__m256i input1, __m256i input2) noexcept {
  Vc::AVX2::int_v i1(input1);
  Vc::AVX2::int_v i2(input2);
  Vc::AVX2::int_v three(3);
  Vc::AVX2::int_v n(1);
  auto d = (i1 - i2 + three) % 3;
  auto did_draw = _mm256_cmpeq_epi32(input1, input2);
  auto did_lose = _mm256_cmpeq_epi32(n.data(), d.data());

  auto win_score = i2 + 6;
  auto los_score = i2 + 0;
  auto dra_score = i2 + 3;

  auto win_los =
      _mm256_blendv_epi8(win_score.data(), los_score.data(), did_lose);
  return _mm256_blendv_epi8(win_los, dra_score.data(), did_draw);
}

__attribute__((const)) __m128i score(__m128i input1, __m128i input2) noexcept
    asm("_ZGVcN4vv__Z5scoreii");
__attribute__((const)) __m128i score(__m128i input1, __m128i input2) noexcept {
  Vc::SSE::int_v i1(input1);
  Vc::SSE::int_v i2(input2);
  Vc::SSE::int_v three{3};
  auto d = (i1 - i2 + three) % 3;
  Vc::SSE::int_v n{1};
  auto did_dra = _mm_cmplt_epi32(d.data(), n.data());
  auto did_los = _mm_cmpeq_epi32(n.data(), d.data());
  auto win_score = i2 + 6;
  auto los_score = i2 + 0;
  auto dra_score = i2 + 3;
  // NB: using epi8 seems totally wrong here, but the _cmp before sets all bits
  // in the mask, such that the upper and nower byte of a 16 bit int will be
  // treated the same.
  auto win_los = _mm_blendv_epi8(win_score.data(), los_score.data(), did_los);
  return _mm_blendv_epi8(win_los, dra_score.data(), did_dra);
}

__attribute__((const)) data_t score(data_t input1, data_t input2) noexcept
    asm("_Z5scoreii");
__attribute__((const)) data_t score(data_t input1, data_t input2) noexcept {
  data_t d = (input1 - input2 + 3) % 3;
  if (d == 0) {
    // return 1;
    return 3 + input2;
  } else if (d == 1) {
    // return 10;
    return 0 + input2;
  } else if (d == 2) {
    // return 100;
    return 6 + input2;
  }
}

#undef SCORES_IMPL
