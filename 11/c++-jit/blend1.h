#pragma once
#include <cstdint>
#include <immintrin.h>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

// ACTUALLY SOMETHING WRONG IN THIS IMPLEMENTATION
// with 
// std::array<int, 8> a{0, 1, 2, 3, 4, 5, 6, 7};
// return (Vc::AVX2::int_v{a.data()} < 7).data();

__m256i blendv_epi32_pp(__m256i const& in1, __m256i const& in2, __m256i const& m) {
  uint8_t ma = _mm256_movemask_ps(_mm256_castsi256_ps(m));

  if (ma == 0) return _mm256_blend_epi32(in1, in2, 0);
#define MACRO(z, n, aux) \
  else if (ma == n) return _mm256_blend_epi32(in1, in2, n);
  BOOST_PP_REPEAT_FROM_TO(0, 127, MACRO, ma)
  else __builtin_unreachable();
}
