#pragma once
#include <immintrin.h>

__m256i blendv_epi32_f(__m256i const& in1, __m256i const& in2, const __m256i& m) {
  return _mm256_castps_si256(_mm256_blendv_ps(
      _mm256_castsi256_ps(in1), _mm256_castsi256_ps(in2),
      _mm256_castsi256_ps(m)));
}
