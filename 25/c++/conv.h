#pragma once
#include <numeric>
#include <string_view>

// TODO
unsigned long fromSNAFU(std::string_view s) {
  return std::accumulate(
      s.begin(), s.end(), 0ul, [](unsigned long acc, const char c) {
        acc *= 5;
        switch (c) {
          case '0':
            acc += 0;
            break;
          case '1':
            acc += 1;
            break;
          case '2':
            acc += 2;
            break;
          case '-':
            acc -= 1;
            break;
          case '=':
            acc -= 2;
            break;
          default:
            __builtin_unreachable();
            break;
        }
        // comment to prevent clang format from merging lines
        return acc;
      });
}
