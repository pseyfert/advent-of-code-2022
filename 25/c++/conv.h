#pragma once
#include <numeric>
#include <string_view>

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

std::string toSNAFU(unsigned long s) {
  std::string reverse_retval;
  for (; s != 0;) {
    // alternative:
    // tmp = std::div(s, 5);
    // s = tmp.quot;
    // if (tmp.rem > 2) s++;
    switch (s % 5) {
      case 2:
        reverse_retval += '2';
        s -= 2;
        break;
      case 1:
        reverse_retval += '1';
        s -= 1;
        break;
      case 0:
        reverse_retval += '0';
        s -= 0;
        break;
      case 3:
        reverse_retval += '=';
        s += 2;
        break;
      case 4:
        reverse_retval += '-';
        s += 1;
        break;
    }
    if (s % 5) {
      throw std::logic_error("my math is wrong");
    }
    s /= 5;
  }
  return std::string(reverse_retval.rbegin(), reverse_retval.rend());
}
