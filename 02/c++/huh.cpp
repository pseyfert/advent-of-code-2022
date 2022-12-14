#include "shared.h"
#include <vector>

void f(const std::vector<data_t>& a, std::vector<data_t>& b, std::vector<data_t>& c) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = score(a[i], b[i]);
  }
}
