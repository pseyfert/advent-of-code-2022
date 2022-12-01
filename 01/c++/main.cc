#include <algorithm>
#include <numeric>
#include <execution>
#include <filesystem>
#include <fstream>
#include <vector>

int res(std::vector<int>& data) {
  partial_sort(
      std::execution::par_unseq, data.begin(), data.begin() + 3, data.end(),
      [](auto l, auto r) { return l > r; });
  printf("part 1 %d\n", data.front());
  return std::reduce(
      std::execution::par_unseq, data.begin(), data.begin() + 3, 0,
      std::plus<int>());
}

std::vector<int> read(const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);
  std::vector<int> retval;
  std::string str;
  int cur{0};
  while (std::getline(instream, str)) {
    if (str.empty()) {
      retval.push_back(cur);
      cur = 0;
    } else {
      cur += std::atoi(str.c_str());
    }
  }
  return retval;
}

int main(int argc, char** argv) {
  auto elves = read(argv[1]);
  printf("part 2 %d\n", res(elves));
  return 0;
}
