#include "main.h"

int main(int, char** argv) {
  auto in = input(argv[1]);
  std::cout << "part1 " << part1(in.second) << '\n';
  std::cout << "part2 " << part2(in.second) << '\n';
  return 0;
}
