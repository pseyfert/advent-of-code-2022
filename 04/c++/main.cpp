#include <iostream>
#include <cstdio>
#include "shared.h"

int main(int argc, char** argv) {
  auto d = input(argv[1]);

  // for (auto p : d) {
  //   std::cout << '[' << p.start_a() << ',' << p.end_a() << "]\t[" << p.start_b()
  //             << ',' << p.end_b() << "]\n";
  // }
  printf("part1: %d\n", part1(d));
  printf("part2: %d\n", part2(d));

  return 0;
}
