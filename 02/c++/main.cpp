#include "shared.h"

auto input(const std::filesystem::path& inpath) {
  std::ifstream instream(inpath);

  SOA::Container<std::vector, unprocessed::skin> retval;

  ranges::for_each(ranges::getlines_view(instream), [&retval](const auto line) {
    auto opponent_move = [](const auto& letter) -> short {
      if (letter == 'A') {
        return 1;
      } else if (letter == 'B') {
        return 2;
      } else if (letter == 'C') {
        return 3;
      } else {
        printf("Error parsing other char: %s\n", letter);
        return -100;
      }
    };
    auto self_move = [](const auto& letter) -> short {
      if (letter == 'X') {
        return 1;
      } else if (letter == 'Y') {
        return 2;
      } else if (letter == 'Z') {
        return 3;
      } else {
        printf("Error parsing self char: %s\n", letter);
        return -100;
      }
    };
    retval.emplace_back(opponent_move(line[0]), self_move(line[2]));
  });

  return retval;
}

int main(int argc, char** argv) {
  auto d = input(argv[1]);
  auto part1_ = part1(d);

  printf("part 1: %d\n", part1_);

  auto part2_ = part2(d);

  printf("part 2: %d\n", part2_);

  return 0;
}
