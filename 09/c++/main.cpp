#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/indices.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include "aoc_utils/to.hpp"
#include "types.h"

int main(int argc, char** argv) {
  const std::filesystem::path& in_path{argv[1]};
  std::ifstream instream(in_path);

  auto steps = aoc_utils::to<std::vector<char>>(ranges::view::concat(
      ranges::getlines_view(instream) |
          ranges::view::transform([](const auto& line) {
            std::string repeat_str =
                line | ranges::view::drop(2) | ranges::to<std::string>();
            auto repeat = std::atoi(repeat_str.c_str());

            return ranges::views::repeat_n(line[0], repeat);
          }) |
          ranges::view::join,
      ranges::views::repeat_n('N', 9)));

  auto s1acc = std::accumulate(
      steps.begin(), steps.end(),
      std::pair<std::unordered_set<pos, myposhash>, state1>{},
      [](std::pair<std::unordered_set<pos, myposhash>, state1> acc, char d) {
        acc.second.move(d);
        acc.first.insert(acc.second.TAIL);
        return acc;
      });

  std::cout << "part 1 " << s1acc.first.size() << '\n';

  std::vector<state2> unrolled_ropes;
  unrolled_ropes.resize(steps.size());

  // NB: The documentation confused me about the associativity that my binary
  // op needs to have. Given that accumulator and input elemnts can have
  // different types, I don't see how associativity can be required. It appears
  // my implementation does the right thing in this non-associative case with
  // `seq`.
  std::inclusive_scan(
      std::execution::seq, steps.begin(), steps.end(), unrolled_ropes.begin(),
      [](const state2& prev, char d) { return prev.pull(d); }, state2{});

  using acc_t = std::vector<pos>;

  auto tail_poses2 = std::transform_reduce(
#ifndef __NVCOMPILER
      std::execution::par_unseq,
#else
      std::execution::unseq,
#endif
      unrolled_ropes.begin(), unrolled_ropes.end(), acc_t{},
      [](const acc_t& lhs, const acc_t& rhs) {
        acc_t retval;
        std::set_union(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
            std::back_inserter(retval));
        return retval;
      },
      [](const state2& s) { return acc_t{s.m_data.back()}; });

  std::cout << "part 2 " << tail_poses2.size() << '\n';
  return 0;
}
