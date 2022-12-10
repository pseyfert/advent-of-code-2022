#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/any_view.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include "aoc_utils/to.hpp"

enum class OpCode { noop, add_1, add_2 };

struct OP {
  OpCode m_code;
  int m_arg;
};

struct system_state {
  int m_reg;
  bool m_light;
};

int main(int argc, char** argv) {
  const std::filesystem::path& in_path{argv[1]};
  std::ifstream instream(in_path);

  auto steps = aoc_utils::to<std::vector<OP>>(

      ranges::getlines_view(instream) |
      ranges::view::transform([](const auto& line) {
        // that has to work w/o buffering in std::string
        auto instr = line | ranges::view::take(4) | ranges::to<std::string>();

        // super unelegant to return repeat(Op)
        if (instr == std::string("noop")) {
          return ranges::any_view(ranges::views::repeat_n(
              OP{.m_code = OpCode::noop, .m_arg = 1}, 1));
        } else if (instr == std::string("addx")) {
          auto arg = line | ranges::view::drop(5) | ranges::to<std::string>();
          auto arg_int = std::atoi(arg.c_str());

          return ranges::any_view(ranges::view::concat(
              // could actually start with noop.
              ranges::views::repeat_n(
                  OP{.m_code = OpCode::add_1, .m_arg = arg_int}, 1),
              ranges::views::repeat_n(
                  OP{.m_code = OpCode::add_2, .m_arg = arg_int}, 1)));
        }
        std::cout << "ERROR instr is " << instr << '\n';
      }) |
      ranges::view::join);

  auto apply = [](system_state register_, OP op) {
    if (op.m_code == OpCode::add_2) {
      register_.m_reg += op.m_arg;
    }
    return register_;
  };

  auto processed =
      ranges::view::enumerate(
          steps | ranges::view::exclusive_scan(
                      system_state{.m_reg = 1}, std::move(apply))) |
      ranges::view::transform([](auto opid_and_state) {
        opid_and_state.first++;
        return opid_and_state;
      });

  auto part1 = ranges::accumulate(
      processed | ranges::view::filter([](auto opid_and_state) {
        auto& step = opid_and_state.first;
        return (
            step == 20 || step == 60 || step == 100 || step == 140 ||
            step == 180 || step == 220);
      }) | ranges::view::transform([](auto opid_and_state) {
        return opid_and_state.first * opid_and_state.second.m_reg;
      })
      //;
      ,
      0, std::plus());

  std::cout << "part1 = " << part1 << '\n';

  return 0;
}
