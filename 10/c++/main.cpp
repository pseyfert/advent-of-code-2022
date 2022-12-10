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

  auto apply = [](int register_, OP op) {
    if (op.m_code == OpCode::add_2) {
      return register_ + op.m_arg;
    } else {
      return register_;
    }
  };

  auto part1 = ranges::accumulate(
      // auto selected =
      ranges::view::enumerate(
          steps | ranges::view::exclusive_scan(1, std::move(apply))) |
          ranges::view::filter([](auto opid_and_state) {
            auto& step = opid_and_state.first;
            return (
                step == 19 || step == 59 || step == 99 || step == 139 ||
                step == 179 || step == 219);
          }) |
          ranges::view::transform([](auto opid_and_state) {
            return (1 + opid_and_state.first) * opid_and_state.second;
          })
      //;
      ,
      0, std::plus());

  std::cout << "part1 = " << part1 << '\n';

  return 0;
}
