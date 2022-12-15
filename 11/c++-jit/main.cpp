#include <cctype>
#include <cstdio>
#include <dlfcn.h>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <regex>
#include <stdlib.h>
#include <type_traits>
// #include <range/v3/algorithm/equal.hpp>
// #include <range/v3/numeric/accumulate.hpp>
#include <range/v3/range/conversion.hpp>
// #include <range/v3/view/any_view.hpp>
#include <range/v3/view/chunk_by.hpp>
// #include <range/v3/view/concat.hpp>
// #include <range/v3/view/drop.hpp>
// #include <range/v3/view/enumerate.hpp>
// #include <range/v3/view/exclusive_scan.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
// #include <range/v3/view/cache1.hpp>
#include <range/v3/view/join.hpp>
// #include <range/v3/view/repeat_n.hpp>
// #include <range/v3/view/take.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/transform.hpp>
// #include <string>
// #include "aoc_ffi_ocr.h"
// #include "aoc_utils/to.hpp"

int main(int argc, char** argv) {
  const std::filesystem::path& in_path{argv[1]};
  std::ifstream instream(in_path);

  auto in_buffer = ranges::getlines_view(instream) | ranges::to_vector;

  std::vector<std::size_t> monkey_ids;

  ranges::for_each(
      ranges::view::chunk_by(
          in_buffer, [](const auto, const auto b) { return !b.empty(); }),
      [&monkey_ids](auto monkey) {
        auto cleaned = monkey | ranges::view::filter(
                                    [](auto line) { return !line.empty(); });

        auto line_iter = cleaned.begin();
        auto monkey_line = *(line_iter++);
        auto starting_setup = *(line_iter++);
        auto operation = std::regex_replace(
            *(line_iter++) | ranges::to<std::string>(),
            std::regex{".*Operation: new"}, "  Vc::AVX2::int_v new_");
        auto test = std::regex_replace(
            *(line_iter++) | ranges::to<std::string>(),
            std::regex{".*divisible by (\\d*)"},
            "  auto test = ((new_ % $1) == 0);");
        auto true_branch = std::regex_replace(
            *(line_iter++) | ranges::to<std::string>(),
            std::regex{".*monkey (\\d*)"}, "$1");
        auto false_branch = std::regex_replace(
            *(line_iter++) | ranges::to<std::string>(),
            std::regex{".*monkey (\\d*)"}, "$1");
        auto mon = *(monkey_line | ranges::view::filter([](auto cha) {
                       return isdigit(cha);
                     })).begin();
        std::FILE* of =
            std::fopen(fmt::format("monkeys/monkey{}.cpp", mon).c_str(), "w");
        // clang-format off
        fmt::print(of, "#include <Vc/Vc>\n");
        fmt::print(of, "\n");
        fmt::print(of, "std::tuple<__m256i, __m256i, std::size_t, std::size_t> Monkey{} (__m256i in) {{\n", mon);
        fmt::print(of, "  Vc::AVX2::int_v old{{in}};\n");
        fmt::print(of, "{};\n", operation);
        fmt::print(of, "  new_ = new_ / 3;\n");
        fmt::print(of, "{}\n", test);
        fmt::print(of, "  return std::tuple(new_.data(), test.data(), {}, {});\n", true_branch, false_branch);
        fmt::print(of, "}}\n");
        // clang-format on
        std::fclose(of);
        monkey_ids.push_back(static_cast<std::size_t>(mon - '0'));
      });

  auto make_targets = monkey_ids | ranges::view::transform([](auto id) {
                        return fmt::format("monkeys/monkey{}.o", id);
                      }) |
                      ranges::view::join(' ') | ranges::to<std::string>();

  auto cmd = fmt::format("make {}\n", make_targets);
  if (auto retval = system(cmd.c_str()); retval) {
    fmt::print("oh no (compile): {}\n", retval);
    return retval;
  }
  cmd = fmt::format("g++ -shared -o libmonkeys.so {}\n", make_targets);
  if (auto retval = system(cmd.c_str()); retval) {
    fmt::print("oh no (link): {}\n", retval);
    return retval;
  }

  void* monkey_lib = dlopen("libmonkeys.so", RTLD_NOW);
  using monkey_ptr =
      std::add_pointer<std::tuple<__m256i, __m256i, std::size_t, std::size_t>(
          __m256i)>::type;
  monkey_ptr monkey0 = reinterpret_cast<monkey_ptr>(dlsym(monkey_lib, "_Z7Monkey0Dv4_x"));

  return 0;
}
