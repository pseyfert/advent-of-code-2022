#undef FMT_CONSTEVAL
#include <cctype>
#include <cstdio>
#include <dlfcn.h>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/chunk_by.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/drop_while.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/split.hpp>
#include <range/v3/view/transform.hpp>
#include <regex>
#include <stdlib.h>
#include <string>
#include <type_traits>
#include "monkey.h"
#include "toi.h"

// #include <range/v3/view/cache1.hpp>
// #include <range/v3/view/repeat_n.hpp>
// #include <range/v3/view/take.hpp>
// #include "aoc_ffi_ocr.h"
// #include "aoc_utils/to.hpp"
// #include <range/v3/algorithm/equal.hpp>
// #include <range/v3/numeric/accumulate.hpp>
// #include <range/v3/view/any_view.hpp>
// #include <range/v3/view/concat.hpp>
// #include <range/v3/view/drop.hpp>
// #include <range/v3/view/exclusive_scan.hpp>

template <typename R>
auto initial_state(R&& init_line) {
  return int_range(
      init_line | ranges::view::filter([](auto c) { return c != ' '; }) |
      ranges::view::drop_while([](auto c) { return c != ':'; }) |
      ranges::view::drop(1) | ranges::view::split(','));
}

int main(int argc, char** argv) {
  const std::filesystem::path& in_path{argv[1]};
  std::ifstream instream(in_path);

  auto in_buffer = ranges::getlines_view(instream) | ranges::to_vector;

  std::vector<std::size_t> monkey_ids;
  std::vector<Monkey> monkeys;

  ranges::for_each(
      ranges::view::chunk_by(
          in_buffer, [](const auto, const auto b) { return !b.empty(); }),
      [&monkey_ids, &monkeys](auto monkey) {
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
        fmt::print(of, "#define Vc_NO_VERSION_CHECK\n");
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

        Monkey& this_monkey = monkeys.emplace_back();
        Vc::AVX2::int_v buffer;
        std::size_t n = 0;
        for (auto [i, v] :
             ranges::view::enumerate(initial_state(starting_setup))) {
          buffer[i] = v;
          n++;
        }
        this_monkey.items.emplace_back(buffer.data());
        this_monkey.actual_size = n;
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

  dlerror();
  void* monkey_lib = dlopen("./libmonkeys.so", RTLD_NOW);
  if (auto err = dlerror(); err) {
    std::cout << err << '\n';
    return 1;
  }
  for (auto [id, monkey] : ranges::view::enumerate(monkeys)) {
    if (monkey_ptr fn = reinterpret_cast<monkey_ptr>(
            dlsym(monkey_lib, fmt::format("_Z7Monkey{}Dv4_x", id).c_str()));
        fn == nullptr) {
      if (auto err = dlerror(); err) {
        std::cout << err << '\n';
      }
      std::cout << "PANIC\n";
    } else {
      if (auto err = dlerror(); err) {
        std::cout << err << '\n';
      }
      monkey.oper = fn;
    }
  }

  print_monkeys(monkeys);

  for (auto [i, m] : ranges::view::enumerate(monkeys)) {
    m.perform_oper(monkeys);
    // fmt::print("after monkey {:-^80}\n", i);
    // print_monkeys(monkeys);
  }
  fmt::print("after round {:=^80}\n", 1);
  print_monkeys(monkeys);
  for (int r = 2; r <= 20; ++r) {
    for (auto [i, m] : ranges::view::enumerate(monkeys)) {
      m.perform_oper(monkeys);
      if (dbg) {
        fmt::print("after monkey {:-^80}\n", i);
        print_monkeys(monkeys);
      }
    }
    fmt::print("after round {:=^80}\n", r);
    print_monkeys(monkeys);
    // if (r == 4) {
    //   dbg = true;
    // } else if (r == 6) {
    //   dbg = false;
    // }
  }

  return 0;
}
