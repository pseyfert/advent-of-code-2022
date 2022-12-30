#include "parse.h"
#include <filesystem>
#include <fstream>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/cache1.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/join.hpp>
#include <range/v3/view/transform.hpp>
#include <string>

std::tuple<std::string, mysizet, mysizet> read(char* argv1) {
  std::ifstream in_stream{std::filesystem::path(argv1)};
  mysizet row_c = 0;
  auto map = ranges::getlines_view(in_stream) | ranges::view::cache1 |
             ranges::view::drop(1) |
             ranges::view::transform([&row_c](auto line) {
               row_c++;
               return line;
             }) |
             ranges::view::join |
             ranges::view::filter([](auto c) { return c != '#'; }) |
             ranges::to<std::string>();
  row_c--;
  mysizet col_c = map.size() / row_c;
  return std::make_tuple(std::move(map), row_c, col_c);
}
