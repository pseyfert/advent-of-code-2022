#include "read.h"
#include <aoc_utils/to.hpp>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/split_when.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include "helpers.cuh"

std::vector<SensorBeaconPair> read(char* fname) {
  std::ifstream in_stream{std::filesystem::path{fname}};

  return aoc_utils::to<std::vector<SensorBeaconPair>>(
      ranges::getlines_view(in_stream) |
      ranges::view::transform([](const auto line) -> SensorBeaconPair {
        auto four_elements =
            ranges::view::filter(
                line,
                [](char c) {
                  return std::isdigit(c) || c == ':' || c == ',' || c == '-';
                }) |
            ranges::view::split_when(
                [](char c) { return c == ':' || c == ','; });
        auto it = four_elements.begin();
        auto sx = std::stoi(ranges::to<std::string>(*it++));
        auto sy = std::stoi(ranges::to<std::string>(*it++));
        auto bx = std::stoi(ranges::to<std::string>(*it++));
        auto by = std::stoi(ranges::to<std::string>(*it++));
        return SensorBeaconPair{
            .S =
                Location{
                    .x = tagged_int<x_tag>(sx), .y = tagged_int<y_tag>(sy)},
            .B = Location{
                .x = tagged_int<x_tag>(bx), .y = tagged_int<y_tag>(by)}};
      }));
}
