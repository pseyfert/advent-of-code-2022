#include <filesystem>
#include <fstream>
// #include <iostream>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view/enumerate.hpp>
#include <range/v3/view/getlines.hpp>
// #include <string>
#include <thrust/host_vector.h>
#include <tuple>
// #include <vector>

#include "io_format.h"

input_data read(char* p) {
  std::ifstream in_stream{std::filesystem::path(p)};

  thrust::host_vector<int> x_vals;
  thrust::host_vector<int> y_vals;

  ranges::for_each(
      ranges::view::enumerate(ranges::getlines_view(in_stream)),
      [&x_vals, &y_vals](const auto row_and_rowid) {
        auto& [rowid, row] = row_and_rowid;
        ranges::for_each(
            ranges::view::enumerate(row),
            [&x_vals, &y_vals, rowid](const auto col_and_colid) {
              auto& [colid, character] = col_and_colid;
              // filter? seems harder to write than this
              if (character == '#') {
                x_vals.push_back(colid);
                y_vals.push_back(rowid);
              }
            });
      });

  return std::make_pair(x_vals, y_vals);
}
