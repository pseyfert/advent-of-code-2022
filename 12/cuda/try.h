#pragma once
#include <vector>

struct input_data {
  std::vector<int> heights;
  std::vector<int> scores;
  int rows;
  int cols;
  int goal_x;
  int goal_y;
};

input_data read();

