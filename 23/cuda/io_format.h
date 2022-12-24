#pragma once
#include <utility>
#include <thrust/host_vector.h>

using input_data = std::pair<thrust::host_vector<int>, thrust::host_vector<int>>;
input_data read(char *);
