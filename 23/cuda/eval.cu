#include "global_funs.cuh"

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

__global__ void empty_spaces(
    int const* current_x, int const* current_y, const int* N_elves, int* result) {
  thrust::pair<const int*, const int*> x_range =
      thrust::minmax_element(thrust::device, current_x, current_x + *N_elves);
  thrust::pair<const int*, const int*> y_range =
      thrust::minmax_element(thrust::device, current_y, current_y + *N_elves);

  auto map_area = (*x_range.second - *x_range.first + 1) *
                  (*y_range.second - *y_range.first + 1);
  *result = map_area - *N_elves;
}
