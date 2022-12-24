#include "global_funs.cuh"
#include "io_format.h"

#include <cub/util_debug.cuh>
#include <stdio.h>
#include <thrust/device_vector.h>

int main() {
  auto input = read("../end_example.txt");

  thrust::device_vector<int> d_x = input.first;
  thrust::device_vector<int> d_y = input.second;

  int* N_elves;
  int* res;
  CubDebugExit(cudaMallocManaged(&N_elves, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&res, sizeof(int)));
  *N_elves = input.first.size();

  empty_spaces<<<1, 1>>>(
      thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()), N_elves, res);

  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  printf("in the example there are %d empty cells\n", *res);

  CubDebugExit(cudaFree(N_elves));
  CubDebugExit(cudaFree(res));

  return 0;
}
