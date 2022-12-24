#include <cstdio>
#include <cub/util_debug.cuh>
#include <thrust/device_vector.h>
#include "global_funs.cuh"
#include "io_format.h"

int main(int, char**) {
  auto input = read("../input.txt");

  thrust::device_vector<int> d_x = input.first;
  thrust::device_vector<int> d_y = input.second;

  thrust::device_vector<int> d_px(input.first.size());
  thrust::device_vector<int> d_py(input.first.size());
  thrust::device_vector<int> d_go(input.first.size());

  int* N_elves;
  int* rm4;
  int* res;
  CubDebugExit(cudaMallocManaged(&N_elves, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&rm4, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&res, sizeof(int)));
  *N_elves = input.first.size();
  *rm4 = 0;

  for (int r = 0; r < 10; ++r) {
    *rm4 = r % 4;

    do_round<<<1, 1>>>(
        N_elves, thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_go.data()),
        thrust::raw_pointer_cast(d_px.data()),
        thrust::raw_pointer_cast(d_py.data()), rm4);
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    CubDebugExit(cudaPeekAtLastError());
  }
  empty_spaces<<<1, 1>>>(
      thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()), N_elves, res);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  printf("In the minimal map there ought to be %d empty cells\n", *res);

  CubDebugExit(cudaFree(N_elves));
  CubDebugExit(cudaFree(rm4));
}
