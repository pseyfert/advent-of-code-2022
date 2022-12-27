#include <cub/util_debug.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "global_funs.cuh"
#include "io_format.h"

TEST(example, part2) {
  auto input = read("../r0.txt");

  thrust::device_vector<int> d_x = input.first;
  thrust::device_vector<int> d_y = input.second;

  thrust::device_vector<int> d_px(input.first.size());
  thrust::device_vector<int> d_py(input.first.size());
  thrust::device_vector<int> d_go(input.first.size());

  int* N_elves;
  int* rm4;
  bool* movement;
  CubDebugExit(cudaMallocManaged(&N_elves, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&rm4, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&movement, sizeof(int)));
  *N_elves = input.first.size();
  *rm4 = 0;
  *movement = 1;
  
  int round_count = 0;

  for (; round_count < 30; ++round_count) {
    *rm4 = round_count % 4;

    do_round<<<1, 1>>>(
        N_elves, thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_go.data()),
        thrust::raw_pointer_cast(d_px.data()),
        thrust::raw_pointer_cast(d_py.data()), rm4);
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    CubDebugExit(cudaPeekAtLastError());

    stationary<<<1, 1>>>(
        thrust::raw_pointer_cast(d_go.data()), N_elves, movement);

    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    CubDebugExit(cudaPeekAtLastError());

    if (!*movement) {
      break;
    }
  }
  EXPECT_EQ(round_count, 19) << "Though maybe off by 1? Who knows?";

  CubDebugExit(cudaFree(N_elves));
  CubDebugExit(cudaFree(rm4));
}
