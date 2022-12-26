#include <cub/util_debug.cuh>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "global_funs.cuh"
#include "io_format.h"

TEST(example, rounds) {
  auto input = read("../r0.txt");
  auto end_1 = read("../r1.txt");
  auto end_2 = read("../r2.txt");
  auto end_3 = read("../r3.txt");
  auto end_4 = read("../r4.txt");
  auto end_5 = read("../r5.txt");

  thrust::device_vector<int> d_x = input.first;
  thrust::device_vector<int> d_y = input.second;

  thrust::device_vector<int> d_px(input.first.size());
  thrust::device_vector<int> d_py(input.first.size());
  thrust::device_vector<int> d_go(input.first.size());

  int* N_elves;
  int* rm4;
  CubDebugExit(cudaMallocManaged(&N_elves, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&rm4, sizeof(int)));
  *N_elves = input.first.size();
  *rm4 = 0;

  do_round<<<1, 1>>>(
      N_elves, thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()),
      thrust::raw_pointer_cast(d_go.data()),
      thrust::raw_pointer_cast(d_px.data()),
      thrust::raw_pointer_cast(d_py.data()), rm4);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  thrust::host_vector<int> h_x = d_x;
  thrust::host_vector<int> h_y = d_y;

  EXPECT_EQ(h_x.size(), h_y.size());
  EXPECT_EQ(input.first.size(), input.second.size());
  EXPECT_EQ(input.first.size(), h_x.size());
  EXPECT_EQ(*N_elves, h_x.size());
  EXPECT_EQ(*N_elves, end_1.first.size());
  EXPECT_EQ(*N_elves, end_1.second.size());

  for (std::size_t i = 0; i < *N_elves; ++i) {
    bool found = false;
    for (std::size_t j = 0; j < *N_elves; ++j) {
      if (h_x[i] == end_1.first[j] && h_y[i] == end_1.second[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Elve " << i << " (on my map) not found in the reference map.";
  }

  // NOW CHECK ROUND 2
  *rm4 = 1;

  do_round<<<1, 1>>>(
      N_elves, thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()),
      thrust::raw_pointer_cast(d_go.data()),
      thrust::raw_pointer_cast(d_px.data()),
      thrust::raw_pointer_cast(d_py.data()), rm4);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  h_x = d_x;
  h_y = d_y;

  EXPECT_EQ(h_x.size(), h_y.size());
  EXPECT_EQ(input.first.size(), input.second.size());
  EXPECT_EQ(input.first.size(), h_x.size());
  EXPECT_EQ(*N_elves, h_x.size());
  EXPECT_EQ(*N_elves, end_2.first.size());
  EXPECT_EQ(*N_elves, end_2.second.size());

  for (std::size_t i = 0; i < *N_elves; ++i) {
    bool found = false;
    for (std::size_t j = 0; j < *N_elves; ++j) {
      if (h_x[i] == end_2.first[j] && h_y[i] == end_2.second[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Elve " << i << " (on my map) not found in the reference map.";
  }

  // NOW CHECK ROUND 3
  *rm4 = 2;

  do_round<<<1, 1>>>(
      N_elves, thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()),
      thrust::raw_pointer_cast(d_go.data()),
      thrust::raw_pointer_cast(d_px.data()),
      thrust::raw_pointer_cast(d_py.data()), rm4);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  h_x = d_x;
  h_y = d_y;

  EXPECT_EQ(h_x.size(), h_y.size());
  EXPECT_EQ(input.first.size(), input.second.size());
  EXPECT_EQ(input.first.size(), h_x.size());
  EXPECT_EQ(*N_elves, h_x.size());
  EXPECT_EQ(*N_elves, end_3.first.size());
  EXPECT_EQ(*N_elves, end_3.second.size());

  for (std::size_t i = 0; i < *N_elves; ++i) {
    bool found = false;
    for (std::size_t j = 0; j < *N_elves; ++j) {
      if (h_x[i] == end_3.first[j] && h_y[i] == end_3.second[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Elve " << i << " (on my map) not found in the reference map.";
  }

  // NOW CHECK ROUND 4
  *rm4 = 3;

  do_round<<<1, 1>>>(
      N_elves, thrust::raw_pointer_cast(d_x.data()),
      thrust::raw_pointer_cast(d_y.data()),
      thrust::raw_pointer_cast(d_go.data()),
      thrust::raw_pointer_cast(d_px.data()),
      thrust::raw_pointer_cast(d_py.data()), rm4);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  h_x = d_x;
  h_y = d_y;

  EXPECT_EQ(h_x.size(), h_y.size());
  EXPECT_EQ(input.first.size(), input.second.size());
  EXPECT_EQ(input.first.size(), h_x.size());
  EXPECT_EQ(*N_elves, h_x.size());
  EXPECT_EQ(*N_elves, end_4.first.size());
  EXPECT_EQ(*N_elves, end_4.second.size());

  for (std::size_t i = 0; i < *N_elves; ++i) {
    bool found = false;
    for (std::size_t j = 0; j < *N_elves; ++j) {
      if (h_x[i] == end_4.first[j] && h_y[i] == end_4.second[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Elve " << i << " (on my map) not found in the reference map.";
  }

  // NOW ROUNDS UNTIL 5
  for (int r = 4; r < 5; ++r) {
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

  h_x = d_x;
  h_y = d_y;

  for (std::size_t i = 0; i < *N_elves; ++i) {
    bool found = false;
    for (std::size_t j = 0; j < *N_elves; ++j) {
      if (h_x[i] == end_5.first[j] && h_y[i] == end_5.second[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Elve " << i << " (on my map) not found in the reference map.";
  }

  CubDebugExit(cudaFree(N_elves));
  CubDebugExit(cudaFree(rm4));
}

TEST(example, round10) {
  auto input = read("../r0.txt");
  auto end_10 = read("../r10.txt");

  thrust::device_vector<int> d_x = input.first;
  thrust::device_vector<int> d_y = input.second;

  thrust::device_vector<int> d_px(input.first.size());
  thrust::device_vector<int> d_py(input.first.size());
  thrust::device_vector<int> d_go(input.first.size());

  int* N_elves;
  int* rm4;
  CubDebugExit(cudaMallocManaged(&N_elves, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&rm4, sizeof(int)));
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

  thrust::host_vector<int> h_x = d_x;
  thrust::host_vector<int> h_y = d_y;

  for (std::size_t i = 0; i < *N_elves; ++i) {
    bool found = false;
    for (std::size_t j = 0; j < *N_elves; ++j) {
      if (h_x[i] == end_10.first[j] && h_y[i] == end_10.second[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Elve " << i << " (on my map) not found in the reference map.";
  }

  CubDebugExit(cudaFree(N_elves));
  CubDebugExit(cudaFree(rm4));
}
