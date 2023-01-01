#include <cooperative_groups.h>
#include <cub/util_debug.cuh>
#include "part2.h"

__device__ std::int32_t yrange;

__device__ std::int32_t y_begin() {
  return threadIdx.y * (yrange / blockDim.y + !!(yrange % blockDim.y));
}
__device__ std::int32_t y_end() {
  return min(
      (threadIdx.y + 1) * (yrange / blockDim.y + !!(yrange % blockDim.y)),
      yrange);
}

__global__ void asdf(Circle* circles, int* N_circles, Location* out) {
  auto block = cooperative_groups::this_thread_block();
  if (threadIdx.y == 0) {
    yrange = 4000001;
  }
  block.sync();
  for (std::int32_t y = y_begin(); y < y_end(); y++) {
    // would love to use `simplify` here, but didn't figure out how to port to
    // device code.
    for (std::int32_t x = 0; x < yrange; ++x) {
      Location l{.x = tagged_int<x_tag>(x), .y = tagged_int<y_tag>(y)};
      bool found = false;
      for (int c = 0; c < *N_circles; ++c) {
        if (circles[c].is_inside(l)) {
          // Fast forward to the end of the current circle's cross section.
          x = int32_t(circles[c].project(l.y)->second);
          found = true;
          break;
        }
      }
      if (!found) {
        *out = l;
      }
    }
  }
}

std::int64_t part2(const std::vector<Circle>& circles) {
  Circle* d_circles;
  Location* best_location;
  int* N_circles;

  CubDebugExit(cudaMalloc(&d_circles, circles.size() * sizeof(Circle)));
  CubDebugExit(cudaMemcpy(
      d_circles, circles.data(), circles.size() * sizeof(Circle),
      cudaMemcpyHostToDevice));
  CubDebugExit(cudaMallocManaged(&best_location, sizeof(Location)));
  CubDebugExit(cudaMallocManaged(&N_circles, sizeof(int)));

  *N_circles = circles.size();

  asdf<<<1, {1, 1024}>>>(d_circles, N_circles, best_location);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  std::int64_t x = std::int32_t(best_location->x);
  std::int64_t y = std::int32_t(best_location->y);

  CubDebugExit(cudaFree(d_circles));
  CubDebugExit(cudaFree(best_location));
  CubDebugExit(cudaFree(N_circles));
  return x * 4000000 + y;
}
