#define CUB_STDERR
#include <cooperative_groups.h>
#include <cub/util_debug.cuh>
#include <cuda/std/barrier>
#include <cuda/std/limits>
#include <experimental/mdspan>

#include <tuple>
#include <vector>

// https://stackoverflow.com/a/14038590
#include <assert.h>
#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

using myspan = std::experimental::mdspan<
    int, std::experimental::extents<
             size_t, std::experimental::dynamic_extent,
             std::experimental::dynamic_extent>>;

__device__ cuda::barrier<cuda::thread_scope_block> barrier;

__global__ void asdf(
    int* h, int* s, cuda::barrier<cuda::thread_scope_block>& barrier,
    int goal_x, int goal_y, int* COLS, int* ROWS) {
  myspan heights(h, *ROWS, *COLS);
  myspan scores(s, *ROWS, *COLS);

  printf("hello 1 from %d, %d\n", threadIdx.y, threadIdx.x);
  auto x_max = scores.extent(1);
  auto y_max = scores.extent(0);
  bool check_xless =
      (threadIdx.x > 0) && (heights)(threadIdx.y, threadIdx.x) <=
                               1 + (heights)(threadIdx.y, threadIdx.x - 1);
  bool check_xmore = (threadIdx.x < x_max - 1) &&
                     (heights)(threadIdx.y, threadIdx.x) <=
                         1 + (heights)(threadIdx.y, threadIdx.x + 1);
  bool check_yless =
      (threadIdx.y > 0) && (heights)(threadIdx.y, threadIdx.x) <=
                               1 + (heights)(threadIdx.y - 1, threadIdx.x);
  bool check_ymore = (threadIdx.y < y_max - 1) &&
                     (heights)(threadIdx.y, threadIdx.x) <=
                         1 + (heights)(threadIdx.y + 1, threadIdx.x);
  for (;;) {
    /* if (threadIdx.x == 0 && threadIdx.y == 0) */
    /*   printf("iter\n"); */
    auto best = cuda::std::numeric_limits<int>::max();
    if ((scores)(goal_y, goal_x) < cuda::std::numeric_limits<int>::max()) {
      printf("abandoning thread for %d, %d\n", threadIdx.y, threadIdx.x);
      barrier.arrive_and_drop();
      break;
    } else {
      if (check_xmore) {
        best = min(best, (scores)(threadIdx.y, threadIdx.x + 1));
      }
      if (check_xless) {
        best = min(best, (scores)(threadIdx.y, threadIdx.x - 1));
      }
      if (check_ymore) {
        best = min(best, (scores)(threadIdx.y + 1, threadIdx.x));
      }
      if (check_yless) {
        best = min(best, (scores)(threadIdx.y - 1, threadIdx.x));
      }
      barrier.arrive_and_wait();
    }

    if (best < cuda::std::numeric_limits<int>::max()) {
      (scores)(threadIdx.y, threadIdx.x) = best + 1;
      printf("reached %d, %d\n", threadIdx.y, threadIdx.x);
      barrier.arrive_and_drop();
      break;
    } else {
      barrier.arrive_and_wait();
    }
  }
}

__global__ void cr(
    int* heights, int* scores, int* goal_x, int* goal_y, int* COLS, int* ROWS) {
  printf("launch kernel\n");

  auto block = cooperative_groups::this_thread_block();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    init(&barrier, (*ROWS) * (*COLS));
  }
  block.sync();
  CubDebug(cudaPeekAtLastError());

  asdf<<<1, {*COLS, *ROWS}>>>(
      heights, scores, barrier, *goal_x, *goal_y, COLS, ROWS);
  cdpErrchk(cudaPeekAtLastError());
  block.sync();
  cdpErrchk(cudaPeekAtLastError());
}

using myspan = std::experimental::mdspan<
    int, std::experimental::extents<
             size_t, std::experimental::dynamic_extent,
             std::experimental::dynamic_extent>>;

#include "try.h"

int main(int, char** argv) {
  auto input = read(argv[1]);
  int* COLS;
  int* ROWS;
  int* goal_x;
  int* goal_y;
  int* scores;
  int* heights;
  CubDebugExit(cudaMallocManaged(&COLS, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&ROWS, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&goal_x, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&goal_y, sizeof(int)));

  *ROWS = input.rows;
  *COLS = input.cols;
  *goal_x = input.goal_x;
  *goal_y = input.goal_y;

  CubDebugExit(cudaMalloc(&scores, (*ROWS) * (*COLS) * sizeof(int)));
  CubDebugExit(cudaMalloc(&heights, (*ROWS) * (*COLS) * sizeof(int)));

  CubDebugExit(cudaMemcpy(
      scores, input.scores.data(), (*ROWS) * (*COLS) * sizeof(int),
      cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(
      heights, input.heights.data(), (*ROWS) * (*COLS) * sizeof(int),
      cudaMemcpyHostToDevice));

  cr<<<1, 1>>>(heights, scores, goal_x, goal_y, COLS, ROWS);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  CubDebugExit(cudaMemcpy(
      input.scores.data(), scores, (*ROWS) * (*COLS) * sizeof(int),
      cudaMemcpyDeviceToHost));

  myspan scores_span(input.scores.data(), *ROWS, *COLS);
  printf("reached goal at %d\n", scores_span(*goal_y, *goal_x));
  return 0;
}
