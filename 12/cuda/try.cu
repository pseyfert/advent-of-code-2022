#define CUB_STDERR
#include <cooperative_groups.h>
#include <cub/util_debug.cuh>
#include <cuda/std/array>
#include <cuda/std/barrier>
#include <cuda/std/limits>
#include <experimental/mdspan>

#include <algorithm>
#include <tuple>
#include <vector>

// https://stackoverflow.com/a/14038590
#include <assert.h>
#define cdpErrchk(ans) \
  { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(
    cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    printf(
        "GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      assert(0);
  }
}

using myspan = std::experimental::mdspan<
    int, std::experimental::extents<
             size_t, std::experimental::dynamic_extent,
             std::experimental::dynamic_extent>>;

__device__ cuda::barrier<cuda::thread_scope_block> barrier;

// TODO could (probably?) also use the original version and spawn a sub-block.
// But I don't want to deal with juggeling the barrier around.
template <int GRAINSIZE_X, int GRAINSIZE_Y>
__global__ void asdf(
    int* h, int* s, cuda::barrier<cuda::thread_scope_block>& barrier,
    int goal_x, int goal_y, int* COLS, int* ROWS) {
  myspan heights(h, *ROWS, *COLS);
  myspan scores(s, *ROWS, *COLS);

  auto x_max = scores.extent(1);
  auto y_max = scores.extent(0);

  std::array<int, (GRAINSIZE_X + 2) * (GRAINSIZE_Y + 2)> one_bigger_each_dir;
  myspan score_buffer(
      one_bigger_each_dir.data(), (GRAINSIZE_Y + 2), (GRAINSIZE_X + 2));
  auto x_offset = GRAINSIZE_X * threadIdx.x;
  auto y_offset = GRAINSIZE_Y * threadIdx.y;
  /* printf( */
  /*     "hello 1 from tidy %d, tidx %d\txoffset %d, yoffset %d\tgrainx %d, " */
  /*     "grainy %d\n", */
  /*     threadIdx.y, threadIdx.x, x_offset, y_offset, GRAINSIZE_X,
   * GRAINSIZE_Y); */

  for (;;) {
    if ((scores)(goal_y, goal_x) < cuda::std::numeric_limits<int>::max()) {
      /* printf( */
      /*     "goal reached abandoning thread for %d, %d\n", threadIdx.y, */
      /*     threadIdx.x); */
      barrier.arrive_and_drop();
      break;
    } else {
      bool can_abort = true;
      for (int x = 0; x < GRAINSIZE_X + 2; ++x) {
        for (int y = 0; y < GRAINSIZE_Y + 2; ++y) {
          auto x_lookup = x + x_offset - 1;
          auto y_lookup = y + y_offset - 1;
          if (x_lookup >= 0 && x_lookup < x_max && y_lookup >= 0 &&
              y_lookup < y_max) {
            score_buffer(y, x) = scores(y_lookup, x_lookup);
            if (score_buffer(y, x) == cuda::std::numeric_limits<int>::max()) {
              can_abort = false;
            }
          } else {
            // shouldn't be read but who knows what bug i have below
            score_buffer(y, x) = cuda::std::numeric_limits<int>::max();
          }
        }
      }
      if (can_abort) {
        /* printf( */
        /*     "grain done abandoning thread for %d, %d\n", threadIdx.y, */
        /*     threadIdx.x); */
        barrier.arrive_and_drop();
        break;
      } else {
        // could also arrive(), but don't want to do even more buffering.
        barrier.arrive_and_wait();
      }
      for (int x = 0; x < GRAINSIZE_X + 2; ++x) {
        for (int y = 0; y < GRAINSIZE_Y + 2; ++y) {
          auto x_lookup = x + x_offset - 1;
          auto y_lookup = y + y_offset - 1;
          if (!(x_lookup >= 0 && x_lookup < x_max && y_lookup >= 0 &&
                y_lookup < y_max))
            continue;
          if (score_buffer(y, x) < std::numeric_limits<int>::max())
            continue;

          bool check_xless = (x_lookup > 0) && (x > 0) &&
                             (heights)(y_lookup, x_lookup) <=
                                 1 + (heights)(y_lookup, x_lookup - 1);
          bool check_xmore = (x_lookup < x_max - 1) && (x < GRAINSIZE_X + 1) &&
                             (heights)(y_lookup, x_lookup) <=
                                 1 + (heights)(y_lookup, x_lookup + 1);
          bool check_yless = (y_lookup > 0) && (y > 0) &&
                             (heights)(y_lookup, x_lookup) <=
                                 1 + (heights)(y_lookup - 1, x_lookup);
          bool check_ymore = (y_lookup < y_max - 1) && (y < GRAINSIZE_Y + 1) &&
                             (heights)(y_lookup, x_lookup) <=
                                 1 + (heights)(y_lookup + 1, x_lookup);

          auto best = cuda::std::numeric_limits<int>::max();
          if (check_xmore) {
            best = min(best, (score_buffer)(y, x + 1));
            /* printf( */
            /*     "reaching %d, %d from right with %d\n", y_lookup, x_lookup,
             */
            /*     best); */
          }
          if (check_xless) {
            best = min(best, (score_buffer)(y, x - 1));
            /* printf( */
            /*     "reaching (thread %d %d) %d, %d from left with %d\n", */
            /*     threadIdx.y, threadIdx.x, y_lookup, x_lookup, */
            /*     (score_buffer)(y, x - 1)); */
          }
          if (check_ymore) {
            best = min(best, (score_buffer)(y + 1, x));
            /* printf( */
            /*     "reaching %d, %d from below with %d\n", y_lookup, x_lookup,
             */
            /*     best); */
          }
          if (check_yless) {
            best = min(best, (score_buffer)(y - 1, x));
            /* printf( */
            /*     "reaching %d, %d from above with %d\n", y_lookup, x_lookup,
             */
            /*     best); */
          }

          if (best < cuda::std::numeric_limits<int>::max()) {
            (scores)(y_lookup, x_lookup) = best + 1;
          }
        }
      }
      barrier.arrive_and_wait();
    }
  }
}

template <int GRAINSIZE_X, int GRAINSIZE_Y>
__global__ void cr(
    int* heights, int* scores, int* goal_x, int* goal_y, int* COLS, int* ROWS,
    int* THREADS_X, int* THREADS_Y) {
  /* printf( */
  /*     "launch kernel with %d xthreads, %d ythreads, %d xgrain, %d ygrain\n",
   */
  /*     *THREADS_X, *THREADS_Y, GRAINSIZE_X, GRAINSIZE_Y); */

  auto block = cooperative_groups::this_thread_block();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    init(&barrier, (*THREADS_X) * (*THREADS_Y));
  }
  block.sync();
  CubDebug(cudaPeekAtLastError());

  asdf<GRAINSIZE_X, GRAINSIZE_Y><<<1, {*THREADS_X, *THREADS_Y}>>>(
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
  auto readback = input.scores;
  int* THREADS_X;
  int* THREADS_Y;
  int* COLS;
  int* ROWS;
  int* goal_x;
  int* goal_y;
  int* scores;
  int* heights;
  CubDebugExit(cudaMallocManaged(&THREADS_X, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&THREADS_Y, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&COLS, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&ROWS, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&goal_x, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&goal_y, sizeof(int)));

  *ROWS = input.rows;
  *COLS = input.cols;

  auto row_spec = std::div(*ROWS, 16);
  auto col_spec = std::div(*COLS, 16);
  auto GRAINSIZE_X = col_spec.quot + !!col_spec.rem;
  auto GRAINSIZE_Y = row_spec.quot + !!row_spec.rem;
  row_spec = std::div(*ROWS, GRAINSIZE_Y);
  col_spec = std::div(*COLS, GRAINSIZE_X);
  *THREADS_X = col_spec.quot + !!col_spec.rem;
  *THREADS_Y = row_spec.quot + !!row_spec.rem;

  *goal_x = input.goal_x;
  *goal_y = input.goal_y;
  printf("Will have to run %d threads\n", (*THREADS_X) * (*THREADS_Y));

  CubDebugExit(cudaMalloc(&scores, (*ROWS) * (*COLS) * sizeof(int)));
  CubDebugExit(cudaMalloc(&heights, (*ROWS) * (*COLS) * sizeof(int)));

  CubDebugExit(cudaMemcpy(
      scores, input.scores.data(), (*ROWS) * (*COLS) * sizeof(int),
      cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(
      heights, input.heights.data(), (*ROWS) * (*COLS) * sizeof(int),
      cudaMemcpyHostToDevice));

  if (GRAINSIZE_X == 1 && GRAINSIZE_Y == 1) {
    printf("tiny grains\n");
    cr<1ul, 1ul><<<1, 1>>>(
        heights, scores, goal_x, goal_y, COLS, ROWS, THREADS_X, THREADS_Y);
  } else if (GRAINSIZE_X == 2 && GRAINSIZE_Y == 1) {
    cr<2, 1><<<1, 1>>>(
        heights, scores, goal_x, goal_y, COLS, ROWS, THREADS_X, THREADS_Y);
  } else if (GRAINSIZE_X == 1 && GRAINSIZE_Y == 2) {
    cr<1, 2><<<1, 1>>>(
        heights, scores, goal_x, goal_y, COLS, ROWS, THREADS_X, THREADS_Y);
  } else if (GRAINSIZE_X == 2 && GRAINSIZE_Y == 2) {
    cr<2, 2><<<1, 1>>>(
        heights, scores, goal_x, goal_y, COLS, ROWS, THREADS_X, THREADS_Y);
  } else if (GRAINSIZE_X == 5 && GRAINSIZE_Y == 2) {
    cr<5, 2><<<1, 1>>>(
        heights, scores, goal_x, goal_y, COLS, ROWS, THREADS_X, THREADS_Y);
  } else if (GRAINSIZE_X == 9 && GRAINSIZE_Y == 3) {
    cr<9, 3><<<1, 1>>>(
        heights, scores, goal_x, goal_y, COLS, ROWS, THREADS_X, THREADS_Y);
  } else {
    printf(
        "Didn't instantiate for that configuration %d  %d\n", GRAINSIZE_X,
        GRAINSIZE_Y);
  }
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  CubDebugExit(cudaMemcpy(
      readback.data(), scores, (*ROWS) * (*COLS) * sizeof(int),
      cudaMemcpyDeviceToHost));

  myspan readback_scores_span(readback.data(), *ROWS, *COLS);
  printf("reached goal at %d\n", readback_scores_span(*goal_y, *goal_x));
  printf("Ran %d threads\n", (*ROWS) * (*COLS));

  // part 2

  myspan input_scores_span(input.scores.data(), *ROWS, *COLS);
  myspan heights_span(input.heights.data(), *ROWS, *COLS);
  for (auto x = 0; x < *COLS; ++x) {
    for (auto y = 0; y < *ROWS; ++y) {
      if (heights_span(y, x) == 0) {
        input_scores_span(y, x) = 0;
      }
    }
  }
  CubDebugExit(cudaMemcpy(
      scores, input.scores.data(), (*ROWS) * (*COLS) * sizeof(int),
      cudaMemcpyHostToDevice));
  cr<9, 3><<<1, 1>>>(
      heights, scores, goal_x, goal_y, COLS, ROWS, THREADS_X, THREADS_Y);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());

  /* CubDebugExit(cudaMemcpy( */
  /*       readback.data(), scores, (*ROWS) * (*COLS) * sizeof(int), */
  /*       cudaMemcpyDeviceToHost)); */

  CubDebugExit(cudaMemcpy(
      readback.data() + *goal_x + (*COLS) * (*goal_y),
      scores + *goal_x + (*COLS) * (*goal_y), sizeof(int),
      cudaMemcpyDeviceToHost));

  printf("part 2: %d\n", readback_scores_span(*goal_y, *goal_x));

  CubDebugExit(cudaFree(THREADS_X));
  CubDebugExit(cudaFree(THREADS_Y));
  CubDebugExit(cudaFree(COLS));
  CubDebugExit(cudaFree(ROWS));
  CubDebugExit(cudaFree(goal_x));
  CubDebugExit(cudaFree(goal_y));
  CubDebugExit(cudaFree(scores));
  CubDebugExit(cudaFree(heights));
  return 0;
}
