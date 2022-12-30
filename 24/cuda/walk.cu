#define CUB_STDERR
#include <cooperative_groups.h>
#include <cub/util_debug.cuh>
#include <cuda/std/barrier>
#include <experimental/mdspan>
#include <string>

// modified from https://stackoverflow.com/a/14038590
#include <assert.h>
#define cdpErrchk(ans) \
  { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(
    cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    printf(
        "%s:%d GPU kernel assert %d: %s \n", file, line, code,
        cudaGetErrorString(code));
    if (abort)
      assert(0);
  }
}

using myspan = std::experimental::mdspan<
    int, std::experimental::extents<
             size_t, std::experimental::dynamic_extent,
             std::experimental::dynamic_extent>>;

__device__ dim3 mapsize;

__device__ std::size_t x_begin() {
  return threadIdx.x * (mapsize.x / blockDim.x + !!(mapsize.x % blockDim.x));
}
__device__ std::size_t x_end() {
  return min(
      (threadIdx.x + 1) * (mapsize.x / blockDim.x + !!(mapsize.x % blockDim.x)),
      mapsize.x);
}

__device__ std::size_t y_begin() {
  return threadIdx.y * (mapsize.y / blockDim.y + !!(mapsize.y % blockDim.y));
}
__device__ std::size_t y_end() {
  return min(
      (threadIdx.y + 1) * (mapsize.y / blockDim.y + !!(mapsize.y % blockDim.y)),
      mapsize.y);
}

__device__ std::size_t upstorm_y(std::size_t y_now, std::size_t round) {
  return (y_now + round) % mapsize.y;
}

__device__ std::size_t leftstorm_x(std::size_t x_now, std::size_t round) {
  return (x_now + round) % mapsize.x;
}

__device__ std::size_t rightstorm_x(std::size_t x_now, std::size_t round) {
  while (round > x_now && round >= mapsize.x) {
    round -= mapsize.x;
  }

  if (round <= x_now)
    return x_now - round;
  return (x_now + mapsize.x) - round;
}

__device__ std::size_t downstorm_y(std::size_t y_now, std::size_t round) {
  while (round > y_now && round >= mapsize.y) {
    round -= mapsize.y;
  }

  if (round <= y_now)
    return y_now - round;
  return (y_now + mapsize.y) - round;
}

__device__ cuda::barrier<cuda::thread_scope_block>::arrival_token print(
    myspan& storm_left, myspan& storm_right, myspan& storm_up,
    myspan& storm_down, myspan& exploration, int& round,
    cuda::barrier<cuda::thread_scope_block>& barrier,
    cuda::barrier<cuda::thread_scope_block>::arrival_token&& token) {
  barrier.wait(std::move(token));
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int x = 0; x < mapsize.x + 2; ++x) {
      printf("#");
    }
    printf("\n");
    for (int y = 0; y < mapsize.y; ++y) {
      printf("#");
      for (int x = 0; x < mapsize.x; ++x) {
        char to_be_put = '.';
        if (storm_left(y, leftstorm_x(x, round)))
          to_be_put = '<';
        if (storm_right(y, rightstorm_x(x, round)))
          to_be_put = '>';
        if (storm_up(upstorm_y(y, round), x))
          to_be_put = '^';
        if (storm_down(downstorm_y(y, round), x))
          to_be_put = 'v';
        if (auto sum = storm_left(y, leftstorm_x(x, round)) +
                       storm_right(y, rightstorm_x(x, round)) +
                       storm_up(upstorm_y(y, round), x) +
                       storm_down(downstorm_y(y, round), x);
            sum > 1)
          to_be_put = sum + '0';
        printf("%c", to_be_put);
      }
      printf("#\n");
    }
    for (int x = 0; x < mapsize.x + 2; ++x) {
      printf("#");
    }
    printf("\n\n");
  }
  return barrier.arrive();
}

__global__ void proceed(int* map_arg, int* X, int* Y) {
  __shared__ myspan map;
  __shared__ myspan storm_left;
  __shared__ myspan storm_right;
  __shared__ myspan storm_up;
  __shared__ myspan storm_down;
  __shared__ myspan exploration;

  auto block = cooperative_groups::this_thread_block();
  __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    mapsize.x = *X;
    mapsize.y = *Y;
    auto storage_size = mapsize.y * mapsize.x;
    auto block_size = blockDim.x * blockDim.y;
    init(&barrier, block_size);
    map = myspan(map_arg, mapsize.y, mapsize.x);
    int* store;
    cdpErrchk(cudaMalloc(&store, storage_size * sizeof(int)));
    storm_left = myspan(store, mapsize.y, mapsize.x);
    cdpErrchk(cudaMalloc(&store, storage_size * sizeof(int)));
    storm_right = myspan(store, mapsize.y, mapsize.x);
    cdpErrchk(cudaMalloc(&store, storage_size * sizeof(int)));
    storm_up = myspan(store, mapsize.y, mapsize.x);
    cdpErrchk(cudaMalloc(&store, storage_size * sizeof(int)));
    storm_down = myspan(store, mapsize.y, mapsize.x);
    cdpErrchk(cudaMalloc(&store, storage_size * sizeof(int)));
    exploration = myspan(store, mapsize.y, mapsize.x);
  }
  block.sync();
  cdpErrchk(cudaPeekAtLastError());

  for (auto y = y_begin(); y < y_end(); ++y) {
    for (auto x = x_begin(); x < x_end(); ++x) {
      storm_left(y, x) = 0;
      storm_right(y, x) = 0;
      storm_up(y, x) = 0;
      storm_down(y, x) = 0;
      if (map(y, x) == '<') {
        storm_left(y, x) = 1;
      } else if (map(y, x) == '^') {
        storm_up(y, x) = 1;
      } else if (map(y, x) == '>') {
        storm_right(y, x) = 1;
      } else if (map(y, x) == 'v') {
        storm_down(y, x) = 1;
      }
    }
  }
  barrier.arrive_and_wait();

  for (auto round = 0; round <= 18; ++round) {
    barrier.wait(print(
        storm_left, storm_right, storm_up, storm_down, exploration, round,
        barrier, barrier.arrive()));
  }

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    cudaFree(storm_up.data_handle());
    cudaFree(storm_down.data_handle());
    cudaFree(storm_left.data_handle());
    cudaFree(storm_right.data_handle());
    cudaFree(exploration.data_handle());
  }
}

__device__ void p() {
  printf(
      "(%d, %d) out of %d x %d\n", threadIdx.x, threadIdx.y, blockDim.x,
      blockDim.y);
}

__global__ void test() {
  p();
}

int main() {
  int* k;
  int* X;
  int* Y;

  std::string raw = ">>.<^<.<..<<>v.><><^v^^>";
  CubDebugExit(cudaMallocManaged(&k, raw.size() * sizeof(int)));
  CubDebugExit(cudaMallocManaged(&X, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&Y, sizeof(int)));
  *X = 6;
  *Y = 4;
  for (int i = 0; i < raw.size(); ++i) {
    k[i] = raw[i];
  }

  proceed<<<1, {2, 2}>>>(k, X, Y);
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());
  CubDebugExit(cudaPeekAtLastError());
}
