#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

// N = y+
// S = y-
// W = x-
// E = x+

struct CollisionDetector : public thrust::binary_function<int, int, bool> {
  int target_x_;
  int target_y_;
  __device__ CollisionDetector(int target_x, int target_y)
      : target_x_(target_x), target_y_(target_y) {}

  __device__ bool operator()(const int x, const int y) const {
    return x == target_x_ && y == target_y_;
  }

  template <typename Tuple>
  __device__ bool operator()(const Tuple& t) const {
    return thrust::get<0>(t) == target_x_ && thrust::get<1>(t) == target_y_;
  }
};

__global__ void collision_check(
    int* go_ahead, int const* proposed_x, int const* proposed_y, int N_elves) {
  const auto this_elve = threadIdx.x;
  const auto target_x = proposed_x[this_elve];
  const auto target_y = proposed_y[this_elve];

  const auto incoming = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(proposed_x, proposed_y),
      thrust::make_zip_iterator(proposed_x + N_elves, proposed_y + N_elves),
      CollisionDetector(target_x, target_y));
  if (incoming == 1)
    go_ahead[this_elve] = 1;
  else
    go_ahead[this_elve] = 0;
}

__device__ bool clear_north(
    const int this_elve, int const* current_x, int const* current_y,
    const int N_elves) {
  const auto N = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve], current_y[this_elve] + 1));
  const auto NE = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] + 1, current_y[this_elve] + 1));
  const auto NW = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] - 1, current_y[this_elve] + 1));
  return N + NE + NW == 0;
}
__device__ bool clear_south(
    const int this_elve, int const* current_x, int const* current_y,
    const int N_elves) {
  const auto S = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve], current_y[this_elve] - 1));
  const auto SE = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] + 1, current_y[this_elve] - 1));
  const auto SW = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] - 1, current_y[this_elve] - 1));
  return S + SE + SW == 0;
}
__device__ bool clear_east(
    const int this_elve, int const* current_x, int const* current_y,
    const int N_elves) {
  const auto E = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] + 1, current_y[this_elve]));
  const auto NE = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] + 1, current_y[this_elve] + 1));
  const auto SE = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] + 1, current_y[this_elve] - 1));
  return E + NE + SE == 0;
}
__device__ bool clear_west(
    const int this_elve, int const* current_x, int const* current_y,
    const int N_elves) {
  const auto W = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] - 1, current_y[this_elve]));
  const auto NW = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] - 1, current_y[this_elve] + 1));
  const auto SW = thrust::count_if(
      thrust::device, thrust::make_zip_iterator(current_x, current_y),
      thrust::make_zip_iterator(current_x + N_elves, current_y + N_elves),
      CollisionDetector(current_x[this_elve] - 1, current_y[this_elve] - 1));
  return W + SW + NW == 0;
}

__global__ void propose_move(
    int* proposed_x, int* proposed_y, int const* current_x,
    int const* current_y, int N_elves) {
  const auto this_elve = threadIdx.x;
}
