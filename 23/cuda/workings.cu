#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

#include <cooperative_groups.h>

#include <cuda/std/array>

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

enum class Direction { North, South, West, East };

template <Direction d>
__device__ bool clear_fn(
    const int this_elve, int const* current_x, int const* current_y,
    const int N_elves) {
  if constexpr (d == Direction::North) {
    return clear_north(this_elve, current_x, current_y, N_elves);
  } else if constexpr (d == Direction::South) {
    return clear_south(this_elve, current_x, current_y, N_elves);
  } else if constexpr (d == Direction::East) {
    return clear_east(this_elve, current_x, current_y, N_elves);
  } else if constexpr (d == Direction::West) {
    return clear_west(this_elve, current_x, current_y, N_elves);
  }
}

struct Preference {
  Direction d_;
  __device__ Preference(Direction d) : d_{d} {}

  __device__ bool clear(
      const int this_elve, int const* current_x, int const* current_y,
      const int N_elves) const {
    if (d_ == Direction::North) {
      return clear_fn<Direction::North>(
          this_elve, current_x, current_y, N_elves);
    } else if (d_ == Direction::South) {
      return clear_fn<Direction::South>(
          this_elve, current_x, current_y, N_elves);
    } else if (d_ == Direction::West) {
      return clear_fn<Direction::West>(
          this_elve, current_x, current_y, N_elves);
    } else if (d_ == Direction::East) {
      return clear_fn<Direction::East>(
          this_elve, current_x, current_y, N_elves);
    }
  }

  __device__ void fill_preference(
      const int this_elve, int const* current_x, int const* current_y,
      int* proposed_x, int* proposed_y) const {
    if (d_ == Direction::North) {
      proposed_x[this_elve] = current_x[this_elve];
      proposed_y[this_elve] = current_y[this_elve] + 1;
    } else if (d_ == Direction::South) {
      proposed_x[this_elve] = current_x[this_elve];
      proposed_y[this_elve] = current_y[this_elve] - 1;
    } else if (d_ == Direction::West) {
      proposed_x[this_elve] = current_x[this_elve] - 1;
      proposed_y[this_elve] = current_y[this_elve];
    } else if (d_ == Direction::East) {
      proposed_x[this_elve] = current_x[this_elve] + 1;
      proposed_y[this_elve] = current_y[this_elve];
    }
  }
};

__global__ void propose_move(
    int* proposed_x, int* proposed_y, int const* current_x,
    int const* current_y, int N_elves, int round_mod_four) {
  const auto this_elve = threadIdx.x;

  // init with NSWE
  // TODO: share withing block
  auto preferences = cuda::std::array<Preference, 7>{
      Preference(Direction::North), Preference(Direction::South),
      Preference(Direction::West),  Preference(Direction::East),
      Preference(Direction::North), Preference(Direction::South),
      Preference(Direction::West)};

  proposed_x[this_elve] = current_x[this_elve];
  proposed_y[this_elve] = current_y[this_elve];
  for (auto pit = preferences.cbegin() + round_mod_four;
       pit != preferences.cbegin() + round_mod_four + 4; pit++) {
    if (pit->clear(this_elve, current_x, current_y, N_elves)) {
      pit->fill_preference(
          this_elve, current_x, current_y, proposed_x, proposed_y);
    }
  }
  /* std::rotate(preferences.begin(), preferences.begin() + 1,
   * preferences.end()); */
}

__global__ void apply_check(
    int* current_x, int* current_y, const int* proposed_x,
    const int* proposed_y, const int* go_ahead) {
  const auto this_elve = threadIdx.x;
  if (go_ahead[this_elve]) {
    current_x[this_elve] = proposed_x[this_elve];
    current_y[this_elve] = proposed_y[this_elve];
  }
}

__global__ void do_round(
    int* N_elves, int* current_x, int* current_y, int* go_ahead,
    int* proposed_x, int* proposed_y, int* round_mod_four) {
  propose_move<<<1, N_elves>>>(
      proposed_x, proposed_y, current_x, current_y, N_elves, round_mod_four);
  // sync
  collision_check<<<1, N_elves>>>(go_ahead, proposed_x, proposed_y, N_elves);
  // sync
  apply_check<<<1, N_elves>>>(
      current_x, current_y, proposed_x, proposed_y, go_ahead);
}
