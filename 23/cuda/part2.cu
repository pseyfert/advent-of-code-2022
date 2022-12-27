#include "global_funs.cuh"
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

__global__ void stationary(const int* go, const int* N_elves, bool* retval) {
  *retval = thrust::any_of(
      thrust::device, go, go + *N_elves, [](const auto i) { return !!i; });
}
