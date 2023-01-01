#include <cub/util_debug.cuh>
#include <gtest/gtest.h>
#include "helpers.cuh"

/* #define EXPECT_DEVICE_GOOD(e) EXPECT_FALSE(CUB_NS_QUALIFIER::Debug((cudaError_t) (e), __FILE__, __LINE__)) */
/* #define ASSERT_DEVICE_GOOD(e) ASSERT_FALSE(CUB_NS_QUALIFIER::Debug((cudaError_t) (e), __FILE__, __LINE__)) */
#define EXPECT_DEVICE_GOOD(e) EXPECT_FALSE(e) << cudaGetErrorString(e)
#define ASSERT_DEVICE_GOOD(e) ASSERT_FALSE(e) << cudaGetErrorString(e)

/* TEST(Circle, DoIHaveOffByOne) { */
__host__ __device__ void DoIHaveOffByOne_impl() {
  Circle c{
      .center = Location{.x = tagged_int<x_tag>(8), .y = tagged_int<y_tag>(7)},
      .radius = 9};
  EXPECT_TRUE(c.is_inside(
      Location{.x = tagged_int<x_tag>(0), .y = tagged_int<y_tag>(7)}));
  EXPECT_TRUE(c.is_inside(
      Location{.x = tagged_int<x_tag>(2), .y = tagged_int<y_tag>(10)}));
  EXPECT_TRUE(c.is_inside(
      Location{.x = tagged_int<x_tag>(8), .y = tagged_int<y_tag>(-2)}));
  EXPECT_TRUE(c.is_inside(
      Location{.x = tagged_int<x_tag>(8), .y = tagged_int<y_tag>(16)}));
  EXPECT_TRUE(c.is_inside(
      Location{.x = tagged_int<x_tag>(8), .y = tagged_int<y_tag>(7)}));
  EXPECT_TRUE(c.is_inside(
      Location{.x = tagged_int<x_tag>(-1), .y = tagged_int<y_tag>(7)}));
  EXPECT_TRUE(c.is_inside(
      Location{.x = tagged_int<x_tag>(17), .y = tagged_int<y_tag>(7)}));

  EXPECT_FALSE(c.is_inside(
      Location{.x = tagged_int<x_tag>(-2), .y = tagged_int<y_tag>(7)}));
  EXPECT_FALSE(c.is_inside(
      Location{.x = tagged_int<x_tag>(18), .y = tagged_int<y_tag>(7)}));
  EXPECT_FALSE(c.is_inside(
      Location{.x = tagged_int<x_tag>(7), .y = tagged_int<y_tag>(22)}));
  EXPECT_FALSE(c.is_inside(
      Location{.x = tagged_int<x_tag>(8), .y = tagged_int<y_tag>(-3)}));
  EXPECT_FALSE(c.is_inside(
      Location{.x = tagged_int<x_tag>(8), .y = tagged_int<y_tag>(17)}));
  EXPECT_FALSE(c.is_inside(
      Location{.x = tagged_int<x_tag>(0), .y = tagged_int<y_tag>(5)}));

  {
    auto p = c.project(tagged_int<y_tag>(17));
    EXPECT_FALSE(p.has_value());
  }
  {
    auto p = c.project(tagged_int<y_tag>(-3));
    EXPECT_FALSE(p.has_value());
  }
  {
    auto p = c.project(tagged_int<y_tag>(7));
    ASSERT_TRUE(p.has_value());
    EXPECT_TRUE(std::int32_t(p->first) == -1);
    EXPECT_TRUE(std::int32_t(p->second) == 17);
  }
  {
    auto p = c.project(tagged_int<y_tag>(10));
    ASSERT_TRUE(p.has_value());
    EXPECT_TRUE(std::int32_t(p->first) == 2);
    EXPECT_TRUE(std::int32_t(p->second) == 14);
  }
}

__global__ void DoIHaveOffByOne_wrapper() {
  DoIHaveOffByOne_impl();
}

TEST(Circle, DoIHaveOffByOne_host) {
  DoIHaveOffByOne_impl();
}

TEST(Circle, DoIHaveOffByOne_device) {
  DoIHaveOffByOne_wrapper<<<1, 1>>>();
  ASSERT_DEVICE_GOOD(cudaPeekAtLastError());
  ASSERT_DEVICE_GOOD(cudaDeviceSynchronize());
  ASSERT_DEVICE_GOOD(cudaPeekAtLastError());
}
