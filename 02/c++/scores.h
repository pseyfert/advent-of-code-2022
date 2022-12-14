#pragma once
#include "shared.h"

#ifndef __NVCOMPILER
#ifndef SCORES_IMPL
__attribute__((const, simd("notinbranch"))) data_t score(data_t, data_t) noexcept asm("_Z5scoreii");
#else
__attribute__((const)) data_t score(data_t, data_t) noexcept asm("_Z5scoreii");
#endif
#endif
