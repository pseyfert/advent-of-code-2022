#pragma once

__global__ void do_round(
    int* N_elves, int* current_x, int* current_y, int* go_ahead,
    int* proposed_x, int* proposed_y, int* round_mod_four);

__global__ void empty_spaces(
    int const* current_x, int const* current_y, const int* N_elves,
    int* result);
