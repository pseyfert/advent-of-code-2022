#pragma once
#include <array>
#include <execution>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

using pos = std::pair<int, int>;

struct myposhash {
  std::size_t operator()(const pos& p) const {
    // file has only 2000 lines, no line has a more than 2 digits, i.e.
    // we won't move more than 200000 in any direction
    // 200000 ~ 2^18
    std::size_t quadrant;
    if (p.first > 0 && p.second > 0)
      quadrant = 0;
    if (p.first < 0 && p.second > 0)
      quadrant = 1;
    if (p.first > 0 && p.second < 0)
      quadrant = 2;
    if (p.first < 0 && p.second < 0)
      quadrant = 3;
    return (std::abs(p.first) << 20) | (std::abs(p.second) << 20) | quadrant;
  }
};

//
//  y ↑, x →
//

struct state1 {
  pos HEAD{0, 0};
  pos TAIL{0, 0};

  void move_up() {
    HEAD.first++;
    if (HEAD.first > TAIL.first + 1) {
      TAIL.first = HEAD.first - 1;
      TAIL.second = HEAD.second;
    }
  }
  void move_down() {
    HEAD.first--;
    if (HEAD.first < TAIL.first - 1) {
      TAIL.first = HEAD.first + 1;
      TAIL.second = HEAD.second;
    }
  }
  void move_right() {
    HEAD.second++;
    if (HEAD.second > TAIL.second + 1) {
      TAIL.second = HEAD.second - 1;
      TAIL.first = HEAD.first;
    }
  }
  void move_left() {
    HEAD.second--;
    if (HEAD.second < TAIL.second - 1) {
      TAIL.second = HEAD.second + 1;
      TAIL.first = HEAD.first;
    }
  }
  void move(char c) {
    if (c == 'U') {
      move_up();
    } else if (c == 'D') {
      move_down();
    } else if (c == 'R') {
      move_right();
    } else if (c == 'L') {
      move_left();
    }
  }
};

#ifdef __NVCOMPILER
void __builtin_unreachable() {}
#endif

struct state2 {
#ifndef __NVCOMPILER
  using state_vec = std::array<pos, 10>;
#else
  using state_vec = std::vector<pos>;
#endif

  state_vec m_data;

  state2 pull(char d) {
    // This is meant to work as a pipeline.
    // i.e.:
    //  * when knot 1 makes step 1, none of the others move.
    //  * when knot 1 makes step 2, knot 2 follows step 1.
    //  * when knot 1 makes step 3, knot 2 follows step 2, knot 3 follows
    //  step 1.
    //
#ifndef __NVCOMPILER
    state_vec retval;
#else
    state_vec retval;
    retval.resize(10);
#endif
    std::adjacent_difference(
        std::execution::par_unseq, m_data.begin(), m_data.end(), retval.begin(),
        [](const pos& cur, const pos& prev) -> pos {
          if (prev.first == cur.first + 2) {
            if (prev.second == cur.second - 2) {
              return {cur.first + 1, cur.second - 1};
            } else if (prev.second == cur.second - 1) {
              return {cur.first + 1, cur.second - 1};
            } else if (prev.second == cur.second + 0) {
              return {cur.first + 1, cur.second};
            } else if (prev.second == cur.second + 1) {
              return {cur.first + 1, cur.second + 1};
            } else if (prev.second == cur.second + 2) {
              return {cur.first + 1, cur.second + 1};
            } else {
              __builtin_unreachable();
            }
          } else if (prev.first == cur.first + 1) {
            if (prev.second == cur.second - 2) {
              return {cur.first + 1, cur.second - 1};
            } else if (prev.second == cur.second - 1) {
              return cur;
            } else if (prev.second == cur.second + 0) {
              return cur;
            } else if (prev.second == cur.second + 1) {
              return cur;
            } else if (prev.second == cur.second + 2) {
              return {cur.first + 1, cur.second + 1};
            } else {
              __builtin_unreachable();
            }
          } else if (prev.first == cur.first + 0) {
            if (prev.second == cur.second - 2) {
              return {cur.first, cur.second - 1};
            } else if (prev.second == cur.second - 1) {
              return cur;
            } else if (prev.second == cur.second + 0) {
              return cur;
            } else if (prev.second == cur.second + 1) {
              return cur;
            } else if (prev.second == cur.second + 2) {
              return {cur.first, cur.second + 1};
            } else {
              __builtin_unreachable();
            }
          } else if (prev.first == cur.first - 1) {
            if (prev.second == cur.second - 2) {
              return {cur.first - 1, cur.second - 1};
            } else if (prev.second == cur.second - 1) {
              return cur;
            } else if (prev.second == cur.second + 0) {
              return cur;
            } else if (prev.second == cur.second + 1) {
              return cur;
            } else if (prev.second == cur.second + 2) {
              return {cur.first - 1, cur.second + 1};
            } else {
              __builtin_unreachable();
            }
          } else if (prev.first == cur.first - 2) {
            if (prev.second == cur.second - 2) {
              return {cur.first - 1, cur.second - 1};
            } else if (prev.second == cur.second - 1) {
              return {cur.first - 1, cur.second - 1};
            } else if (prev.second == cur.second + 0) {
              return {cur.first - 1, cur.second};
            } else if (prev.second == cur.second + 1) {
              return {cur.first - 1, cur.second + 1};
            } else if (prev.second == cur.second + 2) {
              return {cur.first - 1, cur.second + 1};
            } else {
              __builtin_unreachable();
            }
          } else {
            __builtin_unreachable();
          }
        });
    retval[0] = move(d, retval[0]);
    return state2{retval};
  }

  pos move(char d, pos HEAD) {
    if (d == 'U') {
      HEAD.first++;
    } else if (d == 'D') {
      HEAD.first--;
    } else if (d == 'R') {
      HEAD.second++;
    } else if (d == 'L') {
      HEAD.second--;
    } else if (d == 'N') {
    } else {
      __builtin_unreachable();
    }
    return HEAD;
  }
};
