#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/indices.hpp>
#include <string>
#include "types.h"

int main(int argc, char** argv) {
  const std::filesystem::path& in_path{argv[1]};
  std::ifstream instream(in_path);

  std::unordered_set<pos, myposhash> tail_poses1;
  std::unordered_set<pos, myposhash> tail_poses2;

  state1 s1;
  state2 s2;
  for (auto& line : ranges::getlines_view(instream)) {
    std::string repeat_str =
        line | ranges::view::drop(2) | ranges::to<std::string>();
    auto repeat = std::atoi(repeat_str.c_str());
    for ([[maybe_unused]] auto x_ : ranges::view::indices(repeat)) {
      s1.move(line[0]);
      s2 = s2.pull(line[0]);
      // std::cout << line[0] << "\t\t";
      // std::cout << "H(" << s2.m_data[0].first << ','
      //                   << s2.m_data[0].second << ")\t1("
      //                   << s2.m_data[1].first << ','
      //                   << s2.m_data[1].second << ")\t2("
      //                   << s2.m_data[2].first << ','
      //                   << s2.m_data[2].second << ")\t3("
      //                   << s2.m_data[3].first << ','
      //                   << s2.m_data[3].second << ")...\tT("
      //                   << s2.m_data[9].first << ','
      //                   << s2.m_data[9].second << ")\n";
      tail_poses1.insert(s1.TAIL);
      tail_poses2.insert(s2.m_data.back());
    }
  }
  for ([[maybe_unused]] auto x_ : ranges::view::indices(s2.m_data.size() - 1)) {
    s2 = s2.pull('N');
    tail_poses2.insert(s2.m_data.back());
  }
  std::cout << "part 1 " << tail_poses1.size() << '\n';

  std::cout << "part 2 " << tail_poses2.size() << '\n';
  return 0;
}
