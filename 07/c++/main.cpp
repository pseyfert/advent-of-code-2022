#include <algorithm>
#include <filesystem>
#include <fstream>
#include <range/v3/algorithm/find_if.hpp>
#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/getlines.hpp>
#include <range/v3/view/group_by.hpp>
#include <range/v3/view/split.hpp>
#include <range/v3/view/subrange.hpp>
#include <vector>

#include "reinvent_dir.hpp"

int main(int argc, char** argv) {
  const std::filesystem::path& inpath{argv[1]};
  std::ifstream instream(inpath);

  dir fake_fake_root("");
  fake_fake_root.m_subdirs.push_back(dir{""});
  fake_fake_root.m_subdirs.back().m_subdirs.push_back(dir{"/"});

  // wondering how dangerous i'm operating concerning iterator invalidation.
  path cwd{fake_fake_root.m_subdirs.begin()};
  auto data = ranges::getlines_view(instream) | ranges::to_vector;

  ranges::for_each(
      data | ranges::view::group_by([](auto, auto s) {
        bool is_cd = s.starts_with("$ cd");
        return !is_cd;
      }),
      [&cwd](auto lines_for_one_cd) {
        auto line_it = lines_for_one_cd.begin();
        std::string target =
            (*line_it) | ranges::view::drop(5) | ranges::to<std::string>();
        if (target != std::string{".."}) {
          // std::cout << "entering " << target << '\n';
          auto target_dir = ranges::find_if(
              cwd.basename().m_subdirs,
              [&target](dir& d) { return d.m_name == target; });
          if (target_dir == end_reached) {
            std::cout << "CD logic didn't find target\n";
          } else {
            cwd.m_components.push_back(target_dir);
          }
          if (!(ranges::to<std::string>(*(++line_it)).starts_with("$ ls"))) {
            std::cout << "UNEXPECTED no ls after cd\n";
          }
          ranges::for_each(
              ++line_it, lines_for_one_cd.end(), [&cwd](auto ls_line) {
                auto split_line = ls_line | ranges::view::split(' ');
                auto split_line_it = split_line.begin();
                auto dir_or_size =
                    (*(split_line_it++)) | ranges::to<std::string>();
                auto name = (*(split_line_it++)) | ranges::to<std::string>();
                if (dir_or_size == std::string{"dir"}) {
                  // std::cout << "creating dir " << name << '\n';
                  cwd.mkdir(name);
                } else {
                  // std::cout << "creating file " << name << '\n';
                  cwd.basename().m_files.push_back(
                      file{name, std::atoi(dir_or_size.c_str())});
                }
              });
        } else {
          cwd.m_components.pop_back();
          // std::cout << "going up\n";
          if (++line_it != lines_for_one_cd.end()) {
            std::cout << "ASSUMPTION VIOLATED\n";
          }
        }
      });

  // TODO can't really use path as range yet
  // TODO can't really use path as range yet
  // TODO can't really use path as range yet
  // TODO can't really use path as range yet
  // TODO can't really use path as range yet
  // TODO can't really use path as range yet
  // TODO can't really use path as range yet

  std::vector<unsigned long> sizes_1;
  std::vector<unsigned long> sizes_2;

  cwd = path{fake_fake_root.m_subdirs.begin()};
  auto currently_fee = 70000000 - cwd.basename().size();
  auto target_free_up = 30000000 - currently_fee;
  std::cout << "root is " << cwd.basename().size() << " thus need to clear \n"
            << target_free_up << '\n';

  for (; cwd != path_end; ++cwd) {
    if (unsigned long cur_size = cwd.basename().size(); cur_size <= 100000) {
      sizes_1.push_back(cur_size);
    }
    if (unsigned long cur_size = cwd.basename().size(); cur_size >= target_free_up) {
      std::cout << cur_size << "\t\t" << std::string(cwd) << " would work " <<  '\n';
      sizes_2.push_back(cur_size);
    } else {
      std::cout << cur_size << "\t\t" << std::string(cwd) << " would not work " <<  '\n';
    }
  }

  std::cout << "part1 "
            << std::accumulate(
                   sizes_1.begin(), sizes_1.end(), static_cast<unsigned long>(0))
            << '\n';

  std::cout << "part2 " << *std::min_element(sizes_2.begin(), sizes_2.end())
            << '\n';

  // for (auto y : ranges::view::filter(
  //          ranges::subrange(path{fake_fake_root.m_subdirs.begin()},
  //          path_end),
  //          [](auto x) { return true; })) {
  // };

  return 0;
}
