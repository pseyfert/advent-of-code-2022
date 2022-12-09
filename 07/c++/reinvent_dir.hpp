#pragma once
#include <execution>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include "back_aware_iterators.hpp"

struct file;
struct dir;
using dir_vec = back_aware_vector<dir>;
using dir_iter = typename dir_vec::iterator;
using dir_iter_ref = std::reference_wrapper<const dir_iter>;

struct file {
  std::string m_name;
  unsigned long m_size;
};

struct dir {
  dir(std::string_view name) : m_name(name) {}
  std::string m_name;
  dir_vec m_subdirs;
  std::vector<file> m_files;
  unsigned long size() {
    return std::transform_reduce(
               std::execution::unseq, m_files.begin(), m_files.end(),
               static_cast<unsigned long>(0), std::plus{},
               [](auto& f) { return f.m_size; }) +
           std::transform_reduce(
               std::execution::unseq, m_subdirs.begin(), m_subdirs.end(),
               static_cast<unsigned long>(0), std::plus{},
               [](auto& sd) { return sd.size(); });
  };
};

struct path : boost::stl_interfaces::proxy_iterator_interface<
                  path, std::forward_iterator_tag, std::string,
                  std::vector<dir_iter_ref>> {
  std::vector<dir_iter> m_components;

  path() {}
  path(dir_iter i) : m_components{i} {}
  auto operator*() const {
    return reference(m_components.cbegin(), m_components.cend());
  }

  bool operator==(const path& other) const {
    return ranges::equal(m_components, other.m_components);
  }
  path& operator++() {
    if (m_components.empty()) {
      return *this;
    }
    m_components.push_back(m_components.back()->m_subdirs.begin());

    while (!m_components.empty() && (m_components.back() == end_reached)) {
      m_components.pop_back();
      ++m_components.back();
    }
    return *this;
  }
  path operator++(int) {
    path copy = *this;
    ++(*this);
    return copy;
  }
  operator std::string() {
    if (m_components.empty()) {
      return "";
    }
    std::string retval{m_components.front()->m_name};
    for (auto c : ranges::view::drop(m_components, 1)) {
      retval += '/';
      retval += c->m_name;
    }
    return retval;
  }
  dir& basename() {
    if (m_components.empty() || m_components.back() == end_reached) {
      std::cout << "INVALID OPERATION." << std::endl;
    }
    return *(m_components.back());
  }
  void mkdir(std::string_view name) {
    basename().m_subdirs.push_back(dir{name});
  }
};

// todo std::sentinel_for
struct path_end_t {
  bool operator==(path& p) const {
    return p.m_components.empty();
  }
} path_end;
