#pragma once
#include <boost/stl_interfaces/iterator_interface.hpp>
#include <boost/stl_interfaces/sequence_container_interface.hpp>
#include <execution>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <range/v3/algorithm/equal.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/transform.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template <bool BOOL>
struct bool_warner;
template <typename TYPE>
struct type_warner;

template <typename Container>
struct back_aware_tag : std::false_type {
  using type = Container;
};

// todo typename -> concept
template <typename Container>
struct back_aware_const_iter
    : boost::stl_interfaces::iterator_interface<
          back_aware_const_iter<Container>,
          // todo take from container?
          std::random_access_iterator_tag, typename Container::value_type,
          typename Container::const_reference,
          typename Container::const_pointer,
          typename Container::difference_type> {
  const Container* m_container;
  using underlying_iterator = back_aware_tag<Container>::type::const_iterator;
  back_aware_const_iter(){};
  back_aware_const_iter(Container* c, underlying_iterator it)
      : m_container(c), m_iter(it) {}
  underlying_iterator m_iter;
  underlying_iterator& base_reference() {
    return m_iter;
  }
  underlying_iterator base_reference() const {
    return m_iter;
  }
};

// todo typename -> concept
template <typename Container>
struct back_aware_iter
    : boost::stl_interfaces::iterator_interface<
          back_aware_iter<Container>,
          // todo take from container?
          std::random_access_iterator_tag, typename Container::value_type,
          typename Container::reference, typename Container::pointer,
          typename Container::difference_type> {
  using base_type = boost::stl_interfaces::iterator_interface<
      back_aware_iter<Container>,
      // todo take from container?
      std::random_access_iterator_tag,
      typename std::iterator_traits<
          decltype(std::declval<Container>().begin())>::value_type,
      typename std::iterator_traits<
          decltype(std::declval<Container>().begin())>::reference,
      typename std::iterator_traits<
          decltype(std::declval<Container>().begin())>::pointer,
      typename std::iterator_traits<
          decltype(std::declval<Container>().begin())>::difference_type>;
  using iterator_concept = base_type::iterator_concept;
  using iterator_category = base_type::iterator_category;
  using value_type = base_type::value_type;
  using reference = base_type::reference;
  using pointer = base_type::pointer;
  using difference_type = base_type::difference_type;

  Container* m_container{nullptr};

  using underlying_iterator = back_aware_tag<Container>::type::iterator;

  back_aware_iter(){};
  back_aware_iter(Container* c, underlying_iterator it)
      : m_container(c), m_iter(it) {}

  underlying_iterator m_iter;
  underlying_iterator& base_reference() {
    return m_iter;
  }
  underlying_iterator base_reference() const {
    return m_iter;
  }
  operator back_aware_const_iter<Container>() {
    return back_aware_const_iter{m_container, m_iter};
  }
};

template <typename T>
struct back_aware_vector
    : boost::stl_interfaces::sequence_container_interface<
          back_aware_vector<T>,
          boost::stl_interfaces::element_layout::contiguous> {
  using underlying_vector = std::vector<T>;
  using underlying_iterator = underlying_vector::iterator;
  using underlying_const_iterator = underlying_vector::const_iterator;
  using value_type = underlying_vector::value_type;
  using pointer = underlying_vector::pointer;
  using const_pointer = underlying_vector::const_pointer;
  using reference = underlying_vector::reference;
  using const_reference = underlying_vector::const_reference;
  using size_type = underlying_vector::size_type;
  using difference_type = underlying_vector::difference_type;
  using iterator = back_aware_iter<back_aware_vector>;
  using const_iterator = back_aware_const_iter<back_aware_vector>;

  underlying_vector m_vector;
  back_aware_vector() {}
  template <
      std::forward_iterator OtherIter, std::sentinel_for<OtherIter> Sentinel>
  back_aware_vector(OtherIter first, Sentinel last) : m_vector(first, last) {}

  iterator begin() {
    return iterator{this, m_vector.begin()};
  }
  iterator end() {
    return iterator{this, m_vector.end()};
  }

  size_type size() {
    return m_vector.size();
  }
  bool empty() {
    return m_vector.empty();
  }
  void resize(size_type n) {
    return m_vector.resize(n);
  }
  void resize(size_type n, value_type x) {
    return m_vector.resize(n, x);
  }
  void reserve(size_type n) {
    return m_vector.reserve(n);
  }

  auto& operator[](size_type n) {
    return m_vector[n];
  }
  const auto& operator[](size_type n) const {
    return m_vector[n];
  }

  pointer data() {
    return m_vector.data();
  }

  template <typename... Args>
  reference emplace_back(Args&&... args) {
    return m_vector.emplace_back(args...);
  }

  using base_type = boost::stl_interfaces::sequence_container_interface<
      back_aware_vector<T>, boost::stl_interfaces::element_layout::contiguous>;
  using base_type::begin;
  using base_type::end;
  using base_type::erase;
  using base_type::insert;
};

template <typename T>
struct back_aware_tag<back_aware_vector<T>> : std::true_type {
  using type = back_aware_vector<T>::underlying_vector;
};

// todo std::sentinel_for
struct end_reached_t {
  template <typename Container>
  bool operator==(const back_aware_const_iter<Container>& other) {
    return other.m_iter == other.m_container->cend();
  }
  template <typename T>
  bool operator==(const back_aware_const_iter<back_aware_vector<T>>& other) {
    return other.m_iter == other.m_container->m_vector.cend();
  }
  template <typename Container>
  bool operator==(const back_aware_iter<Container>& other) {
    return other.m_iter == other.m_container->end();
  }
  template <typename T>
  bool operator==(const back_aware_iter<back_aware_vector<T>>& other) {
    return other.m_iter == other.m_container->m_vector.end();
  }
} end_reached;

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
