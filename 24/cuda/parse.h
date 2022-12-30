#pragma once
#include <string>
#include <tuple>
#include <cstddef>

using mysizet = int;

std::tuple<std::string, mysizet, mysizet> read(char*);
