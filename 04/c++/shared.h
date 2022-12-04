#include <cstdint>
#include <filesystem>
#include "SOAContainer.h"

using data_t = std::uint16_t;
using score_t = std::uint32_t;

namespace unprocessed {
SOAFIELD_TRIVIAL(start_a, start_a, data_t);
SOAFIELD_TRIVIAL(end_a, end_a, data_t);
SOAFIELD_TRIVIAL(start_b, start_b, data_t);
SOAFIELD_TRIVIAL(end_b, end_b, data_t);
SOASKIN_TRIVIAL(team, start_a, end_a, start_b, end_b);
}  // namespace unprocessed

using container_t = typename SOA::Container<std::vector, unprocessed::team>;

container_t input(const std::filesystem::path& inpath);

score_t part1(const container_t&);
score_t part2(const container_t&);
