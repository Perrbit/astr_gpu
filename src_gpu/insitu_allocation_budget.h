#ifndef ASTR_INSITU_ALLOCATION_BUDGET_H
#define ASTR_INSITU_ALLOCATION_BUDGET_H

#include <cstdint>

namespace astr_insitu {
// A decrease in device-wide free memory includes retained/deferred allocations
// and other processes. Counting other processes is intentionally conservative.
inline bool admit_device_allocation(std::uint64_t bytes, std::uint64_t free,
    std::uint64_t baseline_free, std::uint64_t limit, std::uint64_t reserve) {
  const auto used = baseline_free > free ? baseline_free - free : 0;
  return used <= limit && bytes <= limit - used && free >= reserve && bytes <= free - reserve;
}
}
#endif
