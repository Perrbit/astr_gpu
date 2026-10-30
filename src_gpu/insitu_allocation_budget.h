#ifndef ASTR_INSITU_ALLOCATION_BUDGET_H
#define ASTR_INSITU_ALLOCATION_BUDGET_H

#include <cstdint>
#include <stdexcept>

extern "C" void astr_insitu_local_allocation_recovery(int);

namespace astr_insitu {
class AllocationBudgetExceeded : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Only use around local work followed by a matching product-failure rendezvous.
// In particular, this scope must not enclose an unmatched MPI collective.
class LocalAllocationRecovery {
public:
  LocalAllocationRecovery() { astr_insitu_local_allocation_recovery(1); }
  ~LocalAllocationRecovery() { astr_insitu_local_allocation_recovery(0); }
  LocalAllocationRecovery(const LocalAllocationRecovery&)=delete;
  LocalAllocationRecovery& operator=(const LocalAllocationRecovery&)=delete;
};

// A decrease in device-wide free memory includes retained/deferred allocations
// and other processes. Counting other processes is intentionally conservative.
inline bool admit_device_allocation(std::uint64_t bytes, std::uint64_t free,
    std::uint64_t baseline_free, std::uint64_t limit, std::uint64_t reserve) {
  const auto used = baseline_free > free ? baseline_free - free : 0;
  return used <= limit && bytes <= limit - used && free >= reserve && bytes <= free - reserve;
}
}
#endif
