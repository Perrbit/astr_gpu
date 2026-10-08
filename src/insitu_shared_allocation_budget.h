#ifndef ASTR_INSITU_SHARED_ALLOCATION_BUDGET_H
#define ASTR_INSITU_SHARED_ALLOCATION_BUDGET_H

#include <pthread.h>
#include <cerrno>
#include <cstdint>
#include <stdexcept>

namespace astr_insitu {
struct SharedAllocationBudget {
  pthread_mutex_t mutex;
  std::uint64_t held,peak,admissions,releases,refusals;
};

inline void initialize_shared_allocation_budget(SharedAllocationBudget& budget) {
  pthread_mutexattr_t attributes;
  if(pthread_mutexattr_init(&attributes)!=0)
    throw std::runtime_error("shared allocation mutex attributes");
  int status=pthread_mutexattr_setpshared(&attributes,PTHREAD_PROCESS_SHARED);
  if(!status) status=pthread_mutexattr_setrobust(&attributes,PTHREAD_MUTEX_ROBUST);
  if(!status) status=pthread_mutex_init(&budget.mutex,&attributes);
  pthread_mutexattr_destroy(&attributes);
  if(status) throw std::runtime_error("shared allocation mutex initialization");
  budget.held=budget.peak=budget.admissions=budget.releases=budget.refusals=0;
}

inline void destroy_shared_allocation_budget(SharedAllocationBudget& budget) {
  if(pthread_mutex_destroy(&budget.mutex)!=0)
    throw std::runtime_error("shared allocation mutex destruction");
}

class SharedAllocationLock {
  SharedAllocationBudget* budget;
public:
  explicit SharedAllocationLock(SharedAllocationBudget& value):SharedAllocationLock(&value) {}
  explicit SharedAllocationLock(SharedAllocationBudget* value):budget(value) {
    if(!budget) return;
    const int status=pthread_mutex_lock(&budget->mutex);
    if(status==EOWNERDEAD) {
      pthread_mutex_consistent(&budget->mutex);
      pthread_mutex_unlock(&budget->mutex);
      throw std::runtime_error("shared allocation lock owner died");
    }
    if(status) throw std::runtime_error("shared allocation mutex lock");
  }
  ~SharedAllocationLock() { if(budget) pthread_mutex_unlock(&budget->mutex); }
  SharedAllocationLock(const SharedAllocationLock&)=delete;
  SharedAllocationLock& operator=(const SharedAllocationLock&)=delete;
};

// The caller holds the process-shared lock across admission and reservation.
inline bool reserve_shared_allocation(SharedAllocationBudget& budget,
    std::uint64_t bytes,std::uint64_t available) {
  if(budget.held>available || bytes>available-budget.held) {
    ++budget.refusals;
    return false;
  }
  budget.held+=bytes;
  if(budget.held>budget.peak) budget.peak=budget.held;
  ++budget.admissions;
  return true;
}

inline void release_shared_allocation(SharedAllocationBudget& budget,std::uint64_t bytes) {
  if(bytes>budget.held) throw std::runtime_error("shared allocation release underflow");
  budget.held-=bytes;
  ++budget.releases;
}
}
#endif
