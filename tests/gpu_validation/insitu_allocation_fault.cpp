#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <new>

// Test-only interposition: refuse one exact vector size on one MPI rank.
void* operator new(std::size_t bytes)
{
  const char* size = std::getenv("ASTR_TEST_NEW_BYTES");
  const char* rank = std::getenv("OMPI_COMM_WORLD_RANK");
  const char* selected = std::getenv("ASTR_TEST_NEW_RANK");
  if (size && rank && selected && !std::strcmp(rank,selected) &&
      bytes == std::strtoull(size,nullptr,10)) {
    std::fprintf(stderr,"ASTR TEST allocation refused: rank=%s bytes=%zu\n",rank,bytes);
    throw std::bad_alloc();
  }
  using Allocate = void* (*)(std::size_t);
  static Allocate allocate = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT,"_Znwm"));
  if (!allocate) std::abort();
  return allocate(bytes);
}
