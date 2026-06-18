#pragma once
#include <cstdlib>
#include <cstring>
#include <new>

// Aligned allocation helpers (64-byte for AVX-512 / cache lines).
inline void* tsmm_aligned_alloc(size_t bytes, size_t align = 64) {
    if (bytes == 0) bytes = align;
    // round up to multiple of align (posix_memalign requirement)
    size_t rounded = (bytes + align - 1) / align * align;
    void* p = nullptr;
    if (posix_memalign(&p, align, rounded) != 0) return nullptr;
    return p;
}

inline void tsmm_aligned_free(void* p) {
    free(p);
}
