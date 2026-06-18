#include "tsmm.h"
#include "common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// v5: per-output parallel dot kernel.
//   C = A^T * B, column-major.
//
// Motivation from Intel measurements:
//   The first "small-k packed-A" idea failed on T3/O1 because packing and tiny
//   task overhead dominated. However, its fallback path was unexpectedly useful:
//   it parallelizes directly over every output element and lets the compiler
//   vectorize the contiguous length-k dot product.
//
// Strategy:
//   For each C[r,c], compute dot(A_col_r, B_col_c). A_col and B_col are both
//   contiguous in the chosen column-major layout, so this is simple and robust.
//   It is especially effective when output has enough entries for parallelism
//   but a tiled kernel underutilizes the cores (e.g. T2), and it is kept as a
//   candidate kernel in the benchmark rather than a universal replacement.
// =============================================================================

namespace {
static inline void fallback_tile(const TsmmProblem& p) {
    const int m = p.m, n = p.n, k = p.k;
    const double* A = p.A;
    const double* B = p.B;
    double* C = p.C;
#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (long idx = 0; idx < (long)m * n; ++idx) {
        int r = (int)(idx % m);
        int c = (int)(idx / m);
        double acc = 0.0;
        const double* ap = A + (size_t)r * k;
        const double* bp = B + (size_t)c * k;
        for (int i = 0; i < k; ++i) acc += ap[i] * bp[i];
        C[(size_t)c * m + r] = acc;
    }
}
} // namespace

static void* v5_prepare(int m, int n, int k) {
    (void)m; (void)n; (void)k;
    return nullptr;
}

static void v5_compute(const TsmmProblem& p, void* raw) {
    (void)raw;
    fallback_tile(p);
}

static void v5_destroy(void* raw) {
    (void)raw;
}

extern const TsmmOp TSMM_OP_v5_smallk_packa = {
    "v5_dot_parallel",
    "Per-output parallel contiguous dot kernel (strong for tiny-output huge-k)",
    v5_prepare, v5_compute, v5_destroy
};
