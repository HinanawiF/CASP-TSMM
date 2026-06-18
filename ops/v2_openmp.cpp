#include "tsmm.h"
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// v2: OpenMP multi-threaded, cache-blocked (column-major).
//   C = A^T * B.
// Parallelization strategy:
//   * The output C is m x n. Each (p,q) entry is an independent dot product,
//     so there is no write contention if we partition the OUTPUT space.
//   * We collapse the (n-block, m-block) tile grid and distribute tiles across
//     threads. This gives good load balance for both "M large" and "N large"
//     skinny shapes, which is important for the wildly different required
//     shapes (4000x16000x128, 8x16x16000, 32x16000x16, 144x144x144).
//   * Scheduling policy is selectable via env TSMM_OMP_SCHEDULE
//     (static|dynamic|guided); default we use a dynamic schedule with a small
//     chunk because tile costs are uniform but counts can be huge.
// =============================================================================

namespace {
constexpr int MR = 8;
constexpr int NR = 4;
}

static void* v2_prepare(int, int, int) { return nullptr; }

static inline void micro_tile(const double* A, const double* B, double* C,
                              int m, int k, int ic, int jc, int mb, int nb) {
    if (mb == MR && nb == NR) {
        double acc[MR][NR];
        for (int a = 0; a < MR; ++a)
            for (int b = 0; b < NR; ++b) acc[a][b] = 0.0;
        const double* Acol = A + (size_t)ic * k;
        const double* Bcol = B + (size_t)jc * k;
        for (int i = 0; i < k; ++i) {
            double av[MR], bv[NR];
            for (int a = 0; a < MR; ++a) av[a] = Acol[a * (size_t)k + i];
            for (int b = 0; b < NR; ++b) bv[b] = Bcol[b * (size_t)k + i];
            for (int a = 0; a < MR; ++a)
                for (int b = 0; b < NR; ++b)
                    acc[a][b] += av[a] * bv[b];
        }
        for (int b = 0; b < NR; ++b)
            for (int a = 0; a < MR; ++a)
                C[(size_t)(jc + b) * m + (ic + a)] = acc[a][b];
    } else {
        for (int b = 0; b < nb; ++b)
            for (int a = 0; a < mb; ++a) {
                const double* ap = A + (size_t)(ic + a) * k;
                const double* bp = B + (size_t)(jc + b) * k;
                double acc = 0.0;
                for (int i = 0; i < k; ++i) acc += ap[i] * bp[i];
                C[(size_t)(jc + b) * m + (ic + a)] = acc;
            }
    }
}

static void v2_compute(const TsmmProblem& p, void* /*ctx*/) {
    const int m = p.m, n = p.n, k = p.k;
    const double* A = p.A;
    const double* B = p.B;
    double* C = p.C;

    if (p.layout != TSMM_COL_MAJOR) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < n; ++c) {
                double acc = 0.0;
                for (int i = 0; i < k; ++i)
                    acc += A[(size_t)i * m + r] * B[(size_t)i * n + c];
                C[(size_t)r * n + c] = acc;
            }
        return;
    }

    const int mt = (m + MR - 1) / MR;   // number of row tiles
    const int nt = (n + NR - 1) / NR;   // number of col tiles
    const long ntiles = (long)mt * nt;

#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (long t = 0; t < ntiles; ++t) {
        const int it = (int)(t % mt);
        const int jt = (int)(t / mt);
        const int ic = it * MR;
        const int jc = jt * NR;
        const int mb = (ic + MR <= m) ? MR : (m - ic);
        const int nb = (jc + NR <= n) ? NR : (n - jc);
        micro_tile(A, B, C, m, k, ic, jc, mb, nb);
    }
}

static void v2_destroy(void*) {}

extern const TsmmOp TSMM_OP_v2_openmp = {
    "v2_openmp",
    "OpenMP tiled over output (schedule=runtime, set OMP_SCHEDULE)",
    v2_prepare, v2_compute, v2_destroy
};
