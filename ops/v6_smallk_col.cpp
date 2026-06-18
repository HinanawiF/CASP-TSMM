#include "tsmm.h"
#include "common.h"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// =============================================================================
// v6: small-k, column-wise TSMM kernel.
//   C = A^T * B, column-major.
//
// Motivation:
//   T3=(32,16000,16) and O1=(16,12344,16) are essentially many independent
//   matrix-vector products:
//       C[:, c] = A^T * B[:, c]
//   with k=16. Previous packed-A experiments used too fine a task granularity.
//   This version parallelizes only over B/C columns; each task computes a whole
//   output column using row-block vector accumulators.
//
// For k<=64 and moderate m, packing A from A[r*k+i] to packA[i*m+r] is cheap.
// It converts A accesses for C row blocks into contiguous AVX-512 loads.
// =============================================================================

namespace {
constexpr int RB = 8;
constexpr int SMALLK_MAX_K = 64;
constexpr int SMALLK_MAX_M = 512;

struct V6Ctx {
    bool enabled;
    int m;
    int k;
    double* packA;
};

static inline void fallback_dot(const TsmmProblem& p) {
    const int m = p.m, n = p.n, k = p.k;
#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (long idx = 0; idx < (long)m * n; ++idx) {
        int r = (int)(idx % m);
        int c = (int)(idx / m);
        const double* ap = p.A + (size_t)r * k;
        const double* bp = p.B + (size_t)c * k;
        double acc = 0.0;
        for (int i = 0; i < k; ++i) acc += ap[i] * bp[i];
        p.C[(size_t)c * m + r] = acc;
    }
}

static inline void compute_scalar_col(const double* packA, const double* B,
                                      double* C, int m, int n, int k) {
    const int rb_count = (m + RB - 1) / RB;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < n; ++c) {
        const double* bp = B + (size_t)c * k;
        double* cp = C + (size_t)c * m;
        for (int rb = 0; rb < rb_count; ++rb) {
            int r0 = rb * RB;
            int len = std::min(RB, m - r0);
            double acc[RB];
            for (int r = 0; r < len; ++r) acc[r] = 0.0;
            for (int i = 0; i < k; ++i) {
                double bv = bp[i];
                const double* ap = packA + (size_t)i * m + r0;
                for (int r = 0; r < len; ++r) acc[r] += ap[r] * bv;
            }
            for (int r = 0; r < len; ++r) cp[r0 + r] = acc[r];
        }
    }
}

#if defined(__AVX512F__)
static inline void compute_avx512_col(const double* packA, const double* B,
                                      double* C, int m, int n, int k) {
    const int rb_count = (m + RB - 1) / RB;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < n; ++c) {
        const double* bp = B + (size_t)c * k;
        double* cp = C + (size_t)c * m;
        for (int rb = 0; rb < rb_count; ++rb) {
            int r0 = rb * RB;
            int len = std::min(RB, m - r0);
            if (len == RB) {
                __m512d acc = _mm512_setzero_pd();
                for (int i = 0; i < k; ++i) {
                    __m512d av = _mm512_loadu_pd(packA + (size_t)i * m + r0);
                    __m512d bv = _mm512_set1_pd(bp[i]);
                    acc = _mm512_fmadd_pd(av, bv, acc);
                }
                _mm512_storeu_pd(cp + r0, acc);
            } else {
                double acc[RB];
                for (int r = 0; r < len; ++r) acc[r] = 0.0;
                for (int i = 0; i < k; ++i) {
                    double bv = bp[i];
                    const double* ap = packA + (size_t)i * m + r0;
                    for (int r = 0; r < len; ++r) acc[r] += ap[r] * bv;
                }
                for (int r = 0; r < len; ++r) cp[r0 + r] = acc[r];
            }
        }
    }
}
#endif
} // namespace

static void* v6_prepare(int m, int, int k) {
    V6Ctx* ctx = new V6Ctx;
    ctx->enabled = (k <= SMALLK_MAX_K && m <= SMALLK_MAX_M);
    ctx->m = m;
    ctx->k = k;
    ctx->packA = nullptr;
    if (ctx->enabled) {
        ctx->packA = (double*)tsmm_aligned_alloc((size_t)m * k * sizeof(double));
        if (!ctx->packA) ctx->enabled = false;
    }
    return ctx;
}

static void v6_compute(const TsmmProblem& p, void* raw) {
    V6Ctx* ctx = (V6Ctx*)raw;
    const int m = p.m, n = p.n, k = p.k;
    if (p.layout != TSMM_COL_MAJOR || !ctx || !ctx->enabled) {
        fallback_dot(p);
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long idx = 0; idx < (long)m * k; ++idx) {
        int r = (int)(idx % m);
        int i = (int)(idx / m);
        ctx->packA[(size_t)i * m + r] = p.A[(size_t)r * k + i];
    }

#if defined(__AVX512F__)
    compute_avx512_col(ctx->packA, p.B, p.C, m, n, k);
#else
    compute_scalar_col(ctx->packA, p.B, p.C, m, n, k);
#endif
}

static void v6_destroy(void* raw) {
    V6Ctx* ctx = (V6Ctx*)raw;
    if (!ctx) return;
    tsmm_aligned_free(ctx->packA);
    delete ctx;
}

extern const TsmmOp TSMM_OP_v6_smallk_col = {
    "v6_smallk_col",
    "Small-k column-wise packed-A kernel for T3/O1/O3",
    v6_prepare, v6_compute, v6_destroy
};
