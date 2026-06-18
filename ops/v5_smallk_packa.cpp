#include "tsmm.h"
#include "common.h"

#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// =============================================================================
// v5: small-k packed-A kernel.
//   C = A^T * B, column-major.
//
// Motivation from Intel measurements:
//   T3=(32,16000,16), O1=(16,12344,16), O3=(442,193,11) are dominated by very
//   small k. The normal column-major A layout is good for dot products, but bad
//   for an outer-product style update because A[i, r:r+7] is strided by k.
//
// Strategy:
//   For small k, pack A into A_pack[i*m + r] = A[r*k + i], so rows of A^T are
//   contiguous across r. Then each output column C[:,c] is computed as:
//       C[:,c] = sum_i B[i,c] * A_pack[i,:]
//   using contiguous AVX-512 loads over the C rows. Packing cost is tiny for
//   these shapes (e.g. T3 packs only 512 doubles) and is repaid over large n.
// =============================================================================

namespace {
constexpr int RB = 8;
constexpr int SMALLK_MAX_K = 64;
constexpr int PACKA_MAX_ELEMS = 1 << 20; // 8 MiB worst case, much smaller here.

struct V5Ctx {
    int m, n, k;
    bool use_packa;
    double* packA;
};

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

static inline void compute_scalar_packed(const double* packA, const double* B,
                                         double* C, int m, int n, int k) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (long idx = 0; idx < (long)n * ((m + RB - 1) / RB); ++idx) {
        int rb = (int)(idx % ((m + RB - 1) / RB));
        int c = (int)(idx / ((m + RB - 1) / RB));
        int r0 = rb * RB;
        int rb_len = std::min(RB, m - r0);
        double acc[RB];
        for (int r = 0; r < rb_len; ++r) acc[r] = 0.0;
        for (int i = 0; i < k; ++i) {
            double bv = B[(size_t)c * k + i];
            const double* ar = packA + (size_t)i * m + r0;
            for (int r = 0; r < rb_len; ++r) acc[r] += ar[r] * bv;
        }
        for (int r = 0; r < rb_len; ++r) C[(size_t)c * m + r0 + r] = acc[r];
    }
}

#if defined(__AVX512F__)
static inline void compute_avx512_packed(const double* packA, const double* B,
                                         double* C, int m, int n, int k) {
    const int rb_count = (m + RB - 1) / RB;
#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (long idx = 0; idx < (long)n * rb_count; ++idx) {
        int rb = (int)(idx % rb_count);
        int c = (int)(idx / rb_count);
        int r0 = rb * RB;
        int rb_len = std::min(RB, m - r0);

        if (rb_len == RB) {
            __m512d acc = _mm512_setzero_pd();
            for (int i = 0; i < k; ++i) {
                __m512d av = _mm512_loadu_pd(packA + (size_t)i * m + r0);
                __m512d bv = _mm512_set1_pd(B[(size_t)c * k + i]);
                acc = _mm512_fmadd_pd(av, bv, acc);
            }
            _mm512_storeu_pd(C + (size_t)c * m + r0, acc);
        } else {
            double acc[RB];
            for (int r = 0; r < rb_len; ++r) acc[r] = 0.0;
            for (int i = 0; i < k; ++i) {
                double bv = B[(size_t)c * k + i];
                const double* ar = packA + (size_t)i * m + r0;
                for (int r = 0; r < rb_len; ++r) acc[r] += ar[r] * bv;
            }
            for (int r = 0; r < rb_len; ++r) C[(size_t)c * m + r0 + r] = acc[r];
        }
    }
}
#endif
} // namespace

static void* v5_prepare(int m, int n, int k) {
    V5Ctx* ctx = new V5Ctx;
    ctx->m = m;
    ctx->n = n;
    ctx->k = k;
    ctx->use_packa = (k <= SMALLK_MAX_K && (size_t)m * k <= PACKA_MAX_ELEMS);
    ctx->packA = nullptr;
    if (ctx->use_packa) {
        ctx->packA = (double*)tsmm_aligned_alloc((size_t)m * k * sizeof(double));
        if (!ctx->packA) ctx->use_packa = false;
    }
    return ctx;
}

static void v5_compute(const TsmmProblem& p, void* raw) {
    V5Ctx* ctx = (V5Ctx*)raw;
    const int m = p.m, n = p.n, k = p.k;

    if (p.layout != TSMM_COL_MAJOR || !ctx || !ctx->use_packa) {
        fallback_tile(p);
        return;
    }

    // Pack A as row-major k x m: packA[i*m + r] = A[i + r*k].
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long idx = 0; idx < (long)k * m; ++idx) {
        int r = (int)(idx % m);
        int i = (int)(idx / m);
        ctx->packA[(size_t)i * m + r] = p.A[(size_t)r * k + i];
    }

#if defined(__AVX512F__)
    compute_avx512_packed(ctx->packA, p.B, p.C, m, n, k);
#else
    compute_scalar_packed(ctx->packA, p.B, p.C, m, n, k);
#endif
}

static void v5_destroy(void* raw) {
    V5Ctx* ctx = (V5Ctx*)raw;
    if (!ctx) return;
    tsmm_aligned_free(ctx->packA);
    delete ctx;
}

extern const TsmmOp TSMM_OP_v5_smallk_packa = {
    "v5_smallk_packa",
    "Small-k packed-A kernel: contiguous row updates for T3/O1/O3",
    v5_prepare, v5_compute, v5_destroy
};
