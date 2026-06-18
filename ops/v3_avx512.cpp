#include "tsmm.h"
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// =============================================================================
// v3: AVX-512 + OpenMP. 8x8 outer-product micro-kernel.
//   C = A^T * B, column-major. C[p,q] = dot(A_col_p, B_col_q).
//
// Why this kernel replaces the first v3:
//   The previous AVX-512 version vectorized each individual dot product along k.
//   That made every C element require one expensive horizontal reduction from a
//   ZMM register. The real Intel run showed v3 barely beat v2 on T1 and was
//   worse on small-k shapes, so the reduction overhead was visible.
//
// New strategy:
//   Compute an 8(rows of C) x 8(cols of C) output tile. Each accumulator is a
//   ZMM holding 8 contiguous C rows for one output column. For each k index:
//     1) gather 8 A values A[i, ic:ic+7] into one ZMM;
//     2) broadcast B[i, jc+b] for b=0..7;
//     3) update 8 column accumulators with packed FMAs.
//   At the end each accumulator is already in C's column-major layout and can
//   be stored directly. No horizontal sums are needed.
//
// On non-AVX-512 hardware (e.g. Apple arm64) a scalar fallback with the same
// tiling is compiled, so the file is portable and still benefits from
// auto-vectorization. The intrinsic path is the one tuned for the Intel target.
// =============================================================================

namespace {
constexpr int MR = 8;
constexpr int NR = 8;
}

static void* v3_prepare(int, int, int) { return nullptr; }

#if defined(__AVX512F__)
static inline void micro_avx512(const double* A, const double* B, double* C,
                                int m, int k, int ic, int jc) {
    const __m512i aidx = _mm512_set_epi64(
        7LL * k, 6LL * k, 5LL * k, 4LL * k,
        3LL * k, 2LL * k, 1LL * k, 0LL * k);
    const double* abase = A + (size_t)ic * k;
    const double* bbase = B + (size_t)jc * k;

    __m512d c0 = _mm512_setzero_pd();
    __m512d c1 = _mm512_setzero_pd();
    __m512d c2 = _mm512_setzero_pd();
    __m512d c3 = _mm512_setzero_pd();
    __m512d c4 = _mm512_setzero_pd();
    __m512d c5 = _mm512_setzero_pd();
    __m512d c6 = _mm512_setzero_pd();
    __m512d c7 = _mm512_setzero_pd();

    for (int i = 0; i < k; ++i) {
        const __m512d av = _mm512_i64gather_pd(aidx, abase + i, 8);
        c0 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)0 * k + i]), c0);
        c1 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)1 * k + i]), c1);
        c2 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)2 * k + i]), c2);
        c3 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)3 * k + i]), c3);
        c4 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)4 * k + i]), c4);
        c5 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)5 * k + i]), c5);
        c6 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)6 * k + i]), c6);
        c7 = _mm512_fmadd_pd(av, _mm512_set1_pd(bbase[(size_t)7 * k + i]), c7);
    }

    _mm512_storeu_pd(C + (size_t)(jc + 0) * m + ic, c0);
    _mm512_storeu_pd(C + (size_t)(jc + 1) * m + ic, c1);
    _mm512_storeu_pd(C + (size_t)(jc + 2) * m + ic, c2);
    _mm512_storeu_pd(C + (size_t)(jc + 3) * m + ic, c3);
    _mm512_storeu_pd(C + (size_t)(jc + 4) * m + ic, c4);
    _mm512_storeu_pd(C + (size_t)(jc + 5) * m + ic, c5);
    _mm512_storeu_pd(C + (size_t)(jc + 6) * m + ic, c6);
    _mm512_storeu_pd(C + (size_t)(jc + 7) * m + ic, c7);
}
#endif // __AVX512F__

static inline void micro_scalar(const double* A, const double* B, double* C,
                                int m, int k, int ic, int jc, int mb, int nb) {
    double acc[MR][NR];
    for (int a = 0; a < MR; ++a)
        for (int b = 0; b < NR; ++b) acc[a][b] = 0.0;
    for (int i = 0; i < k; ++i) {
        double av[MR], bv[NR];
        for (int a = 0; a < mb; ++a) av[a] = A[(size_t)(ic + a) * k + i];
        for (int b = 0; b < nb; ++b) bv[b] = B[(size_t)(jc + b) * k + i];
        for (int a = 0; a < mb; ++a)
            for (int b = 0; b < nb; ++b)
                acc[a][b] += av[a] * bv[b];
    }
    for (int b = 0; b < nb; ++b)
        for (int a = 0; a < mb; ++a)
            C[(size_t)(jc + b) * m + (ic + a)] = acc[a][b];
}

static void v3_compute(const TsmmProblem& p, void* /*ctx*/) {
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

    const int mt = (m + MR - 1) / MR;
    const int nt = (n + NR - 1) / NR;
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
#if defined(__AVX512F__)
        if (mb == MR && nb == NR) micro_avx512(A, B, C, m, k, ic, jc);
        else                      micro_scalar(A, B, C, m, k, ic, jc, mb, nb);
#else
        micro_scalar(A, B, C, m, k, ic, jc, mb, nb);
#endif
    }
}

static void v3_destroy(void*) {}

extern const TsmmOp TSMM_OP_v3_avx512 = {
    "v3_avx512",
    "AVX-512 8x8 outer-product tile + OpenMP; scalar fallback",
    v3_prepare, v3_compute, v3_destroy
};
