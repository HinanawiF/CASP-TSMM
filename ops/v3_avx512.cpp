#include "tsmm.h"
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// =============================================================================
// v3: AVX-512 + OpenMP. Shape-dispatched micro-kernels.
//   C = A^T * B, column-major. C[p,q] = dot(A_col_p, B_col_q).
//
// Why two kernels:
//   The first Intel run showed the original 4x4 dot-product kernel was strong
//   for large-output / moderate-k shapes (T1/O4) because it streams contiguous
//   A/B columns. The later 8x8 outer-product kernel removed horizontal sums and
//   helped small-k/small-output shapes (T3/T4/O1/O3), but its strided gather of
//   A is too expensive for huge-k or huge-n cases.
//
// Dispatch:
//   * dot4: 4x4 independent contiguous dot products; best for T1/O4/T2/O2.
//   * outer8: 8x8 outer product; best for short-k / small-output cases.
//
// On non-AVX-512 hardware (e.g. Apple arm64) a scalar fallback with the same
// tiling is compiled, so the file is portable and still benefits from
// auto-vectorization. The intrinsic path is the one tuned for the Intel target.
// =============================================================================

namespace {
constexpr int DOT_MR = 4;
constexpr int DOT_NR = 4;
constexpr int OUTER_MR = 8;
constexpr int OUTER_NR = 8;
constexpr int SCALAR_MAX_MR = 8;
constexpr int SCALAR_MAX_NR = 8;

static inline bool should_use_outer8(int m, int n, int k) {
    // Empirical dispatch from Intel Xeon Platinum 9242 measurements:
    // short-k shapes benefit from removing horizontal sums, except the O4 case
    // where n is enormous and the gather-based outer kernel loses to dot4/v2.
    return (k <= 64 && n <= 20000) || (m <= 256 && n <= 256 && k <= 256);
}
}

static void* v3_prepare(int, int, int) { return nullptr; }

#if defined(__AVX512F__)
static inline double hsum512(__m512d v) {
    return _mm512_reduce_add_pd(v);
}

static inline void micro_avx512_dot4(const double* A, const double* B, double* C,
                                     int m, int k, int ic, int jc) {
    const double* a0 = A + (size_t)(ic + 0) * k;
    const double* a1 = A + (size_t)(ic + 1) * k;
    const double* a2 = A + (size_t)(ic + 2) * k;
    const double* a3 = A + (size_t)(ic + 3) * k;
    const double* b0 = B + (size_t)(jc + 0) * k;
    const double* b1 = B + (size_t)(jc + 1) * k;
    const double* b2 = B + (size_t)(jc + 2) * k;
    const double* b3 = B + (size_t)(jc + 3) * k;

    __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
    __m512d c02 = _mm512_setzero_pd(), c03 = _mm512_setzero_pd();
    __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
    __m512d c12 = _mm512_setzero_pd(), c13 = _mm512_setzero_pd();
    __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
    __m512d c22 = _mm512_setzero_pd(), c23 = _mm512_setzero_pd();
    __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
    __m512d c32 = _mm512_setzero_pd(), c33 = _mm512_setzero_pd();

    int i = 0;
    for (; i + 8 <= k; i += 8) {
        __m512d av0 = _mm512_loadu_pd(a0 + i);
        __m512d av1 = _mm512_loadu_pd(a1 + i);
        __m512d av2 = _mm512_loadu_pd(a2 + i);
        __m512d av3 = _mm512_loadu_pd(a3 + i);
        __m512d bv0 = _mm512_loadu_pd(b0 + i);
        __m512d bv1 = _mm512_loadu_pd(b1 + i);
        __m512d bv2 = _mm512_loadu_pd(b2 + i);
        __m512d bv3 = _mm512_loadu_pd(b3 + i);
        c00 = _mm512_fmadd_pd(av0, bv0, c00);
        c01 = _mm512_fmadd_pd(av0, bv1, c01);
        c02 = _mm512_fmadd_pd(av0, bv2, c02);
        c03 = _mm512_fmadd_pd(av0, bv3, c03);
        c10 = _mm512_fmadd_pd(av1, bv0, c10);
        c11 = _mm512_fmadd_pd(av1, bv1, c11);
        c12 = _mm512_fmadd_pd(av1, bv2, c12);
        c13 = _mm512_fmadd_pd(av1, bv3, c13);
        c20 = _mm512_fmadd_pd(av2, bv0, c20);
        c21 = _mm512_fmadd_pd(av2, bv1, c21);
        c22 = _mm512_fmadd_pd(av2, bv2, c22);
        c23 = _mm512_fmadd_pd(av2, bv3, c23);
        c30 = _mm512_fmadd_pd(av3, bv0, c30);
        c31 = _mm512_fmadd_pd(av3, bv1, c31);
        c32 = _mm512_fmadd_pd(av3, bv2, c32);
        c33 = _mm512_fmadd_pd(av3, bv3, c33);
    }

    double r[DOT_MR][DOT_NR];
    r[0][0]=hsum512(c00); r[0][1]=hsum512(c01); r[0][2]=hsum512(c02); r[0][3]=hsum512(c03);
    r[1][0]=hsum512(c10); r[1][1]=hsum512(c11); r[1][2]=hsum512(c12); r[1][3]=hsum512(c13);
    r[2][0]=hsum512(c20); r[2][1]=hsum512(c21); r[2][2]=hsum512(c22); r[2][3]=hsum512(c23);
    r[3][0]=hsum512(c30); r[3][1]=hsum512(c31); r[3][2]=hsum512(c32); r[3][3]=hsum512(c33);
    for (; i < k; ++i) {
        double a[DOT_MR] = {a0[i], a1[i], a2[i], a3[i]};
        double b[DOT_NR] = {b0[i], b1[i], b2[i], b3[i]};
        for (int x = 0; x < DOT_MR; ++x)
            for (int y = 0; y < DOT_NR; ++y) r[x][y] += a[x] * b[y];
    }
    for (int y = 0; y < DOT_NR; ++y)
        for (int x = 0; x < DOT_MR; ++x)
            C[(size_t)(jc + y) * m + (ic + x)] = r[x][y];
}

static inline void micro_avx512_outer8(const double* A, const double* B, double* C,
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
    double acc[SCALAR_MAX_MR][SCALAR_MAX_NR];
    for (int a = 0; a < SCALAR_MAX_MR; ++a)
        for (int b = 0; b < SCALAR_MAX_NR; ++b) acc[a][b] = 0.0;
    for (int i = 0; i < k; ++i) {
        double av[SCALAR_MAX_MR], bv[SCALAR_MAX_NR];
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

    const bool use_outer = should_use_outer8(m, n, k);
    const int tile_m = use_outer ? OUTER_MR : DOT_MR;
    const int tile_n = use_outer ? OUTER_NR : DOT_NR;
    const int mt = (m + tile_m - 1) / tile_m;
    const int nt = (n + tile_n - 1) / tile_n;
    const long ntiles = (long)mt * nt;

#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (long t = 0; t < ntiles; ++t) {
        const int it = (int)(t % mt);
        const int jt = (int)(t / mt);
        const int ic = it * tile_m;
        const int jc = jt * tile_n;
        const int mb = (ic + tile_m <= m) ? tile_m : (m - ic);
        const int nb = (jc + tile_n <= n) ? tile_n : (n - jc);
#if defined(__AVX512F__)
        if (use_outer && mb == OUTER_MR && nb == OUTER_NR)
            micro_avx512_outer8(A, B, C, m, k, ic, jc);
        else if (!use_outer && mb == DOT_MR && nb == DOT_NR)
            micro_avx512_dot4(A, B, C, m, k, ic, jc);
        else
            micro_scalar(A, B, C, m, k, ic, jc, mb, nb);
#else
        micro_scalar(A, B, C, m, k, ic, jc, mb, nb);
#endif
    }
}

static void v3_destroy(void*) {}

extern const TsmmOp TSMM_OP_v3_avx512 = {
    "v3_avx512",
    "AVX-512 shape-dispatched dot4/outer8 tiles + OpenMP; scalar fallback",
    v3_prepare, v3_compute, v3_destroy
};
