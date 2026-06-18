#include "tsmm.h"
#include "common.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

// =============================================================================
// v3: AVX-512 + OpenMP. Vectorize the dot product ALONG k.
//   C = A^T * B, column-major. C[p,q] = dot(A_col_p, B_col_q).
//
// Why vectorize along k:
//   In column-major storage A_col_p and B_col_q are contiguous length-k
//   vectors. A length-k dot product maps perfectly onto AVX-512 FMA with 8
//   doubles per ZMM register: we keep a vector accumulator of 8 partial sums
//   and do one horizontal reduction per output element.
//
// Register-tiling for reuse:
//   We compute an MR x NR block of C at once. We load MR A-vectors and NR
//   B-vectors per k-step (MR+NR loads) and issue MR*NR FMAs, so every loaded
//   ZMM is reused (MR or NR)-fold. With MR=NR=4 we use 16 accumulators + 8
//   operands = 24 ZMM registers (<= 32 available).
//
// On non-AVX-512 hardware (e.g. Apple arm64) a scalar fallback with the same
// tiling is compiled, so the file is portable and still benefits from
// auto-vectorization. The intrinsic path is the one tuned for the Intel target.
// =============================================================================

namespace {
constexpr int MR = 4;
constexpr int NR = 4;
}

static void* v3_prepare(int, int, int) { return nullptr; }

#if defined(__AVX512F__)
static inline double hsum512(__m512d v) {
    return _mm512_reduce_add_pd(v);
}

static inline void micro_avx512(const double* A, const double* B, double* C,
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
    double r[MR][NR];
    r[0][0]=hsum512(c00); r[0][1]=hsum512(c01); r[0][2]=hsum512(c02); r[0][3]=hsum512(c03);
    r[1][0]=hsum512(c10); r[1][1]=hsum512(c11); r[1][2]=hsum512(c12); r[1][3]=hsum512(c13);
    r[2][0]=hsum512(c20); r[2][1]=hsum512(c21); r[2][2]=hsum512(c22); r[2][3]=hsum512(c23);
    r[3][0]=hsum512(c30); r[3][1]=hsum512(c31); r[3][2]=hsum512(c32); r[3][3]=hsum512(c33);
    // tail
    for (; i < k; ++i) {
        double a[MR] = {a0[i], a1[i], a2[i], a3[i]};
        double b[NR] = {b0[i], b1[i], b2[i], b3[i]};
        for (int x = 0; x < MR; ++x)
            for (int y = 0; y < NR; ++y) r[x][y] += a[x] * b[y];
    }
    for (int y = 0; y < NR; ++y)
        for (int x = 0; x < MR; ++x)
            C[(size_t)(jc + y) * m + (ic + x)] = r[x][y];
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
    "AVX-512 (vectorize along k) + OpenMP tiles; scalar fallback",
    v3_prepare, v3_compute, v3_destroy
};
