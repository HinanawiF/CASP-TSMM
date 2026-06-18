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
// v4: k-dimension parallel reduction for extreme tall-skinny cases.
//   C = A^T * B, column-major.
//
// Motivation from the first Intel run:
//   T2 (m,n,k)=(8,16,16000) and O2=(4,64,606841) have only 128/256 output
//   elements. The output-tile strategy cannot feed 96 cores, so v2/v3 spend most
//   cores idle. This version splits the contraction dimension k across threads:
//     each thread computes a partial C over its k range, then partial C buffers
//     are reduced.
//
// Guardrails:
//   Per-thread C buffers are only reasonable when m*n is small. For large output
//   cases (T1/O4), v4 automatically falls back to output-tile parallelism to
//   avoid allocating huge per-thread matrices.
// =============================================================================

namespace {
constexpr int MR = 8;
constexpr int NR = 4;
constexpr size_t KREDUCE_MAX_C_ELEMS = 65536;  // 512 KiB per thread
constexpr int KREDUCE_MIN_K = 4096;

struct V4Ctx {
    bool use_kreduce;
    int max_threads;
    size_t c_elems;
    double* partials;
};

static int max_threads_runtime() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int choose_k_threads(int k, int max_threads) {
    // Enough work per thread to amortize OpenMP and reduction overhead.
    int t = std::max(1, k / 2048);
    return std::min(max_threads, t);
}

#if defined(__AVX512F__)
static inline double dot_range(const double* a, const double* b, int begin, int end) {
    __m512d acc = _mm512_setzero_pd();
    int i = begin;
    for (; i + 8 <= end; i += 8) {
        __m512d av = _mm512_loadu_pd(a + i);
        __m512d bv = _mm512_loadu_pd(b + i);
        acc = _mm512_fmadd_pd(av, bv, acc);
    }
    double sum = _mm512_reduce_add_pd(acc);
    for (; i < end; ++i) sum += a[i] * b[i];
    return sum;
}
#else
static inline double dot_range(const double* a, const double* b, int begin, int end) {
    double sum = 0.0;
    for (int i = begin; i < end; ++i) sum += a[i] * b[i];
    return sum;
}
#endif

static inline void output_tile_fallback(const TsmmProblem& p) {
    const int m = p.m, n = p.n, k = p.k;
    const double* A = p.A;
    const double* B = p.B;
    double* C = p.C;

#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime)
#endif
    for (long t = 0; t < (long)((m + MR - 1) / MR) * ((n + NR - 1) / NR); ++t) {
        const int mt = (m + MR - 1) / MR;
        const int it = (int)(t % mt);
        const int jt = (int)(t / mt);
        const int ic = it * MR;
        const int jc = jt * NR;
        const int mb = std::min(MR, m - ic);
        const int nb = std::min(NR, n - jc);

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
}
} // namespace

static void* v4_prepare(int m, int n, int k) {
    V4Ctx* ctx = new V4Ctx;
    ctx->max_threads = max_threads_runtime();
    ctx->c_elems = (size_t)m * n;
    ctx->use_kreduce = (k >= KREDUCE_MIN_K && ctx->c_elems <= KREDUCE_MAX_C_ELEMS);
    ctx->partials = nullptr;
    if (ctx->use_kreduce) {
        size_t bytes = (size_t)ctx->max_threads * ctx->c_elems * sizeof(double);
        ctx->partials = (double*)tsmm_aligned_alloc(bytes);
        if (!ctx->partials) ctx->use_kreduce = false;
    }
    return ctx;
}

static void v4_compute(const TsmmProblem& p, void* raw) {
    V4Ctx* ctx = (V4Ctx*)raw;
    const int m = p.m, n = p.n, k = p.k;

    if (p.layout != TSMM_COL_MAJOR) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                double acc = 0.0;
                for (int i = 0; i < k; ++i)
                    acc += p.A[(size_t)i * m + r] * p.B[(size_t)i * n + c];
                p.C[(size_t)r * n + c] = acc;
            }
        }
        return;
    }

    if (!ctx || !ctx->use_kreduce) {
        output_tile_fallback(p);
        return;
    }

    const size_t c_elems = (size_t)m * n;
    const int use_threads = choose_k_threads(k, ctx->max_threads);
    double* partials = ctx->partials;

#ifdef _OPENMP
    #pragma omp parallel num_threads(use_threads)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
#else
        const int tid = 0;
        const int nt = 1;
#endif
        double* local = partials + (size_t)tid * c_elems;
        std::memset(local, 0, c_elems * sizeof(double));

        const int k0 = (int)((long long)k * tid / nt);
        const int k1 = (int)((long long)k * (tid + 1) / nt);
        for (int c = 0; c < n; ++c) {
            const double* bp = p.B + (size_t)c * k;
            for (int r = 0; r < m; ++r) {
                const double* ap = p.A + (size_t)r * k;
                local[(size_t)c * m + r] = dot_range(ap, bp, k0, k1);
            }
        }
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long idx = 0; idx < (long)c_elems; ++idx) {
        double sum = 0.0;
        for (int t = 0; t < use_threads; ++t)
            sum += partials[(size_t)t * c_elems + (size_t)idx];
        p.C[idx] = sum;
    }
}

static void v4_destroy(void* raw) {
    V4Ctx* ctx = (V4Ctx*)raw;
    if (!ctx) return;
    tsmm_aligned_free(ctx->partials);
    delete ctx;
}

extern const TsmmOp TSMM_OP_v4_kreduce = {
    "v4_kreduce",
    "Parallel reduction over k for tiny-output huge-k TSMM; tiled fallback",
    v4_prepare, v4_compute, v4_destroy
};
