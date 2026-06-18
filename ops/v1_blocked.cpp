#include "tsmm.h"
#include "common.h"

// =============================================================================
// v1: Cache-blocked serial implementation (column-major).
//   C = A^T * B.
// Key ideas:
//   * Both A's column p and B's column q are contiguous length-k vectors, so
//     the dot product C[p,q] streams two vectors with unit stride -> good for
//     hardware prefetch and SIMD auto-vectorization.
//   * We block over (m, n) so that a panel of A columns and a panel of B
//     columns stays hot in L1/L2 while we compute their MR x NR dot products.
//   * The innermost work is an MR x NR register tile accumulated over k, which
//     maximizes register reuse (each loaded A/B element is used MR or NR times).
// =============================================================================

namespace {
constexpr int MR = 4;   // rows of C per register tile (A columns)
constexpr int NR = 4;   // cols of C per register tile (B columns)
constexpr int KC = 256; // k-blocking for cache residency
}

static void* v1_prepare(int, int, int) { return nullptr; }

static void v1_compute(const TsmmProblem& p, void* /*ctx*/) {
    const int m = p.m, n = p.n, k = p.k;
    const double* A = p.A;
    const double* B = p.B;
    double* C = p.C;

    if (p.layout != TSMM_COL_MAJOR) {
        // Fallback path for row-major (kept simple; col-major is the fast path).
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < n; ++c) {
                double acc = 0.0;
                for (int i = 0; i < k; ++i)
                    acc += A[(size_t)i * m + r] * B[(size_t)i * n + c];
                C[(size_t)r * n + c] = acc;
            }
        return;
    }

    // zero C
    for (size_t idx = 0; idx < (size_t)m * n; ++idx) C[idx] = 0.0;

    for (int kc = 0; kc < k; kc += KC) {
        const int kb = (kc + KC < k) ? KC : (k - kc);
        for (int jc = 0; jc < n; jc += NR) {
            const int nb = (jc + NR < n) ? NR : (n - jc);
            for (int ic = 0; ic < m; ic += MR) {
                const int mb = (ic + MR < m) ? MR : (m - ic);

                if (mb == MR && nb == NR) {
                    // Full MRxNR register tile.
                    double acc[MR][NR];
                    for (int a = 0; a < MR; ++a)
                        for (int b = 0; b < NR; ++b) acc[a][b] = 0.0;

                    const double* Acol = A + (size_t)ic * k + kc;
                    const double* Bcol = B + (size_t)jc * k + kc;
                    for (int i = 0; i < kb; ++i) {
                        double av[MR], bv[NR];
                        for (int a = 0; a < MR; ++a) av[a] = Acol[a * (size_t)k + i];
                        for (int b = 0; b < NR; ++b) bv[b] = Bcol[b * (size_t)k + i];
                        for (int a = 0; a < MR; ++a)
                            for (int b = 0; b < NR; ++b)
                                acc[a][b] += av[a] * bv[b];
                    }
                    for (int b = 0; b < NR; ++b)
                        for (int a = 0; a < MR; ++a)
                            C[(size_t)(jc + b) * m + (ic + a)] += acc[a][b];
                } else {
                    // Edge tile.
                    for (int b = 0; b < nb; ++b)
                        for (int a = 0; a < mb; ++a) {
                            const double* ap = A + (size_t)(ic + a) * k + kc;
                            const double* bp = B + (size_t)(jc + b) * k + kc;
                            double acc = 0.0;
                            for (int i = 0; i < kb; ++i) acc += ap[i] * bp[i];
                            C[(size_t)(jc + b) * m + (ic + a)] += acc;
                        }
                }
            }
        }
    }
}

static void v1_destroy(void*) {}

extern const TsmmOp TSMM_OP_v1_blocked = {
    "v1_blocked",
    "Cache-blocked serial with MRxNR register tile (col-major)",
    v1_prepare, v1_compute, v1_destroy
};
