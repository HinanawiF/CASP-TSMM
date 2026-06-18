#include "tsmm.h"

// =============================================================================
// v0: Naive serial reference. Demonstrates row-major AND column-major storage.
//   C = A^T * B.
// This is the textbook triple loop; it is intentionally simple so it can serve
// as a correctness oracle and a performance baseline.
// =============================================================================

static void* v0_prepare(int /*m*/, int /*n*/, int /*k*/) {
    return nullptr;  // no preprocessing needed
}

static void v0_compute(const TsmmProblem& p, void* /*ctx*/) {
    const int m = p.m, n = p.n, k = p.k;
    const double* A = p.A;
    const double* B = p.B;
    double* C = p.C;

    if (p.layout == TSMM_COL_MAJOR) {
        // A[i + j*k], B[i + j*k], C[r + c*m]
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < m; ++r) {
                double acc = 0.0;
                const double* ap = A + (size_t)r * k;
                const double* bp = B + (size_t)c * k;
                for (int i = 0; i < k; ++i) acc += ap[i] * bp[i];
                C[(size_t)c * m + r] = acc;
            }
        }
    } else {
        // ROW-MAJOR: A[j + i*m]? No: A is k x m row-major => A[i*m + j].
        //   A[i,j] = A[i*m + j], B[i,c] = B[i*n + c], C[r,c] = C[r*n + c]
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                double acc = 0.0;
                for (int i = 0; i < k; ++i)
                    acc += A[(size_t)i * m + r] * B[(size_t)i * n + c];
                C[(size_t)r * n + c] = acc;
            }
        }
    }
}

static void v0_destroy(void* /*ctx*/) {}

extern const TsmmOp TSMM_OP_v0_naive = {
    "v0_naive",
    "Naive serial triple loop (row/col major)",
    v0_prepare, v0_compute, v0_destroy
};
