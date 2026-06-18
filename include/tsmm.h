#pragma once
#include <cstddef>

// =============================================================================
// TSMM: Tall-Skinny Matrix Multiplication
//   C = A^T * B
//   A in R^{k x m}, B in R^{k x n}, C in R^{m x n}, double precision.
//   FLOPs = 2 * m * n * k
//
// Storage convention (default): COLUMN-MAJOR.
//   In column-major both A and B have the contraction dimension k as the
//   leading (stride-1) dimension, so column p of A and column q of B are
//   contiguous length-k vectors and  C[p,q] = dot(A_col_p, B_col_q).
//     A[i + j*k]  (i in [0,k), j in [0,m), lda = k)
//     B[i + j*k]  (i in [0,k), j in [0,n), ldb = k)
//     C[p + q*m]  (p in [0,m), q in [0,n), ldc = m)
//
// Algorithm files only contain a prepare() and a compute() function; the
// benchmark / driver code is kept completely separate (see benchmark.cpp).
// =============================================================================

enum TsmmLayout {
    TSMM_COL_MAJOR = 0,
    TSMM_ROW_MAJOR = 1,
};

struct TsmmProblem {
    int m;
    int n;
    int k;
    const double* A;   // k x m
    const double* B;   // k x n
    double*       C;   // m x n  (output)
    TsmmLayout    layout;
};

// An operator implementation. Decoupled into:
//   prepare(): one-time preprocessing for the given dimensions (e.g. packing
//              buffers). May return an opaque context pointer or nullptr.
//   compute(): performs C = A^T * B.
//   destroy(): releases the context produced by prepare().
struct TsmmOp {
    const char* name;
    const char* desc;
    void* (*prepare)(int m, int n, int k);
    void  (*compute)(const TsmmProblem& p, void* ctx);
    void  (*destroy)(void* ctx);
};

// ----- Registry (implemented in registry.cpp) -----
int            tsmm_num_ops();
const TsmmOp*  tsmm_get_op(int idx);
const TsmmOp*  tsmm_find_op(const char* name);
