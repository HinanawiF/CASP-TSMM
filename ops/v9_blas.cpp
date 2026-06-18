#include "tsmm.h"

// =============================================================================
// v9: Vendor BLAS reference (dgemm).  Used both as a performance baseline and
// as the correctness oracle.
//   C = A^T * B  with A (k x m), B (k x n), C (m x n).
//
// Backend is selected at compile time:
//   -DTSMM_BLAS_MKL        -> Intel MKL        (target machine)
//   -DTSMM_BLAS_ACCELERATE -> Apple Accelerate (local macOS)
//   -DTSMM_BLAS_OPENBLAS   -> OpenBLAS / generic CBLAS
// =============================================================================

#if defined(TSMM_BLAS_MKL)
  #include <mkl.h>
  #define HAVE_BLAS 1
#elif defined(TSMM_BLAS_ACCELERATE)
  #define ACCELERATE_NEW_LAPACK
  #include <Accelerate/Accelerate.h>
  #define HAVE_BLAS 1
#elif defined(TSMM_BLAS_OPENBLAS)
  #include <cblas.h>
  #define HAVE_BLAS 1
#else
  #define HAVE_BLAS 0
#endif

static void* v9_prepare(int, int, int) { return nullptr; }

static void v9_compute(const TsmmProblem& p, void* /*ctx*/) {
#if HAVE_BLAS
    const int m = p.m, n = p.n, k = p.k;
    const double alpha = 1.0, beta = 0.0;
    if (p.layout == TSMM_COL_MAJOR) {
        // Column-major: A is k x m (lda=k), op(A)=A^T -> m x k.
        //   C(m x n) = A^T(m x k) * B(k x n)
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, n, k, alpha,
                    p.A, k,   // lda = k
                    p.B, k,   // ldb = k
                    beta, p.C, m);
    } else {
        // Row-major: A stored k x m row-major (lda=m), op(A)=A^T -> m x k.
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m, n, k, alpha,
                    p.A, m,   // lda = m (row-major k x m)
                    p.B, n,   // ldb = n (row-major k x n)
                    beta, p.C, n);
    }
#else
    (void)p;
#endif
}

static void v9_destroy(void*) {}

extern const TsmmOp TSMM_OP_v9_blas = {
    "v9_blas",
    "Vendor BLAS dgemm (MKL/Accelerate/OpenBLAS) - baseline & oracle",
    v9_prepare, v9_compute, v9_destroy
};
