#include "tsmm.h"
#include <cstring>

// Each operator lives in its own translation unit and exports a TsmmOp object.
extern const TsmmOp TSMM_OP_v0_naive;
extern const TsmmOp TSMM_OP_v1_blocked;
extern const TsmmOp TSMM_OP_v2_openmp;
extern const TsmmOp TSMM_OP_v3_avx512;
extern const TsmmOp TSMM_OP_v9_blas;

static const TsmmOp* kOps[] = {
    &TSMM_OP_v0_naive,
    &TSMM_OP_v1_blocked,
    &TSMM_OP_v2_openmp,
    &TSMM_OP_v3_avx512,
    &TSMM_OP_v9_blas,
};

int tsmm_num_ops() {
    return (int)(sizeof(kOps) / sizeof(kOps[0]));
}

const TsmmOp* tsmm_get_op(int idx) {
    if (idx < 0 || idx >= tsmm_num_ops()) return nullptr;
    return kOps[idx];
}

const TsmmOp* tsmm_find_op(const char* name) {
    for (int i = 0; i < tsmm_num_ops(); ++i)
        if (std::strcmp(kOps[i]->name, name) == 0) return kOps[i];
    return nullptr;
}
