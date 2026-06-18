#include "tsmm.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// Benchmark / evaluation driver.  Completely separate from the algorithm files:
// it only talks to ops through the TsmmOp interface (prepare/compute/destroy).
//
// Protocol (per project spec):
//   * Warm up 10 iterations, then time 20 iterations and take the average.
//   * Metric: GFLOPS = 2*m*n*k / (avg_seconds * 1e9).
//   * Correctness verified against the vendor BLAS op (v9_blas).
//   * Speedup vs BLAS gemm is reported; geometric mean of speedups summarizes.
//   * Results are emitted as JSON for the web dashboard.
// =============================================================================

namespace {

constexpr int WARMUP = 10;
constexpr int REPEAT = 20;

struct Shape { int m, n, k; const char* tag; bool required; };

// required : (m,n,k)=(4000,16000,128),(8,16,16000),(32,16000,16),(144,144,144)
// optional : (16,12344,16),(4,64,606841),(442,193,11),(40,1127228,40)
const Shape kShapes[] = {
    {4000, 16000, 128,  "T1_4000x16000x128",  true },
    {8,    16,    16000,"T2_8x16x16000",      true },
    {32,   16000, 16,   "T3_32x16000x16",     true },
    {144,  144,   144,  "T4_144x144x144",     true },
    {16,   12344, 16,   "O1_16x12344x16",     false},
    {4,    64,    606841,"O2_4x64x606841",    false},
    {442,  193,   11,   "O3_442x193x11",      false},
    {40,   1127228,40,  "O4_40x1127228x40",   false},
};

double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void fill_random(double* p, size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

// Frobenius-norm relative error vs reference:  ||ref-got||_F / ||ref||_F.
// This is the standard GEMM verification metric; it avoids false failures from
// per-element catastrophic cancellation (output entries near zero).
double max_rel_err(const double* ref, const double* got, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = ref[i] - got[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    if (den < 1e-30) return std::sqrt(num);
    return std::sqrt(num / den);
}

struct Result {
    std::string op;
    std::string task;
    int m, n, k;
    bool required;
    double gflops;
    double avg_ms;
    double rel_err;
    bool   correct;
    double speedup;   // vs blas
};

// Run one (op, shape). Returns timing+gflops. ref may be null (no check).
Result run_one(const TsmmOp* op, const Shape& s, TsmmLayout layout,
               const double* A, const double* B,
               const double* ref /*may be null*/) {
    Result r;
    r.op = op->name; r.task = s.tag;
    r.m = s.m; r.n = s.n; r.k = s.k; r.required = s.required;
    r.speedup = 0.0;

    const size_t cN = (size_t)s.m * s.n;
    double* C = (double*)tsmm_aligned_alloc(cN * sizeof(double));
    std::memset(C, 0, cN * sizeof(double));

    TsmmProblem p{ s.m, s.n, s.k, A, B, C, layout };
    void* ctx = op->prepare(s.m, s.n, s.k);

    for (int w = 0; w < WARMUP; ++w) op->compute(p, ctx);

    double t0 = now_sec();
    for (int it = 0; it < REPEAT; ++it) op->compute(p, ctx);
    double t1 = now_sec();

    double avg = (t1 - t0) / REPEAT;
    r.avg_ms = avg * 1e3;
    double flops = 2.0 * s.m * s.n * s.k;
    r.gflops = flops / (avg * 1e9);

    if (ref) {
        r.rel_err = max_rel_err(ref, C, cN);
        r.correct = (r.rel_err < 1e-9);
    } else {
        r.rel_err = 0.0; r.correct = true;
    }

    op->destroy(ctx);
    tsmm_aligned_free(C);
    return r;
}

void emit_json(const char* path, const std::vector<Result>& results,
               const char* backend, int threads, double geo_required,
               double geo_all) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return; }
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"backend\": \"%s\",\n", backend);
    std::fprintf(f, "  \"threads\": %d,\n", threads);
    std::fprintf(f, "  \"warmup\": %d,\n  \"repeat\": %d,\n", WARMUP, REPEAT);
    std::fprintf(f, "  \"geomean_speedup_required\": %.4f,\n", geo_required);
    std::fprintf(f, "  \"geomean_speedup_all\": %.4f,\n", geo_all);
    std::fprintf(f, "  \"results\": [\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        std::fprintf(f,
            "    {\"op\":\"%s\",\"task\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
            "\"required\":%s,\"gflops\":%.3f,\"avg_ms\":%.4f,"
            "\"rel_err\":%.3e,\"correct\":%s,\"speedup\":%.4f}%s\n",
            r.op.c_str(), r.task.c_str(), r.m, r.n, r.k,
            r.required ? "true" : "false", r.gflops, r.avg_ms,
            r.rel_err, r.correct ? "true" : "false", r.speedup,
            (i + 1 < results.size()) ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
}

const char* backend_name() {
#if defined(TSMM_BLAS_MKL)
    return "MKL";
#elif defined(TSMM_BLAS_ACCELERATE)
    return "Accelerate";
#elif defined(TSMM_BLAS_OPENBLAS)
    return "OpenBLAS";
#else
    return "none";
#endif
}

} // namespace

int main(int argc, char** argv) {
    // CLI: [--layout col|row] [--out results.json] [--only required|all]
    TsmmLayout layout = TSMM_COL_MAJOR;
    const char* out = "web/results.json";
    bool only_required = false;
    std::vector<std::string> op_filter;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--layout") && i + 1 < argc) {
            layout = std::strcmp(argv[++i], "row") == 0 ? TSMM_ROW_MAJOR : TSMM_COL_MAJOR;
        } else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) {
            out = argv[++i];
        } else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) {
            only_required = std::strcmp(argv[++i], "required") == 0;
        } else if (!std::strcmp(argv[i], "--op") && i + 1 < argc) {
            op_filter.push_back(argv[++i]);
        }
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    const TsmmOp* blas = tsmm_find_op("v9_blas");

    printf("=== TSMM Benchmark ===\n");
    printf("layout=%s  backend=%s  threads=%d  warmup=%d repeat=%d\n",
           layout == TSMM_COL_MAJOR ? "col" : "row",
           backend_name(), threads, WARMUP, REPEAT);
    printf("%-12s %-22s %8s %10s %10s %8s %8s\n",
           "op", "task", "GFLOPS", "avg_ms", "rel_err", "ok", "speedup");

    std::vector<Result> results;

    for (const Shape& s : kShapes) {
        if (only_required && !s.required) continue;

        const size_t aN = (size_t)s.k * s.m;
        const size_t bN = (size_t)s.k * s.n;
        const size_t cN = (size_t)s.m * s.n;
        double* A = (double*)tsmm_aligned_alloc(aN * sizeof(double));
        double* B = (double*)tsmm_aligned_alloc(bN * sizeof(double));
        fill_random(A, aN, 1234 + s.m + s.k);
        fill_random(B, bN, 5678 + s.n + s.k);

        // Compute reference with BLAS once.
        double* ref = (double*)tsmm_aligned_alloc(cN * sizeof(double));
        std::memset(ref, 0, cN * sizeof(double));
        double blas_gflops = 0.0;
        if (blas && blas->prepare) {
            TsmmProblem rp{ s.m, s.n, s.k, A, B, ref, layout };
            void* rc = blas->prepare(s.m, s.n, s.k);
            blas->compute(rp, rc);  // fill ref
            // time blas for speedup base
            for (int w = 0; w < WARMUP; ++w) blas->compute(rp, rc);
            double t0 = now_sec();
            for (int it = 0; it < REPEAT; ++it) blas->compute(rp, rc);
            double t1 = now_sec();
            double avg = (t1 - t0) / REPEAT;
            blas_gflops = (2.0 * s.m * s.n * s.k) / (avg * 1e9);
            blas->destroy(rc);
        }

        for (int oi = 0; oi < tsmm_num_ops(); ++oi) {
            const TsmmOp* op = tsmm_get_op(oi);
            if (!op_filter.empty() &&
                std::find(op_filter.begin(), op_filter.end(), op->name) == op_filter.end())
                continue;

            // skip naive on the giant optional shapes (too slow)
            const double work = 2.0 * s.m * s.n * (double)s.k;
            if (!std::strcmp(op->name, "v0_naive") && work > 2e10) {
                printf("%-12s %-22s %8s (skipped: too large for naive)\n",
                       op->name, s.tag, "-");
                continue;
            }

            const bool is_blas = !std::strcmp(op->name, "v9_blas");
            Result r = run_one(op, s, layout, A, B, is_blas ? nullptr : ref);
            if (is_blas) { r.correct = true; r.rel_err = 0.0; }
            r.speedup = (blas_gflops > 0.0) ? r.gflops / blas_gflops : 0.0;
            results.push_back(r);

            printf("%-12s %-22s %8.2f %10.3f %10.2e %8s %8.3f\n",
                   op->name, s.tag, r.gflops, r.avg_ms, r.rel_err,
                   r.correct ? "OK" : "FAIL", r.speedup);
        }

        tsmm_aligned_free(A);
        tsmm_aligned_free(B);
        tsmm_aligned_free(ref);
        printf("\n");
    }

    // geometric mean of speedups for our best non-blas op (v3) is interesting,
    // but spec asks for geomean of speedup vs MKL -> compute per-op geomeans.
    // Here we report geomean over the "v3_avx512" op (the optimized target).
    auto geomean_for = [&](const char* opname, bool req_only) {
        double logsum = 0.0; int cnt = 0;
        for (const Result& r : results) {
            if (r.op != opname) continue;
            if (req_only && !r.required) continue;
            if (r.speedup > 0.0) { logsum += std::log(r.speedup); ++cnt; }
        }
        return cnt ? std::exp(logsum / cnt) : 0.0;
    };

    double geo_req = geomean_for("v3_avx512", true);
    double geo_all = geomean_for("v3_avx512", false);

    printf("=== Geomean speedup (v3_avx512 vs %s) required=%.3f all=%.3f ===\n",
           backend_name(), geo_req, geo_all);

    emit_json(out, results, backend_name(), threads, geo_req, geo_all);
    printf("Wrote %s\n", out);
    return 0;
}
