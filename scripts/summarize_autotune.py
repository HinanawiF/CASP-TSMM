#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


PROBLEM_ORDER = [
    ("T1_4000x16000x128", "4000x16000x128", True),
    ("T2_8x16x16000", "8x16x16000", True),
    ("T3_32x16000x16", "32x16000x16", True),
    ("T4_144x144x144", "144x144x144", True),
    ("O1_16x12344x16", "16x12344x16", False),
    ("O2_4x64x606841", "4x64x606841", False),
    ("O3_442x193x11", "442x193x11", False),
    ("O4_40x1127228x40", "40x1127228x40", False),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", required=True, help="Directory containing autotune JSON files")
    ap.add_argument("--mkl-csv", required=True, help="MKL baseline thread/layout sweep CSV")
    ap.add_argument("--out", default="autotune_summary.csv", help="Output CSV path")
    return ap.parse_args()


def load_mkl(path):
    rows = list(csv.DictReader(open(path, newline="")))
    best_all = {}
    best_row = {}
    best_col = {}
    meta_all = {}
    for _task, problem, _required in PROBLEM_ORDER:
        rs = [r for r in rows if r["problem"] == problem]
        if not rs:
            raise SystemExit(f"missing MKL baseline for {problem}")

        def key(r):
            return float(r["gflops"])

        b_all = max(rs, key=key)
        b_row = max([r for r in rs if r["layout"] == "row-major"], key=key)
        b_col = max([r for r in rs if r["layout"] == "col-major"], key=key)
        best_all[problem] = float(b_all["gflops"])
        best_row[problem] = float(b_row["gflops"])
        best_col[problem] = float(b_col["gflops"])
        meta_all[problem] = b_all
    return best_all, best_row, best_col, meta_all


def parse_json_name(path):
    # run_autotune.sh writes e.g. row_t24_static.json, col_t96_dynamic_8.json.
    m = re.match(r"(row|col)_t(\d+)_(.+)\.json$", path.name)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), m.group(3)


def load_candidates(scan_dir):
    best = {}
    all_rows = []
    for path in sorted(Path(scan_dir).glob("*.json")):
        layout_from_name, threads_from_name, schedule = parse_json_name(path)
        data = json.load(open(path))
        layout = data.get("layout") or layout_from_name
        threads = int(data.get("threads") or threads_from_name or 0)
        repeat = int(data.get("repeat") or 0)
        warmup = int(data.get("warmup") or 0)
        for r in data.get("results", []):
            if r.get("op") == "v9_blas":
                continue
            if not r.get("correct", False):
                continue
            task = r["task"]
            op = r["op"]
            gflops = float(r["gflops"])
            rec = {
                "task": task,
                "op": op,
                "gflops": gflops,
                "avg_ms": float(r["avg_ms"]),
                "layout": layout,
                "threads": threads,
                "schedule": schedule,
                "warmup": warmup,
                "repeat": repeat,
                "file": str(path),
            }
            all_rows.append(rec)
            if task not in best or gflops > best[task]["gflops"]:
                best[task] = rec
    return best, all_rows


def geomean(values):
    vals = [v for v in values if v > 0.0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    args = parse_args()
    best_mkl_all, best_mkl_row, best_mkl_col, meta_all = load_mkl(args.mkl_csv)
    best, _all_rows = load_candidates(args.scan_dir)

    out_rows = []
    req_ratios = []
    all_ratios = []
    for task, problem, required in PROBLEM_ORDER:
        if task not in best:
            print(f"[warn] missing candidate result for {task}")
            continue
        rec = best[task]
        mkl_best = best_mkl_all[problem]
        ratio = rec["gflops"] / mkl_best
        if required:
            req_ratios.append(ratio)
        all_ratios.append(ratio)
        out_rows.append({
            "task": task,
            "problem": problem,
            "required": required,
            "best_op": rec["op"],
            "best_layout": rec["layout"],
            "best_threads": rec["threads"],
            "best_schedule": rec["schedule"],
            "best_gflops": f"{rec['gflops']:.6f}",
            "best_avg_ms": f"{rec['avg_ms']:.6f}",
            "mkl_best_gflops": f"{mkl_best:.6f}",
            "mkl_best_layout": meta_all[problem]["layout"],
            "mkl_best_threads": meta_all[problem]["threads"],
            "vs_mkl_best": f"{ratio:.6f}",
            "source_file": rec["file"],
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fieldnames = [
            "task", "problem", "required", "best_op", "best_layout", "best_threads",
            "best_schedule", "best_gflops", "best_avg_ms", "mkl_best_gflops",
            "mkl_best_layout", "mkl_best_threads", "vs_mkl_best", "source_file",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print("Best candidate per shape vs scanned MKL baseline")
    for r in out_rows:
        print(
            f"{r['task']:22s} {r['best_op']:16s} "
            f"{float(r['best_gflops']):9.2f} GFLOPS  "
            f"layout={r['best_layout']:3s} t={str(r['best_threads']):>2s} "
            f"vs_mkl={float(r['vs_mkl_best']):.3f}"
        )
    print(f"geomean_required_vs_mkl_best={geomean(req_ratios):.6f}")
    print(f"geomean_all_vs_mkl_best={geomean(all_ratios):.6f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
