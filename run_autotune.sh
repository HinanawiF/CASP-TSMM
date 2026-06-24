#!/bin/bash
#SBATCH --job-name=tsmm_tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output=tsmm_tune_%j.out
#SBATCH --error=tsmm_tune_%j.err

# Sweep our TSMM kernels with the same thread/NUMA policy used by the MKL
# baseline sweep. This is intended to find the best shape-aware choice under a
# fairer "best thread count + best layout" evaluation protocol.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source /public5/soft/modules/module.sh
module purge
module load intel/2022.1 2>/dev/null || true
module load gcc/10.2.0

make clean
if command -v icpx >/dev/null 2>&1; then
    make TARGET=intel CXX=icpx
else
    echo "[warn] icpx not found; falling back to g++"
    make TARGET=intel CXX=g++
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
export KMP_AFFINITY=granularity=fine,compact,1,0

THREADS="${THREADS:-1 2 4 8 16 24 48 96}"
LAYOUTS="${LAYOUTS:-row col}"
SCHEDULES="${SCHEDULES:-static dynamic,8}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-5}"
OUTDIR="${OUTDIR:-web/autotune_${SLURM_JOB_ID:-manual}}"

# Candidate kernels only. v9_blas is still used internally as correctness
# reference, but is not emitted as a candidate result.
OPS=(
  --op v1_blocked
  --op v2_openmp
  --op v3_avx512
  --op v4_kreduce
  --op v5_dot_parallel
  --op v6_smallk_col
)

mkdir -p "$OUTDIR"

bind_cmd() {
    local t="$1"
    if [ "$t" -le 24 ]; then
        echo "numactl --cpunodebind=0 --membind=0"
    elif [ "$t" -le 48 ]; then
        echo "numactl --cpunodebind=0-1 --interleave=0-1"
    else
        echo "numactl --cpunodebind=0-3 --interleave=all"
    fi
}

echo "=== TSMM autotune sweep ==="
echo "threads:   $THREADS"
echo "layouts:   $LAYOUTS"
echo "schedules: $SCHEDULES"
echo "warmup=$WARMUP repeat=$REPEAT outdir=$OUTDIR"

for layout in $LAYOUTS; do
  for sched in $SCHEDULES; do
    for t in $THREADS; do
      export OMP_NUM_THREADS="$t"
      export MKL_NUM_THREADS="$t"
      export OMP_SCHEDULE="$sched"

      sched_tag="${sched//,/_}"
      out="$OUTDIR/${layout}_t${t}_${sched_tag}.json"
      echo "=== layout=$layout threads=$t schedule=$sched ==="
      read -r -a numa <<< "$(bind_cmd "$t")"
      "${numa[@]}" ./tsmm_bench \
        --layout "$layout" \
        --warmup "$WARMUP" \
        --repeat "$REPEAT" \
        --out "$out" \
        "${OPS[@]}"
    done
  done
done

python3 scripts/summarize_autotune.py \
  --scan-dir "$OUTDIR" \
  --mkl-csv mkl_baseline_threads_33801149.csv \
  --out "$OUTDIR/autotune_summary.csv"

echo "=== autotune done ==="
echo "summary: $OUTDIR/autotune_summary.csv"
