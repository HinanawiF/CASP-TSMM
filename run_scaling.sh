#!/bin/bash
#SBATCH --job-name=tsmm_scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output=tsmm_scale_%j.out

# =============================================================================
# Strong-scaling sweep on the Intel target: run the required shapes at several
# thread counts and several OpenMP scheduling policies. Produces one JSON per
# configuration so the report can plot scalability and schedule comparisons.
# =============================================================================
set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# 平台要求: 使用 module 前先 source。
source /public5/soft/modules/module.sh
module purge
module load intel/2022.1 2>/dev/null || true
module load gcc/10.2.0

make clean
if command -v icpx >/dev/null 2>&1; then
    make TARGET=intel CXX=icpx
else
    make TARGET=intel CXX=g++
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export MKL_DYNAMIC=FALSE
mkdir -p web/scaling

for sched in static "dynamic,8" guided; do
  for t in 1 2 4 8 16 24 48 96; do
    export OMP_NUM_THREADS=$t
    export MKL_NUM_THREADS=$t
    export OMP_SCHEDULE=$sched
    tag="t${t}_${sched//,/_}"
    echo "=== threads=$t schedule=$sched ==="
    numactl --interleave=all ./tsmm_bench --layout col --only required \
        --out "web/scaling/${tag}.json" \
        --op v2_openmp --op v3_avx512 --op v9_blas
  done
done
echo "=== scaling sweep done ==="
