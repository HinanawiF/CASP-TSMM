#!/bin/bash
#SBATCH --job-name=tsmm
##SBATCH --partition=PARTITION       # 取消注释并填入分区名(登录后 `sinfo` 查看);不确定先留注释用默认队列
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96           # 节点独占, 96 核 = 2 socket x 48 核 = 4 NUMA node
#SBATCH --exclusive
#SBATCH --output=tsmm_%j.out
#SBATCH --error=tsmm_%j.err

# =============================================================================
# BSCC 并行超算云 (bscc-t6) 上的 TSMM 评测作业提交脚本。
#   目标机: Intel Xeon Platinum 9242, 节点独占 96 核, 4 NUMA node
#           (CPU 0-23 / 24-47 / 48-71 / 72-95)。用全部 4 个 NUMA / 96 核并行。
#
# 平台要点(见《并行处理课程计算平台使用方法》):
#   * 登录节点只用于编辑/编译, 禁止直接跑程序; 计算必须用 sbatch/srun 提交。
#   * 计算节点不能联网, 依赖只能靠 module load(不能临时下载)。
#   * 使用 module 前必须先 source 平台的 module.sh。
#
# 提交方式:  sbatch run_slurm.sh
# 查看状态:  squeue            取消作业:  scancel <JOBID>
# 结果文件:  tsmm_<JOBID>.out  以及  web/results.json
# =============================================================================

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ---- 环境 (平台要求: 先 source 再 module load) ------------------------------
source /public5/soft/modules/module.sh
module purge
# 优先用 Intel 编译器 + MKL(性能最佳); 若无可用 intel 模块则回退 gcc(见下方 make)。
module load intel/2022.1 2>/dev/null || true   # 提供 icc/icpx + MKL, 设置 MKLROOT
module load gcc/10.2.0                          # 课程默认编译器 / libgomp
# 若 intel 模块未导出 MKLROOT, 在此显式指定(用 `module show intel/2022.1` 查路径):
# export MKLROOT=/public5/soft/.../mkl

echo "=== node info ==="
lscpu | grep -E "Model name|NUMA|^CPU\(s\)|Socket" || true
numactl --hardware | head -20 || true

# ---- 编译 (Intel 目标: MKL + AVX-512) ---------------------------------------
# 默认用 icpx(Intel 编译器, 与平台 PPT 的 icc 示例一致, 配 MKL 性能最好)。
# 若 icpx 不可用, 改成下面注释的 g++ 那行。
make clean
if command -v icpx >/dev/null 2>&1; then
    make TARGET=intel CXX=icpx
else
    echo "[warn] icpx 不可用, 回退 g++"
    make TARGET=intel CXX=g++
fi

# ---- 线程 / 亲和性 ----------------------------------------------------------
export OMP_NUM_THREADS=96
export OMP_PROC_BIND=spread         # 把线程铺到 4 个 NUMA node
export OMP_PLACES=cores
export OMP_SCHEDULE=dynamic,8       # tile 调度策略(可换 static / guided 做对比)
export MKL_NUM_THREADS=96
export MKL_DYNAMIC=FALSE

mkdir -p web

# numactl --interleave: 让大矩阵 A/B/C 的内存跨 4 个 NUMA node 交织,
# 避免单内存控制器成为大 shape 的瓶颈。
echo "=== run: all shapes ==="
numactl --interleave=all ./tsmm_bench --layout col --out web/results.json

echo "=== done. results in web/results.json ==="
