# CASP 并行计算大作业：TSMM 优化

**TSMM (Tall-Skinny Matrix Multiplication)**：计算 `C = Aᵀ × B`，其中
`A ∈ ℝ^{k×m}`，`B ∈ ℝ^{k×n}`，`C ∈ ℝ^{m×n}`，双精度。FLOPs = `2·m·n·k`。

本仓库实现了一个**算法与评测解耦**的框架：每个算子单独一个 `.cpp`，只暴露
`prepare()` / `compute()` / `destroy()`；评测、正确性校验、计时、网页展示全部
和算法分离。本机（Apple Silicon）与目标机（Intel Xeon AVX-512）**同一份源码**，
靠 Makefile 自动选择后端。

---

## 目录结构

```
casp/
├── include/
│   ├── tsmm.h          # 算子接口 (TsmmOp / TsmmProblem)、存储约定、注册表声明
│   └── common.h        # 对齐内存分配等工具
├── ops/                # 算子实现，每个版本一个文件，互相独立
│   ├── v0_naive.cpp    # 串行三重循环（支持行主序/列主序）—— 基准 & 正确性参照
│   ├── v1_blocked.cpp  # Cache 分块 + MR×NR 寄存器分块（串行，列主序）
│   ├── v2_openmp.cpp   # OpenMP：按输出 tile 并行，调度策略可选 (OMP_SCHEDULE)
│   ├── v3_avx512.cpp   # AVX-512 内核（沿 k 向量化）+ OpenMP；非 x86 自动标量回退
│   ├── v9_blas.cpp     # 厂商 BLAS dgemm（MKL / Accelerate / OpenBLAS）—— 对比 & 正确性 oracle
│   └── registry.cpp    # 算子注册表
├── benchmark.cpp       # 评测驱动（warmup 10 + repeat 20、GFLOPS、加速比、几何平均、JSON 输出）
├── web/
│   ├── index.html      # 实时网页看板（读取 results.json，自动刷新）
│   └── results.json    # 评测结果（运行后生成）
├── Makefile            # 跨平台构建（自动识别 macOS / Intel 目标）
├── serve.sh            # 本机起 http 服务看网页（无需 docker）
├── run_slurm.sh        # 目标机 slurm 提交：4 NUMA / 96 CPU，MKL，全部 shape
├── run_scaling.sh      # 目标机 slurm：线程扩展性 + 调度策略对比 sweep
└── README.md
```

## 评测的矩阵规模

```
required : (m,n,k) = (4000,16000,128), (8,16,16000), (32,16000,16), (144,144,144)
optional : (m,n,k) = (16,12344,16), (4,64,606841), (442,193,11), (40,1127228,40)
```

## 评测协议（与要求一致）

1. 每个 (算子, 规模) 先**预热 10 轮**，再**计时 20 次取平均**。
2. 指标 `GFLOPS = 2·m·n·k / (平均耗时·1e9)`。
3. 正确性：与厂商 BLAS dgemm 对比，用 Frobenius 范数相对误差 `‖ref-got‖_F/‖ref‖_F < 1e-9`。
4. 加速比 = 算子 GFLOPS / BLAS gemm GFLOPS；对各任务取**几何平均**作为参考。

---

## 本机运行（macOS / Apple Silicon）

依赖：`brew install libomp`（OpenMP），BLAS 用系统自带 Apple Accelerate。

```bash
cd casp
make run            # 编译并跑全部 shape，结果写入 web/results.json
make run-required   # 只跑 required shape
./serve.sh 8000     # 浏览器打开 http://localhost:8000/ 看实时看板
```

> ⚠️ 本机是 arm64，没有 AVX-512：`v3_avx512` 会编译成**标量回退**路径，所以本机上它
> 不快是正常的。它的 intrinsic 内核是为 **Intel 目标机**准备的，本机只验证正确性与可编译性。
> 本机的 BLAS 参照是 Accelerate，目标机才是 MKL，因此本机的“加速比”仅供参考。

## 目标机运行（BSCC 并行超算云 bscc-t6：Intel Xeon Platinum 9242，4 NUMA / 96 CPU）

> 平台不是直接 ssh，而是通过网页/客户端登录（见《并行处理课程计算平台使用方法》）。

**登录**
1. 打开超算云平台网页，用课程账号登录 → 点 **SSH** 图标 → 选课程分配的超算账号 → 连接。
2. 进入 `CAS_PP2026` 文件夹。

**上传代码**（计算节点不能联网，登录节点联网需代理）
- 用平台桌面的 **快传 / WinSCP** 把整个 `casp/` 上传到自己的个人目录，**不要用命令行 scp 直连**。
- 个人目录命名规范：姓名拼音 + 学号后三位，例如 `wys_000`：
  ```bash
  cd CAS_PP2026 && mkdir wys_000   # 换成你自己的
  ```

**编译**（登录节点只能编译，禁止跑程序；编译进程≤2 个）
```bash
source /public5/soft/modules/module.sh   # 平台要求: module 前必须先 source
module load intel/2022.1 gcc/10.2.0
cd CAS_PP2026/wys_000/casp
make TARGET=intel CXX=icpx               # 优先 Intel 编译器+MKL; 无 icpx 则 CXX=g++
```

**提交作业**（计算必须经 slurm 提交到计算节点）
```bash
sbatch run_slurm.sh        # 全部 shape，MKL + AVX-512，96 线程，跨 NUMA interleave
sbatch run_scaling.sh      # 线程扩展性 + 调度策略 sweep（写到 web/scaling/）
squeue                     # 查看作业状态
scancel <JOBID>            # 取消作业
# 结果: tsmm_<JOBID>.out（slurm 日志） + web/results.json
```
> `run_slurm.sh` 已内置 `source module.sh`、icpx/g++ 自动检测、NUMA 亲和。
> 分区名（`--partition`）默认留空用默认队列；如需指定先 `sinfo` 查看再回填脚本顶部。

**交互式快速验证**（可选，小作业）
```bash
srun -n 1 ./tsmm_bench --only required   # 单进程交互式跑一下
```

**取回结果看网页**（在本机执行）
- 用快传/WinSCP 把目标机的 `web/results.json` 下载回本机 `casp/web/`，然后 `./serve.sh 8000` 看实时看板。

构建关键点（`make TARGET=intel`）：
- `-march=native -mavx512f -mavx512dq -mfma`，启用 `v3_avx512` 的 intrinsic 内核；
- `-DTSMM_BLAS_MKL` 链接 MKL（`MKLROOT` 由 `module load intel` 提供）；
- 线程亲和：`OMP_PROC_BIND=spread` + `OMP_PLACES=cores` 把线程铺到 4 个 NUMA node；
- `numactl --interleave=all` 让大矩阵 A/B/C 的内存跨 4 个 NUMA node 交织，避免单内存控制器瓶颈。

> 集群默认不能用 docker；网页看板是纯静态 `python3 -m http.server`，把目标机产出的
> `web/results.json` 拷回本机用 `./serve.sh` 查看即可。

---

## 各版本优化思路（用于报告）

| 版本 | 优化点 | 说明 |
|------|--------|------|
| v0_naive | 基准 | 教科书三重循环，行/列主序均支持，作为正确性 oracle 与性能下界 |
| v1_blocked | 访存分块 | 列主序下 A、B 的列都是连续的长度-k 向量；MR×NR 寄存器分块最大化寄存器复用，KC 分块保证 cache 驻留 |
| v2_openmp | 多线程 | 对输出 tile 网格并行（输出空间无写冲突），`schedule(runtime)` 可通过 `OMP_SCHEDULE` 切换 static/dynamic/guided 做调度对比与扩展性分析 |
| v3_avx512 | 向量化 | 沿 k 用 AVX-512 FMA（8 double/ZMM），4×4 寄存器分块共 24 个 ZMM 累加器，尾部标量收尾；非 x86 自动回退标量 |
| v9_blas | 对比/参照 | MKL/Accelerate/OpenBLAS dgemm，既是性能上限对比也是正确性参照 |

### 后续优化方向（待目标机实测反馈）
- A/B 的 **packing**（把要用的列重排成连续 panel，提高 TLB / 预取效率），尤其对 K 大的 shape；
- 针对极端瘦高 shape（如 `8×16×16000`、`4×64×606841`）改成沿 k 做 **并行归约 + 树形求和**，而不是按输出 tile 并行；
- `m`/`n` 很小但 `k` 巨大的情况，BLAS 也难打满，可考虑多线程切 k 段；
- 调 MR/NR/KC 参数匹配目标机 L1=32K / L2=1M / L3=36M；
- 内核汇编 / prefetch 微调（可选项）。

> 工作流：本机先保证正确性 + 框架跑通 → 目标机 slurm 实测 → 把 `results.json` / 扩展性数据
> 反馈回来 → 据此调参数与并行策略再迭代。
