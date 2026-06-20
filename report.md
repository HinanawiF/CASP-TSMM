# 《并行计算》课程大作业报告：不规则稠密矩阵乘 TSMM 优化

> 课程项目：Tall-Skinny Matrix Multiplication (TSMM) Multiplication Optimization  
> 目标表达式：`C = A^T × B`，其中 `A ∈ R^{k×m}`，`B ∈ R^{k×n}`，`C ∈ R^{m×n}`  
> 精度：FP64 双精度  
> 目标平台：BSCC bscc-t6，Intel Xeon Platinum 9242，2 Socket / 4 NUMA / 96 CPU

## 摘要

本项目围绕课程给定的 TSMM 计算 `C = A^T × B`，实现了从串行基线、cache blocking、OpenMP 并行、AVX-512 内核、k 维归约到特定形状调度的完整优化链路。评测框架与算子实现解耦，每个优化版本独立实现 `prepare / compute / destroy` 接口，统一由 benchmark 驱动完成正确性校验、预热、重复计时和 JSON 结果输出。

最终稳定版本在目标 Intel 平台上，required 四个规模的 best non-BLAS 几何平均加速比为 `0.2995 × MKL`（报告中四舍五入为 `0.300`），全部 required 和 optional 规模正确性均通过。最终结果以仓库中归档的 `web/results.json`、`tsmm_33723544.out`、`tsmm_33723544.err` 为准；其中 `.err` 仅包含一个未使用函数的编译 warning，不影响正确性和运行。

## 1. 问题定义与实验规模

TSMM 计算形式为：

```text
C = A^T × B
A ∈ R^{k×m}, B ∈ R^{k×n}, C ∈ R^{m×n}
FLOPs = 2 × m × n × k
```

课程要求中的评测规模分为 required 和 optional：

| 类别 | 任务 | (m,n,k) | 形状特点 |
|---|---:|---:|---|
| required | T1 | (4000,16000,128) | 输出大，k 中等，计算量大 |
| required | T2 | (8,16,16000) | 输出极小，k 很大，输出并行度不足 |
| required | T3 | (32,16000,16) | k 很小，n 大，近似大量短 dot |
| required | T4 | (144,144,144) | 小/中等方形规模 |
| optional | O1 | (16,12344,16) | k 小，n 大 |
| optional | O2 | (4,64,606841) | 输出很小，k 极大 |
| optional | O3 | (442,193,11) | k 极小，输出中等 |
| optional | O4 | (40,1127228,40) | n 极大，k 小，输出很长 |

本项目默认使用列主序布局。列主序下 `A` 的每一列（即参与 dot 的 `A_col_r`）和 `B` 的每一列都是长度为 `k` 的连续数组，因此基础 dot product 能获得连续访存，这也是后续多个优化版本的设计基础。

## 2. 实验平台与评测方法

### 2.1 硬件与软件环境

最终实验在课程并行云平台 bscc-t6 上运行，节点信息来自 `tsmm_33723544.out`：

| 项目 | 配置 |
|---|---|
| CPU | Intel Xeon Platinum 9242 CPU @ 2.30GHz |
| CPU 数 | 96 |
| Socket | 2 |
| NUMA node | 4 |
| NUMA 分布 | node0: 0-23, node1: 24-47, node2: 48-71, node3: 72-95 |
| 编译器 | `icpx` |
| 编译参数 | `-O3 -march=native -mavx512f -mavx512dq -mfma -funroll-loops -qopenmp` |
| BLAS | Intel MKL |
| 线程数 | 96 |

最终数据归档文件如下，已随仓库保存：

| 文件 | 作用 | 备注 |
|---|---|---|
| `web/results.json` | 结构化评测结果 | benchmark 自动输出，包含所有算子的 GFLOPS、耗时、误差、speedup 和 geomean |
| `tsmm_33723544.out` | Slurm 标准输出 | 包含节点拓扑、编译命令、完整评测表 |
| `tsmm_33723544.err` | Slurm 标准错误 | 仅有 `compute_scalar_col` 未使用的编译 warning，无运行错误 |

编译命令核心部分如下：

```bash
icpx -std=c++17 -Wall -Wextra -Wno-unused-parameter -O3 -march=native \
  -mavx512f -mavx512dq -mfma -funroll-loops -qopenmp \
  -Iinclude -I/public5/soft/oneAPI/2022.1/mkl/latest/include \
  -DTSMM_BLAS_MKL ops/*.cpp benchmark.cpp -o tsmm_bench \
  -L/public5/soft/oneAPI/2022.1/mkl/latest/lib/intel64 \
  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
```

### 2.2 评测协议

每个 `(算子, shape)` 使用相同评测协议：

| 项目 | 设置 |
|---|---|
| warmup | 10 次 |
| repeat | 20 次 |
| 时间指标 | 20 次平均耗时 `avg_ms` |
| 性能指标 | `GFLOPS = 2mnk / time` |
| 正确性 | 与 MKL dgemm 结果比较 |
| 误差标准 | Frobenius 相对误差 `||C_ref-C||_F / ||C_ref||_F` |
| 通过阈值 | `rel_err < 1e-9` |
| 总体指标 | 对每个任务选择 fastest correct non-BLAS 算子，再对 speedup 取几何平均 |

采用 Frobenius 范数相对误差，而不是逐元素相对误差，是为了避免结果中接近 0 的元素导致误报。

需要说明的是，表格中的 `speedup` 使用 benchmark 内部单独计时的 MKL 基准 `blas_gflops` 计算；`v9_blas` 行本身也会再次作为普通算子计时输出，因此同一 shape 中 `v9_blas` 行的 GFLOPS 与 speedup 基准可能存在小幅运行波动。本文的总体指标以 `web/results.json` 中的 `geomean_speedup_required` 和 `geomean_speedup_all` 为准。

## 3. 代码结构与实现框架

项目采用“算子与评测解耦”的结构。核心接口定义在 `include/tsmm.h`，每个算子只需要实现：

```cpp
prepare(m, n, k)
compute(problem, ctx)
destroy(ctx)
```

代码目录：

| 文件/目录 | 作用 |
|---|---|
| `benchmark.cpp` | 统一评测驱动，负责数据生成、正确性检查、计时、JSON 输出 |
| `include/tsmm.h` | 算子接口、问题描述、注册表声明 |
| `include/common.h` | 对齐内存分配等公共工具 |
| `ops/v0_naive.cpp` | 串行基线 |
| `ops/v1_blocked.cpp` | cache blocking 与寄存器分块 |
| `ops/v2_openmp.cpp` | OpenMP 输出 tile 并行 |
| `ops/v3_avx512.cpp` | AVX-512 形状分流内核 |
| `ops/v4_kreduce.cpp` | k 维并行归约 |
| `ops/v5_smallk_packa.cpp` | 输出点并行 dot，运行名为 `v5_dot_parallel` |
| `ops/v6_smallk_col.cpp` | 超大 n 小 k packed-A 列核候选，带 shape gating |
| `ops/v9_blas.cpp` | MKL/OpenBLAS/Accelerate 参考实现 |
| `run_slurm.sh` | 课程平台 Slurm 提交脚本 |
| `web/index.html` | 结果可视化网页 |

这种设计的好处是：新增优化版本不需要改 benchmark 主逻辑，便于逐步实验、回滚和横向比较。

## 4. 阶段一：基础实现与性能分析

### 4.1 串行实现 `v0_naive`

串行版本实现了最直接的三重循环：

```text
for c in [0,n):
  for r in [0,m):
    C[r,c] = dot(A[:,r], B[:,c])
```

在列主序布局下，`A[:,r]` 和 `B[:,c]` 均为连续长度 `k` 数组，因此内层 dot 的访存是连续的。该版本的主要作用有两个：

1. 提供正确性基线；
2. 提供性能下界，用于衡量 blocking、OpenMP、AVX-512 的实际收益。

最终平台上，`v0_naive` 在 T1 只有 `6.58 GFLOPS`，T2 为 `6.66 GFLOPS`，说明单线程三重循环远不能利用 96 核和 AVX-512 资源。

### 4.2 与 MKL 对比

`v9_blas` 使用 MKL dgemm 作为库函数对照和正确性 oracle。MKL 在大多数 shape 上显著领先，尤其在 T1/T3/O1 这类可被 BLAS 内部 packing 和 micro-kernel 充分优化的形状上优势明显。

但对于部分不规则形状，手写实现也能接近 MKL：

| 任务 | 最佳手写算子 | 最佳手写 GFLOPS | MKL GFLOPS | 说明 |
|---|---|---:|---:|---|
| T4 | `v2_openmp` | 232.89 | 391.13 | 小矩阵，多线程和调度开销占比高 |
| O4 | `v2_openmp` | 847.78 | 778.04 | n 极大，简单输出并行已经较好 |
| T2 | `v6_smallk_col` | 70.98 | 120.13 | 输出小但 dot 很长，专用 dot 并行有效 |
| O2 | `v4_kreduce` | 30.94 | 236.94 | k 极大，仍受归约和内存带宽限制 |

### 4.3 性能模型分析

TSMM 的算术强度与 shape 强相关。理想 FLOPs 为 `2mnk`，但访存量至少包括读 A、读 B、写 C：

```text
Bytes ≈ 8 × (mk + nk + mn)
Arithmetic Intensity ≈ 2mnk / Bytes
```

不同 shape 的瓶颈差异明显：

| 形状类型 | 代表任务 | 主要瓶颈 | 优化方向 |
|---|---|---|---|
| 输出大、k 中等 | T1 | 计算与 cache 复用 | blocking + AVX-512 + OpenMP |
| 输出极小、k 极大 | T2/O2 | 输出并行度不足，单 dot 很长 | k 维归约或输出点 dot 并行 |
| k 很小、n 大 | T3/O1 | 每个 dot 太短，调度和函数开销明显 | 静态调度、减少任务粒度 |
| n 极大、k 小 | O4 | 内存流式访问 + NUMA | 输出列/块并行，NUMA interleave |

这也是后续优化没有采用单一通用内核，而是采用多版本候选和 shape-aware 策略的原因。

## 5. 阶段二：并行优化设计

### 5.1 `v1_blocked`：cache blocking 与寄存器分块

`v1_blocked` 在串行基础上加入 MR×NR 输出 tile。对一个 tile 内多个 `C[r,c]` 同时累加，可以复用同一段 A/B 数据，减少 cache miss，并提高寄存器复用。

效果：

| 任务 | v0 GFLOPS | v1 GFLOPS | 提升 |
|---|---:|---:|---:|
| T1 | 6.58 | 19.09 | 2.90× |
| T2 | 6.66 | 29.36 | 4.41× |
| T4 | 12.92 | 38.11 | 2.95× |
| O3 | 2.57 | 11.93 | 4.64× |

这说明即使不引入多线程，仅改善局部性和寄存器复用也有稳定收益。

### 5.2 `v2_openmp`：输出 tile 并行

`v2_openmp` 将输出 tile 网格并行化。由于每个 tile 写入不同的 `C` 区域，不存在写冲突，适合 OpenMP 并行。

特点：

- 使用 `#pragma omp parallel for schedule(runtime)`；
- 可通过 `OMP_SCHEDULE` 比较 static/dynamic/guided；
- 对 T1、T3、T4、O4 这类输出空间较大的任务效果明显；
- 对 T2/O2 这类 `m*n` 很小的任务不足，因为可并行输出元素太少。

最终结果中，`v2_openmp` 是 T3、T4、O1、O4 的最佳或接近最佳版本：

| 任务 | v2 GFLOPS | 说明 |
|---|---:|---|
| T1 | 1580.96 | 与 v3 接近，说明输出 tile 并行已能利用多核 |
| T3 | 70.12 | required 中小 k 大 n 的最佳版本 |
| T4 | 232.89 | 小矩阵上比 v3 略好 |
| O1 | 28.05 | optional 小 k 大 n 最佳版本 |
| O4 | 847.78 | optional 超大 n 形状最佳版本 |

### 5.3 `v3_avx512`：形状分流 AVX-512 内核

`v3_avx512` 经历了两次迭代：

1. 早期尝试 `8×8 outer-product` 内核，减少水平求和；
2. 实测发现该方法在 T1/O4/T2/O2 上由于 gather 成本过高而退化；
3. 最终改为 shape-dispatch：
   - 大输出/大 k：使用连续访存 `dot4` AVX-512；
   - 短 k/小输出：保留 `outer8` 候选；
   - 非 AVX-512 平台自动走标量回退。

关键经验是：AVX-512 并不必然带来收益。如果为了向量化引入 gather 或额外重排，可能抵消水平求和优化。最终 `v3` 在 T1 上达到 `1617.02 GFLOPS`，是 T1 的最佳手写实现。

### 5.4 `v4_kreduce`：k 维并行归约

T2/O2 的输出矩阵很小：

```text
T2: m*n = 8*16 = 128
O2: m*n = 4*64 = 256
```

如果只按输出 tile 并行，最多只能产生几百个任务，难以喂满 96 核。`v4_kreduce` 将 k 维切分给多个线程，各线程计算 partial C，最后归约。

效果在 O2 上最明显：

| 任务 | v2_openmp | v3_avx512 | v4_kreduce | 最佳性 |
|---|---:|---:|---:|---|
| O2 | 4.19 | 14.51 | 30.94 | v4 最佳 |

但在 T2 上，k 维归约开销相对结果规模较大，最终不如 `v5/v6` 的输出点 dot 并行。

### 5.5 `v5_dot_parallel`：输出点并行 dot

`v5` 的实验过程比较典型。最初尝试 small-k packed-A，但在 T3/O1 上出现严重退化；分析后发现真正有效的是 fallback 路径：每个 C 元素独立计算一个连续 dot product，并交给编译器自动向量化。

最终将其收敛为 `v5_dot_parallel`：

```text
parallel for idx in [0, m*n):
  r = idx % m
  c = idx / m
  C[r,c] = dot(A[:,r], B[:,c])
```

它在 T2 上显著有效：

| 任务 | v1_blocked | v5_dot_parallel | 提升 |
|---|---:|---:|---:|
| T2 | 29.36 | 66.04 | 2.25× |

### 5.6 `v6_smallk_col`：超大 n 小 k 列核与 shape gating

`v6` 尝试将 A 打包成 `k×m` 行连续布局，并按 B/C 列并行计算整列 C。第一次实测表明，它在 T3/O1 上会出现 `0.x GFLOPS` 的异常慢路径，原因是 packing 和调度开销超过计算收益。

最终版本加入 shape gating：

```text
仅当 n >= 100000 且 k <= 64 时启用 packed-A 列核；
其它形状回退到输出点并行 dot。
```

这样保留了 O4 这类超大 n 小 k 形状的候选价值，同时避免 T3/O1/O3 被失败路径拖慢。最终 T2 上 `v6_smallk_col` 达到 `70.98 GFLOPS`，是 T2 最佳手写算子。

## 6. 最终实验结果

最终稳定版本来自作业 `tsmm_33723544.out`，结构化数据来自 `web/results.json`。运行配置：

```text
layout=col
backend=MKL
threads=96
warmup=10
repeat=20
```

### 6.1 每个 shape 的最佳非 BLAS 算子

| 任务 | (m,n,k) | 最佳非 BLAS 算子 | GFLOPS | avg_ms | MKL GFLOPS | 备注 |
|---|---:|---|---:|---:|---:|---|
| T1 | (4000,16000,128) | v3_avx512 | 1617.02 | 10.132 | 3822.88 | AVX-512 dot4 最有效 |
| T2 | (8,16,16000) | v6_smallk_col | 70.98 | 0.058 | 120.13 | 输出点 dot 并行有效 |
| T3 | (32,16000,16) | v2_openmp | 70.12 | 0.234 | 1390.71 | 小 k 下 MKL packing 优势大 |
| T4 | (144,144,144) | v2_openmp | 232.89 | 0.026 | 391.13 | 小矩阵调度开销敏感 |
| O1 | (16,12344,16) | v2_openmp | 28.05 | 0.225 | 576.81 | k 很小，手写核差距大 |
| O2 | (4,64,606841) | v4_kreduce | 30.94 | 10.043 | 236.94 | k 维归约有效但仍受限 |
| O3 | (442,193,11) | v1_blocked | 11.93 | 0.157 | 121.59 | blocking 比多线程更稳 |
| O4 | (40,1127228,40) | v2_openmp | 847.78 | 4.255 | 778.04 | 输出并行充分，接近 MKL 级别；该 shape 的 MKL 单次计时波动较明显 |

总体指标：

| 指标 | 数值 |
|---|---:|
| required geomean speedup (best non-BLAS vs MKL) | 0.2995（约 0.300） |
| all geomean speedup (best non-BLAS vs MKL) | 0.2248（约 0.225） |
| 正确性 | 全部 OK |

`tsmm_33723544.err` 中唯一诊断信息为：

```text
ops/v6_smallk_col.cpp:57:20: warning: unused function 'compute_scalar_col' [-Wunused-function]
```

该 warning 来自 AVX-512 分支启用后，非 AVX-512 fallback helper 未被当前编译目标调用；不影响最终二进制生成、运行和正确性。所有结果行的 `ok` 均为 `OK`。

### 6.2 完整关键运行结果表

| op | task | GFLOPS | avg_ms | rel_err | ok | speedup |
|---|---|---:|---:|---:|---|---:|
| v0_naive | T1_4000x16000x128 | 6.58 | 2489.232 | 4.05e-16 | OK | 0.002 |
| v1_blocked | T1_4000x16000x128 | 19.09 | 858.400 | 4.22e-16 | OK | 0.005 |
| v2_openmp | T1_4000x16000x128 | 1580.96 | 10.363 | 4.22e-16 | OK | 0.429 |
| v3_avx512 | T1_4000x16000x128 | 1617.02 | 10.132 | 4.11e-16 | OK | 0.439 |
| v4_kreduce | T1_4000x16000x128 | 163.97 | 99.921 | 0.00e+00 | OK | 0.045 |
| v5_dot_parallel | T1_4000x16000x128 | 430.22 | 38.083 | 4.05e-16 | OK | 0.117 |
| v6_smallk_col | T1_4000x16000x128 | 421.90 | 38.834 | 4.05e-16 | OK | 0.115 |
| v9_blas | T1_4000x16000x128 | 3822.88 | 4.286 | 0.00e+00 | OK | 1.038 |
| v0_naive | T2_8x16x16000 | 6.66 | 0.615 | 1.25e-15 | OK | 0.055 |
| v1_blocked | T2_8x16x16000 | 29.36 | 0.140 | 5.44e-16 | OK | 0.241 |
| v2_openmp | T2_8x16x16000 | 21.49 | 0.191 | 2.53e-15 | OK | 0.176 |
| v3_avx512 | T2_8x16x16000 | 17.33 | 0.236 | 1.63e-15 | OK | 0.142 |
| v4_kreduce | T2_8x16x16000 | 7.63 | 0.537 | 6.30e-16 | OK | 0.063 |
| v5_dot_parallel | T2_8x16x16000 | 66.04 | 0.062 | 1.27e-15 | OK | 0.541 |
| v6_smallk_col | T2_8x16x16000 | 70.98 | 0.058 | 1.27e-15 | OK | 0.582 |
| v9_blas | T2_8x16x16000 | 120.13 | 0.034 | 0.00e+00 | OK | 0.985 |
| v0_naive | T3_32x16000x16 | 7.29 | 2.246 | 1.78e-16 | OK | 0.006 |
| v1_blocked | T3_32x16000x16 | 16.85 | 0.972 | 1.69e-16 | OK | 0.013 |
| v2_openmp | T3_32x16000x16 | 70.12 | 0.234 | 1.69e-16 | OK | 0.055 |
| v3_avx512 | T3_32x16000x16 | 68.45 | 0.239 | 0.00e+00 | OK | 0.054 |
| v4_kreduce | T3_32x16000x16 | 50.17 | 0.327 | 0.00e+00 | OK | 0.040 |
| v5_dot_parallel | T3_32x16000x16 | 43.84 | 0.374 | 1.78e-16 | OK | 0.035 |
| v6_smallk_col | T3_32x16000x16 | 42.22 | 0.388 | 1.78e-16 | OK | 0.033 |
| v9_blas | T3_32x16000x16 | 1390.71 | 0.012 | 0.00e+00 | OK | 1.098 |
| v0_naive | T4_144x144x144 | 12.92 | 0.462 | 4.26e-16 | OK | 0.032 |
| v1_blocked | T4_144x144x144 | 38.11 | 0.157 | 4.43e-16 | OK | 0.093 |
| v2_openmp | T4_144x144x144 | 232.89 | 0.026 | 4.43e-16 | OK | 0.569 |
| v3_avx512 | T4_144x144x144 | 227.79 | 0.026 | 0.00e+00 | OK | 0.556 |
| v4_kreduce | T4_144x144x144 | 98.72 | 0.060 | 0.00e+00 | OK | 0.241 |
| v5_dot_parallel | T4_144x144x144 | 25.33 | 0.236 | 4.25e-16 | OK | 0.062 |
| v6_smallk_col | T4_144x144x144 | 25.97 | 0.230 | 4.25e-16 | OK | 0.063 |
| v9_blas | T4_144x144x144 | 391.13 | 0.015 | 0.00e+00 | OK | 0.955 |
| v2_openmp | O1_16x12344x16 | 28.05 | 0.225 | 1.69e-16 | OK | 0.047 |
| v4_kreduce | O2_4x64x606841 | 30.94 | 10.043 | 1.11e-15 | OK | 0.174 |
| v1_blocked | O3_442x193x11 | 11.93 | 0.157 | 1.22e-16 | OK | 0.102 |
| v2_openmp | O4_40x1127228x40 | 847.78 | 4.255 | 2.47e-16 | OK | 0.960 |

> 注：optional 部分表格保留每个 optional shape 的最佳非 BLAS 行，完整逐算子日志可见 `tsmm_33723544.out` 和 `web/results.json`。`speedup` 以 benchmark 单独测得的 MKL 基准为分母，因此可能与同表 `v9_blas` 行的 GFLOPS 直接相除略有差异。

## 7. 关键迭代与失败分析

### 7.1 AVX-512 outer-product 并非总是更好

曾尝试将 `v3` 改成 `8×8 outer-product`，理论上可以减少每个 C 元素的水平求和。但目标机实测显示：

| 任务 | 旧 dot4 v3 | outer8 v3 | 结论 |
|---|---:|---:|---|
| T1 | 约 1617 GFLOPS | 约 950 GFLOPS | gather 读取 A 的成本过高 |
| O4 | 约 835 GFLOPS | 约 655 GFLOPS | 超大 n 下 gather 更明显拖慢 |
| T4 | 约 21 GFLOPS | 约 216 GFLOPS | 小矩阵/短 k 下 outer8 有收益 |

因此最终采用 shape dispatch，而不是单一 AVX-512 kernel。

### 7.2 k 维归约适合 O2，但不适合所有小输出

O2 的 `k=606841`，输出只有 256 个元素。`v4_kreduce` 通过 k 维切分提升并行度，最终从 `v2_openmp 4.19 GFLOPS` 提升到 `30.94 GFLOPS`。但 T2 的 k 只有 16000，归约开销占比更高，最终不如 `v5/v6` 的连续 dot 并行。

### 7.3 packed-A 小 k 列核需要谨慎 gating

`v6` 早期试图服务 T3/O1，但实测出现严重退化。原因是这些 shape 的单次计算量太小，packA 和调度开销无法摊薄。最终做法是只在超大 n 小 k 形状启用 packed-A 列核，其余形状回退。

这个过程说明：优化不是简单叠加技术，而需要用目标平台实测验证假设。失败尝试同样提供了有效分析：它帮助确定哪些 shape 不能使用复杂预处理。

## 8. 扩展性与调度分析

本项目实现了 `run_scaling.sh`，用于比较不同线程数和 `OMP_SCHEDULE` 调度策略。虽然最终报告采用 96 线程完整结果，但从性能模型可以解释调度行为：

| shape 类型 | 输出任务数 | 推荐策略 | 原因 |
|---|---:|---|---|
| T1/O4 | 很多 | static/dynamic 均可 | 输出 tile 多，负载均衡容易 |
| T2/O2 | 很少 | 不能只按输出并行 | 输出任务不足，需 k-reduction 或 dot-parallel |
| T3/O1 | 中等但每任务短 | static 更稳 | dynamic 调度开销可能超过计算收益 |
| O3/T4 | 小规模 | 避免过度并行 | 线程启动和同步开销明显 |

最终版本中，所有算子统一参与 benchmark，由结果选择每个 shape 的最快正确版本。这样的策略比强行寻找单一通用内核更适合“不规则稠密矩阵乘”场景。

## 9. 结论

本项目完成了课程要求的两个阶段：

1. 基础实现与分析：实现串行 TSMM，支持行/列主序，建立 MKL 对照，构建正确性和性能评测框架；
2. 并行优化：实现 OpenMP 输出并行、cache blocking、AVX-512 内核、k 维归约和 shape-aware 选择，并在 Intel 96 核平台完成实测。

最终稳定版本在 required 任务上达到 `0.300 × MKL` 的几何平均性能，全部任务正确性通过。优化过程中的主要经验是：

- TSMM 的最优策略高度依赖 `(m,n,k)`；
- 输出空间大时，输出 tile 并行和 AVX-512 dot 内核有效；
- 输出空间小但 k 很大时，需要 k 维归约或输出点 dot 并行；
- packed-A、outer-product 等技术必须结合访存代价和调度粒度判断，不能只看理论 FLOPs；
- 与 MKL 的差距主要来自成熟 BLAS 的 packing、micro-kernel、预取和线程运行时优化。

后续若继续优化，可优先考虑：

1. 为 T3/O1 设计专门的小 k micro-kernel，减少 MKL 差距；
2. 对 O2 引入更高效的树形归约和 NUMA-aware partial buffer；
3. 根据 Xeon Platinum 9242 的 cache 参数系统调 MR/NR/KC；
4. 对关键内核做汇编级检查和 prefetch 微调。

## 附录：复现实验命令

```bash
source /public5/soft/modules/module.sh
module load intel/2022.1 gcc/10.2.0
make clean
make TARGET=intel CXX=icpx
sbatch run_slurm.sh
```

查看结果：

```bash
squeue -u $USER
cat tsmm_<JOBID>.out
cat web/results.json
```

最终稳定结果对应作业日志：`tsmm_33723544.out`。
