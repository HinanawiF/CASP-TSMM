# 《并行计算》大作业 PPT 大纲：TSMM 不规则稠密矩阵乘优化

## 1. 标题页

- 题目：不规则稠密矩阵乘 TSMM 优化
- 公式：`C = A^T × B`
- 平台：Intel Xeon Platinum 9242，96 CPU，4 NUMA
- 关键词：OpenMP、AVX-512、cache blocking、k 维归约、shape-aware dispatch

讲述重点：这是一个不规则形状矩阵乘问题，不同 `(m,n,k)` 的性能瓶颈完全不同。

## 2. 问题定义与评测规模

- `A ∈ R^{k×m}`，`B ∈ R^{k×n}`，`C ∈ R^{m×n}`
- FLOPs = `2mnk`
- required：T1/T2/T3/T4
- optional：O1/O2/O3/O4

讲述重点：T1 是大输出，T2/O2 是小输出大 k，T3/O1 是小 k 大 n，O4 是超大 n。

## 3. 平台与评测协议

- BSCC bscc-t6
- 96 CPU，2 Socket，4 NUMA
- 编译器：`icpx`
- BLAS：Intel MKL
- warmup 10，repeat 20
- 正确性：Frobenius 相对误差 `< 1e-9`

讲述重点：所有数据来自同一节点环境；结果已归档在 `web/results.json`、`tsmm_33723544.out/.err`。

## 4. 性能模型：为什么不能只写一个通用 kernel

- 算术强度近似：`2mnk / (8(mk+nk+mn))`
- 输出空间、k 长度、访存连续性共同决定瓶颈
- T2/O2 输出并行度不足
- T3/O1 单个 dot 太短
- T1/O4 更适合输出 tile 并行

讲述重点：优化策略要和 shape 绑定，这是后面多个版本并存的原因。

## 5. 框架设计：算法与评测解耦

- 每个算子独立 `.cpp`
- 统一接口：`prepare / compute / destroy`
- benchmark 统一做数据生成、计时、正确性、JSON 输出
- `v9_blas` 同时作为性能参考和正确性 oracle

讲述重点：这个框架便于逐步加入新优化版本并公平比较。

## 6. 优化路线总览

- `v0_naive`：串行三重循环
- `v1_blocked`：cache blocking + register tile
- `v2_openmp`：输出 tile 并行
- `v3_avx512`：AVX-512 shape dispatch
- `v4_kreduce`：k 维归约
- `v5_dot_parallel`：输出点并行 dot
- `v6_smallk_col`：超大 n 小 k packed-A 候选 + gating

讲述重点：每一版都针对前一版暴露出来的瓶颈。

## 7. 优化一：blocking + OpenMP

- blocking 提升 cache 与寄存器复用
- 输出 tile 没有写冲突，适合 OpenMP
- T1/T3/T4/O4 明显受益
- 局限：T2/O2 输出元素太少，无法喂满 96 核

讲述重点：这是最稳的基础并行策略，也是多个 shape 的最终最佳方案。

## 8. 优化二：AVX-512 与 shape dispatch

- 初版 outer-product 减少水平求和
- 实测发现 gather 成本在 T1/O4/T2/O2 上过高
- 最终方案：
  - 大输出/大 k：连续访存 dot4
  - 短 k/小输出：outer8 候选

讲述重点：SIMD 优化不是越复杂越好，访存模式决定收益。

## 9. 优化三：小输出大 k 的特殊处理

- T2/O2 的 `m*n` 只有 128/256
- `v4_kreduce`：沿 k 维拆分 partial C
- `v5/v6`：每个 C 元素独立 dot，让编译器向量化
- T2 最终从 `29.36 GFLOPS` 提升到 `70.98 GFLOPS`
- O2 最终由 `v4_kreduce` 达到 `30.94 GFLOPS`

讲述重点：小输出问题必须增加并行粒度，不能只按输出 tile 分块。

## 10. 最终结果：每个 shape 的最佳算子

- Required geomean：`0.2995 × MKL`
- All geomean：`0.2248 × MKL`
- 全部 correctness：OK
- T1 最佳：`v3_avx512 1617.02 GFLOPS`
- T2 最佳：`v6_smallk_col 70.98 GFLOPS`
- T3/T4 最佳：`v2_openmp`
- O2 最佳：`v4_kreduce`
- O4 最佳：`v2_openmp 847.78 GFLOPS`

讲述重点：最终策略是按 shape 选择最佳非 BLAS 算子。

## 11. 失败尝试与经验

- pure outer8：理论减少水平求和，但 gather 成本导致 T1/O4 下降
- small-k packA：T3/O1 出现 `0.x GFLOPS`，打包和调度开销过大
- v6 最终加 shape gating，避免失败路径污染结果
- `.err` 仅有 unused helper warning，无运行错误

讲述重点：报告价值不只在最终数字，也在解释为什么某些优化不适用。

## 12. 总结与后续方向

- 完成串行、并行、访存、SIMD、归约优化
- 最终 required geomean `0.2995`
- 不规则 TSMM 的核心是 shape-aware optimization
- 后续可继续做：
  - T3/O1 小 k micro-kernel
  - O2 NUMA-aware tree reduction
  - MR/NR/KC 参数调优
  - prefetch / assembly-level micro tuning

讲述重点：当前版本已经适合提交，后续优化方向清晰。
