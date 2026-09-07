# ASTR 核心变更影响指南

本指南用于在编码前界定改动范围。每种变更都必须先确认 CPU reference 是否合理，再定义 GPU ownership、MPI semantics 和验收证据。发现明确 CPU 逻辑缺陷时停止移植，向用户或维护者报告证据和候选修复；未经决策不得直接复制 bug，也不得擅自改变 CPU 行为。

## 新增算例

| 项目 | 要求 |
|---|---|
| 决策边界 | 明确 flowtype、维数、网格、初场、BC、scheme、physics flags、source、statistics 和输出 oracle |
| CPU 入口 | `src/initialisation.F90::flowinit` 的 dispatch 与专用 initializer；必要时 `src/solver.F90::rhscal` 的 source dispatch |
| GPU 入口 | `src_gpu/case_capability_gpu.cuf::case_capability_gpu` 的显式 predicate；复用通用 kernel，新增 case state 时通过 facade 初始化 |
| Ownership | 初场在 host 生成并一次上传；compute loop 不因新 flowtype 增加全场 round trip |
| MPI | 先列出每个轴是 periodic、physical 还是 decomposed，并确认 local extent 满足 `hm` |
| 最小测试 | CPU smoke、GPU reject-before-enable、NP=1 one-step field、multi-step statistics、NP=2 三个 slab 中与 BC 相关的方向、physical invariant |
| 接受证据 | same-topology CPU/GPU 等价和独立物理量；oversubscription 不作为 scaling 证据 |
| 停止条件 | CPU 初场/BC 不闭合，输入依赖 compact 或 deferred physics，缺少独立 oracle，或 GPU predicate 只能通过放宽未知组合实现 |

新算例 public 名称应描述物理 case，而不是某次验证 phase。临时 forced-3D 或 explicit override 必须在验证脚本和文档中标明，不能替换原始算例语义后仍沿用原结论。

## 新增或修改边界条件

| 项目 | 要求 |
|---|---|
| 决策边界 | 定义 face、法向、传入/传出 characteristic、目标状态、corner priority、ghost depth、filter closure 和 RHS closure |
| CPU 入口 | legacy path 从 `src/bc.F90::boucon` 审计；conservative path 从 config、face algebra、runtime stage 三层审计 |
| GPU 入口 | `src_gpu/boundary_gpu.cuf::apply_boundary_conditions_gpu` 或 conservative face/stage module |
| Ownership | physical face/ghost 由 BC 写入；MPI interface 不得被 physical kernel 覆盖；curved face 优先使用已有 geometric normal/projection |
| MPI | physical rank 执行 BC，internal rank 执行 halo；检查 six-face corner ordering 和 topology-specific active range |
| 最小测试 | algebra probe、face/corner probe、one-stage RHS、one-step field、相关 slab/combined topology、long-step invariant、memcheck |
| 接受证据 | primitive 与 conservative face residual、same-phase field diff、wall/farfield invariant 和统计量 |
| 停止条件 | CPU 路径读取未定义 halo、法向定义与几何不一致、NSCBC incoming-wave policy 未定义，或 CPU 明确 bug 尚未获修复决策 |

边界不是单个 kernel。filter 前后、RHS 前后、qswap、primitive refresh、statistics 和 checkpoint phase 共同组成边界契约。

## 修改数值格式

| 项目 | 要求 |
|---|---|
| 决策边界 | 给出离散公式、order、stencil radius、boundary closure、selector、flux split、sensor coupling 和稳定性适用范围 |
| CPU 入口 | `src/comsolver.F90::solvrinit`, `src/derivative.F90::derivative`, `src/filter.F90::filter`, `src/solver.F90::rhscal` |
| GPU 入口 | `src_gpu/solver_gpu.cuf::solver_gpu`, `src_gpu/gradcal_gpu.cuf::gradcal_gpu`, dispatch in `src_gpu/mainloop_gpu.cuf::time_integration_rk_gpu` |
| Ownership | work arrays 必须 device-resident；in-place stencil 必须证明无覆盖；常量系数优先复用 `constdef` 语义 |
| MPI | stencil radius 决定 halo；solution endpoint 与 raw field halo 分开；curvilinear metric halo 必须同相位 |
| 最小测试 | analytic/operator test、manufactured RHS、one-step and ten-step field、boundary closure、MPI interface、physics benchmark、profile |
| 接受证据 | order/accuracy 与 CPU/GPU equivalence 分开报告；性能以完整 RK 或 end-to-end 为主 |
| 停止条件 | 公式与 selector 不一致，local dimension 小于 stencil，CPU oracle 可疑，或单 kernel 变快但完整 RK 退化且无目标平台证据 |

当前主线不重新开放 compact line solve。将默认 compact 的输入改为 explicit 可以用于新验证，但必须记录格式变化，不能作为原 compact case 的直接复现。

## 新增 device field

| 项目 | 要求 |
|---|---|
| 决策边界 | 定义 shape、halo、lifetime、authoritative side、初始化、更新者、消费者、输出和释放时机 |
| Host 入口 | 对应 host field owner 通常为 `src/commarray.F90::commarray` 或明确 config module |
| GPU 入口 | 在 `src_gpu/commarray_gpu.cuf::alloc_gpu_arrays` 分配，在 `copy_flow_to_gpu` 或专用 facade 初始化 |
| Ownership | 明确是 persistent field、stage work、halo buffer、reduction partial 还是 output snapshot |
| MPI | 只有 stencil 消费者需要的 field 才加入 generic halo；定义 `hm`、component count 和 physical ownership |
| 最小测试 | allocation/capability negative test、initial copy、one update、halo contract、memcheck、Nsight transfer audit |
| 接受证据 | 无 uninitialized read、无隐藏全场 transfer、释放路径完整、field diff 与 consumer result 均通过 |
| 停止条件 | 无法说明 authoritative side，复制由多个模块隐式触发，或仅为临时优化增加完整 4D field 且 end-to-end 无收益 |

新增 field 后更新[运行时与数据流](runtime-and-dataflow.md)的 ownership 表。只有 output/checkpoint 或显式 validation hook 可以将完整 device field 同步至 host。

## 修改 HaloTransport

| 项目 | 要求 |
|---|---|
| 决策边界 | 先冻结 payload semantics，再选择 pageable、pinned、nonblocking 或 device-aware transport |
| CPU reference | `src/parallel.F90::qswap` 及对应 raw-field exchange |
| GPU 入口 | semantics/pack/unpack 在 `src_gpu/halo_exchange_gpu.cuf::halo_exchange_gpu`；transport 在 `src_gpu/halo_transport_gpu.cuf::halo_transport_gpu` |
| Ownership | solver 只请求 halo，不持有 MPI request 或 staging policy |
| MPI | solution 为 `hm+1` 加 endpoint average；filter/diffusion/generic field 为 `hm` 无 average；private tags `21001..21006` |
| 最小测试 | buffer/count/tag probe、NP=2 x/y/z、NP=4 combined、NP=8 `2x2x2`、fallback、memcheck、timeline |
| 接受证据 | same-topology fields/statistics、错误 fallback、无 full-field transfer、真实 overlap timeline、one-rank-per-GPU scaling |
| 停止条件 | transport 改变数值 halo、不同 rank mode 不一致、registration failure 无 collective fallback，或非阻塞 API 没有时间线 overlap |

新 transport 必须保留 host-staged correctness backend。CUDA-aware 或 HIP-aware 失败后能否回退是 production portability 的一部分。

## 新增统计量

| 项目 | 要求 |
|---|---|
| 决策边界 | 先写数学定义、归一化、采样相位、空间/时间平均方向和 MPI reduction 语义 |
| CPU 入口 | `src/statistic.F90::statcal`, `src/statistic.F90::statout` 或专用 diagnostic function |
| GPU 入口 | local kernel/reduction in `src_gpu/statistic_gpu.cuf::statistic_gpu`，facade dispatch in `src_gpu/gpu_runtime.cuf::gpu_write_flow_statistics` |
| Ownership | device 计算 local numerator/denominator，host 只接收 partial/scalar；写文件由 I/O rank 完成 |
| MPI | 先 global sum/max，再归一化；禁止对 rank-local average 再做无权平均 |
| 最小测试 | 手工小场、CPU/GPU scalar、decomposition invariance、zero/degenerate denominator、sampling continuation after restart |
| 接受证据 | 数学定义、实现函数、输出字段和物理解释一致；same field 在 NP=1/多 rank 给出相同 global statistic |
| 停止条件 | 平均方向不具统计均匀性、wall metric/area 不正确、采样 state phase 未定义，或 denominator 可为零但无明确处理 |

统计量图示和论文解释不属于核心代码验收本身。正式科研输出还需给出定义、无量纲化、绘图函数和独立物理解释。

## 通用提交顺序

1. 固定 CPU 行为与适用范围，发现 CPU bug 时先决策。
2. 添加会失败的局部 contract 或 capability test。
3. 实现最小 CPU/GPU/MPI 变更并通过局部 gate。
4. 依次通过 one-step field、multi-step statistics、multi-rank、physical、sanitizer、residency 和 performance gate。
5. 更新 generated inventory、当前架构、数值契约和 validation matrix 中受影响部分。
6. 只暂存相关源码、测试和文档，不加入 runtime output 或 profile binary。

