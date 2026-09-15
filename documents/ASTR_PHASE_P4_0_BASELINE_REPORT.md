# ASTR Phase P4-0 Baseline Report

## Scope And Commit

本报告冻结 P4-0 的本地可信性能基线。测量对应分支
`feature/gpu_dev` 的代码提交 `16441bba458f892e4e9d1bcc06c59dc2bf78aa4b`，
设计基点为 `9e4fd2cbde5365448e23482ac705265278e917a0`。P4-0 只增加
fail-closed benchmark 策略、启动场 I/O 抑制、严格 RK 阶段计时和验证工具，
未改变 `solver_gpu.cuf` 或 `gradcal_gpu.cuf` 的数值算术。`mainloop_gpu.cuf`
中的变更仅包围既有调用并记录阶段时间。

所有性能数据均来自本地双 GPU 工作站，采用 FP64、RK3、六阶显式中心差分、
十阶显式中心滤波、完整五分量 `qwork_d`、`pageable` 主机暂存和逐 kernel
显式同步。它们只用于筛选 P4-1 优化对象，不代表 A800 性能或生产级扩展性。

## Local Hardware And Software

| 项目 | 配置 |
| --- | --- |
| CPU | 2 x Intel Xeon Gold 6248R，48 物理核，96 逻辑 CPU |
| GPU | 2 x NVIDIA RTX 4000 Ada Generation，19195 MiB/卡 |
| NVIDIA 驱动 | 595.91.07 |
| NVHPC | nvfortran 26.1-0 |
| MPI | Open MPI 4.1.9a1 |
| CMake | 4.2.3 |
| 操作系统 | Linux 7.0.0-31-generic x86_64 |

构建统一从项目根目录 `CMakeLists.txt` 配置。CPU 和 GPU 可执行文件分别位于
`build_cpu_p4/bin/astr` 和 `build_gpu_p4/bin/astr`。

## Benchmark Lifecycle Result

`ASTR_GPU_BENCHMARK_NO_FIELD_IO=1` 只允许 CUDA 构建中的三维、全周期、GPU
TGV，并要求 `ASTR_GPU_RK_TIMING=1`。非法布尔值、CPU 路径、非 TGV、非三维、
非全周期边界、rank 间配置不一致均集体终止。策略只跳过生成网格后的
`writegrid` 和初始场 `writeflfed`，不改变统计量、常规 checkpoint、restart
或普通运行的输出语义。

计时基线和阶段归因中的每个标签均采用一个独立 warm-up 进程，再启动五个
独立保留进程。
每个保留进程配置 `MAXSTEP=20`，共推进 21 次，丢弃第一个进程内样本并保留
20 个 slowest-rank 完整 RK 样本。四种计时拓扑均有五行结果，每行恰有
20 个有限样本。以下目录中未发现 `grid*.h5` 或 `flowfield*.h5`：

- `tests/gpu_validation/out/p4_0_256_baseline`
- `tests/gpu_validation/out/p4_0_256_phases`
- `tests/gpu_validation/out/p4_0_nsys/case`

普通 CPU/GPU 场比较仍生成启动和 checkpoint HDF5，证明 benchmark 开关没有
改变默认输出路径。

## CPU GPU Field Equivalence

场比较采用 `128^3` TGV、`643e`、滤波与黏性项均开启、完整 FP64
滤波工作区。表中数值是所有守恒量 `q1:q5` 的最大绝对差。门槛为
`atol=rtol=1e-10`，所有报告均为有限值并通过。

| MPI 布局 | 步数 | 最大守恒场差 | 结果 |
| --- | ---: | ---: | --- |
| NP=1，`1x1x1` | 1 | `2.8421709430404007e-13` | PASS |
| NP=1，`1x1x1` | 10 | `2.8421709430404007e-13` | PASS |
| NP=1，`1x1x1` | 100 | `6.2527760746888816e-13` | PASS |
| NP=2，`2x1x1` | 10 | `3.1263880373444408e-13` | PASS |
| NP=2，`1x2x1` | 10 | `3.1263880373444408e-13` | PASS |
| NP=2，`1x1x2` | 10 | `2.8421709430404007e-13` | PASS |

这里验证的是 CPU/GPU 同相位场等价，不替代长时间 TGV 物理统计验证。

## NP1 And NP2 Slab Timing

性能基线采用 `256^3` TGV。完整 RK 时间先在每个独立进程内取 20 个
slowest-rank 样本的中位数，再对五个进程中位数取中位数。加速比统一定义为
同机 NP=1 GPU 中位时间除以相应 NP=2 中位时间；效率为加速比除以 2。

| GPU/MPI 布局 | 完整 RK 中位时间 (s) | 相对 NP=1 加速比 | 强扩展效率 | 五轮相对极差 | 吞吐率 (cell-RK/s) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NP=1，`1x1x1` | `0.5960130865` | `1.000` | `100.000%` | `10.553%` | `2.814907e7` | `9768` |
| NP=2，`2x1x1` | `0.6137393010` | `0.971118` | `48.556%` | `1.530%` | `2.733606e7` | `7621` |
| NP=2，`1x2x1` | `0.5937514960` | `1.003809` | `50.190%` | `1.146%` | `2.825629e7` | `7621` |
| NP=2，`1x1x2` | `0.5908108075` | `1.008805` | `50.440%` | `0.789%` | `2.839693e7` | `7621` |

NP=2 的计算量减半后，完整 RK 时间几乎不变。NP=1 五轮相对极差达到
`10.553%`，说明本地节点存在明显噪声。因此表中差异只支持“当前双卡不能
形成有效强扩展”的判断，不用于区分 y-slab 与 z-slab 的细小性能差异。

## RK Phase Attribution

阶段计时单独运行，不进入上表。每个 `prepare` 行含五轮、每轮 21 次推进，
共 105 个 slowest-rank 样本；其余阶段每次推进执行三个 RK 子步，共 315 个
样本。`prepare` 是包含第一子步 filter 和 solution halo 的包容时间，不能与
嵌套阶段相加。

| 阶段 | NP=1 中位时间 (ms) | NP=2 `2x1x1` 中位时间 (ms) | NP2/NP1 |
| --- | ---: | ---: | ---: |
| prepare，包容时间 | `104.847` | `122.644` | `1.170` |
| filter | `26.462` | `37.579` | `1.420` |
| solution halo | `4.053` | `29.978` | `7.396` |
| convection | `37.179` | `18.843` | `0.507` |
| diffusion flux | `18.311` | `9.568` | `0.523` |
| diffusion halo | `3.490` | `42.406` | `12.151` |
| diffusion RHS | `34.832` | `17.690` | `0.508` |
| RK update | `10.252` | `5.202` | `0.507` |

convection、diffusion flux、diffusion RHS 和 RK update 接近理想减半，说明
局部计算缩放正常。solution halo 和 diffusion halo 分别增加约 `25.925 ms`
和 `38.916 ms`，filter 也增加约 `11.117 ms`。暴露通信与滤波 halo 路径抵消了
两卡计算收益。diffusion halo 是当前 x-slab 中最大的最终优化对象，但 P4-1
仍先用结构较简单的 solution halo 验证单轴异步状态机。

## Nsight Systems Timeline Evidence

诊断 trace 为 NP=2 `2x1x1`、`256^3`、`MAXSTEP=2`、显式同步和 pageable
主机暂存：
`tests/gpu_validation/out/p4_0_nsys/np2_xslab.nsys-rep`。整个进程生命周期中，
Nsight Systems 记录 352 次 `cudaMemcpy` API 调用、840 次
`cudaDeviceSynchronize`、506 个 H2D 操作、168 个 D2H 操作，以及两个 rank
合计 204 次 `MPI_Sendrecv`。总量包含启动初始化，不能直接解释为 RK halo
传输量。

RK 区间内的一次 rank 0 solution-halo 交换给出以下事件序列。时间是 trace
起点后的秒数。

| 操作 | 起止时间 (s) | 观察 |
| --- | --- | --- |
| 左/右 pack kernels | `18.905984` - `18.906456` | pack 完成后才开始主机读回 |
| 两次 D2H | `18.906475` - `18.939359` | pageable 同步复制 |
| 两次 `MPI_Sendrecv` | `18.940532` - `18.966859` | tags `21001/21002`，阻塞在主线程 |
| 两次 H2D | `18.967175` - `18.975042` | MPI 完成后才回传设备 |
| unpack kernel | `18.975053` - `18.976180` | H2D 完成后启动 |

该时间线与当前代码中的
`pack -> sync -> D2H -> exchange_host_pair -> H2D -> unpack -> sync` 顺序一致。
本 trace 没有证明计算与通信重叠；它证明该 halo 传输在显式模式下串行暴露。

## P4-1 Decision

**Decision: GO**

| 门槛 | 证据 | 结果 |
| --- | --- | --- |
| 计时目录无场 HDF5 | 三个 P4-0 输出树均为 0 个禁止文件 | PASS |
| 每种拓扑五个独立保留进程 | 四份 timing TSV 均为五行 | PASS |
| 每进程 20 个有限 slowest-rank RK 样本 | 每行 `rk_samples=20`，时间均有限 | PASS |
| 八个阶段标签完整 | NP=1 每日志 462 条，NP=2 每日志 924 条；严格解析通过 | PASS |
| NP=1 与三种 NP=2 slab 场误差不超过 `1e-10` | 最大值 `6.2527760746888816e-13` | PASS |
| P4-0 未修改数值 kernel 算术 | 基点至测量提交的核心算术文件无差异 | PASS |

P4-1 首先实现 x 方向 solution halo 的 `pinned-pipeline` 单轴状态机，用它验证
异步 pack/D2H、MPI request 生命周期、H2D/unpack 和依赖事件。完整 `qwork`
filter halo 随后接入，`sigma_d` 与 `qflux_d` diffusion halo 在状态机通过后
接入。每一步都必须重新通过 NP=1、x/y/z slab 场等价和完整 RK 门槛；只有
完整 RK 改善才保留优化。
