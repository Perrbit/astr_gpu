# ASTR GPU 性能优化推进计划

状态更新：2026-09-07。P1、P2 和 P3 本机阶段均已收口；P0 A800 基线仍待执行。

## 1. 目标

本计划面向当前 ASTR CUDA Fortran 实现，在不改变既定数值方法和物理语义的前提下，提高单 GPU 计算效率，并为后续多 GPU 通信优化和激波边界层算例性能优化建立统一流程。

性能优化必须同时满足以下条件：

- CPU/GPU 数值结果继续通过既有误差门槛；
- RK3、边界处理、MPI halo 和滤波时序不变；
- 主时间循环保持 GPU 变量常驻；
- 性能结论来自完整 RK 计时，不以单个 kernel 的 NCU replay 时间代替；
- 每项优化具有明确的 NCU 或 Nsight Systems 瓶颈证据；
- 失败候选保留测试记录，但不保留在正式计算路径中。

当前源码起点为提交：

```text
0d684a84bb689654bfc5a53a6928140c3a8e6fdb
perf(gpu): close TGV profiling and candidate validation
```

该提交是本轮性能优化的冻结起点，不是当前仓库 HEAD。P2 正式实现收口于
`566b5bc`，P3 HaloTransport 收口于 `547a398`。后续 OpenSBLI 功能提交不得
被重新标注为上述历史性能数据的执行版本。

## 2. 当前性能基线

当前可靠的单 GPU 性能结果来自 RTX 4000 Ada 工作站上的三维周期 TGV：

| 项目 | 固定设置 |
|---|---|
| 网格 | `256x256x256` |
| MPI | NP=1，`1x1x1` |
| 精度 | FP64 |
| 对流和黏性导数 | 六阶显式中心差分 `643e/643e` |
| 滤波 | 十阶显式中心滤波，开启 |
| 时间推进 | RK3 |
| 同步 | 每个 kernel 后显式同步 |
| 输出 | 计时区间内关闭 checkpoint 和全场输出 |
| 重复方式 | 一次进程预热，五个独立计时进程 |

| 指标 | 原始 GPU 基线 | 当前 GPU 实现 | 变化 |
|---|---:|---:|---:|
| 完整 RK 中位时间 | `0.762855769 s` | `0.580044424 s` | 降低 `23.964%` |
| GPU 版本内部加速比 | `1.0x` | `1.31517x` | 吞吐提高 `31.517%` |
| 重复测试离散度 | `2.105%` | `2.027%` | 通过 5% 门槛 |
| 峰值显存 | `9766 MiB` | `9738 MiB` | 降低 `0.287%` |

这里的 `1.31517x` 只表示优化前后两个 GPU 版本的差异，不表示相对 CPU NP=1 的整体加速比。后续论文和汇报必须分别报告：

1. GPU 优化增益：优化后 GPU 与冻结 GPU 基线比较；
2. 应用整体加速比：优化后 GPU 与同口径 CPU NP=1 比较；
3. 多 GPU 扩展效率：NP=N 与 NP=1 GPU 比较，并要求一个 MPI rank 对应一张物理 GPU。

## 3. 已保留的优化

当前正式路径已经保留以下优化：

1. 周期 x/y/z 对流、diffusion-RHS 和滤波 kernel 使用线性线程映射，减少无效线程，并改善 Fortran 存储顺序下的连续访问。
2. 守恒量转原始量时复用局部速度和压力，避免写入全局数组后立即重新读取。
3. 完全周期且开启滤波的 TGV 跳过冗余的第一阶段原始量刷新；物理边界算例继续执行原刷新路径。
4. x 周期 primitive halo kernel 让 halo 位置成为最快变化的线程索引，并复用局部原始量。
5. 六阶显式导数统一使用 `constdef::num1d60`，避免 `/60.d0` 在多个设备调用点生成重复 FP64 倒数指令。
6. 建立完整 RK 计时、12-kernel NCU 矩阵、Nsight Systems 常驻分析、Compute Sanitizer 和候选验收 runner。

`num1d60` 优化已经获得正常执行和 NCU 两类证据：

- diffusion-flux 正常执行时间降低约 `31.10%`；
- convection x/y/z 总时间降低约 `28.77%`；
- `gradcal` 时间降低约 `29.73%`；
- 对应六阶导数热点中的 `MUFU.RCP64H` 被消除；
- 该轮总 GPU kernel 时间降低约 `10.05%`。

## 4. 已拒绝或暂不采用的优化

### 4.1 diffusion-flux 线程块变更

将 diffusion-flux 从 `(8,8,8)` 改为 `(32,4,4)` 后，完整 RK 时间慢约 `0.093%`。该变化没有达到性能门槛，已经撤销。

### 4.2 大量取消显式同步

受控 selective-sync 实验将性能区间内的 `cudaDeviceSynchronize` 从 275 次降低到 15 次，但配对测试中位时间从 `0.682327627 s/RK` 变为 `0.683028141 s/RK`，慢约 `0.103%`。另一个 compute-only 候选慢约 `3.426%`，离散度达到 `6.677%`。

因此：

- `explicit` 继续作为正式默认模式；
- `selective` 只保留为 TGV 受限实验入口；
- 当前硬件上的主要矛盾是 kernel 执行，而不是主机同步调用开销；
- 后续候选不得把取消同步和计算 kernel 优化混在同一次测试中。

### 4.3 当前不进入的方向

以下方向不属于当前性能优化范围：

- `--use_fast_math`、`-Mfprelaxed`、`-fast`；
- FP32、TF32、混合精度和 Tensor Core 替换；
- 修改六阶显式差分或十阶显式滤波系数；
- compact 差分和 compact 滤波；
- 跨 RK、边界、滤波或 MPI 阶段的 kernel fusion；
- CUDA Graphs；
- GPU HDF5/checkpoint 输出；
- 仅凭 GPU 利用率或单个 NCU replay 时间宣布整体加速。

## 5. Phase P0：A800 基线冻结

状态：暂缓。A800 排队任务不作为本机 P1 探索的前置条件；本机候选仍以
`0d684a8`、RTX 4000 Ada 和 `0.580044424 s/RK` 冻结基线判定。任何本机
结论都不得外推为 A800 结论。

### 5.1 目的

RTX 4000 Ada 的测试结果不能直接外推到 A800。A800 的 FP64 吞吐、显存带宽、寄存器和 occupancy 平衡不同，热点排序可能发生变化。所有后续正式候选应优先以 A800 同机数据作为基线。

### 5.2 操作

1. 从提交 `0d684a8` 建立新的 A800 构建目录，不复用工作站 CMake cache。
2. 使用项目根目录 `CMakeLists.txt`、NVHPC RELEASE 和 `ASTR_WITH_CUDA=ON` 构建。
3. 记录 GPU 型号、UUID、驱动、NVHPC、MPI、Nsight、功率状态和编译命令。
4. 运行 `256^3` NP=1 五次完整 RK 基准。
5. 运行相同算例的 Nsight Systems 常驻分析。
6. 运行 12-kernel Nsight Compute 矩阵。
7. 运行 NP=1/NP=2 十步场和统计量比较以及 Compute Sanitizer。
8. 在显存允许时增加 `512^3` NP=1 测试，用于判断更大问题规模下的带宽和计算饱和状态。`512^3` 结果作为规模效应证据，不替代 `256^3` 固定验收算例。

### 5.3 CPU 基线补充

建立同口径 CPU NP=1 完整 RK benchmark：

- 使用相同网格、步数、数值格式、滤波和扩散设置；
- 排除初始化、checkpoint、HDF5 和结束输出；
- 至少运行三个独立进程，条件允许时运行五次；
- 报告中位时间和离散度；
- 只用该结果计算 CPU NP=1 到 GPU NP=1 的整体加速比。

### 5.4 验收

- 五次 GPU 测试离散度不超过 `5%`；
- NP=1/NP=2 CPU/GPU 误差不超过 `1e-10`；
- 常驻区间没有禁止的全场 H2D/D2H；
- memcheck 为 0 errors；
- A800 baseline timing TSV、NSYS、NCU 和环境记录完整保存。

## 6. Phase P1-C1：convection 和 diffusion-flux 读取复用

状态：已于 2026-09-06 按替代终止条件关闭，没有候选进入正式源码。

### 6.1 当前证据

NCU 显示 convection 和 diffusion-flux 同时具有较高计算吞吐、TEX throttle 和 short-scoreboard stall。它们重复读取速度、压力、Jacobian 和网格度量，是下一项优先候选。

### 6.2 候选原则

- 在单个线程内复用已经读取的 primitive 和 metric 值；
- 合并数学上完全相同的局部计算；
- 不新增全场临时数组；
- 不改变浮点运算顺序时优先；确需改变局部求值顺序时必须先说明误差预期；
- 不改变既定 x/y/z block shape；
- shared memory 只能在 NCU 证明跨线程重复读取足够高时作为第二候选。

### 6.3 重点指标

- executed instructions；
- global load sectors 和 bytes；
- L1/L2 命中率；
- TEX throttle；
- short/long scoreboard stall；
- registers/thread 和 local-memory spill；
- 完整 RK 中位时间。

### 6.4 验收

- 完整 RK 中位时间相对 A800 冻结基线至少降低 `3%`；
- 候选离散度不超过 `5%`；
- 峰值显存增长不超过 `5%`；
- NP=1/NP=2 十步 TGV 场和统计量通过 `1e-10`；
- 12-kernel 矩阵没有隐藏其他常用 kernel 的显著回退。

### 6.5 候选结果

固定基线为 `0.580044424 s/RK`。三个候选均使用 `256^3`、NP=1、FP64、
`643e/643e`、十阶显式滤波、扩散开启和逐 kernel 显式同步。每项候选在
测试后均已从计算路径撤销。

| 候选 | NCU 机制证据 | 五轮完整 RK 中位时间 | 相对基线 | 结论 |
|---|---|---:|---:|---|
| C1-1 `flux_at_global` 局部标量复用 | `conv_x` 的 registers、instructions 和 excessive L2 均不变；编译器已完成等价复用 | `0.589632722 s` | 慢 `1.653%` | 拒绝 |
| C1-2 扩散常量倒数/组合系数前移 | `diffusion_flux` 从 `27.430 ms` 降至 `25.268 ms`，instructions 从 `430.70 M` 降至 `411.05 M` | `0.581123491 s` | 慢 `0.186%` | 单核有效但整步无收益，拒绝 |
| C1-3 复用 `qwork_d` 存储方向通量 | 六个新核合计约 `53.02 ms/stage`，DRAM throughput 为 `87.27%` 至 `93.79%` | `0.638189768 s` | 慢 `10.024%` | 新增全场写读成为带宽瓶颈，拒绝 |

三项候选的 run-to-run spread 分别为 `1.832%`、`1.345%` 和 `1.653%`，
峰值显存均未增长。C1-1 证明手工局部变量不能优于 NVHPC 已有公共子表达式
消除。C1-2 证明单核优化不能替代完整 RK 验收。C1-3 证明当前显式同步约束下，
以全场工作数组换取通量复用不适合 RTX 4000 Ada。

证据目录：

- `tests/gpu_validation/out/p1_c1_baseline_conv_diff_0d684a8/`
- `tests/gpu_validation/out/p1_c1_candidate1_flux_local_reuse/`
- `tests/gpu_validation/out/p1_c1_candidate1_flux_local_reuse_ncu/`
- `tests/gpu_validation/out/p1_c1_candidate2_combined/`
- `tests/gpu_validation/out/p1_c1_candidate2_combined_ncu/`
- `tests/gpu_validation/out/p1_c1_candidate3_stored_conv/`
- `tests/gpu_validation/out/p1_c1_candidate3_stored_conv_ncu/`

下一阶段进入 P1-C2。C1-2 的逐点常量系数前移可在将来与更大范围的
diffusion 重构组合测试，但在完整 RK 未达到 `3%` 前不得单独保留。

## 7. Phase P1-C2：diffusion-RHS 寄存器生命周期

### 7.1 当前证据

三个 diffusion-RHS kernel 当前约为 128 registers/thread，achieved occupancy 约为 32%。优化目标不是单纯追求更高 occupancy，而是在不增加全局流量的条件下减少长生命周期临时量和 spill。

### 7.2 候选顺序

1. 缩小局部变量作用域，减少不同时使用的临时量同时存活。
2. 复用只读系数和已经加载的数据。
3. 检查内联设备函数是否导致过多临时量。
4. 只有前述方法无效时，测试受控的局部计算拆分。

不得为了降低寄存器数量而增加全场中间数组或额外 H2D/D2H。

### 7.3 验收

采用与 C1 相同的完整 gate。寄存器下降或 occupancy 上升只是机制证据，若完整 RK 改善不足 `3%`，该候选不进入正式路径。

### 7.4 筛选结果

P1-C2 已关闭，没有保留计算候选。所有测试均使用固定的 512 线程方向块、
FP64、`256^3` TGV 和逐 kernel 显式同步。

| 候选 | NCU 机制证据 | 完整 RK | 结论 |
|---|---|---:|---|
| 六组通量数组复用为一组正负缓冲 | 三方向仍为 128 registers/thread，三核合计仅降低约 `0.974%` | 未进入整步门 | 寄存器生命周期未改变，拒绝 |
| 全文件 `maxregcount=96` | occupancy 仍约 `32%`，出现约 `9.55 M` spill 指令 | 未进入整步门 | 未跨过双块驻留阈值，拒绝 |
| 全文件 `maxregcount=64` | occupancy 提升到 `61-63%`，但出现约 `146.46 M` spill 指令；三核合计慢 `35.45%` | 未进入整步门 | spill 抵消 occupancy，拒绝 |
| 周期 diffusion-RHS 拆为动量核和能量核 | 动量核仍为 128 registers/thread，能量核为 96-98 | `0.601406599 s/RK`，慢 `3.68%` | 增加 launch、同步和度量读取，拒绝 |

结论是当前固定 512 线程块下，寄存器数只有降至不高于 64 才能增加 block
residency，但编译器限寄存器会造成不可接受的 local-memory spill。后续若重新
打开该方向，必须通过更大范围的算法重组降低活跃状态，而不能继续调整
`maxregcount`。

## 8. Phase P1-C3：y/z 十阶滤波访存

### 8.1 当前证据

x 滤波 occupancy 约为 60%，y/z 滤波约为 32%，并具有明显 TEX 和 scoreboard stall。Fortran 第一维连续存储使 y/z stencil 天然存在较大访问跨度。

### 8.2 候选顺序

1. 调整线程到数据的映射，但保持既定 block shape。
2. 减少同一线程对 stencil 数据的重复读取。
3. 检查只读缓存和 L2 reuse。
4. 只有 NCU 证明复用收益足够时，测试共享内存 tile。
5. 保留当前全 `q` ping-pong 实现为默认正确性路径。

此前规划的“单个 3D work array”继续作为额外低显存方案，而不是替代当前 ping-pong。该方案主要解决可计算网格规模，不预设其速度更快。它必须单独测量额外 pass、复制和同步成本。

### 8.3 验收

- 首先通过滤波模块级 CPU/GPU 差值；
- 再通过 TGV、物理壁面和曲线网格滤波回归；
- 最后应用完整性能 gate；
- 只有完整 RK 改善达到门槛才作为性能默认实现。

### 8.4 筛选结果

P1-C3 已关闭，没有保留性能核。全周期专用 halo 核证明去除物理边界分支可将
y/z 寄存器从 80 降到 58/60，并将单核时间分别降低约 `11.45%` 和 `12.39%`，
但五轮完整 RK 中位时间为 `0.582806489 s`，相对冻结基线慢 `0.48%`。

逐分量 shared tile 将 excessive L2 从约 `25.5 MiB` 降到 `5-6 MiB`，但十次
block barrier 和附加寻址令 y/z 核变慢到 `10.270/11.379 ms`。一次装载五个
分量的 tile 将 barrier 降为一次，y/z 为 `8.931/9.433 ms`，两核合计只改善
约 `4.94%`，折算完整 RK 的理论收益约 `0.5%`，未进入五轮验收。共享内存
版本还会使 z 核 DRAM throughput 升至约 `83.4%`。

上述“拒绝”结论限定于当前 RTX 4000 Ada、当前 NVHPC/CUDA 工具链和冻结的
`0.580044424 s/RK` 基线。它表示候选不应成为当前工作站默认实现，不表示这些
机制在 A800 上必然无效。A800 的 FP64 吞吐、显存带宽、寄存器驻留阈值和缓存
行为不同，因此保留各候选的补丁机制、NCU 报告和输出目录，待 A800 环境稳定后
逐项重建。复测必须先建立同机五轮完整 RK 基线，再检查相同 kernel 的寄存器、
spill、吞吐和缓存指标；只有完整 RK 中位时间改善至少 `3%` 且全部正确性门通过，
候选才可恢复。不得直接使用工作站上的单 kernel 百分比推断 A800 收益。

十阶滤波继续使用原全 `q` ping-pong 默认实现。六个有效滤波系数集中到
`constdef.F90` 的编译期 `parameter`，GPU 通过常数乘法使用；NCU SASS 已确认
热点中为立即数 `DMUL/DFMA`，不存在由这些固定分数生成的运行时 FP64 除法。

## 9. Phase P2：激波和 SBLI 专项优化

TGV 不能代表 selective-Roe 和特征重构的成本。激波路径需要独立性能基线。

### 9.1 基准算例

- 保留 TGV 作为规则 stencil 性能基准；
- 使用三维周期挤出 Shu-Osher 作为可控激波占比和 kernel 定位基准；
- 使用已验证的三维高超声速边界层/SBLI 路径作为完整激波性能基准；
- 分离物理正确性、CPU/GPU 数值一致性和性能三类结论。

P2-0 不修改数值 kernel。Shu-Osher 默认网格为 `256x64x32`，SBLI 默认网格为
`256x192x32`。两者采用 NP=1、逐 kernel 显式同步、20 steps、一次进程预热和
五轮独立计时。mask 激活比例通过独立的一步 dump 离线计算，不在计时循环中加入
reduction、D2H 或文本输出。checkpoint 和场输出频率必须大于 `maxstep`。
SBLI 性能初场使用已验证的解析斜激波叠加场，使首个 RK 即覆盖 selective-Roe
路径；上边界仍使用相同斜激波目标的 `bctype=52` NSCBC。无激波初场继续保留为
边界注入正确性回归，不作为激波 kernel 性能基准。

P2-0 同时解除一个非数值 capability 耦合：已验证的 characteristic Shu-Osher
路径不再要求设置 `ASTR_SHOCK_SENSOR_DUMP` 才能运行。`lchardecomp=t` 已经要求
每个 RK 计算 sensor 和 mask，dump 仅是可选诊断输出，不能作为数值能力开关。
仅传感器验证使用的 S0-A4/A5 仍保持 dump 限制。

### 9.2 优化对象

- characteristic interface-flux kernel 的寄存器和 spill；
- 激波传感器、扩展 mask 和特征通量之间的重复读取；
- 激波区域比例变化引起的线程分支成本；
- adaptive 单 kernel 与 physical/shock 分离 kernel 的实际成本；
- 传感器 halo 交换和通量计算之间的等待时间。

不得在没有测量激波单元占比和 branch 指标前引入压缩界面列表。前缀扫描和不规则索引可能比线程发散更昂贵。

### 9.3 验收

- Sod/Shu-Osher 精度与超调门槛不退化；
- SBLI CPU/GPU 场、壁面压力、摩擦和热流诊断不退化；
- 单 kernel 改善必须转化为完整 RK 改善；
- 多 rank 下 shock-sensor halo 和 characteristic flux 保持一致。

### 9.4 P2-0 基线结论

P2-0 已在 RTX 4000 Ada 上完成，完整证据见
`documents/ASTR_PHASE_P2_BASELINE_REPORT.md`。Shu-Osher `256x64x32` 和 SBLI
`256x192x32` 的五轮完整 RK 中位时间分别为 `0.151583855 s` 和
`0.523264916 s`，离散度分别为 `1.933%` 和 `0.363%`。两者 shock-active
节点比例分别为 `9.3385%` 和 `3.1713%`。

Nsight Systems 显示三方向 characteristic interface-flux 占两算例总 kernel
时间约 `94.1%` 和 `87.5%`，sensor 与 mask expansion 合计不足 `1%`。y 向
特征通量使用 128 registers/thread，occupancy 约 `32%`，并产生大量 local
spill；branch efficiency 约 `99.94-99.95%`。因此 P2-1 首选 y 向“全域物理
通量 + active 界面特征覆盖”双 kernel 筛选，优化假设是隔离高资源特征路径，
而不是消除严重线程发散。该候选修改数值 kernel，实施前需单独审批。

### 9.5 P2-1A y 向双 kernel 筛选结果

P2-1A 已完成实现、正确性、memcheck、NSYS、NCU 和五轮完整 RK 验收，并按
失败规则撤销源码。Shu-Osher `256x64x32` 从 `0.151583855 s/RK` 增至
`0.184307279 s/RK`，慢 `21.588%`；SBLI `256x192x32` 从
`0.523264916 s/RK` 增至 `0.529303436 s/RK`，慢 `1.154%`。两项离散度分别为
`0.928%` 和 `0.176%`，负收益不是运行噪声。

NCU 显示物理基底 kernel 仍为 128 registers/thread、`32.54%` occupancy 和约
`63.96 M` local-memory spilling requests；特征覆盖 kernel 另有约 `3.90 M`
spilling requests。拆分未实现预期的低资源物理路径，反而增加一次完整界面网格
遍历和显式同步。因此不推进 x/z 同类拆分，生产路径继续使用原单 kernel 实现。
完整证据见 `documents/ASTR_PHASE_P2_BASELINE_REPORT.md`。

上述拒绝仅适用于当前 RTX 4000 Ada 与当前 NVHPC/CUDA 工具链。A800 的 FP64、
寄存器驻留和缓存行为不同，P2-1A 可在冻结 A800 本机基线后重建复测；但必须继续
满足完整 RK 改善至少 `3%`、正确性门全部通过的条件，不能外推工作站单 kernel
指标。

### 9.6 P2-1B 非激波物理通量五分量复用

P2-1B 保留单个 sensor-coupled interface kernel，不采用 P2-1A 的双 kernel
结构。非激波分支原先对五个守恒分量分别调用标量 Steger-Warming 分裂函数，
导致每个 stencil 点重复计算五次网格度量归一化、声速、特征值分裂和平方根。
新路径在每个 stencil 点一次生成五分量物理通量，并复用该线程原本已经存在的
`split_plus(5,7)` 与 `split_minus(5,7)`。未新增全场数组、kernel、同步或
H2D/D2H。

候选保留 `sqrt(tmp)/mach` 声速定义、FP64 运算、MP7/Roe 选择语义和固定线程块。
物理 x/y 方向的非激波和 Roe 分支均令 `npdc=3` 显式使用 `-hm:dim+hm`，与
原标量物理实现一致，避免未来出现内部 MPI rank 时错误钳制 halo stencil。

RTX 4000 Ada 五轮完整 RK 结果如下：

| 算例 | P2-0 基线 | P2-1B | 改善 | spread | 决策 |
|---|---:|---:|---:|---:|---|
| Shu-Osher `256x64x32` | `0.151583855 s` | `0.109288716 s` | `27.902%` | `0.244%` | 保留 |
| SBLI `256x192x32` | `0.523264916 s` | `0.268521579 s` | `48.684%` | `0.431%` | 保留 |

SBLI y 核 NCU replay 从 `103.30 ms` 降至 `33.53 ms`，local spill requests 从
`68,911,722` 降至 `15,059,748`。寄存器仍为 128/thread，说明收益来自消除
重复标量物理分裂工作并缩短局部临时量压力，而不是通过寄存器上限提高 occupancy。

十步 SBLI 完整场继续通过，primitive 与重构守恒量最大差分别约为 `1.02e-14`
和 `6.00e-15`。此前在线 `massflux` 的 `9.05e-9` 差异已定位为统计相位错误：CPU
在 `rkfirst` 前完成 `bctype=52` 上边界 x/z 横向滤波，GPU 统计曾在滤波前取样。
GPU 现复用 `qsave_d` 保存完整 RK 状态，构造同相位滤波状态完成统计，再恢复状态
执行正式 RK。十步 `massflux` 最大差降至 `5.03e-13`，且不改变场演化或增加全场
device 数组。完整证据与复现路径见
`documents/ASTR_PHASE_P2_BASELINE_REPORT.md`。

### 9.7 P2-1B 修正后热点

修正后 Nsight Systems 中，三方向 characteristic interface flux 在 Shu-Osher
和 SBLI 分别占 kernel 时间 `91.7%` 和 `73.1%`。raw sensor 与 expanded mask
合计仅 `1.2%` 和 `1.4%`，其完整 RK 理论上限不足 `3%`，不进入候选。

SBLI x/y/z 特征通量 NCU 均为 128 registers/thread，local spilling requests
分别为 `19,472,739`、`15,059,748` 和 `6,762,124`；Shu-Osher x 为
`8,485,620`。该证据支持再筛选一个减少 thread-local split storage 的候选，
但不支持 mask fusion 或压缩 active-interface list。

### 9.8 P2-1C 单 split 数组筛选结果

P2-1C 让正、负分裂通量顺序复用一个 `split_flux(5,7)`，并用 5 个标量保存正向
特征重构值。一步 Shu-Osher 和 SBLI 正确性通过。同为每轮 10 个 RK 样本的五轮
正式比较仅改善 `1.975%` 和 `1.208%`，低于 `3%`。SBLI y 核寄存器仍为
128/thread，spill requests 反而从
`15,059,748` 增至 `18,007,902`。候选源码已撤销，原始 timing、NCU 和源码副本
保留在 `tests/gpu_validation/out/p2_full_p21c_split_reuse_*`。

### 9.9 Shock-Sensor Halo 等待结论

NP=2 y-slab 的 `cuda,mpi` 时间线按每张 GPU 配对 raw sensor 与 expanded mask。
Shu-Osher 的 halo 关键路径为 `7.706589 ms`，占 profile 内完整 RK 的 `3.488%`；
SBLI 为 `8.765948 ms`，占 `1.421%`。Shu-Osher 中 P2 可改的 sensor pack/unpack
kernel 只占完整 RK 约 `0.074%`。其余是 pageable host staging 与 blocking MPI，
属于 P3 通信后端，不在 P2 内用 kernel 改动处理。

### 9.10 P2 关闭状态

最终生产路径通过顶层 CPU/GPU 构建、54 个 Python 测试、全部 shell 语法检查、
Sod/Shu-Osher/SBLI 十步对照、Shu-Osher NP=2 x/y/z 与 NP=8 `2x2x2`、SBLI
壁面统计及 Compute Sanitizer 0 errors。恢复后五轮复测为：

| 算例 | 最终中位时间 | spread | 峰值显存 | 相对 P2-0 |
|---|---:|---:|---:|---:|
| Shu-Osher `256x64x32` | `0.110301406 s/RK` | `0.958%` | 810 MiB | 改善 `27.234%` |
| SBLI `256x192x32` | `0.267333482 s/RK` | `0.770%` | 1,500 MiB | 改善 `48.910%` |

P2 已关闭。正式源码只保留 P2-1B。P2-1A/P2-1C 均拒绝，sensor/mask 无 3%
收益上限，halo 剩余机会转交 P3。

## 10. Phase P3：多 GPU HaloTransport 优化

2026-09-07 本机验收完成。默认仍为 pageable blocking；保留可选 `pinned`
及仅适用于全周期 stored diffusion 内部区域的 `pinned-overlap`。
独立 nonblocking 因收益不足拒绝，CUDA-aware 因当前软件栈准入失败暂缓。
以下早期筛选记录保留，最终验收结论见本文件末尾和 P3 基线报告。

当前 host-staged blocking 交换继续作为可移植正确性基线。solution/sponge 的
qswap-compatible 路径保留 `hm+1` 和接口面平均，filter/qwork、diffusion 和
shock-sensor 路径保留固定 `hm`。优化按以下顺序进行：

1. pageable host buffer 改为 pinned host buffer；
2. blocking host-staged MPI 改为 nonblocking host-staged MPI；
3. 分离内部区域和 halo 邻近区域，评估计算通信重叠；
4. 在 NVIDIA 平台评估 CUDA-aware MPI；
5. 为未来 AMD/HIP/DCU 保留 host-staged 后端和设备感知接口边界。

通信重叠会改变同步和阶段组织，不得与 P1 kernel 优化同时实施。它需要单独设计、审批和性能基线。

### 10.1 硬件和完成状态

P3 已完成本机最终验收，详细证据见 `ASTR_PHASE_P3_IMPLEMENTATION_PLAN.md`
和 `ASTR_PHASE_P3_BASELINE_REPORT.md`。最终结论为：

- pageable blocking 保留为默认、可移植的正确性基线；
- pinned blocking 作为可选后端保留，九个正式组合的完整 RK 时间减少
  `2.829%--23.728%`；
- pinned-overlap 仅对完全周期 stored-diffusion 内部区域启用，TGV x/y/z
  相对配对 pinned 的增量为 `2.616%/3.085%/2.944%`；
- 独立 paired nonblocking 最大可复现增量为 `2.425%`，未达到 `3%` 门槛，
  已从正式源码撤下；
- CUDA-aware MPI 在当前 HPC-X 软件栈的大消息或 sanitizer 门槛失败，暂不接入；
- 物理边界闭合、shock sensor 和 SBLI 不直接复用周期扩散的内部区 overlap。

- 性能测试必须一个 MPI rank 对应一张物理 GPU；
- 两张 GPU 上的 NP=4/8 oversubscription 只能作为正确性测试；
- NP=4/8 扩展效率必须在具有相应物理 GPU 数量的平台上测试。

### 10.2 验收

- 所有 x/y/z 和组合拓扑继续通过 halo 正确性门槛；
- 通信时间或完整 RK 时间有可重复下降；
- overlap 必须由时间线证明，而不是根据非阻塞 API 名称推断；
- CUDA-aware 路径失败时可以回退到 host-staged 正确性后端。

## 11. 候选执行流程

每个候选使用：

```bash
CANDIDATE_ID=<candidate-id> \
BASELINE_REF=<frozen-git-ref> \
BASELINE_TIMINGS=/absolute/path/to/baseline_timings.tsv \
TARGET_KERNELS='<kernel names>' \
ALLOWED_PATHS='<comma-separated source paths>' \
HYPOTHESIS='<measured bottleneck and expected mechanism>' \
GATE_SET=full \
OUT_DIR=/tmp/astr_candidate_<candidate-id> \
  tests/gpu_validation/run_gpu_optimization_candidate_gate.sh
```

执行顺序：

1. 检查候选元数据和改动范围；
2. 从顶层 CMake 构建 CPU/GPU；
3. NP=1 和 NP=2 十步场/统计量比较；
4. 五次完整 RK benchmark；
5. 与冻结基线比较时间、离散度和显存；
6. Nsight Systems 常驻检查；
7. 12-kernel NCU 因果检查；
8. Compute Sanitizer；
9. 补跑候选影响模块对应的物理边界、曲线网格或激波回归矩阵。

## 12. 停止条件

出现以下任一情况时停止当前候选，不继续堆叠优化：

- CPU/GPU 误差超过对应门槛；
- 数值格式、边界或 MPI 语义发生未批准变化；
- 新增禁止的全场数据传输；
- Compute Sanitizer 报错；
- 五次测试离散度超过 `5%`；
- 完整 RK 改善不足 `3%`；
- NCU 指标改善但完整 RK 没有改善；
- 候选只在 oversubscription rank 配置中表现更快；
- 需要通过放宽误差、改变精度或启用 relaxed math 才能获得加速。

失败后保留报告和日志，源码候选单独回退。不得删除失败证据，也不得将多个失败候选混合后重新测试。

## 13. 推荐执行顺序

近期执行顺序固定为：

```text
P0  A800 基线与 CPU NP=1 同口径基线（暂缓，不阻塞本机探索）
 -> P1-C1 convection/diffusion-flux 读取复用（已关闭，无保留候选）
 -> P1-C2 diffusion-RHS 寄存器生命周期（已关闭，无保留性能候选）
 -> P1-C3 y/z 滤波访存（已关闭，仅保留 constdef 常量集中化）
 -> P2 激波和 SBLI 专项优化（已关闭，仅保留 P2-1B）
 -> P3 多 GPU HaloTransport 优化（已完成本机收口）
```

下一性能阶段为 P0 A800 NP=1/2/4 基线和强扩展测试。609x255x9 OpenSBLI
薄层用于物理验证，不单独承担四卡三维扩展结论。在 P0 完成前，不对 A800
做性能结论。取消显式同步、kernel fusion、CUDA Graphs 和混合精度均不属于
既有验收结论，需要单独立项审批。

## 14. 交付物

本节先保留 P3 执行期间的过程记录，最终状态以末尾“P3 最终本机验收”为准。

P3 CUDA-aware 准入更新：当前 HPC-X 2.25.1 软件栈的小消息设备缓冲区测试通过，
但默认 UCX 大消息出现接收数据未更新。关闭 IPC 后逐元素比较通过，
Compute Sanitizer 仍未达到零错误门槛，提前绑定设备也未解决。
本轮停止该候选接入，不给出其性能结论，也不据此断言硬件不支持。
独立探针、配置、失败日志和冻结哈希见 `ASTR_PHASE_P3_BASELINE_REPORT.md`。
默认 pageable 回退不变，pinned 与 pinned-overlap 的最终完整回归仍待收口。
当前收口检查点已通过三种后端、三个算例、五种拓扑的 45 组十步逐场及统计量比较，
以及九组小网格 memcheck，18 份 rank 日志均为零错误。
SBLI 使用完整 RK 同相位比较。该内存检查只限定于 host-only MPI 配置。
随后已补齐 18 处物理端点 `MPI_PROC_NULL` 接收上传保护，重新通过 45 组主回归、
九组开启滤波和扩散的物理壁面专项，以及 18 组 memcheck 的 36 份零错误 rank 日志。
生产 pack/unpack 冲突接口值专项随后也已通过：三种后端分别运行 NP=2/3，
直接检查生产模块的三轴平均、所有权和分量边界，另有九份 memcheck rank 日志零错误。
最终源码的 TGV 256^3 x-slab 新测量中，pinned 完整 RK 时间比 pageable 下降 13.839%。
overlap 组 spread 为 8.020%，整组不确定，不能剔除单次结果后声称通过。
仍需完整配对重测和其余方向、算例的最终性能复核，尚不宣布 P3 完成。
Shu-Osher/SBLI 三方向的最终源码性能组现已完成，Shu z 的高波动组也完成了
完整反向重测。有效组中 pinned 的 RK 时间降幅分别为 2.829%--7.611% 和
7.823%--23.728%，最大单卡显存增幅 3.436%。这两类算例的 overlap 分支未启用。
剩余 TGV y/z 和 TGV x 配对重测的性能收口。

请求完成状态证据已补齐：仅用于 profiling 的 PMPI/NVTX 包装库记录真实
MPI_Testall 返回值，不增加轮询，也不改变求解器。受控双 rank 探针确认
未完成/已完成标记有效，空请求不被计入。最终源码 TGV 256^3 y-slab
捕获到 4237 个未全部完成标记和 24 个全部完成标记位于同进程扩散 kernel
期间。这证明该次插桩执行存在请求与计算重叠，不代表网络带宽或正式整步收益。
所有正式性能测试禁用该包装库。证据、重现方法和分析器测试见 P3 报告及验证 README。

### P3 最终本机验收

- 九个正式组合采用最终冻结源码、NP=2 双物理 GPU、五次完整进程重复。
  pinned 相对 pageable 的完整 RK 时间减少 2.829%--23.728%。
- overlap 相对配对 pinned 的 TGV x/y/z 增量为 2.616%/3.085%/2.944%。
  y 达到 3% 单点门槛，其他有效组没有超过 1% 的稳定退化。
- TGV x 两组噪声结果全部保留，最后一次完整配对重测通过稳定性门槛。
  共审计 34 组、204 个含预热的进程日志，未剔除单次结果。
- 最大单卡显存增长 3.436%。主机锁页内存另记，不与显存混淆。
- 最终源码 45 组 CPU/GPU 对比、九组物理壁面检查、生产 halo 精确合同及
  host-only MPI 的零错误 sanitizer 矩阵通过。68 个 Python 测试、96 个
  shell 语法检查和顶层 CPU/GPU 构建通过。
- 新增冻结 L0、小网格 overlap、NSYS 插桩及重链接版本的直接逐场对照，
  六个原始场和五个重构守恒场最大差值均为零。

逐 kernel 显式同步、FP64、RK3、数值格式及边界/halo 语义保持不变。
此结论仅覆盖本机 host-staged 路径，不代表 A800、多节点或 CUDA-aware
准入，不将短步 SBLI 回归作为真实物理验证。正式性能始终归属于冻结二进制
`aee22855...`；重新链接后的文件哈希及对照结果单独记录。详细表格、失败组、
输入哈希和证据路径见 `ASTR_PHASE_P3_BASELINE_REPORT.md` 的最终验收章节。
现有 SQLite 的请求句柄不能单独证明通信在核执行期间尚未完成。

每个通过的阶段必须提交：

- 冻结基线 Git reference；
- 环境与编译选项记录；
- 原始 timing TSV 和汇总报告；
- NP=1/NP=2 场与统计量误差报告；
- Nsight Systems 报告和常驻审计；
- Nsight Compute 原始报告、source CSV 和热点矩阵；
- Compute Sanitizer 日志；
- 候选保留或拒绝的明确结论；
- 更新后的性能报告和可复现命令。

性能报告必须明确区分 GPU 内部优化增益、CPU 到 GPU 的整体加速比和多 GPU 扩展效率。
