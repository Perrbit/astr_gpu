# ASTR GPU Current Status and Next Targets

更新日期：2026-09-07

## 1. 总体结论

ASTR 已从单卡 TGV CUDA 验证程序推进为支持规则网格、静态曲线网格、
多 MPI rank、显式激波格式和多类物理边界的非反应流 GPU 求解框架。
当前最成熟的是 CPU/GPU 数值等价性、固定 halo 的多 rank 正确性和主计算循环
变量常驻。P2 激波路径优化和 P3 HaloTransport 本机优化已经收口，OpenSBLI
参考边界、生产接线和 restart 路径已经运行。生产级 SBLI 物理验证、A800
真实四卡扩展效率和通用曲线特征边界尚未闭合。

本文档同步至提交 `c31840e`。性能测量仍归属于各专项报告中记录的冻结二进制，
不得把文档更新后的提交哈希重新标注为原性能结果的执行版本。

## 2. 已完成的 GPU 能力

### 2.1 数值方法

| 类别 | 已验证 GPU 能力 |
|---|---|
| 时间推进 | 三阶段三阶 Runge-Kutta |
| 中心格式 | 六阶显式中心差分及黏性扩散 |
| 滤波 | 十阶显式中心滤波及物理边界闭合 |
| 迎风格式 | 一阶 Steger-Warming、WENO7、MP7 |
| 激波处理 | Ducros 传感器、激波区域选择性 Roe 特征空间 MP7 |
| 物性 | 单组分理想气体、Sutherland 变黏性 |
| 网格 | Cartesian 网格和静态单块三维结构化曲线网格 |

不支持紧致差分、紧致滤波、三对角或五对角矩阵求解。

### 2.2 已验证算例

| 算例类别 | 当前状态 |
|---|---|
| Taylor-Green Vortex | 周期、扩散、滤波、统计量和多 rank 已验证 |
| 三维挤出 Vortex Transport | 显式周期路径已验证 |
| HIT | 确定性随机初场、周期路径和多 rank 已验证 |
| Channel flow | 固定驱动力、壁面、扩散和 100-step 多 rank 统计量已验证 |
| Lid-Driven Cavity | 强制三维显式版本及多轴物理边界滤波已验证 |
| Rayleigh-Taylor Instability | 显式三维版本、重力源项和多 rank 已验证 |
| Sod、Shu-Osher | WENO7、MP7、Ducros 和选择性 Roe 路径已验证 |
| OpenShock | 受限开放边界、NSCBC 和 sponge 路径已验证 |
| 平板边界层和高速边界层 | Mach 3/5、Sutherland、相似解入口和曲线网格已验证 |
| 受控 SBLI | 人工压缩入口、激波传感器和选择性 Roe 耦合已验证 |
| OpenSBLI Katzer 层流 SBLI | 609x255x9 参考输入、专用边界生产接线、NP=2 长跑和 step 60000 restart 已接通；外部物理验证未完成 |

这里的“已验证”主要指 CPU/GPU 数值等价、边界不变量和内存安全通过，
不等同于每个算例都已完成实验或 DNS 数据库级物理验证。

### 2.3 MPI 和变量常驻

- 已覆盖 NP=1、NP=2 三种 slab、NP=4 三种 plane 和 NP=8 `2x2x2`。
- TGV 已进行 NP=16 和 NP=32 的双卡共享正确性 smoke test。
- 生产目标仍是一 MPI rank 对应一 GPU。双卡上的 rank oversubscription 不能作为
  四卡或八卡扩展性能证据。
- 无输出 RK 区间内，主流场、守恒变量、RHS、通量、滤波工作区、曲线度量和
  激波传感器常驻 GPU。
- HaloTransport 默认采用 pageable host-staged blocking；保留可选 pinned blocking，
  以及仅适用于完全周期 stored-diffusion 内部区域的 pinned-overlap。
- 独立 nonblocking 候选因端到端收益不足被拒绝；CUDA-aware MPI 因当前软件栈
  的大消息或 sanitizer 准入失败而暂缓。
- 统计量在 GPU 完成归约，只向主机返回少量标量。
- HDF5、checkpoint 和完整场输出仍由 CPU 负责，不属于当前 GPU 化范围。
- 正确性阶段保持每个 CUDA kernel 后显式同步。

## 3. 边界条件支持范围

| `bctype` | 当前 GPU 支持范围 |
|---:|---|
| `1` | x/y/z 周期边界 |
| `11` | x-min 给定入口，完整更新 `rho/u/v/w/p/T/q` |
| `12` | 受限 x-min 特征入口，仅用于已验证组合 |
| `21` | 当前主要支持轴对齐 x-max 出口；非轴对齐曲面主动拒绝 |
| `22` | 受限 Cartesian NSCBC 出口；非正交曲线网格主动拒绝 |
| `31` | RTI 使用的固定边界 |
| `41` | 曲线六面零吹吸静温无滑移壁面；另支持 y-min 物理法向吹吸 |
| `42` | 曲线 x/y 绝热无滑移壁面；z 向因 CPU 无分支而拒绝 |
| `411/421` | y 向滑移/无滑移组合，速度按局部几何法向投影 |
| `50` | 曲线六面零外推 |
| `51` | 受限轴对齐远场组合 |
| `52` | 受限 upper-y NSCBC 远场组合 |
| `60` | 曲线六面对称，速度投影到局部切平面 |

当前不能宣称任意曲线 NSCBC、任意方向开放边界或缺失 CPU 分支的 GPU 支持。
受支持组合和拒绝条件记录在
`documents/ASTR_CURVILINEAR_GPU_COMPLETENESS_PLAN.md`。

## 4. 静态曲线网格状态

CURVE-C0 至 C21 已完成以下能力：

1. 周期曲线度量、自由流保持和解析度量收敛。
2. 曲线网格显式对流、扩散、滤波及统计量。
3. 曲线物理壁面的几何投影和 y-min 法向吹吸。
4. Ducros、选择性 Roe、Sutherland、NSCBC 和 x-max sponge 的受控组合。
5. 三网格、两时间步 Mach 5 层流边界层物理一致性验证。
6. 开放边界能力盘点以及未支持组合的 CUDA 启动前拒绝。
7. 使用同一组 CPU/GPU 二进制完成全部支持矩阵、拒绝项、三方向内存检查和
   大网格常驻审计。

CURVE-C20 的入口状态最大误差为 `4.4408920985006262e-15`，EOS 残差为
`2.2204460492503131e-16`，CPU/GPU `q5` 最大误差为
`7.1054273576010019e-15`。目标 Compute Sanitizer 检查为
`ERROR SUMMARY: 0 errors`。

CURVE-C21 共执行 23 个聚合阶段，全部通过。136 份逐场报告中的最大误差为
`q5=6.2527760746888816e-13`，128 份统计量报告中的最大误差为
`massflux=2.5093260802577788e-11`，必需边界不变量的最大残差为
`7.6170181273482740e-12`。70 项有限场检查全部通过，最小数值 Jacobian 为
`2.5296638468231652e-7`。

该结论限定为静态、单块、三维结构化曲线网格。多块网格、非共形接口、
动网格、ALE/GCL 和浸入边界不在当前完成范围。

## 5. 性能证据

### 5.1 历史 Channel CPU/GPU 对比

`128^3` channel、`maxstep=100`、无场输出测试，以 NP=1 CPU 为统一基线，
历史 GPU 整体加速比约为 `28.7x` 至 `43.1x`。该结果来自早期工作站测试，
应作为历史性能证据，而不是当前最终基准。

### 5.2 CURVE-C15/C21 双 GPU 扩展复测

| 配置 | 中位 wall time |
|---|---:|
| NP=1，单 GPU | `58.490 s` |
| NP=2，双 GPU x-slab | `38.989 s` |

- 双 GPU 相对单 GPU 加速比：`1.5002x`。
- 双 GPU 并行效率：`75.01%`。
- GPU 利用率达到 100%。
- 无输出 RK 区间没有大于或等于 64 KiB 的完整场传输。
- NP=1/NP=2 峰值设备内存约为 `9876/5573 MiB`。

该测试证明当前曲线高速边界层路径实现了变量常驻和有效双卡扩展，
但不是相对 CPU 的统一性能基准，也不能外推到四卡或八卡。

详细证据见 `documents/GPU_VALIDATION_MATRIX.md`。

### 5.3 P2 激波和 SBLI 专项优化

在 RTX 4000 Ada 本机冻结基线上，最终保留的 P2-1B 物理通量复用使
Shu-Osher `256x64x32` 和受控 SBLI `256x192x32` 的完整 RK 中位时间相对
P2-0 分别减少 `27.234%` 和 `48.910%`。这些结果是 GPU 内部实现优化，
不是相对 CPU 的整体加速比，也不代表 OpenSBLI 参考算例的物理验证。

### 5.4 P3 HaloTransport 本机收口

- NP=2 双物理 GPU 的九个正式组合中，pinned 相对 pageable 的完整 RK 时间
  减少 `2.829%` 至 `23.728%`。
- TGV x/y/z slab 的 pinned-overlap 相对配对 pinned 增量分别为
  `2.616%/3.085%/2.944%`。
- pageable blocking 保持默认可移植基线；pinned 和周期扩散 overlap 为可选后端。
- 该结论仅覆盖本机双 GPU host-staged 路径，不外推到 A800、多节点或
  CUDA-aware MPI。

## 6. 后续目标

### 已完成里程碑：CURVE-C21 总回归

以上终止条件均已满足。完整报告见
`documents/ASTR_CURVE_C21_REGRESSION_REPORT.md`。

### 当前目标：完成 OpenSBLI 层流 SBLI 物理验证

当前采用 OpenSBLI Katzer Mach 2 层流 SBLI 作为外部参考。参考输入、保守边界、
度量一致 eps、GPU 主循环和 checkpoint restart 已接通。step 60000 restart
短程等价达到舍入误差量级，长时间续算正在进行。

终止条件：

- 完成至少三套网格和两种时间步验证。
- CPU/GPU 同相位场和统计量满足既定误差阈值。
- 比较激波位置、壁面压力、摩阻、热流和分离长度。
- 激波传感器和选择性 Roe 区域具有可追溯诊断输出。
- restart 接缝不存在统计量跳变；若长期轨迹分离，须比较同相位完整场、
  z 向均匀度和传感器区域，区分浮点扰动放大与未保存状态。
- 结论能够区分数值等价性与物理收敛性。

### 已完成里程碑：P2/P3 本机性能收口

P2 和 P3 已完成本机验收。保留 pageable、pinned 和受限 overlap 三层后端，
不继续堆叠未达到完整 RK 收益门槛的候选。物理边界、shock sensor 和 SBLI
路径的通信重叠作为后续独立阶段，不视为当前 P3 已覆盖能力。

### 下一目标：A800 四卡真实性能与扩展验证

在一 MPI rank 对应一 GPU 的环境中，以 TGV 和具有充分三维规模的曲线
HBL/SBLI 分别代表规则计算路径和复杂物理路径。609x255x9 OpenSBLI 薄层
用于物理验证，不单独承担四卡三维扩展结论。

终止条件：

- NP=1/2/4 均有一次预热和至少五次独立重复计时。
- 数值正确性、初始化/输出时间和纯 RK 吞吐分别报告。
- 给出强扩展效率及计算、通信、传输和同步占比。
- 同一冻结输入比较 pageable、pinned 和适用时的 overlap；选择性同步只能
  作为独立候选，不能替换显式同步正确性基线。

### 后续目标：按具体工况扩展通信和曲线开放边界

- 先把计算/通信重叠从周期 stored diffusion 扩展到明确选定的非周期物理路径。
- 只有新算例明确需要时，再实现基于局部物理法向的特征入口或远场。
- 不得直接把 Cartesian `12/22/51/52` 分支推广到任意曲面。
- 为 AMD/HIP/DCU 先验证 backend-neutral facade 加 `ISO_C_BINDING` 的小型
  HIP kernel 原型，不直接重写整个 CUDA Fortran 后端。

## 7. 暂缓范围

以下能力不进入近期目标：

- compact 差分和 compact 滤波；
- RANS/LES；
- 多组分、化学反应和燃烧；
- GPU HDF5 和 checkpoint 写场；
- 多块非共形网格；
- ALE/GCL 动网格；
- 浸入边界法。

## 8. 当前推荐顺序

1. 完成 OpenSBLI `t=13000/26000` 时间收敛、三网格两时间步和外部物理比较。
2. 在 A800 四卡平台建立 NP=1/2/4 同口径性能与强扩展基线。
3. 以真实 profile 结果决定是否扩展非周期/SBLI 通信重叠和选择性同步。
4. 建立一个 CUDA/HIP 后端边界原型，验证未来 DCU 路径的实际可行性。
5. 由具体算例需求决定是否扩展曲线特征边界。
6. 只有项目需求重新打开时，才进入组分、化学、湍流、IBM 或多块网格移植。

当前交付范围应表述为“ASTR 单组分非反应显式格式 GPU 求解器”。在暂缓能力
重新打开并通过各自物理验收前，不使用“整个 ASTR 已完成 GPU 移植”的结论。
