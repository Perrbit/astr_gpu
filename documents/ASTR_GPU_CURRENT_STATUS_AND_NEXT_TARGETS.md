# ASTR GPU Current Status and Next Targets

更新日期：2026-09-12。当前代码基线：`feature/gpu_dev`，提交
`e0de98ea1607d3649239754fdc7629534f093327` 加当前未提交 CURVE-C23 工作树。

## 1. 总体结论

ASTR 已从单卡 TGV CUDA 验证程序推进为支持规则网格、静态曲线网格、
多 MPI rank、显式激波格式和多类物理边界的非反应流 GPU 求解框架。
当前最成熟的是 CPU/GPU 数值等价性、固定 halo 的多 rank 正确性、静态
单块曲线网格以及主计算循环变量常驻。P1 TGV、P2 激波路径和 P3
HaloTransport 本机性能阶段已经收口，OpenSBLI 参考边界和 restart 已接通，
GPU 常驻动态时序入口已完成 D0 至 D2 数值接入和阶段一致性闭环。紧凑 GPU
在线生产统计 D4 已完成软件与数值验收。

当前交付范围应表述为“ASTR 单组分非反应显式格式 GPU 求解器”。生产级
SBLI 物理验证、A800 四卡真实性能、真实湍流入口和通用曲线特征边界尚未闭合。
在暂缓能力重新打开并通过独立验收前，不使用“整个 ASTR 已完成 GPU 移植”
的结论。

性能测量归属于各专项报告记录的冻结二进制。不得把本状态文档的提交哈希
重新标注为历史性能结果的执行版本。

## 2. Git 与架构基线

按项目此前约定，以 `df1961bc` 的父提交作为纯 CPU 基线。到当前提交共形成
35 个后续提交，tracked 变更覆盖 412 个文件。该文件数包含源码、验证工具、
参考数据和文档，不表示 412 个独立 GPU 功能。

主要提交阶段如下。

| 提交范围 | 主要进展 |
|---|---|
| `df1961b` 至 `451fe0e` | TGV 多 rank、2dvort、HIT、Channel 和墙面边界 |
| `a684edb` 至 `aecb18f` | LDC、RTI、显式激波格式、HBL、选择性 Roe 和 NSCBC |
| `63fe7a8` 至 `5a800c7` | 静态三维曲线网格 CURVE-C0 至 C21 收口 |
| `30a5f4d` 至 `547a398` | SBLI 路径和 P1/P2/P3 性能优化 |
| `c31840e` | OpenSBLI checkpoint restart 工作流 |
| `9fad55b` 至 `e73bcd9` | 混合马赫数入口、常驻动态入口和 A800 TGV campaign |

当前架构保持以下边界。

- 顶层 `CMakeLists.txt` 是统一 CPU/GPU 构建入口。
- `ASTR_WITH_CUDA` 决定 binary 是否包含 CUDA backend，`use_gpu` 仍由输入文件
  在运行时选择 CPU 或 GPU 路径。
- CPU 负责输入、分区、网格和初场、外层 step、controller 及文件生命周期。
- `src/` 通过窄 facade 调用 `src_gpu/`，CUDA kernel 和 device 数据所有权集中在
  GPU backend。
- HDF5、checkpoint 和完整场输出保持 CPU-owned；输出边界的完整 D2H 不属于
  计算循环常驻失败。
- 当前 backend 是 CUDA Fortran。HIP/DCU 仍是目标架构，不是已实现能力。

## 3. 已完成的 GPU 能力

### 3.1 数值方法

| 类别 | 已验证 GPU 能力 |
|---|---|
| 时间推进 | 三阶段三阶 Runge-Kutta |
| 中心格式 | 六阶显式中心差分及黏性扩散 |
| 滤波 | 十阶显式中心滤波及物理边界闭合 |
| 迎风格式 | 一阶 Steger-Warming、WENO7、MP7 |
| 激波处理 | Ducros 传感器、激波区域选择性 Roe 特征空间 MP7 |
| 物性 | 单组分理想气体、Sutherland 变黏性 |
| 源项 | Channel 固定驱动力和 RTI 重力源项 |
| Sponge | 受限 x-max `layer` sponge |
| 网格 | Cartesian 网格和静态单块三维结构化曲线网格 |

不支持紧致差分、紧致滤波、三对角或五对角矩阵求解。GPU 当前只准入 RK3；
CPU 已有的 RK4 不属于当前 GPU 合同。

### 3.2 已验证算例

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
| OpenSBLI Katzer 层流 SBLI | `609x255x9` 输入、专用边界、NP=2 长跑和 step 60000 restart 已接通；外部物理验证未完成 |
| 动态入口 BL | 四切片三次插值、D2 restart 组合及逐 RK 级 NP=1/4 数值对比已通过；真实湍流入口尚未建立 |

这里的“已验证”主要指 CPU/GPU 数值等价、边界不变量和内存安全通过，
不等同于每个算例都已完成实验或 DNS 数据库级物理验证。

### 3.3 MPI 和变量常驻

- 已覆盖 NP=1、NP=2 三种 slab、NP=4 三种 plane 和 NP=8 `2x2x2`。
- TGV 已进行 NP=16 和 NP=32 的双卡共享正确性 smoke test。
- 生产目标仍是一 MPI rank 对应一 GPU。双卡 rank oversubscription 不能作为
  四卡或八卡扩展性能证据。
- 无输出 RK 区间内，主流场、守恒变量、RHS、通量、滤波工作区、曲线度量和
  激波传感器常驻 GPU。
- solution `q` 的有效外 halo 是 `hm`，但 qswap-compatible packet 为 `hm+1`，
  用于重复接口面平均；滤波、扩散和传感器 raw field 固定交换 `hm`，不平均
  接口面。
- HaloTransport 默认采用 pageable host-staged blocking；保留可选 pinned blocking，
  以及仅适用于完全周期 stored-diffusion 内部区域的 pinned-overlap。
- 独立 nonblocking 候选因端到端收益不足被拒绝；CUDA-aware MPI 因当前软件栈
  的大消息或 sanitizer 准入失败而暂缓。
- 统计量在 GPU 完成归约，只向主机返回少量标量。
- HDF5、checkpoint 和完整场输出仍由 CPU 负责，不属于当前 GPU 化范围。
- 默认正确性路径保持每个 CUDA kernel 后显式同步。

### 3.4 动态时序入口

当前 `bctype=11,turbinf=intp` 路径已经实现：

- x-min 物理边界 rank 常驻四个五变量 y-z 切片；
- 每个 RK 子步在 GPU 上使用与 CPU 相同的均匀时间间隔三次插值；
- 跨时间窗时只读取并 H2D 上传被替换的一个二维切片；
- 静态 `turbinf=prof` 与动态 `turbinf=intp` 使用独立 kernel；
- NP=1、NP=2 x/y/z slab、NP=4 `4x1x1` 和 `2x2x1` 数值门槛通过；
- 最大 primitive 和 conservative 误差分别为约 `7.11e-15` 和 `1.78e-14`；
- NP=1 Compute Sanitizer 报告零错误。

动态入口 D2 已增加非多项式时间序列，覆盖切片节点、跨切片和单步跨多个切片，
并覆盖 NP=4 `2x2x1`、CURVE 和十阶显式滤波组合。CPU/GPU 连续路径及共享
CPU checkpoint 重启路径的 complete-RK 守恒量最大绝对误差约为 `2e-14`。
逐 RK 级验证比较 `pre_rhs` 和 `post_update`：NP=1 的 6 份快照以及 NP=4 的
24 份逐 rank 快照全部通过，最大绝对误差均为
`1.4210854715202004e-14`。

上游 checkpoint 保存 primitive 场和辅助时间状态，不序列化完整 `q`/halo
内存状态。无滤波路径的连续/重启严格等价达到约 `1e-14`；显式滤波路径不把
连续/重启差异作为 GPU 缺陷，而是用同一 CPU checkpoint 分别重启 CPU/GPU，
比较相同重建阶段。该门槛已经通过。真实湍流入口统计仍属于 D3，尚未完成。

## 4. 边界条件支持范围

| `bctype` | 当前 GPU 支持范围 |
|---:|---|
| `1` | x/y/z 周期边界 |
| `11` | x-min 给定入口，完整更新 `rho/u/v/w/p/T/q`；支持静态和受控动态入口 |
| `12` | 受限 x-min 特征入口，仅用于已验证组合 |
| `21` | 当前主要支持轴对齐 x-max 出口；非轴对齐曲面主动拒绝 |
| `22` | 受限 Cartesian NSCBC 出口；非正交曲线网格主动拒绝 |
| `31` | RTI 使用的固定边界 |
| `41` | 曲线六面零吹吸静温无滑移壁面；另支持 y-min 物理法向吹吸 |
| `42` | 曲线 x/y 绝热无滑移壁面；z 向因 CPU 无分支而拒绝 |
| `411/421` | y 向滑移/无滑移组合，速度按局部几何法向投影 |
| `50` | 曲线六面零外推 |
| `51` | 受限轴对齐远场组合 |
| `52` | 受限 upper-y NSCBC 远场组合、五方程常数目标模式，以及静态曲面无黏源项平衡非反射模式 |
| `60` | 曲线六面对称，速度投影到局部切平面 |

当前不能宣称任意曲线 NSCBC、任意方向开放边界或缺失 CPU 分支的 GPU 支持。
受支持组合和拒绝条件记录在
`documents/ASTR_CURVILINEAR_GPU_COMPLETENESS_PLAN.md`。

## 5. 静态曲线网格状态

CURVE-C0 至 C21 已完成以下能力：

1. 周期曲线度量、自由流保持和解析度量收敛。
2. 曲线网格显式对流、扩散、滤波及统计量。
3. 曲线物理壁面的几何投影和 y-min 法向吹吸。
4. Ducros、选择性 Roe、Sutherland、NSCBC 和 x-max sponge 的受控组合。
5. 三网格、两时间步 Mach 5 层流边界层物理一致性验证。
6. 开放边界能力盘点以及未支持组合的 CUDA 启动前拒绝。
7. 使用同一组 CPU/GPU 二进制完成支持矩阵、拒绝项、三方向内存检查和
   大网格常驻审计。

CURVE-C21 共执行 23 个聚合阶段，全部通过。136 份逐场报告中的最大误差为
`q5=6.2527760746888816e-13`，128 份统计量报告中的最大误差为
`massflux=2.5093260802577788e-11`，必需边界不变量的最大残差为
`7.6170181273482740e-12`。70 项有限场检查全部通过，最小数值 Jacobian 为
`2.5296638468231652e-7`。x/y/z Compute Sanitizer 均报告零错误。

该结论限定为静态、单块、三维结构化曲线网格。多块网格、非共形接口、
动网格、ALE/GCL 和浸入边界不在当前完成范围。

CURVE-C22 新增 `ASTR_NSCBC_FARFIELD_MODE=nonreflecting`。该模式在曲面
upper-y `bctype=52` 上采用局部 eta 法向，并在特征空间令传入波抵消度量项和
横向通量源项。它不使用远场目标状态、经验松弛长度、边界 Mach 归约或旧式
x/z 边界面滤波。三网格声波反射系数随加密由 `0.01620` 降至 `0.00394`，
且小于匹配 compatibility 对照的四分之一。NP=1/2/4/8 八种拓扑全部通过，
最大 CPU/GPU 最终场差为 `2.21e-13`；NP=1/2 memcheck 为零错误。当前结论仍
限于无黏、五方程、静态单块曲面上边界，不代表完整 Navier-Stokes GCBC。

CURVE-C23 在相同 upper-y `bctype=52` 合同上加入六阶显式黏性残差的传入
特征源项修正。CPU 仅保存扩散前的五分量边界 RHS，GPU 使用常驻设备端面
缓冲区；扩散后由 RHS 差值得到离散黏性贡献，并沿 CURVE-C22 的局部 eta
特征基只扣除传入分量。十步均匀流最大 CPU/GPU 场差为 `7.99e-15`，Mach 5
曲线 HBL 二十步最大场差为 `T=5.06262e-14`，八种 NP=1/2/4/8 拓扑均通过。
NP=1/2 内存检查为零错误；两步 profile 中 capture/correction 各执行六次，
计算循环内无大于或等于 64 KiB 的场传输。该结论仍只覆盖静态单块、单组分、
五方程和 upper-y 面，不代表通用完整黏性开放边界。
黏性声学三网格 `64x48x64`、`80x60x80` 和 `96x72x96` 的 CPU 反射系数为
`0.01499173359`、`0.005955882611` 和 `0.003584736637`，随网格加密单调下降；
最大 CPU/GPU 反射差和最终场差分别为 `3.24e-13` 和 `2.13e-13`。

## 6. 性能证据

### 6.1 TGV 单 GPU P1

RTX 4000 Ada 上的 `256^3` FP64 TGV，NP=1、六阶显式中心、十阶显式滤波、
扩散和 RK3 条件下，完整 RK 中位时间从 `0.762855769 s` 降至
`0.580044424 s`，对应时间减少 `23.964%` 和 GPU 内部实现加速 `1.31517x`。
该结果不是相对 CPU 的整体加速比，也不能外推到 A800。

保留优化包括方向遍历、primitive 局部复用、冗余刷新消除和使用
`constdef::num1d60` 的编译期常量乘法。选择性同步把一个 profile 区间的同步
次数从 275 降至 15，但完整 RK 没有收益，因此显式同步仍为默认。

### 6.2 历史 Channel CPU/GPU 对比

`128^3` channel、`maxstep=100`、无场输出测试，以 NP=1 CPU 为统一基线，
历史 GPU 整体加速比约为 `28.7x` 至 `43.1x`。该结果来自早期工作站测试，
应作为历史性能证据，而不是当前最终基准。

### 6.3 CURVE-C21 双 GPU 扩展复测

| 配置 | 中位 wall time |
|---|---:|
| NP=1，单 GPU | `58.490 s` |
| NP=2，双 GPU x-slab | `38.989 s` |

- 双 GPU相对单 GPU加速比为 `1.5002x`，并行效率为 `75.01%`。
- GPU 利用率达到 100%。
- 无输出 RK 区间没有大于或等于 64 KiB 的完整场传输。
- NP=1/NP=2 峰值设备内存约为 `9876/5573 MiB`。

该测试证明当前曲线高速边界层路径实现了变量常驻和有效双卡扩展，
但不是相对 CPU 的统一性能基准，也不能外推到四卡或八卡。

### 6.4 P2 激波和 SBLI 专项优化

在 RTX 4000 Ada 本机冻结基线上，最终保留的 P2-1B 物理通量复用使
Shu-Osher `256x64x32` 和受控 SBLI `256x192x32` 的完整 RK 中位时间相对
P2-0 分别减少 `27.234%` 和 `48.910%`。这些结果是 GPU 内部实现优化，
不是相对 CPU 的整体加速比，也不代表 OpenSBLI 的物理验证。

### 6.5 P3 HaloTransport 本机收口

- NP=2 双物理 GPU 的九个正式组合中，pinned 相对 pageable 的完整 RK 时间
  减少 `2.829%` 至 `23.728%`。
- TGV x/y/z slab 的 pinned-overlap 相对配对 pinned 增量分别为
  `2.616%/3.085%/2.944%`。
- pageable blocking 保持默认可移植基线；pinned 和周期 diffusion overlap
  为可选后端。
- 该结论仅覆盖本机双 GPU host-staged 路径，不外推到 A800、多节点或
  CUDA-aware MPI。

### 6.6 A800 campaign

TGV-only campaign 于 2026-09-08 提交为 Slurm job `451398`。该作业随后在
`gpu01` 的首个计算节点预检中因缺少 `libsz.so.2` 失败，没有产生 A800 runtime
evidence。任务仍包含 A800 CFL/显存预检、`256^3` CPU/GPU 基线、`512^3`
NP=1/2/4 强扩展、pageable 对照、NP=4 NSYS 和分段推进到约 `t=20` 的生产计算。
作业脚本已新增计算节点 `ldd` 门槛和 host-staged MPI 固定配置；重新提交前仍需
提供自包含的 HDF5/SZIP 运行时依赖。

## 7. 显式滤波工作区策略

### 7.1 当前默认：完整五变量 ping-pong

当前已验证实现分配
`qwork_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:numq)`。对 `numq=5`，每次
完整滤波按以下顺序执行：

```text
q_d     --x filter--> qwork_d
qwork_d --y filter--> q_d
q_d     --z filter--> qwork_d
qwork_d --copy------> q_d
```

每个方向之间执行对应 raw halo 更新并显式同步。该实现一次处理五个守恒变量，
kernel 和 MPI 启动次数较少，已覆盖 TGV、Channel、LDC、RTI、物理边界和 CURVE
回归，因此继续作为默认和发布基线。

设本 rank 含 halo 的三维点数为

```text
Nlocal = (im + 2*hm + 1) * (jm + 2*hm + 1) * (km + 2*hm + 1)
```

则 FP64 完整 `qwork_d` 占用约为 `5*8*Nlocal` 字节。

### 7.2 已实现候选：单分量三维工作数组

已新增一个可选低显存实现，保留现有完整 `qwork_d` 路径，不直接替换。候选只
分配 `filter_work_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm)`，依次处理
`m=1,...,5`：

```text
exchange all q_d components x-filter halo once
for m = 1, 5
  q_d(:,:,:,m)  --x filter--> filter_work_d
  exchange filter_work_d y halo
  filter_work_d --y filter--> q_d(:,:,:,m)
  exchange q_d(:,:,:,m) z halo
  q_d(:,:,:,m)  --z filter--> filter_work_d
  filter_work_d --copy------> q_d(:,:,:,m)
end for
```

十阶中心系数、`0-6-6-6-8-10` 物理边界闭合、x/y/z 顺序、`hm` raw halo
语义和每个 kernel 后显式同步均不得改变。不同守恒变量之间没有 stencil 数据
依赖，因此按变量串行处理在数学上可保持当前分离滤波算子。运行时通过
`ASTR_GPU_FILTER_WORKSPACE=scalar` 启用；缺省值 `full` 保持原路径。

单分量工作区占用约为 `8*Nlocal` 字节，即当前 `qwork_d` 的 20%，工作区显存
减少 80%。对 NP=1、`512^3`、`hm=5` 的近似分配，完整工作区约为 `5.3 GiB`，
单分量工作区约为 `1.1 GiB`，可减少约 `4.2 GiB`。

主要代价是五个变量分别启动 kernel 和 halo 通信。在保持显式同步的条件下，
kernel 次数和 MPI 消息数最多接近当前的五倍，虽然传输总字节数近似不变。
因此该方案首先定位为显存不足时的可选模式，不预设其完整 RK 性能优于当前路径。

### 7.3 备选与拒绝方案

| 方案 | 工作区显存 | 主要判断 |
|---|---:|---|
| 完整五变量 `qwork_d` | 100% | 当前默认，验证最完整，启动和通信次数较少 |
| 单分量 `filter_work_d` | 20% | 新增推荐候选，优先保证低显存和数值等价 |
| 2 或 3 分量 chunk | 40% 或 60% | 单分量性能不足时的折中，需要额外 chunk 和尾块验证 |
| 复用完整 `qrhs_d` | 接近零新增体数据 | 生命周期和 NSCBC 边界 RHS 耦合风险高，不作为第一实现 |
| 直接原位或滚动覆盖 | 很低 | CUDA block 间缺少全局同步，会产生新旧 stencil 混读，拒绝 |

单分量实现已输出实际工作区分配字节数。完整 device 内存构成仍需单独审计，
大型 SBLI 的显存瓶颈可能来自通量、梯度或边界辅助数组，不能预设滤波工作区
是唯一主导项。

### 7.4 单分量模式验收门槛

1. 保留完整 `qwork_d` 默认路径和全部现有回归入口。
2. 采用 GPU backend 内部运行时选项选择 `full` 或 `scalar`，默认必须为 `full`。
3. 比较 CPU/full-GPU/scalar-GPU 三方同相位场，不能只比较统计量。
4. 覆盖 TGV NP=1、三种 NP=2 slab、至少一个 NP=4 plane 和 NP=8 `2x2x2`。
5. 覆盖 `41/42` 物理边界、LDC 多轴边界、Channel 和代表性 CURVE 滤波路径。
6. 验证 physical closure、MPI raw halo、有限性、Compute Sanitizer 和无输出常驻。
7. 实测工作区字节数至少减少 75%，并分别报告 kernel/MPI 次数和完整 RK 时间。
8. 即使数值门槛通过，只有完整 RK 性能达到单独定义的生产门槛后，才允许将
   scalar 模式用于常规生产；不得据低显存结果替换默认 full 模式。

### 7.5 2026-09-10 单分量实现证据

- NVHPC 26.1 CUDA 编译通过。
- NP=1 TGV 1/10 step 的 CPU/GPU 统计量和逐场比较通过；10 step 守恒量
  最大绝对差为 `2.84e-13`。
- NP=1 full-GPU/scalar-GPU 在 1 step 后逐场按零容差完全一致。
- NP=2 x/y/z slab、NP=4 `2x2x1` 和 NP=8 `2x2x2` 统计比较通过。
- x/y/z 三方向 `bctype=41` 的 `32^3` 一步逐场比较通过。
- `32^3` TGV Compute Sanitizer memcheck 报告零错误。
- LDC 五步、Channel 五步、曲线 `bctype=42` x 壁面 NP=1 和 y 壁面 NP=2
  的滤波加黏性逐场与统计量比较通过。Channel 验证脚本改为使用 CPU 完整 RK
  快照后，scalar 的 `q5 L_inf=4.6185277824406512e-14`。
- `128^3`、`hm=5`、NP=1 的 full/scalar 实测工作区分别为
  `107,424,760/21,484,952` bytes，减少 `80%`。
- 本地 `128^3`、NP=1、显式同步五轮完整 RK 中位时间为 full
  `0.081408774 s`、scalar `0.078219996 s`，scalar 在 RTX 4000 Ada 上快
  `3.916%`；采样进程峰值显存从 `1686 MiB` 降至 `1604 MiB`。
- `64^3` 两个推进步的 NSYS 进程级计数为 full/scalar kernel
  `188/332`、MPI `262/262`。这说明 scalar 增加 kernel 启动数；NP=1 的总 MPI
  调用数未增加，不能据此外推多 rank halo 消息成本。

当前结论是 scalar 已完成本机正确性、显存、性能和代表性物理边界准入。
`full` 继续作为默认和生产基线；A800 的 full/scalar 实测、`512^3` 显存余量和
分段重启尚未完成，因此不能把本机收益写成 A800 或生产结论。

## 8. 后续目标

### 8.1 A800 四卡真实性能与扩展

修复运行时依赖后重新提交 TGV campaign，完成 T0 至 T7，并满足以下终止条件：

- T0 证明单 A800 可以分配 `512^3`，并给出平台实测 `dt_CFL=1`；
- NP=1/2/4 均有一次预热和至少五次独立重复计时；
- 分别报告初始化/输出、纯 RK、halo、同步和 MPI 时间；
- 给出强扩展效率、显存峰值和实际 GPU 占用；
- 同一冻结输入比较 pageable、pinned 和适用时的 overlap；
- TGV 分段推进到约 `t=20`，并与 DLR Re=1600 统计历史比较。

### 8.2 OpenSBLI 层流 SBLI 物理验证

当前采用 OpenSBLI Katzer Mach 2 层流 SBLI 作为公开外部参考。参考输入、
保守边界、度量一致 eps、GPU 主循环和 checkpoint restart 已接通。下一步需要：

- 完成至少三套网格和两种时间步验证；
- CPU/GPU 同相位场和统计量满足既定误差阈值；
- 比较激波位置、壁面压力、摩阻、热流和分离长度；
- 输出激波传感器和选择性 Roe 区域的可追溯诊断；
- 检查 restart 接缝、长期 z 向均匀度和传感器 mask；
- 分开陈述数值等价、时间/网格收敛和外部物理一致性。

### 8.3 动态入口 D3 至 D4

D2 已完成。验收脚本分别保留 complete-RK 组合矩阵和逐 RK 级二进制快照矩阵，
并将上游 checkpoint 语义与 GPU 同相位数值门槛分开记录。

1. D3：建立独立 Mach 5 湍流平板前驱，验收 `Re_theta`、`Re_tau`、平均剖面、
   Reynolds 应力、时间相关和周期连续性。
2. D4：在 GPU 常驻累积壁压、摩阻、热流、Favre 平均和 Reynolds 应力，减少
   高频完整三维输出。该项的软件、数值、安全和本机性能门槛已完成。
3. 入口统计通过后，才启动生产级湍流 SBLI；静态平均入口不能承担该物理结论。

D4 采用紧凑 GPU 在线统计作为生产架构。设计合同见
`docs/superpowers/specs/2026-09-10-gpu-compact-production-statistics-design.md`。
CPU 独立参考与 GPU sidecar 在静态/动态 NP=1、z-slab NP=2、过滤曲线网格
NP=4、三维分解 NP=8 和同拓扑 restart 上均通过 `atol=rtol=1e-10`，实际最大
绝对差为 `1.02e-14`。Compute Sanitizer 报告零泄漏、零内存错误和零竞争。
Nsight Systems 在统计核之后未发现超过 64 KiB 的完整场 H2D/D2H 传输。
RTX 4000 Ada 的五轮配对完整步中位时间为关闭统计 `0.075387564 s`、开启统计
`0.075807734 s`，开销 `0.557%`。这些结果完成 D4 本机软件验收，不替代 D3
湍流统计收敛，也不代表 A800 生产性能。

### 8.4 性能、通信与同步

- 只根据 A800 profile 决定是否扩展非周期/SBLI 通信重叠。
- 选择性同步保持独立候选，不能替换显式同步正确性基线。
- RTX 4000 Ada 上拒绝的 P1 候选可在 A800 重新测试，但必须重新冻结同机基线。
- 仅在新的 MPI 软件栈通过大消息正确性和 sanitizer 准入后恢复 CUDA-aware MPI。

### 8.5 曲线开放边界与 HIP/DCU

- 已完成第一条基于局部物理法向的曲线 upper-y 非反射远场切片。
- 已完成该 upper-y 切片的首个黏性特征源项耦合，保留原六阶显式扩散实现。
- 只有具体算例给出物理法向合同后，才扩展到其他开放面或更一般黏性项。
- 不得把 Cartesian `12/22/51/52` 分支按编号直接推广到任意曲面。
- 先用 backend-neutral facade 和 `ISO_C_BINDING` 验证导数、滤波、halo pack/unpack
  的小型 HIP/DCU 原型，再决定是否建设完整第二 backend。

### 8.6 混合精度 Phase MP

当前生产默认仍为全 FP64。MP1 至 MP3 已实现四个相互排斥的实验候选：周期、物理
空间 `543e` WENO7/MP7 的标量 `flux_work`，显式 `643e` 的诊断导数
`dvel/dtmp`，以及显式黏性通量 `sigma/qflux`。运行时通过
`ASTR_GPU_MIXED_CANDIDATE=flux|derivative|viscous_flux|characteristic_flux`
选择一组 FP32 工作区；
未设置时保持兼容默认 `flux`。所有 MPI rank 必须选择同一候选。

新增方向采用“FP64 权威状态加可选 FP32 临时工作区”，详细设计见
`docs/superpowers/specs/2026-09-12-gpu-mixed-precision-workspace-design.md`：

- `q_d/qrhs_d/qsave_d`、RK 更新、原始变量、`jacob/dxi`、物理法向始终保持 FP64；
- MPI halo、checkpoint、Roe 特征系、NSCBC/GCBC、CURVE-C23 黏性源项和统计归约
  首阶段保持 FP64；
- 运行时默认 `ASTR_GPU_PRECISION_MODE=fp64`，实验模式为
  `ASTR_GPU_PRECISION_MODE=mixed_workspace`；
- 已实现候选为 `flux_work_d`、`dvel_d/dtmp_d`、`sigma_d/qflux_d` 和周期
  `flux_characteristic_work_d`；后续候选为 `shock_sensor_d` 和滤波工作区；
  每次只改变一组；
- FP16、BF16、TF32 和 Tensor Core 算法重构不属于本阶段。

2026-09-12 本机 MP1 证据如下：

- `32^3`、20 步 WENO7/MP7 的 CPU 与 GPU FP64 最大守恒场差均为
  `2.84e-13`；GPU FP64 与 mixed 的最大 `q5` 差分别为 `8.87e-7` 和
  `1.20e-6`，统计量最大差小于 `3e-11`。
- WENO7 的 NP=2 `2x1x1` 和 NP=4 `2x2x1` 通过，通信仍交换 FP64 halo。
- mixed 路径 Compute Sanitizer 非法访存为零。完整 leak-check 发现统计模块已有
  6 个 `512 B` 退出期未释放临时量，与新工作区调用栈无关，需单独修复。
- `128^3` 每 rank 工作区由 `21,484,952` 降至 `10,742,476` bytes，采样峰值显存
  由 `1578` 降至 `1568 MiB`。
- 五轮配对完整 RK 中位时间为 FP64 `1.219752923 s`、mixed
  `1.224175191 s`。mixed 慢 `0.363%`，没有本地加速证据。

首个候选结论为“可行但不晋级”：保留实验模式用于显存受限场景和 A800 复测，
不替换 FP64 默认。当前证据只覆盖短时光滑周期 TGV，不代表声学、曲线网格、
物理边界、激波、长期统计或生产算例已经通过混合精度验证。

2026-09-12 本机 MP2 证据如下：

- `derivative` 将诊断导数工作区从 FP64 改为 FP32。求解器扩散项不读取该存储，
  而是从 FP64 `vel/tmp` 独立重算梯度；统计核在读取时显式提升到 FP64。
- `viscous_flux` 在 FP64 中计算应力和热流，只将最终 `sigma/qflux` 写入 FP32；
  扩散 RHS 读取时提升到 FP64。FP32 场打包到现有 FP64 MPI 缓冲，接收后再收窄，
  MPI tag、计数、邻居顺序和 halo 载荷语义未改变。
- `derivative` 的 TGV NP=1/2/4 场量无差异，Cartesian/CURVE HBL 的运行时壁面
  统计最大差约 `3.91e-13`，后处理最大差约 `3.70e-15`。Compute Sanitizer
  报告零错误。
- `viscous_flux` 的 10-step TGV 最大守恒场差为 `3.70e-13`，统计量最大差为
  `6.11e-16`；NP=2 `2x1x1` 和 NP=4 `2x2x1` 通过。Cartesian HBL 最大场差
  `4.26e-12`、后处理差 `1.97e-12`；CURVE-C23 均匀场、声学扰动和黏性 HBL
  均通过。Compute Sanitizer 报告零错误。
- `64^3` 五轮交错计时中，`derivative` 的 FP64/候选中位完整 RK 时间为
  `8.706775/8.891394 ms`，候选慢 `2.120%`；工作区字节为
  `26,364,000/13,182,000`。`viscous_flux` 为 `8.736809/9.417326 ms`，候选慢
  `7.789%`；工作区字节为 `30,375,000/15,187,500`。两者均精确节省 50% 的
  对应工作区存储，但当前本地 GPU 没有整步加速证据。

MP2 两个候选均分类为 `local-pass-not-promoted`：保留作显存受限和 A800 复测的
实验选项，不改变 FP64 生产默认，也不允许组合启用。现有短时 HBL 是同相位一致性
检查，不应表述为稳态边界层物理验证。

2026-09-12 本机 MP3-A 数值门槛进展如下：

- `characteristic_flux` 仅在全周期、无黏、无滤波、五方程单组分 RK3 的
  `543e` MP7 特征重构路径准入。Roe 平均、特征矩阵、重构代数、Ducros 原始
  传感器、整数 mask、`qrhs_d` 和 RK 状态保持 FP64；仅五分量界面通量最终存储为
  FP32，RHS 读取时提升回 FP64。
- S0-A6 的 NP=1 pilot 按十倍观测误差和 `1-2-5` 上取整冻结场与统计量绝对门槛
  均为 `2e-6`，相对门槛为零。正式 A6 最大场误差为 `1.5973888878306752e-7`，
  最大统计量误差为 `1.0126266403176487e-7`。
- S0-A7/A8/A9 的 x/y/z slab 与 S0-A10 `2x2x2` 均通过同一冻结门槛。四个 MPI
  门槛的 GPU FP64 与候选原始传感器逐点相同，mask mismatch 均为零；最大场误差
  为 `1.8279774849361274e-7`，最大统计量误差为 `1.0126264982091016e-7`。
- A10 的八个 MPI rank 在本地共享两张 GPU，只证明三方向 halo 与分解路径的短时
  正确性，不构成一 rank 一 GPU 的扩展性能证据。

MP3-A 的 Compute Sanitizer 报告 `ERROR SUMMARY: 0 errors`。`400x16x16` 五轮
交错计时得到 FP64/候选完整 RK 中位时间 `0.023973636/0.025151473 s`，候选慢
`4.913%`。对应工作区由 `11,984,760` 降至 `5,992,380` bytes，精确节省 50%；
采样峰值显存由 `656` 降至 `650 MiB`，峰值利用率为 `95%/88%`。

据此，MP3-A 分类为 `local-pass-not-promoted`：本地短时数值、三方向 MPI halo、
非法访存和交错计时流程已经闭合，且具有可核算的工作区显存收益，但没有本地整步
加速证据。OpenSBLI 生产路径、物理边界、扩散、滤波、CURVE 和长期统计仍未通过
该候选验收，因此不改变 FP64 生产默认。

周期物理空间上风通量还新增了一个与精度正交的 P2 性能候选。运行时
`ASTR_GPU_FLUX_PAIR_MODE=fused` 将正负 Steger-Warming 重构合并为单个方向核，
共享八个唯一模板点的状态计算；默认 `split` 和原六个分裂核完整保留。该候选不
改变权威 FP64 状态、MPI halo、边界、特征重构、激波、滤波或扩散路径，并继续在
每个 kernel 后显式同步。

2026-09-12 本机 `128^3` WENO7 五轮交错测试中，FP64 完整 RK 中位时间由
`1.219539164 s` 降至 `0.845358029 s`，降低 `30.682%`，对应 `1.44263x`。
WENO7/MP7 最大场差为 `5.68e-14`，NP=2/4 与 memcheck 通过。该路径当前为本地
通过但未晋级的可选优化，仍需 A800 NP=1/2/4、长时 TGV 和 restart 验收。

补充的 `256^3` 五轮交错测试得到 split/fused 中位时间
`8.678503483/6.052904717 s/RK`，离散度 `1.223%/1.136%`，整步降低
`30.254%`，对应 `1.43378x`。FP64 最大 `q5` 差为 `1.14e-13`，统计量一致。
mixed split/fused 的稀疏 FP32 舍入差最大为 `4.20e-9`，两条 mixed 路径各自相对
FP64 的最大 `q5` 差均为 `8.72e-7`，继续满足 MP1 的 `2e-6` 门槛。

性能候选采用宽松保留政策。发现阶段不设固定加速比或显存下降百分比。数值门槛
通过后，只要存在超出测量噪声的可重复正收益、可核算的显存下降、更大可运行网格，
或对未来 HIP/DCU 后端有实际价值，即可保留为实验候选。小幅收益记为“可行但不晋级”，
不直接淘汰。进入生产默认仍需在 A800 NP=1/2/4 上证明整步不退化，并在速度、显存、
问题容量、能耗或可移植性中至少提供一项实测价值。

Phase MP 分为：MP0 冻结 FP64 基线与测量噪声，MP1 筛选光滑流工作区，MP2 处理
黏性和曲线网格工作区，MP3 处理激波敏感工作区，MP4 最后处理十阶滤波工作区，
MP5 在 A800 上给出晋级、保留实验或拒绝结论。OpenSBLI 的 FP64 外部物理验证是
MP3 激波敏感候选的前置条件。

## 9. 暂缓范围

以下能力不进入近期目标：

- compact 差分和 compact 滤波；
- RANS/LES；
- 多组分、化学反应和燃烧；
- GPU HDF5 和 checkpoint 写场；
- 多块非共形网格；
- ALE/GCL 动网格；
- 浸入边界法。

## 10. 当前推荐顺序

1. 收口 A800 TGV T0 至 T7，建立 NP=1/2/4 正式性能和强扩展证据。
2. 补齐首个 MP1 候选和 P2 通量正负对融合的长时 TGV、restart 与 A800 NP=1/2/4
   复测；保持 FP64 与 `split` 为默认，分别测试后再评估组合。
3. 完成 OpenSBLI 三网格、两时间步和外部物理比较，为激波敏感混合精度建立 FP64 物理基线。
4. 将已本地收口的 MP2 候选带到 A800 NP=1/2/4，并推进 MP3 至 MP5 的激波敏感、
   滤波和生产晋级验证。
5. 完成动态入口 D3 前驱统计收敛，并在 A800 上复核已完成的 D4 常驻统计性能，
   再进入生产级湍流 SBLI。
6. 完成单分量滤波的 LDC、Channel、`42`、CURVE 和完整 RK 性能验收，再决定
   是否允许其进入低显存生产任务。
7. 以 A800 profile 决定非周期/SBLI overlap 和选择性同步是否继续。
8. 建立 CUDA/HIP 后端边界原型，验证未来 DCU 路径。
9. 由具体算例需求决定是否扩展曲线特征边界。
10. 只有项目需求重新打开时，才进入组分、化学、湍流、IBM 或多块网格移植。
