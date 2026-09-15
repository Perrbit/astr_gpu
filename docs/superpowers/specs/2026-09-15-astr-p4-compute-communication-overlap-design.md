# ASTR P4 计算与通信重叠设计

_面向本地双 GPU 筛选和后续 A800 单节点强扩展验收，2026-09-15_

---

## 📋 决策摘要

本阶段采用主机暂存双流流水线。它在不依赖 CUDA-aware MPI 的条件下，将
GPU pack、异步 D2H、MPI halo、异步 H2D 和 unpack 与不依赖 halo 的内部区域
计算重叠。现有 `pageable`、`pinned` 和 `pinned-overlap` 后端保持不变，新增
`pinned-pipeline` 作为受控实验后端。

本阶段先修复性能基准的无效 HDF5 生命周期开销，再建立可归因时间线，最后
实现异步流水线。A800 上正在运行的任务不终止、不修改，也不作为本地实现的
交互式测试环境。

验收目标保持为 A800 单节点 `512x512x512` TGV、FP64、完整五变量滤波工作区，
拓扑最优 NP=1 到 NP=4 强扩展效率不低于 70%，逐场误差不超过 `1e-10`。
本地双 GPU 只承担实现筛选、正确性、Compute Sanitizer 和 Nsight Systems
重叠证据，不产生 A800 性能结论。

## 🔍 已确认的问题

### A800 基准生命周期失真

任务 `458942` 的 NP=1 完整 RK 中位时间约为 `1.75868 s`。NP=2 x-slab
warm-up 的完整 RK 均值约为 `1.32896 s`，对应初步强扩展效率 `66.17%`。
其中首阶段准备效率约为 `74.85%`，后续积分效率约为 `62.95%`。NP=2 达到
70% 效率需要将完整 RK 时间再降低约 `5.47%`。这些数值来自未完成矩阵，
只能用于定位，不能作为最终论文数据。

当前 benchmark 每个独立进程都会重新执行生成网格和初始场写出。
`gridgeneration::gridgen` 在生成网格后调用 `writegrid`，
`initialisation::flowinit` 无条件调用 `writeflfed`。NP=2 warm-up 因而写出约
`3.24 GB` 网格和约 `6.48 GB` 流场，进程生命周期接近四小时，而 22 个 RK
步只占约 29 秒。`FEQCHKPT > MAXSTEP` 仅关闭时间循环内 checkpoint，不能关闭
这两次启动写出。

### 当前重叠范围不足

[`halo_transport_gpu.cuf`](../../../src_gpu/halo_transport_gpu.cuf) 中的
`exchange_host_pair` 只有在传入 callback 时才使用非阻塞 MPI。当前 callback
只用于完全周期、开启黏性项且使用 stored diffusion 路径时的第一条活动分解轴。
它仅把 MPI request 存活期与三个 diffusion-RHS 内部 kernel 重叠。

以下阶段仍然串行：

- pack kernel 与强制设备同步
- device-to-host 赋值
- 大部分 solution 和 filter halo
- 第二、第三条分解轴的通信
- host-to-device 赋值与 unpack kernel
- `sigma_d` 之后的 `qflux_d` halo

因此 `pinned-overlap` 是局部 MPI wait 隐藏，不是完整通信流水线。

### 全局同步消除只作为受限候选

既有单卡实验将 `cudaDeviceSynchronize` 大幅减少后没有得到完整 RK 收益。
本阶段允许 `dependency` 以受限 opt-in 候选重新测量，但不将全局“尽量不同步”
设为推荐路径。同步只根据跨流、MPI host buffer、halo消费和主机读回依赖移除，
默认流上的计算顺序仍由 CUDA stream 保证；没有本地整步收益时继续保留
`explicit` 默认。

## 🏗️ 目标架构

### 运行模式

| 配置 | 用途 | 默认值 | 约束 |
| --- | --- | --- | --- |
| `ASTR_GPU_HALO_TRANSPORT=pageable` | 兼容基线 | 默认 | 阻塞主机暂存 |
| `ASTR_GPU_HALO_TRANSPORT=pinned` | 固页基线 | 可选 | 阻塞 MPI |
| `ASTR_GPU_HALO_TRANSPORT=pinned-overlap` | P3 基线 | 可选 | 仅 stored diffusion 局部重叠 |
| `ASTR_GPU_HALO_TRANSPORT=pinned-pipeline` | P4 候选 | 新增、可选 | 首期仅周期 TGV；可用 `explicit` 做 A/B |
| `ASTR_GPU_SYNC_MODE=explicit` | 调试与回归 | 默认 | 每个 kernel 后设备同步 |
| `ASTR_GPU_SYNC_MODE=dependency` | P4 候选 | 新增、可选 | 仅与 `pinned-pipeline` 联用 |
| `ASTR_GPU_BENCHMARK_NO_FIELD_IO=1` | 计算基准 | 新增、默认关闭 | 仅 GPU TGV 且启用 RK timing |

所有 MPI rank 必须选择相同模式。非法组合、rank 间配置不一致、固定主机内存
注册失败或异步状态机非法迁移均 fail closed。注册失败时不得静默宣称流水线
有效，可以明确回退到 `pageable`，同时输出统一的失活原因。

### 组件职责

| 组件 | 新职责 | 不承担的职责 |
| --- | --- | --- |
| `gpu_check` | 启动错误检查、流边界同步、依赖事件等待 | 决定数值阶段顺序 |
| `halo_transport_gpu` | request/context 生命周期、MPI进度、固页缓冲注册 | pack/unpack 数值索引 |
| `halo_exchange_gpu` | 各轴 pack/unpack、异步拷贝和事件连接 | 选择内部区计算 kernel |
| `mainloop_gpu` | 启动内部区、等待 halo、启动边界条带 | 直接管理 MPI request |
| benchmark driver | 五轮计时、I/O禁用、完整元数据与结果校验 | 修改生产输出配置 |
| profile analyzer | 阶段时间、MPI/kernel交叠和同步计数 | 用单核时间代替完整 RK |

### Halo context

每个活动方向和字段族拥有独立 context，至少保存：

- 方向、字段族、消息长度和固定 tag
- 两个接收与两个发送 request
- device send/receive buffer 和 pinned host send/receive buffer
- pack/D2H 完成事件和 H2D/unpack 完成事件
- 当前状态、激活标志和失败原因

context 不拥有权威流场。缓冲区在 request 完成和消费事件完成前不得复用。
首期保持单主机线程调用 MPI，不要求 `MPI_THREAD_MULTIPLE`。

```mermaid
stateDiagram-v2
    accTitle: Halo Context Lifecycle
    accDescr: Lifecycle of one directional halo exchange from receive posting through asynchronous staging, MPI progress, device upload, and boundary readiness

    [*] --> Idle: 初始化
    Idle --> ReceivePosted: 发布 Irecv
    ReceivePosted --> DevicePacking: 启动 pack 和 D2H
    DevicePacking --> SendReady: D2H 事件完成
    SendReady --> MpiActive: 发布 Isend
    MpiActive --> HostComplete: Waitall 完成
    HostComplete --> DeviceUploading: 启动 H2D 和 unpack
    DeviceUploading --> HaloReady: unpack 事件完成
    HaloReady --> Idle: 边界条带消费完成
    ReceivePosted --> Failed: CUDA 或 MPI 错误
    DevicePacking --> Failed: CUDA 或 MPI 错误
    MpiActive --> Failed: CUDA 或 MPI 错误
    DeviceUploading --> Failed: CUDA 或 MPI 错误
    Failed --> [*]: MPI Abort
```

### 单方向流水线

接收缓冲区与发送缓冲区相互独立，因此 `MPI_Irecv` 可以在 pack 前发布。
pack 与 D2H 位于 communication stream，内部区域 kernel 位于 compute stream。
两者只读取同一权威输入时允许并发，不允许未声明的跨流读写竞争。

```mermaid
sequenceDiagram
    accTitle: Host Staged Halo Pipeline
    accDescr: Host-staged halo exchange overlaps GPU packing and MPI progress with halo-independent interior computation, then releases boundary computation after unpack completion

    participant host as Host thread
    participant comm as CUDA communication stream
    participant compute as CUDA compute stream
    participant mpi as MPI runtime

    host->>mpi: 发布两个 Irecv
    host->>comm: 启动 pack
    host->>comm: 启动异步 D2H
    par 内部区域计算
        host->>compute: 启动 interior kernel
    and 发送准备
        comm-->>host: D2H event ready
        host->>mpi: 发布两个 Isend
    end
    loop 计算尚未结束
        host->>mpi: MPI Testall 推进
        host->>compute: 查询 compute event
    end
    mpi-->>host: 四个 request 完成
    host->>comm: 启动异步 H2D
    host->>comm: 启动 unpack
    comm-->>compute: halo-ready event
    host->>compute: 启动 boundary-strip kernel
    compute-->>host: 阶段完成
```

### 多方向扩展

首个实现一次只激活一条分解轴，用于验证状态机和单轴 slab。随后扩展为每轴
独立 context，使 `2x2x1`、`2x1x2` 和 `1x2x2` 能同时提前发布接收。不同轴
继续使用现有私有固定 MPI tag 区间，不复用仍处于 active 状态的 request 或
buffer。

多方向并发不能改变数值操作顺序。filter 的 x、y、z pass 仍依次完成，只在
每个 pass 内将内部区计算与该方向 halo 重叠。solution 与 diffusion flux 可在
字段依赖允许时提前发布接收，但 boundary-strip kernel 必须等待对应方向的
halo-ready event。

## 🔄 同步与错误语义

### 保留的同步边界

`dependency` 模式仍在以下位置建立明确完成关系：

1. MPI 读取 pinned send buffer 前等待 D2H 完成事件
2. boundary-strip kernel 消费 receive buffer 前等待 unpack 完成事件
3. host 统计量、验证快照、checkpoint 或文件输出读取 device 数据前
4. 完整 RK 计时采样结束前
5. 程序退出和缓冲区释放前

kernel launch 后继续立即调用 `cudaGetLastError`。异步执行错误在最近的流或
事件边界报告，并同时输出该 stream 最后一个 kernel label。不得用删除错误
检查换取性能。

### 可以移除的同步

只有满足以下条件时才移除 `cudaDeviceSynchronize`：

- 前后 kernel 位于同一 stream，CUDA顺序已经建立依赖
- 并发 stream 对同一数组只有 read-read 或访问不重叠区域
- host 不读取未完成的 device 或 pinned buffer
- MPI 不访问尚未完成 D2H 的 send buffer
- 后续消费方具有显式 event 依赖

没有依赖图证明的同步保持不动。物理边界、CURVE、shock、chemistry、统计量和
文件输出路径在首期继续使用 `explicit`。

## ⚙️ 分阶段实施

### P4-0 基准可信化

1. 增加默认关闭的 `ASTR_GPU_BENCHMARK_NO_FIELD_IO`
2. 仅在 GPU 周期 TGV 且启用 RK timing 时跳过 `writegrid` 和初始 `writeflfed`
3. 保持生产输入、checkpoint、统计量和普通运行输出语义不变
4. benchmark 检查运行目录内没有新 `grid.h5` 和 `flowfield.h5`
5. 为 prepare、filter、solution halo、convection、diffusion flux、diffusion halo、
   diffusion RHS 和 RK update 建立可解析阶段标记
6. 本地 `256^3` NP=1/2 x、y、z slab 各完成五轮基线

P4-0 是后续优化的阻塞门槛。未消除HDF5生命周期开销或无法形成阶段归因时，
不得改通信实现。

### P4-1 单轴异步状态机

1. 新增 `pinned-pipeline` 和 `dependency`，保持默认配置不变
2. 实现 per-axis context、异步 pack/D2H、MPI进度、异步H2D/unpack
3. 首先接入周期 TGV 的 solution halo
4. 再接入完整 qwork filter halo
5. 最后接入 `sigma_d` 和 `qflux_d` diffusion halo
6. 对 x、y、z slab 分别验证实际重叠和完整 RK 时间

### P4-2 内部区与边界条带

1. 为每个候选阶段明确 interior 下标范围和宽度
2. interior kernel 不读取任何远端 halo
3. boundary-strip kernel 只覆盖受对应方向 halo 影响的网格点
4. 每个网格点每阶段只写一次权威输出
5. 先保持算术表达式和求值顺序不变，再单独评估 kernel 数量增加的代价

### P4-3 多轴和消息聚合

1. 为二维分解同时维护两个 context
2. 为三维正确性测试维护三个 context
3. 单独评估合并 `sigma_d(1:6)` 和 `qflux_d(1:3)` 的九变量消息
4. 只有完整 RK 改善时保留消息聚合
5. 本地 NP=4/8 共享双卡只用于正确性，不报告扩展效率

### OpenCFD-SCU 交叉审计后的采用边界

OpenCFD-SCU 的可复用价值主要位于任务图和模板数据移动，不位于其 MPI
传输函数。ASTR 按下表选择性吸收，所有候选仍须经过逐场、sanitizer、时间线和
完整 RK 门槛。

| OpenCFD-SCU 做法 | ASTR 决策 | 当前执行阶段 |
| --- | --- | --- |
| 内点、边界和通信分 stream，用 event 建立依赖 | 采用其任务图思想，保留 ASTR 的状态机接口 | P4-1 至 P4-3 |
| y/z 模板使用共享内存转置和 warp shuffle | 仅在 NCU 证明非合并访存或 long-scoreboard 主导后建立原型 | P4-5 候选 |
| 共享输入的有限 kernel 融合 | 只允许生产者与直接消费者的小范围融合，检查寄存器和 occupancy | P4-5 候选 |
| 化学活跃单元列表和积分状态分组 | 概念纳入 C5，使用高效 scan/compaction，不复制原循环计数代码 | C5 后续 |
| 逐场 `MPI_Sendrecv` 和每方向强制 stream 同步 | 不采用；ASTR 保留聚合消息、`MPI_Irecv/Isend/Testall` | 禁止回退 |
| `atomicAdd` 累加方向 RHS | 不采用；保持每阶段单写或显式有序累加 | 禁止引入 |
| `world_rank % device_count` 绑定 GPU | 不采用；继续依赖节点本地 rank 或调度器绑定 | 运行时约束 |
| 每次归约或化学调用临时分配 | 不采用；工作区必须初始化分配并循环复用 | 内存约束 |

单轴 x/y/z slab 与每轴独立 context 均已完成。solution halo 会为所有活动轴完成
pack、D2H 并发布 MPI，再按 x、y、z 顺序等待和 unpack，以保持 CPU 共享面与边棱
覆盖语义。filter 因三方向 ping-pong 数据依赖仍按轴顺序执行。融合 diffusion 会
遍历所有活动轴，但只在首个活动轴通信期间执行一次内部 RHS；其余轴当前串行完成，
尚未宣称多轴 diffusion 完全重叠。

### P4-4 A800 验收

本地门槛通过并形成干净提交后，才准备新的 A800 作业。该作业不复用当前
`458942`，也不修改其目录。正式矩阵至少包含：

| 类型 | 网格 | MPI | 拓扑 |
| --- | --- | --- | --- |
| 强扩展 | `512x512x512` | NP=1 | `1x1x1` |
| 强扩展 | `512x512x512` | NP=2 | 三种 slab |
| 强扩展 | `512x512x512` | NP=4 | 三种 slab 与三种双轴分解 |
| 时间线 | `512x512x512` | NP=2/4 | 各自最优拓扑 |

## ✅ 验证门槛

### 基准逻辑

- `ASTR_GPU_BENCHMARK_NO_FIELD_IO` 默认关闭
- 非TGV、CPU、未启用RK timing或rank配置不一致时 fail closed
- 性能运行不生成网格或流场HDF5
- 五轮计时仍保留独立进程、warm-up和20个有效RK样本
- 计时结果拒绝缺rank、缺step、非有限值和重复冲突记录

### 数值正确性

- FP64完整工作区保持权威
- NP=1/2 的 x、y、z slab 完成 1、10 和 100 步逐场比较
- `q` 与原始变量最大绝对误差不超过 `1e-10`
- TGV动能、拟涡能和耗散率统计差不超过既有门槛
- NP=4/8共享双卡完成多轴状态机、固定tag、halo内容及 1/10/100 步逐场比较；
  该配置只提供正确性证据，不提供扩展效率证据

### 并发安全

- Compute Sanitizer memcheck 为 0 errors
- racecheck 不报告 pipeline 新增的数据竞争
- request 未完成前不能复用host或device buffer
- context 退出时没有active request、未销毁event或已注册host buffer
- 任一 rank 失败时使用统一 MPI abort 路径

### 性能证据

- Nsight Systems 同时采集 CUDA、MPI 和阶段标记
- 报告 pack、D2H、MPI wait、H2D、unpack和interior kernel时间
- 直接报告MPI request与interior kernel的时间交集
- 报告 `cudaDeviceSynchronize`、stream/event wait和MPI调用次数
- 本地候选相对 `pinned-overlap/explicit` 回退不得超过2%
- 本地没有收益但具有真实重叠证据的候选只保留为opt-in，不晋升默认
- A800最终NP=4强扩展效率不低于70%

## 2026-09-15 本地实现进展

`pinned-pipeline` 已覆盖周期 TGV 的 x/y/z 单轴、双轴和三轴分解。默认
`explicit` 以及 `pageable`、`pinned`、`pinned-overlap` 后端保持不变，
`dependency` 继续是显式选择的实验同步模式。transport 现为每轴独立 context，
每个活动轴各自持有通信流、事件、四个 MPI request、计数、邻居和私有固定 tag。
共享计算流只负责当前唯一的独立计算组，不承担 MPI context 所有权。

普通字段 halo 缓冲仍按六分量分配。启用 pipeline 时，每个活动 MPI 轴的字段缓冲
扩展为九分量，用于聚合 `sigma_d(1:6)` 和 `qflux_d(1:3)`；非活动轴不分配该容量，
默认后端的常驻显存不变。

融合 diffusion halo 将原来的两轮
`pack -> D2H -> MPI -> H2D -> unpack` 合并为一轮。Nsight Systems 的三步
`256^3` trace 中，D2H 调用由 `168` 次降为 `132` 次，H2D 调用由 `506` 次降为
`470` 次。原 diffusion 路径的 `72` 次阻塞 `MPI_Sendrecv` 被 `36` 次
`MPI_Irecv/MPI_Isend` 事务替代。点对点总消息字节保持不变，因此该候选减少的是
消息和主机暂存启动次数，不是 halo 数据量。

首个融合事务中，`MPI_Isend` 在相对时间约 `8.96 ms` 发布，三个 diffusion-RHS
内点 kernel 从 `9.04 ms` 运行至 `25.44 ms`，request 在约 `36.70 ms` 完成。
约 `16.4 ms` 内点计算位于通信活跃窗口。主机通过计算完成事件与 `MPI_Testall`
共同推进，不再依赖对默认流的查询。

本地 `256^3`、20 个保留 RK 样本、五个独立进程的结果如下：

| 路径 | 完整 RK 中位时间 | 五轮相对极差 | 状态 |
| --- | ---: | ---: | --- |
| pinned 基线 | `0.529989956 s` | `0.606%` | 回退基线 |
| 初始 pipeline 状态机 | `0.533358806 s` | `1.366%` | opt-in |
| 融合 diffusion 加事件推进 | `0.509744836 s` | `0.815%` | 本地候选通过 |

最终候选相对 pinned 基线降低 `3.820%`，相对初始 pipeline 降低 `4.427%`。
相对 P4-0 pageable x-slab 的 `0.613739301 s` 降低 `16.944%`。以 P4-0 NP=1
GPU 的 `0.5960130865 s` 为本地统一基线，当前 NP=2 加速比为 `1.1692x`，并行
效率为 `58.46%`。它仍未达到 A800 的 `70%` 目标，且不能把本地排序外推到 A800。

三轴数值门槛采用 `128^3`、NP=2。x/y/z 的 1、10 和 100 步 CPU/GPU 同相位场
比较均通过 `1e-10` 门槛。10 步最大守恒场差依次为 `3.1264e-13`、
`3.1264e-13` 和 `3.4106e-13`；100 步依次为 `6.5370e-13`、`6.8212e-13`
和 `7.1054e-13`。y/z 的 10 步 `dependency` 复测得到与 `explicit` 相同的差值。
生产 halo 合约在 NP=2 和 NP=3 下精确覆盖三轴、周期重复邻居、物理端点、
共享面平均和 1/3/5/6 分量交换。双 rank Compute Sanitizer 对三轴合约报告
`0 errors`、`0 bytes leaked`，racecheck 报告 `0 errors, 0 warnings`。

每轴 context 接入后，NP=4 `2x2x1` 和 NP=8 `2x2x2` 的 `128^3`、1/10/100 步
CPU/GPU 同相位逐场比较全部通过。100 步最大守恒场差分别为 `7.1054e-13` 和
`7.6739e-13`；两种拓扑在 `dependency` 模式下的 10 步结果与 `explicit` 相同。
纯 transport `contexts` 合约确认两个轴的 MPI request 在任一轴等待前均已发布。
NP=4 `2x2x1` 完整求解器的 full leak-check 在四个 rank 上均为 `0 errors`、
`0 bytes leaked`，racecheck 均为 `0 errors, 0 warnings`。当前证据仍不覆盖物理
边界、CURVE、激波或 chemistry 的异步化，也不构成共享双卡配置的性能证据。

系统重启后的同一时段本地 `256^3`、NP=2、20 个保留 RK 样本、五个独立进程
配对结果如下。两组都使用 `explicit`，因此比较的是 pipeline 任务图与聚合通信，
不是取消同步的收益。

| slab | pinned 中位时间 | pipeline 中位时间 | 降低 | 基线/候选相对极差 | 显存增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| y `1x2x1` | `0.510051923 s/RK` | `0.479824547 s/RK` | `5.926%` | `0.969% / 0.869%` | `32 MiB` |
| z `1x1x2` | `0.508967702 s/RK` | `0.478513546 s/RK` | `5.984%` | `1.057% / 0.646%` | `33 MiB` |

y 的峰值 GPU 利用率由 `83%` 增至 `93%`，z 由 `94%` 增至 `96%`。这些结果
支持在本地保留 y/z pipeline，但不替代 A800 配对矩阵，也不能作为多轴扩展效率。

### P4-4 同步消除与通信启动审计

`dependency` 模式现已真正省略非强制 `cudaDeviceSynchronize`，同时保留
host 读回、归约和显式 `force_sync` 边界。默认流在每次 pipeline transaction
记录 source-ready event，通信流和独立计算流在读取 `q_d`、`qwork_d`、
`sigma_d/qflux_d` 前显式等待。solution halo 窗口中的 `zero_rhs_kernel` 也等待
该 event，因为上一 RK stage 的更新 kernel 仍读取 `qrhs_d`；只把它视为独立
输出会留下跨 stage 读写竞争。

同为三步 `256^3`、NP=2 x-slab 的 Nsight Systems trace 中，kernel 和 memcpy
调用数保持 `708/602` 不变，`cudaDeviceSynchronize` 从 `492` 次降至 `36` 次，
减少 `92.683%`。最终 trace 包含 `102` 次 `cudaStreamWaitEvent`，其 CUDA API
总时间约 `0.285 ms`。双 rank 的 10 步逐场比较继续通过，最大守恒场差为
`3.1264e-13`；memcheck 为 `0 errors`，racecheck 为 `0 errors, 0 warnings`。

本机当前负载状态与历史基准存在漂移，因此新增了
`explicit + pinned-pipeline` 诊断组合，以便同一可执行文件、同一时段做配对
A/B。当前五轮结果如下：

| 同时段路径 | 完整 RK 中位时间 | 五轮相对极差 | 相对 explicit pipeline |
| --- | ---: | ---: | ---: |
| `explicit + pinned-pipeline` | `0.628957548 s` | `0.941%` | 基线 |
| `dependency + pinned-pipeline` | `0.641986121 s` | `6.473%` | 慢 `2.071%` |

历史 pipeline 候选为 `0.509744836 s/RK`，比当前同时段 explicit 基线快
`23.387%`。这说明绝对时间受当前主机和 GPU 环境影响，不能用跨时段值归因；
同时段 A/B 仍表明全局取消同步没有本地加速证据。`dependency` 继续保留为 opt-in
候选，默认和当前推荐运行模式仍为 `explicit`，等待 A800 配对复测后再决定。

另一个候选把 diffusion-RHS 内点 kernel 提前到 D2H wait 之前。逐场、memcheck
和 racecheck 均通过，但五轮中位时间为 `0.677846560 s/RK`，相对同时段 explicit
基线慢 `7.773%`。时间线显示 pack kernel 仍约 `0.62 ms`，不是回退来源；D2H
结束到 H2D 开始的 MPI 区间中位数由 `26.685 ms` 增至 `52.080 ms`。提前计算
缩短了真正慢 MPI 阶段可隐藏的计算窗口，因此该调度已从执行路径回退。

当前 host halo 数组由 `ensure_*_buffers` 一次分配，在 transport 初始化时用
`cudaHostRegister` 一次注册，在每个时间步循环复用，并在 finalize 时统一
`cudaHostUnregister`。循环内不存在 `cudaMallocHost/cudaFreeHost`，所以改用
`cudaHostAlloc` 或 `MPI_Alloc_mem` 不会消除当前路径上的逐步分配开销。持久
`MPI_Send_init/MPI_Recv_init` 请求暂未实现：现有 trace 中 `MPI_Irecv/MPI_Isend`
创建总开销不足 `1 ms`，主要代价是大消息 host staging 和传输尾部。该项保留为
A800 MPI 栈上的独立候选，不与同步策略混合验收。

每轴 context 重构后的同一时段 x-slab `256^3`、NP=2、五轮配对结果如下。每轮
保留 10 个完整 RK 样本，并丢弃首个进程内 warm-up 样本。

| 路径 | 完整 RK 中位时间 | 五轮相对极差 | 峰值 GPU 利用率 |
| --- | ---: | ---: | ---: |
| `pinned + explicit` | `0.529076911 s` | `0.820%` | `83%` |
| `pinned-pipeline + explicit` | `0.506394356 s` | `0.763%` | `92%` |
| `pinned-pipeline + dependency` | `0.504894878 s` | `0.336%` | `92%` |

pipeline explicit 相对 pinned 降低 `4.287%`。dependency 相对 pinned 降低
`4.571%`，但相对 pipeline explicit 只再降低 `0.296%`，不足以改变默认同步策略。
因此当前推荐仍为 `pinned-pipeline + explicit`；`dependency` 保持 opt-in。
本机只有两张 GPU，不能在一秩一卡条件下给出双轴或三轴性能结论。下一门槛是在
四卡 A800 上完成 NP=4 plane/cube 时间线和完整 RK 配对，再决定是否并行推进多个
diffusion 轴或实现持久 MPI request。

## 🚫 非目标

本阶段不包含：

- 修改六阶中心差分、十阶中心滤波或RK3公式
- CUDA-aware MPI、NVSHMEM、NCCL或GPUDirect RDMA生产接入
- 多节点扩展
- 物理边界、CURVE、shock或chemistry路径的异步化
- GPU HDF5、checkpoint或场输出移植
- 为减少通信而缩短固定 `hm` halo层数
- 用GPU利用率或单kernel时间替代完整RK验收

## ⚠️ 风险与回退

| 风险 | 检测 | 回退 |
| --- | --- | --- |
| MPI缺少异步进度 | request完成状态和时间线 | 主机持续 `MPI_Testall` |
| 拷贝与kernel不能重叠 | CUDA copy-engine时间线 | 保留pinned阻塞路径 |
| 拆分kernel增加launch成本 | NP=1与NP=2完整RK | 候选保持opt-in或撤销 |
| 多流发生数据竞争 | racecheck与逐场误差 | 回退到explicit同步 |
| 固页注册失败 | 所有rank统一状态 | 明确记录后回退pageable |
| A800与本地排序不同 | A800配对矩阵 | 不外推本地性能结论 |

## 📚 内部依据

- [GPU性能优化计划](../../../documents/ASTR_GPU_PERFORMANCE_OPTIMIZATION_PLAN.md)
- [P3基线报告](../../../documents/ASTR_PHASE_P3_BASELINE_REPORT.md)
- [P3实施计划](../../../documents/ASTR_PHASE_P3_IMPLEMENTATION_PLAN.md)
- [A800 TGV缩放矩阵计划](../../../documents/ASTR_A800_TGV_SCALING_MATRIX_PLAN.md)
- [GPU并行架构维护文档](../../../documents/maintenance/parallel-gpu-architecture.md)
