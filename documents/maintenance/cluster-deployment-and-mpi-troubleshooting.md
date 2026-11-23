# ASTR 超算部署与多节点排障经验

更新日期：2026-11-20。覆盖中科算联 A800 和曙光海光 DCU 两套已使用的环境。本文记录排查顺序和实际案例，不是机器配置清单，也不替代站点规定。两平台的账号、工作根目录、运行库和作业授权分别管理；项目文件、结果、临时目录和私有依赖留在各自获准的工作根目录内。本文不固化个人账号、绝对路径或节点名单。

## 先判断故障在哪一层

| 首个异常 | 优先检查 | 不要先做什么 |
| --- | --- | --- |
| 作业还在 `PENDING`，没有 rank 启动 | 分区、资源请求、优先级、是否绑死节点；以调度状态和分配记录为准 | 不把等待时间算成程序耗时，不重复提交 |
| 编译找不到 `hdf5.mod` 或链接缺符号 | 编译包装器、Fortran ABI、并行 HDF5 及 Fortran/HL 库 | 不靠拷贝一个旧 `.mod` 或临时混入另一套 MPI 解决 |
| 计算节点尚未进入 ASTR，MPI/PMIx 就报错 | 启动器、实际 MPI/PMIx、临时目录、rank 映射；先做同环境的最小 MPI 启停 | 不先改求解器或把错误归为 GPU profiling 权限 |
| 写完 `grid.h5` 后、首步前 DCU VMFault | 核对上一条成功的初始化操作、报错 rank、`q` / `dxi` 上传和首次 halo 的先后 | 不用跨节点带宽解释尚未开始的 halo 通信 |
| 四步都走完，但 MPI 退出非零 | 看 IPC detach、MPI 清理和最终退出码；检查点存在不代表正常结束 | 不把失败样本纳入性能结果 |
| 检查点、切片或原位输出时报错 | 区分 HDF5 写/关闭、目录发布、文件系统配额、渲染依赖及单卡显存峰值 | 不把最后一条 PMIx 信息或输出日志前缀当根因 |
| 已推进多步，首次出现负密度、负内能或 CFL 异常 | 查首个失效步、RK 阶段、位置、局部 CFL 和此前的修复次数 | 不删除正值检查来掩盖数值不稳定 |

## 1. 构建、链接与上机预检

**分开验证登录节点和计算节点。** 登录节点做根 CMake 构建、依赖解析、输入审核、最小 CPU/MPI 与实际输出目录的 I/O 测试。无 GPU 的登录节点不能验证驱动、设备绑定、GPU-aware MPI 或 EGL。计算节点先跑单 rank 设备初始化，再跑目标 rank 布局和代表算例短窗，生产任务沿用同一二进制与运行栈。

| 核对项 | A800 | DCU |
| --- | --- | --- |
| 编译器与目标 | CUDA Fortran 路径核对 NVHPC 和实际 `cc80` 编译/链接；根 CMake 可用 `ASTR_CUDA_ARCHITECTURES=80` | GNU Fortran/MPI 配合 DTK GPUFusion；核对 `ASTR_GPU_CPP_BACKEND=gpufusion` 和实际 `--offload-arch`，历史验收为 `gfx906`；不要传 NVIDIA CC |
| HDF5 与 MPI | 检查并行 HDF5 的 Fortran、HL 和压缩依赖；紧致 TGV 部署实测使用无 szip 的 1.14.5，并核对了 NVHPC 数学库 | 编译器、MPI C/Fortran、并行 HDF5 必须匹配；历史验证栈之一为 GNU 12.2、Open MPI 4.1.5、HDF5 1.14.6 |
| 动态装载 | 同时核对数学库、GCC/NVHPC 运行库；需要原位时另查实际 Catalyst/ParaView、EGL 和载入模块 | 单独核对 MPI 的 `mca_pml_ucx.so`、UCX ROCm 模块和 HSA 运行库。主二进制的 `ldd` 不会列出所有运行时插件 |

检查可信的本次构建产物时，`mpifort --showme:command`、`h5pfc -show`（包装器存在时）、`ldd -r` 和 `readelf -d` 各回答不同问题：实际编译器、HDF5 包装器参数、链接解析、ELF 搜索路径。还要记下根 CMake 的实际 cache、运行环境和二进制 SHA-256。在 A800，曾需要显式指定 HDF5 高层库才能完整链接；关闭大场输出也不会让 HDF5 链接依赖消失。

**升级 UCX 时不要只改 `LD_LIBRARY_PATH`。** DCU 旧 Open MPI 的 PML 插件带绝对 `RPATH`，预检发现它仍解析到 UCX 1.18.1。相同版本的 Open MPI 重新链接到独立 UCX 1.22.0 后，才有有效对照。先核对插件的 `ldd -r`，再从计算节点 rank 日志确认实际加载版本；不要覆盖公共库或原部署。

证据：[A800 构建和节点预检](../ASTR_A800_COMPACT_TGV_SUBMISSION_BC7524F_20261101.md)、[DCU 构建与跨编译器恢复，§9.5](../ASTR_MULTI_BACKEND_MIGRATION_PROGRESS.md)、[GPUFusion 构建工具链](../../tests/gpu_backend/gpufusion.cmake)。

## 2. 调度、MPI 启动与设备绑定

- **先区分资源未分配、启动失败和求解失败。** 短测一般交给 Slurm 自动选同档节点。仅在故障与指定节点有关、确实要做同节点对照时固定节点；若要验证另一台，只排除已测节点即可。曾有一次固定节点的单个启动进程先后报告两个排队编号；调度记录最终确认前一个已取消、后一个仍排队，两者都未分配资源。不要凭编号变动推断重启或成功，更不要在状态未核实前另投。
- **作业控制与日志检查分开。** DCU 上查询、提交、取消等操作按明确授权执行；已有资源授权不自动扩大到别的作业。改派节点先确认证据对应同一申请、尚未开跑，并核对旧分配已释放。排队预估和试提交不是已运行证据。
- **在每个 rank 记录映射，不能只看 rank 0。** 记录 hostname、全局及本地 rank、实际可见设备、PCI BDF、显存档位、CPU 绑定和拓扑。四卡要证实四个 rank 分别占四张物理卡，不能依靠全局 rank 取模或脚本里固定的 `CUDA_VISIBLE_DEVICES=0,1`。开发机用两卡共享运行四个 rank 可查 MPI 正确性，不能当四卡计时。
- **PMIx 权限错误先独立复现。** A800 的 `dstore_base.c:237 NO-PERMISSIONS` 在纯 MPI hostname 启停也出现；仅隔离临时目录无效，`PMIX_MCA_gds=hash` 在该匹配环境消除了诊断。另有任务在程序已退出 0、输出已完成后才打印 PMIx 消息。处理要看出现阶段、主进程返回值和实际产物，不全局屏蔽 PMIx。对可疑的启动目录，核对写权限、共享文件系统及站点对本地临时目录的要求。

证据：[A800 AIR5 启动与运行时预检](../ASTR_A800_AIR5_SBLI_31c01ae_REVIEW.md)、[A800 小网格映射和预检](../ASTR_A800_COMPACT_TGV_LOCAL_PREFLIGHT_RESULTS.md)、[固定节点申请取消日志（本地归档）](../../results/dcu_tgv_vm_fault_df8064f_diagnostic/ucx122_mpi_io_pair/launch_cancelled.log)。

## 3. 多节点通信：先证明走了哪条路

| 路径 | MPI 收到什么 | 如何证明 |
| --- | --- | --- |
| `pinned + TCP/IPoIB` | 页锁定主机缓冲，经 TCP 使用 IPoIB 接口 | 实际节点可达、MPI 使用指定 TCP 接口；接口名为 `ib0` 不等于走原生 IB RDMA |
| `pinned + IB RDMA` | 页锁定主机缓冲，由 UCX 经 IB HCA 传输 | 核对 HCA/端口、实际消息长度的 RDMA 协议与端到端数据 |
| 节点内设备缓冲 MPI | MPI 直接收到设备指针；可走 `cuda_ipc` 或 `rocm_ipc`，也可能内部转主机 | 核对实际指针类型、协议、消息大小和正常退出。A800 作业 460370/460439/460441 有对应负载和 `cuda_ipc/cuda` 证据 |
| 跨节点设备缓冲 MPI | MPI 直接收到设备指针，不保证网卡直接读显存 | 用具体消息尺寸检查实际 GPU 源/目的地、直接 RDMA 或 host-fragment 协议，再做求解器回归 |

**不能照搬前一次作业的网口。** 每次分配重新记录节点间路由、IPoIB 或 IB 端口、HCA 的 PCI/NUMA 位置，再检查各 rank 的实际选择。16 卡 DCU 历史测试先确认所有节点间 MPI 可达，再做代表 halo 消息的主机/设备负载检查。小 ping、`ib0` 存在、设备 MPI 能力查询通过、网卡标称速率或 UCX 列出一个候选传输，都不能单独证明正式算例跨节点走了那条路径。

**避免把不同内存错误混在一起。** `process_vm_readv ... Bad address` 指向主机 CMA/指针类型路径；曾在本地 HPC-X CUDA 通道通过显式模块路径和 `UCX_MEMTYPE_CACHE=n` 修正，但不能据此断言 DCU 同因。DCU 上应另比主机/设备真实负载，按失败相位核对 UCX 模块、协议表和 rank 日志。当前 ROMIO 修复不等于跨节点设备 MPI 已验收，也不要把已验证的主机中转称为 NCCL、设备直连或静默回退。

证据：[A800 CUDA-aware MPI 资格及协议日志索引，§8.9](../ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md)、[CMA 实例与限定修正，§9.3](../ASTR_MULTI_BACKEND_MIGRATION_PROGRESS.md)、[DCU 16 卡 TCP/IPoIB 数据](../../results/dcu_tgv_strong16_308_df8064f_verified_net/collection_128472756/strong_scaling.json)、[DCU 16 卡 pinned + IB RDMA 数据](../../results/dcu_tgv_rdma308_df8064f_paired/collection_128475688/pinned_per_rank_ib_rdma.json)（后两项为本地归档）。

## 4. DCU 256³ TGV：H2D VMFault 与 IPC 解映射

这是两种失败相位。**启动 VMFault** 在 rank 0 上传 `q` 或 `dxi` 时出现，已定位样本早于首个 halo 接收；跨节点曾复现，原故障节点换物理卡后错误仍跟着 rank 0。**IPC 解映射错误** 可以在四步后出现在 `MPI_Finalize`；在 UCX 1.22 关闭 IPC 句柄缓存后，rank 0 在第一步前的传输完成阶段仍有三次 detach 告警，尽管退出码变为 0。因此仅升级 UCX、关闭 IPC 缓存或延后输出不是已验证的修复。

**已验证配置：** 未改动的 `df8064f`，DTK/GPUFusion 26.04，私有 Open MPI 4.1.5/UCX 1.22.0，显式指定 `OMPI_MCA_io=romio321`。保持 device 缓冲、`rocm_ipc` 和正常网格/检查点 I/O。

- 作业 128542295：同一节点和二进制，OMPIO 在启动网格文件写出后、首步前触发 VMFault；ROMIO 正常完成。
- 作业 128542494：另一台同档节点两次独立 device 与一次同输入 pinned 参考均成功。累计三个 device 启动均走完四步，CFL 最大值 0.13048654，未见 VMFault 或 IPC 解映射告警，五个守恒分量对 pinned 的最大缩放绝对差均为 0。
- 这确定了一个可用的运行配置，并未确定驱动内部根因，也未验证 ROMIO + 旧 UCX 1.18.1、跨节点设备直连或生产长跑。若复现，提供节点/作业窗口、rank PID、设备 BDF、fault VA、最后进入的 API、内存区间与运行库版本，由管理员核对驱动/固件及内核故障记录。

**308³ 扩展经验：** 作业 128543441 的短测通过后，四卡正式配置仍在启动时崩溃。同节点对照发现，保留原网格及各 rank 的读取子块，只将启动网格从 MPI-IO 改为独立只读即可绕开该配置的故障；独立复现程序本身未触发，不能把根因归为 HDF5 单独缺陷。源码通过显式 `ASTR_GRID_HDF5_READ=independent` 提供该路径，不改变默认、halo 或检查点写入。作业 128548233 的两次无插桩四卡与 pinned 终态差为 0；作业 128548513 用同一二进制通过 8/16 卡短验收及 1/2/4/8/16 卡各三轮计时。这里的跨节点结论来自本批实际分配和消息尺寸，不外推至其他环境或生产长跑。[308³ 诊断与验收（本地归档）](../../results/dcu_tgv_308_readgrid_diagnostic/INCIDENT.md)。

另有一项独立 DCU 限制：固定冷态 AIR5 夹具的 `general` 化学核出现 VMFault，显式 `specialized` 在定义的短测范围内通过。不要把这类化学核故障归到 TGV I/O，也不要把切模式写成自动回退或物理验证。[AIR5 已知问题](../ASTR_MULTI_BACKEND_MIGRATION_PLAN.md) · [TGV 完整诊断、配置和原始日志（本地归档）](../../results/dcu_tgv_vm_fault_df8064f_diagnostic/PLATFORM_INCIDENT.md)。

## 5. I/O、检查点和内存

| 实际问题 | 经验 |
| --- | --- |
| A800：HDF5 写完，但 NFS 目录发布报 `renameat2(RENAME_NOREPLACE)` / `EINVAL(22)` | 这是目录原子发布能力，不是 HDF5 库缺失或数值错误。488899 用实际输出路径复现；经批准的目录发布修订及单/双份保留预检通过。新平台要在目标文件系统测一次小批次的写、关、发布、保留和重启，不在其他挂载点代测 |
| A800：双 rank 检查点写出卡住 | AIR5 登录节点同输入对照中，默认 MPI-IO 超过 6 分钟仍未进入推进；选择 `romio321` 后完整单步约 59.02 s。该结果不定位底层变慢原因，也不是 RK 加速比 |
| A800：原位产品小网格可用，原尺寸首帧失败 | 489559 是 Viskores 分配触发设置的附加显存预算，不能写成 CUDA 物理 OOM。先分离程序预算拒绝、显存/驱动真实分配失败、Slurm 主机内存杀进程和磁盘配额，再按产品测峰值；需要时先用已验收的二维截面、壁面量，把高显存等值面/流线留给另测 |
| A800：同一求解器能运行，但新渲染产品被拒绝 | 489552 的首个致命错误是平台加载了旧设备 Mapper，裁剪能力与当前源码不一致，不是 MPI 或 HDF5。核对实际加载的插件、版本与产品参数；设备绑定和 EGL 探针通过也不能代替具体产品探针 |
| DCU：NVHPC 写出的检查点在 GNU 环境读成开关变化 | 既有格式里 Fortran `LOGICAL` 真值曾用 -1 或 1。控制元数据改用固定 32 位整数 0/1，读取时接受已验证的旧 -1，并要求真开关变化仍被拒绝。不能据此宣称所有旧格式、精度或后端可精确互通 |
| 多节点：估算显存足够，仍可能不过资源门槛 | 显存按**每物理卡**检查，主机内存按**每节点**检查。预算包含 halo、度量、工作区、统计、MPI 缓冲、输出及渲染峰值；四卡显存不是一个内存池。先以同分解关闭/开启产品的短测区分求解器和产品增量，不把观测采样当作任意瞬态峰值上界 |

保持已发布的完整检查点和必要资源/辅助状态，另起目录续算，不覆盖原结果。精确续算按当前检查点合同验证；非精确恢复允许重建规定状态，但不能表述为逐值续算。可选后处理产品失败与检查点发布失败是不同合同，不能把关闭后处理当作掩盖写盘故障的办法。

证据：[NFS 发布和 M12 原位产品，§15、§17](../ASTR_M12_C13_GPU_INSITU_INTEGRATION_PLAN.md)、[实际文件系统预检](../../tests/gpu_validation/run_checkpoint_publication_preflight.py)、[输出与续算合同](../../USER_GUIDE.md)、[跨编译器布尔恢复，§9.5](../ASTR_MULTI_BACKEND_MIGRATION_PROGRESS.md)。

## 6. 数值失败和性能记录

- **输入错误先于内核调试。** 历史 CRLF 文本曾让 NVHPC 读出的 `flowtype` 或网格文件名带回车，表现为初始化未执行、CFL 为零或 Inf。先核对输入 LF、实际 `use_gpu`、网格和剖面路径，然后再看数值核。以当前 `input.case`、`input.run`、`input.output` 合同为准，不照抄历史启动文件。
- **数值退出保留首个失效现场。** A800 作业 489914 的首个记录是壁面吹吸区附近 MP7-LD 界面的负密度；之前检查点已成功发布，不能归咎于 I/O 或 PMIx。另有固定步长续算在 CFL 超 1 之前就已有非零 crashfix。自适应步长能限制已接入的对流/声学 CFL，不保证消除负壁面状态、边界缺陷或未接入的黏性/化学限制。
- **性能计时和诊断分开。** 复验先核对同一二进制、输入、分解、物理推进量和数值结果，然后关闭大场、图片、检查点及高频诊断，保留必要标量。分开报告 RK、完整步、I/O 和初始化；用各 rank 在窗口内累计耗时的最大值，不把分项重叠时间相加，也不要混用 RK 与完整步。每组独立预热、重复并报告中位数与范围。少量统计仍是 I/O。
- **性能工具失败不等于求解器失败。** Nsight Compute 的 `ERR_NVGPUCTRPERM` 属于计数器权限；按站点权限由管理员处理，不改系统设置。DCU 曾因 `libroctracer_tool.so` 缺失导致 rocprof 无轨迹而 ASTR 正常结束；另行检查实际工具链和可用的 hipprof。在正常计时中不要加载 profiler，不用 GPU 利用率单值推断加速或通信占比。

证据：[M12 负密度定位](../ASTR_M12_LONG_RUN_FAILURE_DIAGNOSIS.md)、[固定步长的历史失败，§1.2](../ASTR_ADAPTIVE_TIMESTEP_PLAN.md)、[DCU 计时与工具链记录](../ASTR_GPU_PERFORMANCE_OPTIMIZATION_PLAN.md)、[分层验证与 Nsight 排障](validation-and-troubleshooting.md)。

## 复用的最短流程

1. 冻结算例输入、二进制、运行库环境和版本；在登录节点核对链接，在实际输出目录做一次最小发布/恢复。
2. 计算节点先证实设备绑定、MPI 启停。目标跨节点路径先做真实大小的主机或设备 MPI 收发校验，再在同一分配内跑代表算例短窗。
3. 短窗同时确认首步、数值状态、实际通信协议、输出和退出。发生故障只改一个因素，保留失败日志。通过后才扩大卡数、网格或生产时长。
4. 上报或交接保留：作业号与时间窗、节点和 rank/设备/HCA 映射、源码及二进制身份、MPI/HDF5/UCX/驱动版本、生效环境、首个错误、退出状态、最后一个完整检查点和对照记录。
