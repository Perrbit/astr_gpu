---
title: M12 MP7-LD NSCBC Local Performance - Plan
type: perf
date: 2026-10-12
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# M12 MP7-LD NSCBC Local Performance - Plan

## Goal Capsule

- Objective: 降低本地中等网格 M12 算例的完整推进耗时，给出可复现的单卡、双卡成本及瓶颈解释。
- Means: 采用已选择的局部 x 加密网格，先测量 MP7-LD、通量保正及 NSCBC 的组合成本，再逐项优化计算与通信，见 KTD1 至 KTD6。
- Authority: 用户明确决定及 `AGENTS.md` 优先。本计划不授权修改物理模型、数值算法或现有外部任务。
- Execution: 当前 `feature/gpu_dev`、单会话、本地双卡。不创建 worktree，不操作 Git，不连接或干扰超算任务。
- Stop: 数值门槛失败即停止该候选的性能验收。需要改变离散、保正判定、吹吸、步长或重启契约时提交人工决策。U6 仅按 R9 的条件推进。
- Handoff: U1-U5 本地范围完成，U6 未触发。正式计时和优化结果存于 `tests/gpu_validation/out/m12_mp7_nscbc_perf_20261012_1355/`。
- Acceptance amendment (2026-10-12): 用户批准放宽性能收益门槛。有明确降成本机制、通过数值及内存检查的候选可保留为可选实现，不再要求整步收益达到固定比例。该修订优先于初始 goal 文本中的 5%、四轮胜出和不退化筛选条件，不放宽科学正确性要求。

### Execution Status (2026-10-12)

证据根目录为上文 `m12_mp7_nscbc_perf_20261012_1355/`，最终验证二进制为
`device_prefix_fix/bin/astr`，SHA256
`20f7da833963751e5c9905511b2b1fd324f1eab0302c0794827b13e7963a2f78`。

| 单元 | 当前状态与证据 |
|---|---|
| U1 | 完成。原初场十步通过；当前二进制合法重放到 step880，与先前原点状态/RNG 逐位一致。W0 测量20..119，无限制事件；W1 预热880..899、测量900..999，共922次 rank-node/stage 事件，首个激活步928，最大 CFL=0.26613229852191767。 |
| U2 | 完成当前成本归因。参考/组合的 W0 和激活窗各八步 NSYS、每配置两 rank，以及首三个 MP 面核的 NCU 已采集。`nsys_aligned_summary.json`、`ncu_metrics.json` 保留原报告来源。 |
| U3 | 完成布局与同阶段正负分裂缓存候选。重构及保正代数不变；激活窗百步最终场/RNG/统计逐位一致，一个激活步36份阶段快照一致。缓存每 rank 新增300533040字节；两候选默认仍 directional/off。 |
| U4 | 完成受影响短测。MP 专用面交换改传完整设备容量缓冲，以 count 选择前缀，修复 NVHPC 在 MPI 前生成主机临时拷贝的问题。device-aware/UCX 的 NP1 与 NP2 x/y/z 十步场/RNG/统计及界面检查通过，七份当前候选 rank memcheck 零未抑制错误。y/z 短窗未激活限制器，不扩称激活矩阵或生产准入。 |
| U5 | 完成。W0/W1 各五轮交错，20步预热+100步测量。RK 中位耗时0.722261→0.548940、0.722818→0.545877 s/步，降低24.00%/24.48%；完整推进步降低23.54%/24.02%。NP1 一组成本参考 RK 1.245502→0.989601 s/步，不当作严格强扩展或五轮证据。原始慢轮保留，数值、内存、资源及71项相关测试通过，见 `summary.json` 与 `PERFORMANCE_REPORT.md`。 |
| U6 | 未触发。当前四配置六个对齐周期内，未被设备工作覆盖的同步等待占比均不超过0.408%，不足以支持本轮扩大 dependency 准入。保留 explicit。 |

设备通信时间线每 rank 每周期记录48次 P2P 拷贝、102469248字节；
D2H 由233712628降到2356字节，H2D 由103026072降到556824字节。
这支持本机设备间传输已实际发生，不证明其他平台的物理传输协议。
MPI 暴露等待与暴露同步可能重叠，不能相加。缓存把部分重复计算转为
显存流量，不能用单核或 profiler replay 时间代替完整步收益。

详细报告为证据目录的 `PERFORMANCE_REPORT.md`。初始 userdefine 构建错误、
NVHPC 切片崩溃及 ob1 外部 API memcheck 失败均保留，不列为通过证据。
`final_np1_verify10` 因协调器中断缺少资源结束记录，已由
`final_np1_verify10_qualified` 替代，未补造验收记录。

正式20次 NP2 运行每窗口统计字节及事件数一致，无 HDF5 输出。
每 GPU 进程显存峰值参考2.811、组合3.076 GiB；W1 末轮参考 RK=0.827437，
明显高于前四轮，未删除、未重测，原因未确定。保留完整范围及 MAD。
只推荐在本轮本机无渲染配置中显式选择组合，所有原默认保持不变。
不由本地短测宣称 A800、稠密激波限制区、生产 SBLI 或新同步模式已验收。

---

## Product Contract

### Summary

针对 M12 平板边界层，在保留既有 MP7-LD、通量保正和 NSCBC 组合的前提下，完成本地性能诊断、热点优化和双卡通信对照。
先建立该组合在 476 网格上的新基线，不把其他边界、其他网格或历史程序的耗时直接当成优化前时间。

### Problem Frame

已有 450 网格的 NSCBC 加保正长测，但网格改变了吹吸采样，不能直接代表本轮工作负载。
现有 476 网格加保正十步预检采用外推边界，且限制器未激活，也不能证明本轮组合的成本。
优化需区分 MP7 重构、保正固定开销、激活后的搜索成本、边界处理和 MPI 等待，不能仅凭某个 kernel 变快推断整步收益。

### Requirements

**Case**

- R1. 采用 476×130×96 个区间，即 477×131×97 个全域节点，按 `prepare_m12_local_case.py:REFINED_INTERVALS` 取原始网格节点，不插值原初场或入口剖面。
- R2. 保留表中数值、物性和边界配置，测量期间不改变它们。

| 项目 | 冻结配置 |
|---|---|
| 流动与网格 | `bl`，原 M12 的静态正交拉伸网格，物理范围不变 |
| 参考量 | Ma=12，Re=10000，参考温度 594.3 K，Sutherland 黏度 |
| 对流 | `743e`，`recon_schem=5`，`lchardecomp=t`，`bfacmpld=0.5`，`shkcrt=0.01` |
| 黏性与推进 | `643e`，RK3，FP64，`dt=0.02` |
| x− / x+ | `11,prof` / `22,1.d0` |
| y− / y+ | `41,10.d0` / `52` |
| z− / z+ | `1` / `1`，周期 |
| NSCBC 策略 | `ASTR_NSCBC_FARFIELD_MODE=compatibility`，不切换 `nonreflecting` 或其他特征策略 |
| 保正 | `ASTR_GPU_MP_POSITIVITY=flux`，保持完整 RHS 的低阶基准和共享界面限制系数算法 |
| 吹吸 | `legacy_random`，幅值 0.12，原空间参数及已允许的非零种子映射 |
| 海绵层 | x+ 50 区间、y+ 20 区间，原定义不变 |
| 滤波及修复 | 主体 `lfilter=f`、`lcracon=f`；保留 NSCBC 自身横向滤波 |
| 初始执行模式 | `ASTR_GPU_SYNC_MODE=explicit`，`ASTR_GPU_HALO_TRANSPORT=pinned` |

**Measurement And Correctness**

- R3. 每一组基准/候选必须从同网格、同拓扑、同完整步状态及同随机数状态开始，并具有相同的吹吸调用序列。
- R4. 正式计时窗关闭原位渲染、三维场、切片、中间检查点和逐场验证转储，保留少量统计及必要失败诊断。
- R5. 主指标为包含 RK 首阶段准备、三子步、限制器、边界及必要通信的 RK 时间，另报含固定少量统计的完整步时间。
- R6. 不改变运算顺序的候选须与冻结参考在相同阶段的权威场、限制器状态和随机状态逐值一致，物理节点严格有限且 rho、温度及内能为正。
- R7. 使用已有完整步 CFL 定义检查 CFL≤1，不自动降低 dt 或修改吹吸来让测试通过。
- R8. 保留低阶基准和受限更新的失败检查，不以延迟错误判定、关闭限制器或减少必要 halo 交换取得性能收益。
- R9. U6 已批准作为本轮候选，仅在 U2 证实同步成本显著后实施。原显式同步默认值、旧检查点契约及其他算例准入不因本任务自动放宽。

### Key Decisions

- 采用局部 x 加密中等网格，而非复用 450 网格成熟场，以保留吹吸区原 x 向分辨率。Governs R1, R3. (session-settled: user-directed; chosen over 450x130x96: retain the original streamwise forcing-band resolution.)
- 优化求解器推进，不同时调试可视化参数。Governs R4, R5.

### Scope Boundaries

本轮不做 CPU 加速比、不做 A800 性能推断、不声称生产 DNS 或转捩物理验收。
不引入混合精度、紧致格式、主滤波、其他保正算法或新的 NSCBC 物理条件。
不重构检查点格式，不修改只读源算例 `../M12_C13`。
CUDA Graph、跨多个 RK 阶段的大范围融合及生产规模可视化留待瓶颈证据支持后另议，不预先实现。

---

## Planning Contract

### Key Technical Decisions

- KTD1. 优先复用 `prepare_m12_local_case.py` 的网格取样与 `run_m12_boundary_compare.py` 的边界配置，在独立性能驱动中关闭其默认最终场和切片输出。落实 R1、R2，不复制网格生成逻辑。
- KTD2. 冻结当前工作区源码快照、构建缓存、可执行文件、输入及运行环境。HEAD 当前为 `2afcd49`，但存在未提交改动，不能仅用 HEAD 描述基线。
- KTD3. 采用 W0 未激活和 W1 激活两类窗口，分别报告固定开销与激活开销。W1 必须观察到实际限制事件，不能用初始十步代替。
- KTD4. 主配对采用 NP=2、`2,1,1`。NP=1 用于单卡成本参考，NP=2 的 y/z 分解仅在通信优化后做短正确性检查，不默认展开大矩阵。
- KTD5. 先用现有整体计时和 Nsight Systems 归因，再对排名靠前的最多三个 kernel 做 Nsight Compute。现有 `ASTR_GPU_PHASE_TIMING` 只准入周期 TGV，不能直接用于本算例。
- KTD6. 先优化保持运算顺序的复用、布局和通信调度。每次仅改变一个因素，kernel、整步和扩展收益分开判断。
- KTD7. 依赖同步扩展按 R9 纳入条件性候选，不是切换现有环境变量即可使用的能力。`gpu_runtime.cuf` 的周期 TGV 准入及 `output_runtime.F90` 的显式同步要求均需纳入 U6 设计。(session-settled: user-approved; chosen over retaining explicit synchronization for every candidate: evaluate unnecessary synchronization without widening exact-restart support.)

### Workload Windows

W0 从原初场建立，经 20 个完整步预热，测量后续 100 步并记录限制器激活情况。
如果它已激活，不将其标为未激活窗口，而报告实测类型。

W1 从本轮正确性通过的 476/NSCBC/flux 路径发展获得。
先按最多 2000 个完整步准备，找到存在限制事件的固定 100 步窗口，并保留该窗口前 20 步的完整步检查点供同二进制复用。
准备阶段只保存必要的基准检查点，不保存场序列或图片。
若准备窗内没有限制事件，报告该事实并询问是否延长，不提高扰动幅值、不修改初场来强迫激活。
该 2000 步是首次准备规模，不是生产收敛目标。

新二进制不能直接读取旧可执行文件指纹绑定的检查点。
优先在同一验证二进制内选择参考/候选实现，并重新生成它自身的窗口检查点。
跨二进制或跨拓扑无法合法恢复时，从相同原初场、种子映射和调用顺序确定性重放到同一基准步，并核对窗口初始状态。
不编辑检查点、不用调试器改指纹，不默认随机状态可跨拓扑等价。
NP=1 与 NP=2 若不能保持同一吹吸实现，只作各自同拓扑配对，不给出严格同工作负载的强扩展效率。

### Cost Attribution

| 路径 | 首轮检查对象 | 尚未证实的性能假设 |
|---|---|---|
| MP7-LD | `solver_gpu.cuf:mp_ld_interface_flux`、`mp7ld_gpu`、`mp_faces_kernel` | 重复加载、特征变换和局部数组引起计算或寄存器开销 |
| 保正搜索 | `mp_ratio_kernel`、`mp_face_budget` | 六面候选检查、最多64次二分及线程分支可能占用显著时间 |
| 保正存储 | `high_d`、`delta_d`、`ratio_d`、RHS 修正 | 三方向五分量 high/delta 存储可能产生较高显存带宽压力 |
| 通信 | `exchange_mp_planes`、比值 halo 刷新、解场/黏性 halo | 打包、host staging、MPI 等待及强制同步可能串行化 |
| 小量归约 | `check_mp_failure`、逐阶段限制节点计数、NSCBC Mach 归约 | 小数据回读及集合通信可能形成 GPU 空闲间隙 |
| 边界与黏性 | `boundary_gpu.cuf`、`mainloop_gpu.cuf`、黏性核 | 横向滤波、物性刷新及全域遍历可能构成固定成本 |

这些是测量对象，不是已经确定的瓶颈。
NSCBC 的 Mach 归约在不同策略中用途不同，只有证明选定策略不消费其结果且无必要副作用后才考虑省略。
不能因为主滤波关闭就删除 NSCBC 横向滤波，也不能以缓存复用混用滤波前后的物性。

---

## Implementation Units

### U1. P0 - Freeze The New Baseline

**Goal:** 建立 R1、R2 组合的可重复性能驱动和合法基准状态。

**Dependencies:** 无。实现前先确认本地无其他计算负载，不终止用户进程。

**Files:** `tests/gpu_validation/prepare_m12_local_case.py`、`run_m12_boundary_compare.py`，新增 `tests/gpu_validation/run_m12_mp7_nscbc_performance.py`、`test_m12_mp7_nscbc_performance.py`；必要时仅为窗级计时扩展 `src_gpu/gpu_runtime.cuf`。

**Approach:** 复用 KTD1，按 KTD2 冻结参考。先完成10步原初场检查，再建立 KTD3 的窗口；输出驱动记录实际完整步身份、种子和后端，不沿用旧脚本的验证转储开关。

**Test Scenarios:**
- 476 网格的吹吸区保留原节点，出口海绵位置不变，边界确为 `11/22/41/52/1/1`。
- 同拓扑两次相同输入/种子推进到窗口起点，权威状态及随机状态一致。
- 计时目录不产生三维场、切片、图片或中间检查点，预热不计入结果。
- 非法重启指纹被拒绝，不把旧外推或无保正的检查点静默转换成新基线。

**Verification:** 10步有限性、保正和 CFL 检查通过；窗口可复现且激活类型明确。此阶段不宣布加速。

### U2. P1 - Attribute End-To-End Cost

**Goal:** 给出 W0/W1 的耗时分解，确定值得优化的三个主要对象。

**Dependencies:** U1。

**Files:** `src_gpu/gpu_runtime.cuf`、`mainloop_gpu.cuf`、`mp_positivity_gpu.cuf`、`boundary_gpu.cuf`，U1 的驱动和测试；复用 `tests/gpu_validation/summarize_gpu_rk_performance.py`。

**Approach:** 优先用现有 RK、每 rank RK 和完整步计时。对固定窗口采集5至10步 Nsight Systems，对主要核少量定点采集 Nsight Compute；编译保持优化并提供行信息，不使用禁用设备优化的调试构建作性能数据。
需要标注时优先增加不插入设备同步的 NVTX 区间，不直接扩大 TGV 专用阶段计时准入。

**Test Scenarios:**
- RK prepare 与 integration 均计入总时间，多个 rank/阶段记录可按身份完整配对。
- 标注开关不改变权威场，profiling 运行与正式无 profiling 计时明确分开。
- GPU 空闲、MPI 等待、内存传输及 kernel 时间不重复相加为虚假的总耗时。

**Verification:** 每个候选有实际成本占比、源码位置和收益上限依据。缺少计数器权限时报告限制，继续使用时间线，不重复失败的完整采集。

### U3. P2 - Optimize Measured Compute Hotspots

**Goal:** 降低已测热点成本，保持 R6、R8。

**Dependencies:** U2。

**Files:** `src_gpu/solver_gpu.cuf`、`mp_positivity_gpu.cuf`、`boundary_gpu.cuf`，新增 `tests/gpu_validation/test_m12_mp7_performance_contract.py`；必要时为窄设备探针扩展 `src/CMakeLists.txt`。

**Approach:** 按 U2 排名最多先做三个单项候选。优先同阶段物性/度量复用、连续读写、方向布局、临时数组寄存器压力及合法的 kernel 合并。
检查 `src/constdef.F90` 的常量定义，但不把除法改乘倒数当成必然逐值等价的重写。
不减少二分次数、不改变安全退让、不将失效候选截断为零。

**Test Scenarios:**
- 未激活界面 RHS 逐值不变，激活界面的 theta、通量和最终更新与参考一致。
- 同时覆盖近零内能和非有限中间运算，保持原失败行为。
- 原固定状态的单阶段、10步及100步配对通过，再测候选耗时。

**Verification:** 通过 Verification Contract；kernel 收益与整步收益分别记录。运算顺序必须改变才能继续的候选暂停，提交独立误差门槛，不自行放宽 R6。

### U4. P3 - Optimize Two-GPU Communication

**Goal:** 减少已测串行通信成本，保持界面一致性及限制预算。

**Dependencies:** U2；组合测试在 U3 候选结论后进行。

**Files:** `src_gpu/mp_positivity_gpu.cuf`、`halo_exchange_gpu.cuf`、`halo_transport_gpu.cuf`、`device_mpi_transport_gpu.cuf`，U1/U3 的测试。

**Approach:** 先比较 pinned 与已具备代码路径的 device-aware，再按证据评估 pinned-pipeline 或专用面交换的非阻塞调度。
保正模块已经预分配 host/device 缓冲，不假设它每步分配 pinned 内存。
高阶通量/delta 的共享面一致化、邻节点 theta 取最小值以及最终 halo-only 刷新不能省略。
与独立内点计算重叠时明确发送缓冲完成、接收完成和使用者依赖，不直接复用原缓冲或修改边界调用顺序。

**Test Scenarios:**
- NP=2 x/y/z 分解各做短参考/候选同拓扑场比较，包括限制器激活状态和物理边界邻域。
- 同一共享界面两侧的高阶/delta 通量及最终 theta 一致，halo 刷新不额外平均节点。
- MPI 能力查询通过后仍进行真实设备缓冲交换验证，不把 CUDA-aware 编译支持当成该路径已验收。
- 只减少统计日志的候选单独计时，必要失败归约仍在消费错误状态前完成。

**Verification:** R6 至 R8 通过；报告通信量、等待及整步变化。后端失败保留证据和 pinned 参考，不静默回退后继续宣称 device-aware 成功。

### U5. P4 - Close The Performance Evidence

**Goal:** 给出可复现的最终建议配置及边界说明。

**Dependencies:** U3、U4；U6 若按 R9 实施，纳入相同验收。

**Files:** U1 驱动及测试、`documents/ASTR_GPU_PERFORMANCE_OPTIMIZATION_PLAN.md`，本轮结果目录中的 `summary.json` 和性能报告。

**Approach:** 按下文统一口径测五轮，交错参考与候选顺序。正式结果无 Nsight、Sanitizer 或逐场转储开销。对最终配置完成100步同状态比较和受影响路径的有界内存检查。

**Test Scenarios:**
- 比较器拒绝混合网格、边界、激活类型、步数和计时范围的结果。
- 单卡装不下时如实标记缺失，不缩网格后计算单/双卡扩展效率。
- 整步无收益的单核优化不成为默认，不删除仍有跨平台潜力且已通过正确性的明确可选实现。

**Verification:** 满足 Definition of Done；失败或中性结果如实汇总，不用最优单轮替代总体结果。

### U6. Conditional - Admit Dependency Synchronization

**Goal:** 在明确依赖边界下减少非必要设备同步，落实 R9、KTD7。

**Dependencies:** U2 证明同步是重要成本，批准范围见 R9、KTD7。U1 至 U4 不因本项的条件性实施而阻塞。

**Files:** `src_gpu/gpu_check.cuf`、`gpu_runtime.cuf`、`mainloop_gpu.cuf`、`mp_positivity_gpu.cuf`、`boundary_gpu.cuf`，必要的 `src/output_runtime.F90` 最小接口改动，U1/U3 测试。

**Approach:** 先列出 MP 面构造、NSCBC 滤波/物性刷新、halo、状态回读和错误检查的生产者/消费者依赖。
仅为 R1、R2 组合设计候选准入，保留 explicit 参考及每次启动检查，保留 MPI 使用设备缓冲、host 回读、失败判定和计时边界所需的完成保证。
若新输出的同步约束需要调整，首轮只考虑不读取/写入检查点的性能运行；旧精确重启的显式同步要求和文件格式不放宽。
候选从原初场重放到同一测量步，发展段不计时。非显式模式的精确续接另行批准。

**Test Scenarios:**
- explicit 与 dependency 同拓扑、同随机序列逐值一致，原其他未准入路径仍拒绝。
- 以同一已捕获状态检查 NSCBC 两次横向滤波之间的 halo 和滤波后物性，不使用过期缓存。
- 设备/主机缓冲在 MPI 完成前不被复用，必要错误仍在状态消费前报告。

**Verification:** 无检查点性能路径通过 R6 至 R8 和内存检查，才将其作为本地可选性能配置。不宣称重启路径已随之准入。

---

## Verification Contract

### Timing

每个固定窗口先预热20步，再计时100个完整步，参考/候选交错重复5轮。
RK 主值为各 rank 的 prepare+integration 在整窗内求和后取 rank 最大值，再除以100；不能将每步 rank 最大值之和与最大 rank 的窗总值混为同一口径。
启用 `ASTR_GPU_RANK_RK_TIMING=1` 获取每 rank 数据，`ASTR_GPU_COMPLETE_STEP_TIMING=1` 的现有输出仅为 I/O rank，本轮完整步窗口另做一次窗级最大 rank 汇总。
不在每个 kernel 或每步增加计时专用 MPI barrier；必要的窗口起止完成保证保持一致。
统计和日志频率须在参考/候选间一致。

报告每轮原始时间、五轮中位数、范围、MAD/中位数、峰值显存和限制事件数。
优化加速比为同拓扑参考中位时间除以候选中位时间。
严格相同工作负载时才计算单卡到双卡的 S2=T1/T2、E2=S2/2，适用限制见 R3、KTD4。

性能结果与代码保留分开判断，不设置最小收益比例或最少胜出轮数。
有明确理论降成本机制且通过数值、内存和资源检查的候选可保留为可选实现，即使本地中性或变慢，也记录其额外成本和可能受益的工作负载，不宣称已经加速。
五轮原始数据、中位数、范围、MAD/中位数、胜出轮数和完整步变化仍全部报告。MAD/中位数超过3%时标记波动，不为了满足筛选阈值反复重跑，也不将微小时间差作为稳定收益。
默认推荐以本机实测完整步收益为依据；未证明收益的候选默认关闭。理论依据不替代实测，也不授权改变数值门槛。

### Checks

| 检查 | 范围与完成条件 |
|---|---|
| 构建 | 经根 `CMakeLists.txt` 构建 Release；记录实际 NVHPC、MPI、CUDA、HDF5、GPU 架构和链接库，不以旧二进制代表新源码 |
| 驱动 | 新增 `test_m12_mp7_nscbc_performance.py` 检查配置冻结、输出关闭、计时窗及比较器拒绝不一致输入 |
| 数值 | R6、R7；仅受影响路径的单阶段、10步、100步同状态对照，不重复无关全项目矩阵 |
| 限制器 | 未激活 RHS、激活 theta/通量、完整低阶基准和受限 RK 状态均检查，节点事件数注明可能包含跨 rank 共享节点 |
| 现有合同 | 运行受影响的 `test_nscbc_physics_contract.py`、`test_m12_boundary_compare.py` 和 `test_summarize_gpu_rk_performance.py` |
| 内存 | 改动设备索引、存储或异步生命周期时做小窗口 Compute Sanitizer memcheck；其时间不纳入性能结果 |
| 环境 | 初始运行检查动态库、MPI/GPU 绑定和 nvitop/NVML 实际占用；不为内核优化反复重做无关平台部署审计 |

---

## Definition of Done

1. R1、R2 新组合的基线成立，窗口初始状态、吹吸随机状态和计时范围可复现。
2. W0/W1 成本有明确标签；若未得到激活窗口，不能声称完成激活限制器性能验收。
3. 完成已测热点的候选实现与对照，逐项记录理论机制、实际收益或代价、适用条件及保留结论。不再以至少一个候选达到5%作为完成条件；无实测收益时交付诊断和可选实现，不虚报加速。
4. 建议配置通过数值及受影响内存门槛，说明单卡/双卡比较是否属于严格同工作负载。
5. 数值或内存检查失败的实验不得启用；有理论依据且正确性通过的候选予以保留。已验证但本地无收益的候选以明确可选、默认关闭方式保留，并记录成本与限制。
6. 结果存于 `tests/gpu_validation/out/m12_mp7_nscbc_perf_<run-id>/`，包括快照来源、环境、日志、原始计时和报告，不加入 Git。
7. U6 未因性能证据触发实施时记录原因，不宣称依赖同步已验收。提交、推送、超算部署或新的自主 goal 都需用户另行要求。

---

## Sources And Decisions

- 网格与随机取样差异：`documents/ASTR_M12_LONG_RUN_FAILURE_DIAGNOSIS.md` 的局部 x 加密及原网格分辨率章节。
- 已完成450网格长测：`tests/gpu_validation/out/m12_nscbc_mp_positive_10000_20261010/long_2860_10000/receipt.json`，仅作历史可行性证据。
- 476网格外推加保正预检：`tests/gpu_validation/out/m12_extrapolation_xrefined_preflight_20261011/run_0_10/receipt.json`，不是本轮 NSCBC 基线。
- 保正算法与交换：`src_gpu/mp_positivity_gpu.cuf` 的 `build_mp_convection_gpu`、`limit_mp_update_gpu`、`exchange_mp_planes`。
- 边界策略与修复范围：`documents/ASTR_GPU_NSCBC_PHYSICS_REPAIR_PLAN.md`、`src_gpu/boundary_gpu.cuf`。
- 计时及同步限制：`src_gpu/gpu_runtime.cuf`、`gpu_check.cuf`、`gpu_phase_timing.cuf`、`src/output_runtime.F90:begin_output_runtime`。

已确认采用 476×130×96 区间网格，并将 U6 纳入条件性候选。是否实际实施由 U2 的同步成本证据决定，边界见 R9。
