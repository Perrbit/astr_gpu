# ASTR 原生 FP32 实施记录

更新：2026-10-22；实施起始：2026-10-20。分支：feature/gpu_dev。源码起点：7a2ded9。
实施依据：ASTR_NATIVE_FP32_DESIGN.md，用户已批准 FP0–FP8 并启用 goal。
仅本地实施，不操作 Git 或超算任务；不设 FP32/FP64 偏差上限。

## 1. 当前进度

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| FP0 | 完成 | 构建期选择、公共 kind、C++ 类型及 CPU/CUDA ABI 探针通过，FP64 基线正常 |
| FP1 | 最小门槛完成 | CPU 非反应调用链构建通过；16³ 显式 TGV 的 NP=1/2 两步推进有效，FP64 原基线不变 |
| FP2 | 最小门槛完成 | 32³ 十步显式矩阵、同状态 RHS、两类通信和 scalar/full 通过；当前 FP64 GPU 回归不变，代表 Memcheck 通过 |
| FP3 | 最小门槛完成 | Ma12、RNG、激活保正/修复通过；历史窗口两种精度、scalar/full生产滤波与同状态修复通过，代表Memcheck零错误 |
| FP4 | 最小门槛完成 | TGV/Ma12完整十步矩阵、两种私有通信及代表Memcheck通过；不代替后续恢复、原位和中等规模计时 |
| FP5 | 最小门槛完成 | 两类32³显式CURVE的几何、法向、同状态梯度及相容恒定场通过；诊断开关不改变推进，代表Memcheck通过 |
| FP6 | 最小门槛完成 | 代表续算、统计段合并、场序列实读及自适应跨精度通过；追加实际渲染/归档/自适应同精度联合续算通过 |
| FP7 | 设备最小门槛完成，兼容入口待决策 | 两种精度的TGV/CURVE两设备管线、平面、壁面均值及续算通过；M12标准入口、只读预览和设备驻留观测通过，代表Memcheck零错误；旧CPU VTK数值路径不满足严格FP32，见§27 |
| FP8 | 性能和长窗后处理完成，紧致FP32长时可信性不足 | 32配置、96次纯RK计时及4个40000步TGV均完成；显式FP32拟涡能峰差0.176%，紧致为31.049%且质量/总能量漂移约22%；分析见§31及独立报告，不据运行正常宣称精度通过 |

## 2. FP0 实际证据

根 CMake 增加 ASTR_NUMERICAL_PRECISION=fp64|fp32，默认 fp64；
ASTR_REAL32 定义仅控制显式 kind，不使用全局实数转换编译选项。
公共模块为 src/astr_precision.F90，C++ 对应类型为 src/astr_precision.h。
五变量 FP32 不与 AIR5/旧 Cantera 构建组合，原 FP64 化学路径未修改。

测试使用 tests/gpu_validation/native_precision_probe.F90/.cpp：

| 检查 | 实际结果 |
| --- | --- |
| FP32 CPU Fortran/C++ ABI | 通过，数值4字节，计时8字节 |
| FP32 CUDA Fortran/C++ ABI及小核算术 | 通过，数值4字节，计时8字节 |
| 默认FP64 CUDA Fortran/C++ ABI及小核算术 | 通过，数值8字节，计时8字节 |
| FP64 CPU主程序构建 | 通过，build_release_restart_cpu |
| FP64 CUDA主程序构建 | 通过，build_native_fp0_fp64_gpu |

先建立修改前的16³周期TGV CPU基线，再运行加入公共模块后的同一输入：
显式643e、黏性及滤波开启、dt=1e-3、2个完整步，无整场文件输出。
flowstate.dat各列差值为0；step0的stage3完整更新q快照逐字节一致。
新增模块没有改变该受测FP64路径。

相同小算例的FP64 CPU/GPU统计对照通过原FP64检查：动能最大绝对差
7.7854389601839102e-15、拟涡能1.8318679906315083e-15、耗散率
2.1412992906588713e-17。GPU末步CFL上界约0.0815540，正常结束。
这些不是原生FP32完整推进的误差或物理验证结果。

原始目录：tests/gpu_validation/out/native_fp32_fp0_20261020/。
其中baseline_cpu保留原二进制、输入和快照，after_cpu/after_gpu保留对照；
两个fp64_*_compare.txt记录统计比较。构建及测试均通过项目根CMake。

验证过程中修正了新增CTest的注册位置：原根CMake在进入src/后才启用CTest，
因此新测试移至根CTest块注册，不修改既有算例测试的语义。
已有CRLF源码采用cr-at-eol口径检查diff，不为清理换行重写无关行。

## 3. 公共类型命名调整

用户暂停整体goal后，单独批准将本轮精度参数rk改名为real_kind。
对应C类型名采用c_real_kind，性能时钟采用clock_kind，所有已迁移声明、
字面量及接口探针同步更新；不保留rk兼容别名，不修改旧mpiio的公开rk。
用户随后将goal恢复为active；完成此次重命名验证后继续既定FP1–FP8，不改变实施范围。
重命名后FP32 CPU/CUDA的native_precision_abi均通过，FP64 CPU主程序重新构建通过。
16³两步FP64 CPU回放的统计量差为0，所查step0/stage3更新q快照仍逐字节一致，
证据为同一根目录的after_rename_cpu及rename_fp64_baseline_compare.txt。
公共状态、物理时间与系数的目标kind检查通过，计时状态仍为FP64。
该次重命名验证时，完整FP32 CPU主程序仍未构建通过，后续结果见下文。

## 4. FP1 CPU 与公共调用链

通过根 CMake 构建 build_native_fp0_fp32_cpu/bin/astr。迁移数值声明、
系数字面量及固定双精度专用内在函数，保持公式、循环和表达式顺序不变。
性能计时单独使用 clock_kind。未修改 CPU 物理模型或数值闭合。

| 调用链 | 类型适配位置 | 保留的计算 |
| --- | --- | --- |
| 主状态与初始化 | commarray、commvar、initialisation、gridgeneration、geom、默认 UDF | 原网格、初态和度量构造 |
| 导数、通量与滤波 | derivative、flux、riemann、filter、comsolver、solver | 原显式及紧致公式、RK 与滤波顺序 |
| CPU 通信 | parallel、astr_precision_mpi | 原 halo/共享节点规则；浮点消息改为匹配 MPI_REAL4/MPI_REAL8 |
| 原位与输出接口 | insitu_fields、insitu_geometry、统计/调度、output_fields、checkpoint_state_io 等 | 先完成编译所需类型闭包；完整功能留待 FP6/FP7 验收 |
| 旧独立 MPI 文件接口 | mpiio | 无实际读写调用；保留其固定 FP64 的 rk 和 real64 名称 |

公共 real_to_word/word_to_real 用同宽整数位型再转换为 int64 元数据，
避免 FP32 直接 transfer 到 int64 的未定义填充位。探针覆盖正负零、±1、
最小正常值与最大有限值；逐位往返通过，FP64 元数据布局不变。
C/C++ 共用 astr_precision.h 的构建精度类型，数值参数 ABI 同步。

实际短测：16³ 周期 TGV、643e、黏性/十阶显式滤波开启、dt=1e-3，
maxstep=1 对应零起算的两个完整步；关闭整场/检查点输出。
NP=2 使用已有 ASTR_FORCE_MPI_TOPOLOGY=2,1,1，非自动拓扑分配。

| 检查 | 结果 |
| --- | --- |
| FP32 CPU NP=1 | 两步正常结束，CFL 上界约 0.0815541；7 份 q 快照有限、密度及内能为正 |
| FP32 CPU NP=2 x-slab | 两步正常结束，14 份 q 快照有限、密度及内能为正 |
| FP32 初态 NP=1/2 | 重组物理节点后逐值一致 |
| FP32 第一步 stage3 更新 q，NP=1/2 | 最大绝对差 1.52587890625e-5，最大缩放差 1.1920928955078125e-7，仅报告 |
| FP32 网格 HDF5 | x/y/z 均为 float32，17×17×17，全部有限 |
| 当前 FP64 CPU 与修改前基线 | flowstate.dat 各数值列差为 0；全部 16 份阶段 q/RHS 快照逐字节一致 |
| FP32/FP64 q 比较 | 7 份同相位快照最大绝对差 8.017324603315501e-5，最大缩放差 4.7743780529074797e-7，仅报告 |
| 快照读取/报告工具 | 9 项单元测试通过，显式声明载荷精度，不按文件大小猜测 |

首个统计采样的 FP32/FP64 最大绝对差：动能 5.69369999758873e-9，
拟涡能 3.662010000793359e-8，耗散率 2.815076100015374e-9。
物理时间差 4.7497000039628534e-11。近零分量的逐点相对差较大，
因此同时保留绝对差和分量缩放差，不把单一相对差解释成整体场误差。
FP32 运行报告 ieee_underflow/ieee_inexact；未出现 invalid/divide-by-zero，
全部已读取 q 及由其重建的内能有限，不隐藏这些浮点标志。

本轮检查曾将同目录的 RHS 当作 q 读取而报头部错误；核对写入接口后，
按 initialized/pre_rhs/post_update 标签选择 q，未改写快照格式或放宽有限性检查。

证据仍位于 tests/gpu_validation/out/native_fp32_fp0_20261020/：
native_fp32_cpu、native_fp32_cpu_np2、fp1_after_fp64_cpu，及 fp1_fp64_baseline_compare.txt、
fp32_cpu_state_validity.txt、fp32_cpu_field_report.txt、fp32_cpu_statistics_report.txt、
fp32_cpu_np2_report.txt。小规模结果不替代 Ma12、CURVE、精确续算或性能验证。

## 5. FP2 当前进展

GPU 常驻数组、共享系数与核的数值类型正在改为 real_kind；同步调整主机/设备
halo 缓冲、MPI 浮点类型、pinned 注册长度及已触及的存储字节统计。
原生 FP32 不启用旧 mixed_workspace 候选，运行标识为 fp32；
FP64 构建保留既有选项和默认行为。内部默认模式名称改为 GPU_PRECISION_NATIVE，
不把编译为 FP32 的程序标为 FP64。

已通过的独立证据：

- FP32 CUDA ABI/小核及 MPI 探针通过；默认 FP64 CUDA 的同组探针也通过。
- FP32 NP=2 主机交换探针覆盖周期及 MPI_PROC_NULL、5/6 层、1/3/6 分量，精确值一致。
- FP32 pinned 内存注册、GPU 往返和释放，1/37/4096/262144 元素均通过。
- FP32 pinned-overlap 主机交换及独立工作回调通过，12 次回调。
- 本地 MPI 提示无 HCA 可供 HCOLL 初始化，实际共享内存通信正常完成；未据此声称 IB 或远程通信通过。

日志为 fp32_halo_pageable.log、fp32_halo_pinned.log、fp32_halo_pinned_overlap.log、
fp32_gpu_precision_mode.log，构建记录为 build_fp2_gpu.log。
完整 FP32 CUDA 主程序已构建通过，16³ 两步 NP=1 GPU TGV 正常结束，
显式同步、scalar 滤波、pageable halo，CFL 上界约 0.0815541。
7 份 q 快照均有限且物理节点密度/内能为正，初态与 FP32 CPU 逐值一致；
阶段最大绝对差 7.62939453125e-5、最大缩放差 4.274508600354049e-7，仅报告。

首轮 NP=2 在第二步开始前的动能统计处失败，不能记为完整推进通过。
GDB 将失败定位到 statistic_gpu::gpu_kenergy_scalar 的临时设备数组释放。
进一步核查发现统计、边界及流向通量归约还保留小写 mpi_real8，
会向 4 字节结果区写入 8 字节，损坏相邻存储。已统一改用 mpi_real_type，
重建后原 NP=2 用例两步正常结束，释放错误不再出现，未改变归约算法或增加容差。
Fortran 大小写不敏感，本轮补查使用大小写无关匹配；非化学 GPU 数值消息中
已无残留 MPI_REAL8/MPI_DOUBLE_PRECISION。CPU 计时、旧固定 FP64 MPIIO 和
未纳入 FP32 的化学消息不在这一替换范围。
NP=2 原始日志为 native_fp32_gpu_np2/run.log，OB1 对照仍复现，
GDB 证据在 fp2_np2_gdb/run.log。首轮阶段快照差值报告不能替代完成和内存安全门槛。

修复后的 14 份 NP=2 q 快照有限且物理节点密度/内能为正。
FP32 CPU/GPU NP=2 采样统计的绝对差：动能 7.45053133077711e-9、
拟涡能 3.907985046680551e-14、耗散率 1.1059455409302178e-9，仅报告。
对应阶段场的最大缩放差仍为 4.274508600354049e-7。
证据为 fp2_np2_after_mpi_fix、fp2_np2_after_mpi_fix_validity.txt、
fp2_cpu_gpu_np2_after_mpi_fix_report.txt。

Compute Sanitizer 的首次设置包含 UCX/Open MPI CUDA 指针查询，不能作为干净的
内存检查。为检查实际 host-staged 路径，诊断启动采用 OB1、self/vader/tcp、
osc=pt2pt，设置 opal_cuda_support=0，并排除 cuda/hcoll/ucc collective。
这只改变本轮诊断命令，不修改求解器通信默认值，也不屏蔽 Sanitizer API 报错。
其中 NP=1 scalar 滤波两步 Memcheck 为 0 errors；NP=2 x-slab full 滤波两步
两个进程均为 0 errors，退出码 0。证据在 fp2_np1_memcheck_host_mpi 和
fp2_np2_memcheck_full_host_mpi_v2。此前失败日志保留，不能把该结果扩展成
CUDA-aware、所有拓扑或所有数值核的内存安全结论。
当前没有完整 CUDA-aware halo 或紧致求解通过结论。

后续32³正式矩阵结果见下文，16³预检不替代该正式矩阵。
输出/检查点精度身份、跨精度恢复、统计段合并及原位 C++ 数值核仍需专门完成，
不因本轮接口类型调整而视作已交付。

## 6. FP2 正式矩阵与收口

执行入口为 tests/gpu_validation/run_native_precision_tgv_gate.py。
FP32、FP64 各自运行 32³ 区间周期 TGV、643e/643e、dt=1e-3、10个完整步。
每种精度覆盖 NP=1 和 NP=2 x/y/z，分别关闭/开启滤波，共8组 CPU/GPU 配对；
另在 NP=2 x-slab、滤波开启时运行 full 工作区，与 scalar 结果比较。
每种精度17次短运行均正常完成，不是性能计时。

首/末完整步的三阶段采集 q、RHS 及 pre_rhs 的 rho/u/v/w/p/T，
核对采样身份、文件宽度、有限性和物理节点正值。新增原始变量快照只在既有
ASTR_VALIDATION_RHS_PREFIX 显式开启时写出，不增加默认生产 I/O。

原始变量有效域为物理节点和轴向模板使用的面 halo。现有 GPU 转换也计算不参与
这些算子的边角存储，那里可因 q 的零填充得到 NaN；本轮确认32³首个快照的
54500个非有限值均在不使用的边角，物理节点与面 halo 无非有限值。
未改求解器、未填零，也不将整个原始变量存储声明为有限。读取器增加对应反例测试，
有效面 halo 或物理节点出现非有限值仍失败。原始变量范围检查不替代后续精确续算验证。

| 证据 | 结果 |
| --- | --- |
| FP32 CPU/GPU，全部8组 | 完成10步；所采 q/原始变量有效，最大 q 绝对差3.204345703125e-4、缩放差3.4695258364081383e-6，仅报告 |
| FP64 CPU/GPU，全部8组 | 维持绝对差≤1e-10检查；实际最大 q 绝对差3.126388037344441e-13、缩放差2.886579864025407e-15 |
| FP32 与 FP64 各自 scalar/full | 所采 q 最大绝对差均为0 |
| 当前FP64 GPU与FP0原二进制输出 | 原13份阶段快照逐字节一致，flowstate.dat各数值列一致 |
| 同状态算子检查 | 不滤波的step0/RK1，CPU/GPU q和轴向输入halo逐位相同；对流/完整RHS与其黏性增量最大绝对差：FP32为1.0766088962554932e-6，FP64为2.1798968879993552e-15 |
| FP32 GPU与匹配FP64 GPU | 全部8组最大 q 绝对差4.2980047260243737e-4、缩放差4.421996441368584e-6，仅报告 |

同状态检查直接复用上述真实场，没有另写推进器。黏性增量定义为所存完整 RHS
减去所存对流 RHS，包含累加舍入，不能称为未累加的原始黏性核输出。
报告同时保留每分量峰值和位置；近零量不使用局部量作为归一化分母。

生产 halo_exchange_contract_test 已按构建精度适配。FP32 用可精确表示的编码
区别rank/分量/节点，NP=2在pageable、pinned、pinned-pipeline、device-aware下
均通过x/y/z、周期/MPI_PROC_NULL、共享节点平均和纯halo复制的逐值检查。
沿用探针的11分量合成消息和1/3/5/6分量字段，不将它解释为FP32 AIR5物理支持。
CUDA-aware 的通过只说明本地该MPI环境下的数据正确性，不说明零主机暂存或性能收益。

对当前二进制重新检查：16³两步 NP=1 scalar、NP=2 x-slab full 的 Memcheck
分别为0 errors和两进程各0 errors。沿用第5节明确的host-staged诊断命令，
未屏蔽工具错误。比较/读取器共14项单元测试通过。

正式结果位于：

- tests/gpu_validation/out/native_fp32_fp2_fp32_tgv_20261020_v2/report.json
- tests/gpu_validation/out/native_fp32_fp2_fp64_tgv_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp0_20261020/fp2_cross_precision_report.json
- 同一目录的fp2_halo_contract_fp32_*.log、fp2_fp64_gpu_baseline_compare.txt
- 同一目录的fp2_latest_memcheck_scalar_np1、fp2_latest_memcheck_full_np2

首轮读取器把不使用的边角也列为有效域而失败的目录继续保留，不覆盖为通过记录。
FP2完成不代表紧致、Ma12、CURVE、文件恢复、原位或中等规模性能通过。
下一步按原计划进入FP3，不改变整体goal范围。

## 7. FP3 Ma12 初始化与浮点标志诊断

选定的 HBL UDF 已复制到 user_define_module/examples/userdefine_hbl.F90，并仅适配
real_kind；原 /home/dell/workspace/M12_C13 保持只读。四个专用 CPU/GPU、FP32/FP64
构建均通过，默认 UDF 和 FP64 默认精度不变。

首个 FP32 入口被原固定 1e-10 的 EOS 反算一致性检查误拒绝。49 个剖面节点的
最大相对往返舍入为 7.976738913839654e-8。现以四次乘除的标准舍入上界
gamma4=4u/(1-4u)、u=epsilon/2 与原 1e-10 取大值，仅调整该输入一致性检查。
FP32 上界为 2.384186643666908e-7，FP64 仍为 1e-10。
生产 thermal 函数的独立 eosr 探针先复现误拒绝，修复后两个精度均通过，且拒绝
1% 不一致温度；没有改 EOS、剖面值、物理目标或状态下限。

修复后 Ma12 CPU FP32、96x48x32 区间、dt=0.005、10 个完整步正常完成。
使用 743e/643e、原随机吹吸、incoming_only NSCBC、crashfix 开启、滤波关闭。
首末步三阶段及初始化共13份 q 快照有限，物理节点密度和内能为正，末步
CFL 上界 0.026406249031424522，所报告修复次数均为零。此处尚不是 FP3 收口。

该 Release 运行退出时报告 ieee_invalid。GDB 直接设置 MXCSR 后截获于
solver::chardecomp 的 vdivps 指令，而非负密度的平方根：两个实际密度为
0.0925896764、0.104310788，除法分子 SIMD 通道为
[0.304285526,0.322971821,0,0]，分母为
[0.627257347,0.645943642,0,0]。NVHPC -O2 合并标量运算后，未使用的两个通道
执行0/0。优化调试信息显示的 ro_r 负垃圾值并非真实参数，寄存器指向的实际值
为正。相同源代码 -O0 对照在 invalid trap 开启下完成10步，未触发 invalid。
不据此修改 Roe 算法、添加截断或更改默认编译参数；Release 标志仍如实报告。
该证据定位首个 invalid，不声称已枚举所有第三方或 SIMD 非活动通道异常。

原始证据目录为 tests/gpu_validation/out/native_fp32_fp3_m12_20261020：
fp3_profile_roundtrip_diagnosis.txt、fp3_thermal_roundtrip_* 日志、
fp32_np1_filterf_v2/cpu，以及 fp32_invalid_trace/sse_invalid_lanes.log、
debug_sse_invalid_location.log。GDB 在程序正常结束后执行 bt 会返回非零，
必须区分调试器退出码与已完成的求解器10步。初次通过 inferior call 设置 fenv
未改变恢复后的 MXCSR，不能用其无 SIGFPE 证明无异常；有效截获采用直接寄存器设置。

## 8. FP3 Ma12 十步矩阵

执行入口为 tests/gpu_validation/run_native_precision_m12_gate.py，复用原有 M12
资源抽取和输入生成器，由真实 ASTR 推进，不使用 Python CFD 推进器。
网格为96x48x32区间、dt=0.005、10个完整步；743e MP7-LD、643e黏性、
bc=[11,22,41,52,1,1]、incoming_only 常数远场、原 legacy_random 吹吸及 crashfix。
输出诊断只有首/末步三阶段和初始化，不启用整场序列或检查点。

每种精度各覆盖NP=1、NP=2 x分解的滤波关/开CPU/GPU配对，共8次运行；
另以NP=2滤波开启的GPU full工作区对照scalar，共9次，均完成10步。
原始变量检查只覆盖物理节点和实际MPI/周期轴向halo，明确排除物理域外幽灵存储；
不把未参与算子的外部存储填零或冒称整个分配域有效。

| 项目 | 实际结果 |
| --- | --- |
| FP32 CPU/GPU 守恒量阶段比较 | 最大绝对差1.1362135410308838e-5、最大分量缩放差9.273829814167288e-6，仅报告 |
| FP64 CPU/GPU 守恒量阶段比较 | 最大绝对差3.774758283725532e-15，通过原2e-10分量缩放门槛 |
| FP32 原始变量 CPU/GPU | 最大绝对差3.9768218994140625e-4，仅报告 |
| FP32/FP64 GPU守恒量 | 相同输入配置的阶段最大绝对差1.133987251651647e-5，仅报告 |
| scalar/full | 两种精度各自的所采守恒量最大差均为0 |
| 密度、内能、压力、温度 | 所采物理节点均有限且为正；原始变量有效轴向halo有限 |
| CFL | 两种精度、所有本轮配对的最大CFL上界不超过0.026752 |
| 传感器 | FP32单rank滤波开启的一个采样相位有1个mask节点不同；其他本轮FP32配对采样mask一致，差值如实记录 |
| 同阶段物性一致性 | 以所采rho/u/v/w/p重新构造q的离线最大绝对差：FP32为8.373039250830772e-8，FP64为2.220446049250313e-16 |

物性重构在离线FP64中诊断舍入和相位，并非FP32求解器回退；其常数按相应构建
精度取值。上述普通短测的CPU critical-node和GPU crashfix报告均为零实际修复，
因此不能据此宣称激活修复路径已验证。flowstate.dat按既有文本精度比较，
不将它代替FP6的完整统计累计状态恢复检查。

另以GPU NP=2在两种精度运行未滤波MP flux保正各10步，所报告limited_rank_nodes
均为零。与未启用该可选路径的GPU结果最大差：FP32为8.344650268554688e-7，
FP64为1.1102230246251565e-16；该路径会重算通量，零限幅不等于整条可选路径
必须逐位等同于另一通量组织。不能把GPU flux保正称为CPU已经具有的同一限制器。
FP32 NP=2的该10步保正路径经Memcheck，两进程均为0 errors、退出码0；
仍使用排除CUDA/HCOLL/UCC指针查询的host-staged诊断启动，不扩展为设备直传内存安全结论。

原随机吹吸的独立探针在FP32/FP64、rank种子0/1下均通过：独立抽样顺序逐值一致、
RNG状态保存恢复一致、身份查询不推进RNG、首调用生命周期一致；两张GPU各自三次
消费已给定壁面速度逐值一致。FP32和FP64随机数的数值序列不要求跨精度相同。

正式目录：

- tests/gpu_validation/out/native_fp32_fp3_m12_fp32_matrix_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp3_m12_fp64_matrix_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp3_m12_20261020/cross_precision_fields.json
- 同目录primitive_q_consistency_report.json、mp_flux_short_report.json
- 同目录fp32_np2_flux_memcheck、fp32/fp64_wall_rng_rank*.log和wall_upload_gpu*.log

## 9. FP4 单线求解预检

compact_line_solver_probe已按real_kind调整数值类型、显式kind字面量、存储字节数。
保留原FP64误差/残差检查，FP32记录差值和残差但不施加FP64数值上限；有限性、
输入RHS不被改变和既有有效性检查仍保留。

长度127/128/129、255/256/257、511/512/513，导数/滤波两类系数、
五种端点类型（含周期）及1/5/11个RHS，每个精度/后端均270组。
CPU采用原独立因子分解参考，GPU采用生产分块PCR求解器。

| 构建/后端 | 最大CPU/GPU缩放解差 | 最大归一化残差 |
| --- | --- | --- |
| FP32 CPU参考 | 0 | 1.9457877442619065e-7 |
| FP32 GPU | 5.572807822318282e-6 | 7.313843752854154e-7 |
| FP64 CPU参考 | 0 | 4.0065237402204433e-16 |
| FP64 GPU | 1.0640423132570715e-14 | 6.930989818401768e-16 |

CPU行的解差为与自身参考比较，不代表与解析制造解无误差；解析误差原值在CSV中。
证据为tests/gpu_validation/out/native_fp32_fp4_line_20261020/report.json及四份CSV/日志。
这只是单线组件，不代表跨rank全局两层PCR、完整紧致导数/滤波和完整步矩阵通过。

随后两种精度的完整紧致导数组件各16组、紧致滤波组件各108组均通过。
FP32导数最大缩放差1.1324882507324219e-6、残差9.462413430583183e-8；
滤波最大缩放差7.86777036410058e-6、残差8.428929163528665e-7。
FP64对应最大缩放差为1.1657341758564144e-15、1.7398121743692514e-14，
两种精度的scalar/full滤波差均为0。原物理端点保持检查仍逐值要求一致。
收敛阶和解析误差保留在CSV；FP32不套用FP64的阶数或误差上限。

全局两层PCR组件使用compact_distributed_gpu_probe，NP=2分别分解x/y/z，
每种精度覆盖pinned和device-aware两个后端，共12次运行。
每个rank包含324组，覆盖上述9种长度、导数/滤波系数、周期/非周期端点及1/5/11 RHS。
CPU全局求解和GPU全局求解均与原独立因子分解比较，并检查共享节点重建。
所有有效载荷和数值归约使用匹配的mpi_real_type，实际多rank线路的后端标志已核对。
FP32的最大缩放解差2.9887521577620646e-6、残差6.505114811261592e-7；
FP64为7.121639655014502e-15、5.248981624808094e-16。
同配置最大累计发送数值载荷为FP32的680字节、FP64的1360字节，
不是整步通信量或性能加速比。设备直传结果只证明本地该环境的数据正确性，
不证明MPI库内部无暂存或跨节点GPUDirect。
分布式证据在tests/gpu_validation/out/native_fp32_fp4_distributed_20261020/report.json，
每次运行保留两个rank的CSV、二进制解和日志。

## 10. FP3 激活路径和历史窗口

mp_ld_reconstruction_probe的数值声明、MPI类型和字面量已适配real_kind。
其FP32差值只报告，原FP64阈值及有限性、严格保正、原始RHS不被未激活限制器修改、
修复历史/RNG/RK原点等离散不变量检查保留。
最终二进制分别在FP32/FP64下通过NP=1完整通量组件、NP=2通信/海绵/保正组件、
NP=2四组有序crashfix、NP=2八组壁面吹吸/输入修复。
有序修复两种精度的CPU/GPU场与缓存最大差均为0，修复标记一致。
这些合成组件确实激活修复及保正，不借普通Ma12短测中的零修复次数冒称覆盖。

NVHPC26.1在该大型测试驱动的kind化内部函数/优化组合上触发fort2 SIGSEGV。
保留数学表达式和二分算法，将五个独立参考函数放入该文件原有测试模块，并为
测试驱动本身设置-Mnofma后，两种精度均由根CMake构建并完成上述重跑。
生产模块和求解器/核仍使用原-O2及原FMA设置，未加运行时回退或修改数值精度。
仅调整结果声明、限制导入、关闭调试信息、增大编译进程栈，以及单独-Mnovect等
尝试未形成通用修复，相关失败日志保留。不能将一个成功诊断命令当成根构建通过。
最终构建日志为build_fp32/fp64_probe_module_nofma.log，重跑日志以_final.log结尾。
另在两种精度分别检查关闭crashfix的非法输入和无有效邻居的修复失败，
均按预期以92退出，且包含对应失败原因；日志为*_failure_detect-only.log、
*_failure_no-neighbor.log。不将FP32的report-only差值政策扩展为允许无效状态继续推进。

CP9已捕获的真实11x11x11窗口另由原CPU显式滤波器分别原生FP32/FP64回放。
两者输入最小内能约1.8627873e-4，滤波后中心内能分别为
-1.6793256509117782e-4、-1.6793346937688837e-4，均重现原滤波失正。
原始文件不变，FP32测试载荷显式由原FP64窗口转换；存储的GPU观察值仍来自旧FP64运行，
不能把与它的FP32差值称为新FP32 GPU同状态回放。
日志位于第9节目录的fp32/fp64_captured_explicit_window_cpu.log。
本轮未更改滤波公式或增加经验保正参数。GPU独立回放的后续结果见第11节。

## 11. FP3 回放与 FP4 完整步收口

新增native_filter_window_probe根CMake测试目标，入口为
tests/gpu_validation/native_filter_window_replay.cuf。原大型MP探针对新增过程的导入
仍触发NVHPC26.1内部崩溃，因此将这一独立任务单独编译，保留原MP探针可构建。
新探针的生产模块和回放过程均使用原-O2，不修改生产FMA或求解器编译设置。

按原捕获的450x130x96尺寸保留物理x/y闭合及周期z，以窗口中心状态常值外延未知区域。
这不是完整历史流场恢复，也不推进CFD；已知中心模板与原窗口保持对应。
两种精度、scalar/full共四次回放，生产X/Y/Z核与原CPU滤波器的最大绝对差：
FP32为2.384185791015625e-7、FP64为4.440892098500626e-16。
中心滤波后内能分别为-1.6793308896012604e-4、-1.6793346937688593e-4，
均重现原失正。随后将同一GPU滤波后状态提供给CPU/GPU原有有序crashfix，
均实际修复1312个外延夹具节点，修复标记和修复后q逐值一致。
1312不是原生产算例的修复节点数，未知区域的常值外延会影响这一数量。
FP32 full工作区经Memcheck并开启完整泄漏检查：0 errors、0 bytes leaked。

独立回放证据：第8节目录下captured_gpu_workspaces/report.json及
captured_gpu_memcheck_fp32_full/run.log。两种工作区均调用生产过滤入口，
方向中间态另外核对，不以旧FP64观察值的单纯转换代替新GPU计算。

FP4首次完整FP32紧致TGV因旧固定FP64比较阈值停止，首个同状态缩放差
7.4505805969238281e-8、残差1.6719852169444493e-7。已按批准合同将测试比较
阈值仅用于FP64，FP32输出RHS/解差/残差，保留有限性及所有权检查；没有修改算法。
同时修正紧致打包缓冲的报告字节数为实际real_bytes，不再把FP32工作区按8字节计数。

完整步矩阵由两个native_precision运行脚本增加可选643c参数复用，不另写推进器。
TGV每种精度17次运行：32^3、10步、dt=1e-3，NP=1及NP=2 x/y/z、滤波关/开，
另加NP=2 x分解full工作区；Ma12每种精度9次运行：96x48x32、10步、dt=0.005，
NP=1/2 x、滤波关/开，另加full工作区。CPU/GPU同阶段状态和原始变量有效。
各相位密度缓存与q1要求精确复制，另在离线FP64中检查能量/EOS重构并保留源kind常数。

| 工况/精度 | CPU/GPU q最大绝对差 | 按单分量参考最大值缩放后的最大差 | 原始变量最大绝对差 |
| --- | --- | --- | --- |
| 紧致TGV FP32 | 1.18255615234375e-2 | 6.587709357403378e-5 | 4.730224609375e-3 |
| 紧致TGV FP64 | 6.849631972727366e-12 | 5.2069165492788064e-14 | 2.7426949600339867e-12 |
| 紧致Ma12 FP32 | 1.1712312698364258e-4 | 9.544980807059458e-5 | 6.173133850097656e-3 |
| 紧致Ma12 FP64 | 4.929390229335695e-14 | 4.01564138498987e-14 | 6.217248937900877e-12 |

FP32只报告差值，不据此宣称达到特定物理精度；FP64保留原误差门槛。
两种精度各工况的scalar/full q差均为0。每种精度再以两工况NP=2 x、滤波开启
检查device-aware紧致私有通信，与pinned比较的26份物理q采样文件均逐位一致。
共同halo仍使用pinned，此处不冒称整步所有通信均已切换device-aware。
FP32 NP=2、滤波开启的完整紧致Ma12十步Memcheck两进程均为0 errors。

证据目录：

- tests/gpu_validation/out/native_fp32_fp4_fp32_tgv_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp4_fp64_tgv_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp4_m12_fp32_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp4_m12_fp64_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp4_device_aware_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp4_compact_m12_memcheck_20261020/run.log

比较/读取器共17项单元测试通过。未启动中等规模性能计时或远程任务。

## 12. FP5 代表性静态 CURVE

32³区间、显式643e/643e、滤波开启，dt=2e-5，推进4个完整步。
每种精度6对CPU/GPU运行：周期曲线TGV的NP=1、NP=2 x分解及周期相容恒定场，
带上下bc41壁面的y-wavy TGV的NP=1、NP=2 y分解及壁面相容恒定场。
壁面恒定场采用零速度、T=273.15、rho=1/273.15，不让非零来流穿越无滑移壁面。

几何文件的坐标/度量使用对应float32/float64类型，Jacobian有限且为正。
检查实际生产GPU梯度与原CPU梯度，以及将相同GPU原始变量提供给CPU后的冻结输入对照；
后者两种精度的梯度差均为0，不表示解析梯度截断误差为0。
设备y法向与同输入CPU源数组逐字节一致，方向朝内；壁面速度为0，壁温与单位法向检查通过。
离散度量恒等式、解析几何截断误差、恒定场漂移与统计差值单独保留在报告中。
所有运行的状态有限、rho/T为正，CFL不超过0.5；FP32差值仅报告。
全部q快照的CPU/GPU最大绝对差为FP32的2.593994140625e-4、FP64的4.263256414560601e-13。

新增ASTR_VALIDATION_GRADIENTS=1仅在已显式开启阶段快照时输出梯度和法向；默认关闭。
两种精度分别对周期NP=2四步进行开/关对照，26份物理q文件逐字节一致，关闭时不生成梯度文件。
FP32 GPU y-wavy、NP=2 y分解、4步、诊断开启的Memcheck：两个rank均0 errors、0 bytes leaked。
读取器及原比较工具共19项单元测试通过。

证据目录：

- tests/gpu_validation/out/native_fp32_fp5_fp32_curve_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp5_fp64_curve_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp5_diag_off_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp5_memcheck_20261020/run.log

## 13. FP6 精度身份及同精度精确恢复

checkpoint_state_io的新写出版本为schema=3，在现有HDF5状态文件/状态组内增加
numerical_kind_bytes整数数据集，取值4或8；不新增状态附属文件。
所有数值数据集继续采用本构建的F32/F64类型。既有发布的schema=2按FP64识别，
并核对实际数据集类型；未发布且缺少精度身份的旧FP32开发探针不冒充旧FP64格式。
头部仍为13个int64，时钟位型按来源精度解码；同精度位型往返及跨精度0.1、负零转换通过。
跨精度转换只在文件边界进行，没有建立全域FP64数值镜像。

文件原语的10项测试覆盖两种精度的同拓扑精确读取、双向非精确读取及重新分区，
跨精度精确读取拒绝、旧FP64版本读取、身份缺失/非法值/载荷类型矛盾和未发布FP32原型拒绝。
这是MPI/HDF5文件原语验收，不是完整跨精度CFD续算。
原checkpoint_state_probe的43项FP64回归全部通过，包含缓存、统计、AIR5状态载荷和继承组。
对应证据为native_fp32_fp6_precision_primitive_20261020.xml、
native_fp32_fp6_precision_primitive_20261020_v2/及native_fp32_fp6_fp64_state_regression_20261020/，
均位于tests/gpu_validation/out/。

真实续算由run_native_precision_restart_gate.py复用现有ASTR运行器：16³、dt=1e-3、
滤波开启，同后端同拓扑连续12步对照5+7步。每种精度分别包含CPU/GPU的
643e NP=1/2，以及643c NP=2，共12种配置、36次求解器运行。
全部state.h5数据集（包含11分量守恒量/缓存和rank_extras）逐字节一致；
control.bin、archives.bin、insitu_control.bin也逐字节一致。
显式8种配置另开启原有三维Reynolds/Favre统计，statistics.h5全部数据集逐字节一致。
紧致4种配置不扩大原位统计现有的显式格式准入。物理节点状态有限，密度、内能为正。
FP32 GPU NP=2从同一step5恢复至step12的Memcheck两个rank均0 errors，
最终权威场及统计与未插桩连续参考仍逐字节一致。

真实矩阵和Memcheck证据：

- tests/gpu_validation/out/native_fp32_fp6_fp32_cpu_exact_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp6_fp32_gpu_exact_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp6_fp64_exact_20261020/report.json
- tests/gpu_validation/out/native_fp32_fp6_fp32_restart_memcheck_20261020/report.json

离线export_checkpoint.py同步识别新精度身份，正确解码时钟，保留源浮点宽度输出
标量、速度向量与坐标，并在XDMF声明实际Precision。29项导出/安装回归通过；
额外从两种精度、CPU/GPU的实际NP=2检查点导出整场及三方向切片，
缓存字段与导出字段逐字节一致，源状态/控制/几何文件的大小及mtime不变。
证据为native_fp32_fp6_export_unit_20261020.xml及native_fp32_fp6_real_exports_20261020/report.json。
未恢复CRC或整场写后回读。

本节记录文件原语与初轮同精度结果，后续完整程序的跨精度结果见第14节。

## 14. FP6 完整 TGV 跨精度续算

新增astr_precision_io，仅用于小型控制记录的源精度读取与显式转换，支持标量、
一维和二维元数据；不通过该接口读取整场，也不建立FP64数值镜像。
适配control.bin、archives.bin、insitu_control.bin、原调度及可变频率/产品调度的读取接口，
写出布局保持本构建的原生宽度。读取原精度的完成步时间和dt，重建目标精度物性、
几何及通信halo。跨精度exact_restart=t在读取控制记录前明确拒绝。
同精度路径仍保留原精确恢复和调度缓存检查。

run_native_precision_cross_restart_gate.py实际运行16³ TGV：先用源精度推进5步，
再由目标精度非精确恢复并推进7步。步数调度和物理时间调度各12种配置，
分别覆盖CPU/GPU、643e NP=1/2及643c NP=2、双向精度切换，共48次求解器运行。
两组矩阵全部通过，物理节点状态有限，密度、内能为正，源运行文件大小及mtime未变。

恢复探针在派生状态和halo重建之后、首个RK之前输出。非精确恢复会调用原qswap，
其x/y/z周期端点平均会改变源文件中原本不完全相等的重复端点。
因此按“源场转换为目标精度，再按原次序执行周期端点平均”建立独立接口参考，
五个守恒分量与实际恢复场全部逐字节一致，不将该探针误称为未经边界处理的原始拷贝。
首次测试将两种相位混比而失败，相关运行保留在cross_matrix_20261020/；
修正的是测试预期，未修改CPU/GPU平均或通信算法。

显式组开启三维Reynolds/Favre统计；恢复后首次RK前的时间权重和质量权重均为0，
证明未静默继承源精度累计。完成后唯一周期节点的覆盖时长约0.0065，
起点为恢复时刻，不是原窗口的0.0005。GPU统计文件的重复末端面为已定义的零填充，
不作为未采样的唯一物理节点；此前全数组正权重断言的失败保留在cross_matrix_20261020_v2/。
最终通过矩阵为cross_matrix_20261020_v3/及cross_time_matrix_20261020/。

时间调度的初轮红测重现：FP32文件中的0.004提升至FP64后，与输入按FP64解析的0.004
不逐位相等，被误判为用户修改配置。新接口按源精度可表示值确认同一配置，不引入经验容差；
跨精度非精确恢复在目标精度下规范化输出周期并重建派生目标序号，
保持已发帧身份、完成步物理时间及dt，不修改物理时间推进。
这样新写出的目标精度控制记录能按同一输入继续恢复，而不固化一个隐式的源精度周期偏移。
原同精度序号/配置损坏拒绝仍保留。独立schedule_probe覆盖跨精度后再次原生精确续接，
全程序另通过两种精度、CPU/GPU NP=1/2的物理时间调度精确矩阵，全部场、统计及控制文件逐字节一致。

另从实际FP64 TGV构造旧新式格式回归：状态/统计/几何改为schema=2，控制记录改为
ASTROC04及原14整数前缀。不声称它是未修改的历史生产文件。
FP64精确恢复与连续参考逐字节一致，FP32非精确恢复也通过接口和有效性检查。
两种精度的元数据转换/ABI探针通过，原FP64可变频率及产品调度11项组件回归通过。
FP64至FP32、GPU NP=2的跨精度七步Memcheck两个rank均0 errors。

本节证据均位于tests/gpu_validation/out/：

- native_fp32_fp6_cross_matrix_20261020_v3/report.json
- native_fp32_fp6_cross_time_matrix_20261020/report.json
- native_fp32_fp6_fp32_exact_control_20261020/report.json
- native_fp32_fp6_fp32_gpu_exact_control_20261020/report.json
- native_fp32_fp6_fp64_exact_control_20261020/report.json
- native_fp32_fp6_fp32_exact_time_20261020/report.json
- native_fp32_fp6_fp64_exact_time_20261020/report.json
- native_fp32_fp6_time_schedule_red_20261020/report.json及time_schedule_green_20261020/report.json
- native_fp32_fp6_legacy_control_20261020/report.json
- native_fp32_fp6_control_regression_20261020.xml
- native_fp32_fp6_cross_memcheck_20261020/report.json

本节之后已补统计段身份与独立FP64合并，见第15节。
本节TGV不激活随机吹吸，不能单凭本节证明其生命周期；Ma12/CURVE和归档的后续结果见第16、17节。

## 15. FP6 统计分段与独立 FP64 合并

在现有statistics.h5内新增statistics_segment，含21个int64，共168字节有效载荷，
另有HDF5元数据开销。不另存整场坐标、额外精确状态附属文件或瞬时统计样本。
记录版本、起点是否已知、首个完成步、起始时间、请求窗口、源时钟位型及字节宽度、
周期唯一节点标识、单位体系、统计/采样定义和几何来源。
几何生成代码身份由根CMake在配置时从网格生成、度量、常量、原位几何及所选UDF源码生成，
不在程序启动时打印，不对运行流场做SHA/CRC扫描。
转换编译期常量时不增加逐字MPI归约；移除初版8次冗余常量检查后，
GPU FP32 NP=1/2的全部状态和统计数据集与移除前仍逐字节一致。

同精度精确恢复保留分段起点；跨精度或显式重新起算统计时，从恢复的完成步建立新段，
保存转换前源时钟的原始位型，不重新初始化物理驱动或RNG。
旧新式统计文件缺少分段记录时，数值累计照常精确恢复，但新记录标明起点未知，
不凭duration或文件时间倒推。实际旧FP32文件的全部既有状态/统计数据集与连续参考逐字节一致，
新标记为unknown。两种精度、CPU/GPU、NP=1/2的新记录精确续算共8种配置通过，
包含权威场、累计统计及控制文件的逐字节比较。

新增scripts/output/merge_statistics_segments.py，独立使用NumPy/h5py，以FP64按块合并
原生FP32/FP64的充分统计量。不进入求解器，不改变在线累加，不新增累计补偿。
Reynolds以覆盖时长加权，Favre以密度时间积分加权，二阶中心矩保留段均值差修正项。
归一化协方差、RMS及密度加权应力由合并后的权重/中心矩得到，不直接平均RMS。
零覆盖点由covered=0明确标记，零值占位不解释为瞬时场或零脉动估计。

离线检查复用已有冻结输入、静态物理/网格资源及生产者几何身份，不回读checkpoint的state.h5。
静态资源逐块比较只在显式离线合并时进行，不新增运行时流场校验和。
按记录排除周期末端重复节点，不根据零值判断节点所有权。
拒绝重复源、重叠累计历史、未知起点、窗口/定义/网格不一致及不匹配的父时钟。
真实缺口保留在报告中，不补造样本。精度切换的原始时间坐标跳变量单独记录，
区分完成步身份连续和原始时钟坐标连续；不宣称混合来源是一次精确计算轨迹。
同时记录窗口两端缺失情况、实际正权重点数和覆盖时长范围。

regional沿用原CPU/GPU对几何空间平均速度信号的时间统计，密度权重固定为1。
其Favre字段只是单位权重信号统计，不是流体区域Favre应力，输出属性及README已明确区分；
实际流体密度加权统计位于point。未修改原区域统计定义。

初轮22项独立单元测试覆盖加权观测、中心矩修正、零覆盖、无效矩、源保护、重叠/缺口、时钟映射、
静态网格资源及8/9字段输入兼容。随后按原readwrite的7/8/9标志读取顺序补齐7字段输入，
旧输入缺省lcomb=f；23项合并测试与29项导出/隔离安装检查合计52项通过，
证据为native_fp32_fp6_tools_input_compat_20261020.xml。
新增工具已纳入根CMake的OutputTools，CPU/GPU当前根构建的隔离安装均通过，CLI不依赖求解器或Catalyst。

真实16³ TGV完成CPU/GPU、NP=1/2、双向精度切换共8种短段合并，源文件未改写。
每组4096个唯一节点全部有正权重，时间/质量权重与源段和逐值一致；
协方差、RMS和应力满足各自定义，矩阵对称、结果有限。
check_merged_statistics另外用围绕合并均值的独立中心化表达式复核，最大差2.220446049250313e-16，
仅为同一充分统计量的两种FP64求值顺序之差，不是FP32对FP64的流动误差。
更新元数据及标签时只重读不可变短测统计，未为报告变化重跑CFD。

本节证据均位于tests/gpu_validation/out/：

- native_fp32_fp6_statistics_tools_verified_20261020.xml
- native_fp32_fp6_merge_real_20261020/report.json
- native_fp32_fp6_merge_real_final_20261020/report.json及algebra_report.json
- native_fp32_fp6_segment_exact_cpu32_20261020/report.json
- native_fp32_fp6_segment_exact_cpu64_20261020/report.json
- native_fp32_fp6_segment_exact_gpu32_20261020/report.json
- native_fp32_fp6_segment_exact_gpu64_20261020/report.json
- native_fp32_fp6_segment_final_gpu32_20261020/report.json
- native_fp32_fp6_segment_unknown_origin_20261020/report.json

## 16. FP6 Ma12 与静态 CURVE 续算

两类32³ CURVE、CPU/GPU、FP32/FP64、NP=1/2共16种配置，通过连续12步与5+7步的
同后端精确续算。周期曲线采用NP=2 x分解，y-wavy bc41采用y分解；dt=2e-5，
逐步CFL上界不超过0.5。全部权威场数据集及三个控制文件逐字节一致。
两类几何另通过CPU/GPU NP=2双向精度切换共8组，恢复探针、时钟身份和源资源不变性均通过。

Ma12使用96×48×32区间、dt=0.005、743e MP7-LD、incoming_only NSCBC，
边界[11,22,41,52,1,1]，原legacy_random吹吸及crashfix开启。
643e黏性/滤波关闭的两种精度、CPU/GPU、NP=1/2共8种配置通过12步与5+7步精确对照。
643c黏性/紧致滤波开启另通过两种精度、CPU/GPU NP=2共4种配置。
场、控制文件及原随机状态逐字节一致；恢复时删除本地grid.2d、flowini2d.h5、
inlet.prof和wallbs.dat，验证实际使用源检查点冻结资源。
Ma12显式组CPU/GPU、NP=1/2双向跨精度共8组通过，首个RK前的守恒量、完成步、
转换后时钟逐位核对，crashfix历史及潜在RNG整数包不变。
不要求跨精度后的浮点随机扰动或后续流场逐值相同。

首次FP64 CPU到FP32的q4参考差异定位为91/92个次正规数在原NVHPC转换环境中归为带符号零，
最大绝对差1.1157460671600988e-39。独立native_precision_probe cast模式使用相同构建参数，
其REAL转换加原周期/共享节点重建与实际恢复场逐位一致。修正的是测试参考，
未修改FTZ、编译优化、求解器公式或物理阈值。

旧hdf5io的read入口使用RDWR打开输入，恢复曾改变源HDF5的mtime。
将h5io_init(mode='read')及八个串行命名读取助手改为RDONLY，保留真正写入口的RDWR。
冻结HDF5及grid.2d设为0444后，上述恢复通过且源文件大小、mtime不变。
这是I/O访问模式修正，不是CPU数值算法修订。
测试层另按原生精度检查dt，跨精度首步使用检查点dt_next，之后仍检查输入dt；
CURVE夹具不再保留生成专用网格后未使用的50 MiB示例副本。未放宽CFL门槛，失败证据保留。

证据位于tests/gpu_validation/out/，以下目录均含report.json：

- native_fp32_fp6_curve_cpu_fp32_readonly_20261020、curve_gpu_fp32_readonly_20261020、curve_fp64_readonly_20261020
- native_fp32_fp6_curve_cross_periodic_20261020、curve_cross_ywavy_20261020
- native_fp32_fp6_m12_cpu_fp32_readonly_20261020、m12_gpu_fp32_readonly_20261020、m12_fp64_readonly_20261020
- native_fp32_fp6_m12_compact_filtered_fp32_20261020、m12_compact_filtered_fp64_20261020
- native_fp32_fp6_m12_cross_native_cast_20261020

同一行的缩写目录均补前缀native_fp32_fp6_；CFL单测见native_fp32_fp6_cfl_kind_gate_20261020/。

## 17. FP6 原生场序列与跨精度历史

场序列metadata仍为11个int64，首字版本1表示原FP64时钟/载荷，版本2表示原生FP32。
既有FP64元数据布局不变。repair_series按身份解码时钟、核对F32/F64字段、坐标及字节数，
生成匹配Precision=4/8的XDMF。combine_series允许不同精度的明确父子段串联，
每帧引用自己的数据与几何宽度；单段内改变精度仍拒绝，不转换整场以伪装成统一存储精度。

运行时总索引ASTR_NATIVE_LINEAGE_2在时钟行增加实际元素字节数，后面仍为数据和几何路径。
读取器保留旧LINEAGE_1的已发布FP64解释。历史时钟保留原FRAME文本，不先经过当前kind再格式化。
父帧类型只在准备目录时读取小量HDF5元数据确定；发布复用已准备记录，不反复回读祖先流场。
FP64到FP32实际恢复中，源step5的0.005变成0.004999999888241291，旧离线比较曾误判父末帧越界。
仅跨精度创建的SEGMENT_3在原文件追加源字节宽度和原始时间位型；父历史截断使用源时间，
新段起点保留转换后时间。同精度仍写SEGMENT_2，工具同时读取1/2/3，不猜测时间、不提高容差。
未新增附属文件、运行时流场CRC或大数组写后校验。

run_native_precision_archive_gate使用16³ TGV、NP=2、12步与5+7步，开启整场和三方向切片，
每类包含六个原始量、速度向量、九个梯度、Q、散度及三个涡量。
CPU/GPU各两种精度的同后端精确场、统计及控制通过，归档开/关不改变场和累计统计。
各后端双向跨精度后，原生帧、分段序列及父链通过ParaView 6.0.1实读，
字段、坐标、向量的类型和值与HDF5参考逐字节一致，源运行未改写。
CPU四组在archive_clock目录完成写入report，随后GPU首轮计数失败，不将该目录整体称为通过。
GPU四组在archive_gpu目录完成全矩阵并标记passed。

GPU首轮失败定位为output_fields_gpu仍将FP32载荷按8字节计数，已改为real_bytes，
并同步打包/私有梯度预算。原cudaMemcpy已使用类型化元素数，不存在这项计数导致的错误回读。
FP32 NP=2带派生归档的七步精确续算Memcheck两rank均0 errors，最终场和统计仍与连续参考逐字节一致。
该16³完整帧20个标量的有效载荷FP32为393040字节，FP64为786080字节，
不含速度向量副本、静态坐标及HDF5元数据，不作为进程显存峰值。

离线序列/父链工具109项通过；FP64原生索引及类型行39项通过；FP32非法父源及类型行17项通过。
NP=2的两种精度、CPU/GPU共四组自适应固定窗口通过精确恢复和输出开/关检查。
额外合成64→32→64的两次切换父链，验证原始时钟、逐帧类型和不可变父资源均正确。
跨精度自适应事件状态及渲染联合配置仍待验证。
上述结果不代表FP7 C++设备分析及渲染已验收。

按设计§6.1补齐旧ASTR_GPU_PRECISION_MODE的输入行为：原生FP32构建接受已有合法值
fp64/mixed_workspace，但只说明一次该选项不改变构建精度，不分配混合工作区。
CPU/GPU NP=2各比较未设置、fp64、mixed_workspace三次两步运行，最终权威场全部逐字节一致，
GPU活动工作区为none、混合工作区字节数为0。未知值仍拒绝；FP64构建的既有选择行为不变。

证据位于tests/gpu_validation/out/：

- native_fp32_fp6_series_parent_clock_20261020.xml
- native_fp32_fp6_native_lineage_fp64_final_20261020.xml
- native_fp32_fp6_typed_lineage_fp32_green_20261020.xml
- native_fp32_fp6_archive_clock_20261020/report.json中的四个CPU完成组
- native_fp32_fp6_archive_gpu_20261020/report.json及gpu_np2_fp32_resumed/memcheck.*.log
- native_fp32_fp6_adaptive_cpu32_20261020.xml
- native_fp32_fp6_adaptive_gpu32_20261020.xml、native_fp32_fp6_adaptive_fp64_20261020.xml
- native_fp32_fp6_two_switch_history_final_20261020.xml
- native_fp32_fp6_legacy_selector_20261020.xml

## 18. FP6 自适应跨精度与紧致平板补项

Ma12 743e/643c、紧致滤波开启、NP=2的CPU/GPU双向跨精度共4组通过，
包含恢复前后的权威场转换、时钟、原随机吹吸整数包、修复历史及只读源资源。
证据为tests/gpu_validation/out/native_fp32_fp6_m12_compact_filtered_cross_20261020/report.json。

run_native_precision_adaptive_gate复用16³ TGV、NP=2、Re=1的输出调度夹具，
同时启用已有衰减事件、步数窗口、物理时间窗口，以及步数/时间两种场产品调度。
源精度先推进5步，目标精度非精确继续到12步，再读取目标精度新检查点继续到14步。
CPU/GPU双向切换4组通过，恢复场和时钟身份正确，监测完成步连续，源运行未改写，
无需restart_output=override，也没有因精度转换重置事件或窗口。

初轮在FP32到FP64的adaptive monitor/control metadata处拒绝。
原因是0.74、0.723及时间参数经FP32保存后，与按FP64解析的相同输入不逐位相同。
adaptive_output和output_archive现按已有source_real_matches规则识别源精度下等价的配置，
保留活动状态、监测值/时间、事件进入时刻及已输出身份；只规范化目标精度周期和重算派生next缓存。
同精度仍严格检查缓存、配置及原有无效输入，不新增经验容差，不改变事件判据或RK推进。
原FP64自适应组件33项通过，另6项因未选择其独立配置/集合探针而跳过，不计入通过数。

证据为tests/gpu_validation/out/native_fp32_fp6_adaptive_cross_20261020/report.json、
native_fp32_fp6_adaptive_cross_red_20261020/及native_fp32_fp6_adaptive_core_regression_20261020.xml。
这些配置未启用真实渲染，不能代替归档/自适应/渲染的最终联合检查。

## 19. FP7 原生积分核与桥接类型基础

insitu_rk45_device.h的轨迹状态、Cash–Karp系数、阶段算术、误差估计及步长控制使用
astr::numerical_real；分子/分母先转为目标类型再按原表达式计算系数。
保留原算法和终止规则。原1e299是状态标记而非物理阈值，FP32下改用可表示的最大有限值，
避免标记溢出成Inf；FP64仍为1e299，subminimum判定同步引用该标记。
native_precision_rk45_probe/.cu检查常向量、圆轨迹、零速度、域外、非有限值及传播终止。
FP32 CPU/GPU均通过，圆轨迹单步解析差约1.9741e-8，只报告、不作为偏差门槛。
FP64 GPU与原VTK的26个步进对照max_error=0，原轨迹连续/分区续接及最短末段检查也通过。

FP32独立核的SASS没有FP64算术或浮点转换指令；同一FP64探针有771条此类静态指令。
该检查仅覆盖实际编译的RK45探针，不推广到所有原位核或第三方渲染。
FP32探针Compute Sanitizer Memcheck为0 errors。宿主GNU构建以
-Wdouble-promotion -Werror=double-promotion和关闭收缩检查该积分核，保留原CUDA --fmad=false。
独立探针不链接Viskores/ParaView分析库，不将已发现的FP64 FloatDefault当作匹配的FP32依赖。

insitu_compact_mesh.h和insitu_mesh_adapter.cpp的数值数组、Fortran入口参数及色标状态数组
采用原生Real；Conduit外部数组明确选择float32或float64，缓冲字节按sizeof(Real)计算。
没有建立FP64数组镜像，保留纯图形调色板、相机边界/API展示标量和性能计时的原类型。
两种精度的桥接库均构建通过，native_insitu_abi_probe实际跨Fortran/C++保存/读取六个标量，
逐位值及二维范围查询通过。编译和该ABI测试不证明完整产品的几何/插值路径已适配。

六项组件检查通过，证据为tests/gpu_validation/out/native_fp32_fp7_components_20261020.xml；
原始SASS及GPU日志、Memcheck见native_fp32_fp7_rk45_20261020/。
当前build_native_fp7_fp32_device仅作为这些组件的构建目录，尚未验收其中的完整求解器/原位程序。
配置复用已有CUDA 13.1/glibc的私有头兼容层，没有修改系统CUDA或原FP64依赖安装。

## 20. FP7 设备分析类型闭包与分区组件

设备分析的数值坐标、字段、Q、切面谓词、壁面积分、CURVE逆映射和流线状态改用公共Real。
NativeVector明确为viskores::Vec<Real,3>，并在编译时要求Viskores的FloatDefault与其一致。
FP32另建paraview-6.1.1-astr-native32-gcc13依赖目录，不覆盖原FP64构建或安装。
构建脚本新增PV_DEVICE_NUMERICAL_PRECISION=fp32/fp64，默认仍为fp64。
当前依赖补丁的ASTR_VISKORES_FP64_MPI开关同时控制Viskores默认类型和其内部MPI：
FP32配置关闭第三方Viskores内部MPI，但ASTR自己的MPI及分布式粒子传递仍启用，
不将该选项误称为关闭求解器的多GPU通信。

流线状态不再将整数计数混放在八个浮点槽中：六个数值槽采用原生类型，
步数和终止状态保持Int32，MPI派生类型用MPI_FLOAT/MPI_DOUBLE加两个MPI_INT32_T字段，
并以结构实际字节数设置消息extent。实际GPU核和MPI往返验证INT32_MAX-rank完整保存。
两种精度各通过6项NP=1/2、单向/双向及x/y/z跨分区组件检查；
FP32的一个双向NP=2解析直线对照最大差为1.1920928955e-6，接缝差为0，差异仅报告。
FP64仍保留原2e-10门槛。FP32双卡y向流线Memcheck两rank均为0 errors。

切面原语以目标类型构造坐标、仿射场及扩展算术，保留原拓扑、绕序、共享键和无零面积检查。
乘积最小指数由目标类型的min_exponent+digits给出，FP64为-968，FP32为-101，
并非增加经验几何容差。两种精度各通过NP=1及NP=2 x/y/z的4项组件测试。
CURVE两种映射的32^3单元逆映射组件也均通过，FP32误差只报告，FP64原门槛不变。

完整Fortran桥接进一步统一c_real_kind，包括物理时间、平面参数、色标状态、视图数值和
CPU/GPU字段参数；M12色标续算读入按源精度转换。整数、指针和性能时钟没有缩短。
主机注册范围、私有halo、统计状态、样本及产品暂存的字节数按real_bytes/sizeof(Real)计算，
不再对原生FP32缓冲固定乘8。实际完整程序及后处理生命周期联测仍待构建完成后验证。
FP64设备产品静态库已编译通过，但这不是最新完整可执行程序或实际图像的验收结论。

组件证据：tests/gpu_validation/out/native_fp32_fp7_trace32_20261020.xml、trace64同名记录，
native_fp32_fp7_plane32_20261020.xml、plane64同名记录、
native_fp32_fp7_inverse32_20261020.xml、inverse64同名记录；
双卡Memcheck及构建日志位于native_fp32_fp7_rk45_20261020/。
新run_native_precision_insitu_gate.py复用既有ASTR运行器，准备TGV、CURVE及同精度续算联测；
目前仅完成脚本语法检查，不计入实际产品通过数。

FP8新增run_native_precision_performance.py，按两类网格、显式/紧致、滤波关/开、
NP=1/2与两种精度组织3轮交错测试，保留10步预热和30步采样。
汇总以每rank窗口累计时间的最大值计算RK步耗时，给出精度比较与各自NP=1/2扩展性。
配对身份、采样窗、中位数及修复活动汇总工具23项通过，证据为native_fp32_fp8_tools_20261020.xml。
尚未执行正式中等规模计时，不从组件运行或编译负载下的短测宣称性能收益。

## 21. FP7 真实字段、产品和联合续算

两种精度的完整CUDA/原位程序已构建通过，FP32使用独立的匹配ParaView/Viskores依赖，
没有在FP32进程混用FP64分析库。补齐了预览源注册入口的c_real_kind时间参数。
图形API、调色板、性能时钟保持既有精度；数值字段、插值、轨迹、Q及几何使用原生类型。

真实32^3 TGV、NP=2检查了CPU和GPU后处理梯度、Conduit/VTK数组类型及五个守恒分量的实际传递。
字段与检查点按相同周期唯一节点映射逐值一致；FP32梯度差为0，Q最大差为5.9604644775e-8，
差异仅报告。此处CPU指GPU推进器上选择的CPU后处理导数，不是解除EGL对GPU绑定的既有限制。
检查曾暴露测试中的三处错误前提：将CPU推进器送入EGL入口、读取已按keep=2退休的检查点、
直接比较已规范化周期复制端点和原副本。分别修正夹具选择、输出周期及节点映射，没有改求解器。

TGV在NP=1/2下的六类实际产品，两种精度和standard-device/direct-device均通过。
周期CURVE和y-wavy的Q、瞬时/两类平均流线，以及y-wavy壁面产品均完成两管线检查，
开启处理不改变权威推进。实际静态斜切面也通过笛卡尔/周期CURVE的两管线检查。
CURVE的输入dt保持2e-5，CFL检查按目标kind的精确值核对，仍保留原CFL<=0.5，
不把FP32的dt舍入误判为输入被修改，也没有增大CFL容限。

已有Re=1短夹具用于激活自适应事件，不作为Re=1600物理精度结论。
两管线、两种精度的12步连续推进与5+7续算联合启用渲染、统计、归档和自适应监测：
权威场、统计及数值控制状态逐值一致，新归档帧浮点载荷逐值一致，JPEG续接逐字节一致，
源检查点未修改。此项包含实际渲染，不再只是FP6的无图形调度组件。

y-wavy壁面均值的两种精度/两管线实际图像及3+1续接通过，状态和统计逐值一致。
壁面统计开启时，既有结束导出每rank约58 KiB的FP32壁面统计文件不属于数值几何回读。
其ASTRWS01家族头中版本1保留FP64布局，版本2标明FP32布局，整数计数及掩码不缩短。
共享测试读取器按版本选择载荷类型；没有新增附属文件或改变默认FP64布局。
FP32双卡完整TGV设备原位路径Memcheck两rank均为0 errors。

主要证据位于tests/gpu_validation/out/：
native_fp32_fp7_derivatives64_v4_20261021、native_fp32_fp7_real32_20261021、
native_fp32_fp7_real64_20261021、native_fp32_fp7_curve64_20261021、
native_fp32_fp7_curve_joint32_20261021、native_fp32_fp7_joint_archives64_20261021、
native_fp32_fp7_wall_safety32_20261021及native_fp32_fp7_wall64_20261021。
平面图像证据在native_fp32_fp7_extensions32/64_20261021的已完成记录中；
该轮后续检查误将壁面标量统计导出归为几何输出，修正分类并补齐格式标记后单独完成了壁面均值。
测试记录native_fp32_fp7_wall_format_20261021.xml验证两种原始壁面输出载荷宽度。

## 22. 纯计时与长窗监测准备

用户于2026-10-21明确将正式性能比较与长时精度评估分开，细则已同步主设计§9及§9.1。
ASTR_GPU_STATISTICS_INTERVAL默认1；TGV/平板设0关闭原有标量统计计算和写出，正整数用于稀疏采样。
槽道统计包含反馈驱动，不允许用该选项将其跳过。独立原位统计及平均累计仍由各自输入关闭。
ASTR_BENCHMARK_BUFFER_OUTPUT默认关闭，启用后将RK计时和crashfix摘要延后输出，
保留真实修复、保正及必要通信。小网格两种精度/显式紧致双卡8项检查中，权威场逐值一致。
原随机吹吸、NSCBC、滤波及crashfix开启的Ma12十步对照中，4组精度/格式组合的场和修复诊断也不变。

ASTR_GPU_BENCHMARK_NO_FIELD_IO增加既有三维平板路径，仍默认关闭，不影响物理边界。
纯计时准备器不复制内部生网格TGV的未用grid.h5，正式窗口不写场或标量统计。
实际双卡小型TGV已检查到无HDF5文件、无flowstate.dat输出；小型Ma12和紧致TGV的预跑有效。
这些预跑含资源采样，不作为正式性能结果。

ASTR_TGV_CONSERVATION默认关闭，仅显式启用时在现有统计相位归约两个原生标量：
密度域均值/roinf和总能量密度域均值/(roinf*uinf^2)，不回读整场，不引入累计补偿。
双卡短测按相同原生归约顺序独立重算，结果逐值一致，开启诊断不改变权威状态。
结束时另报告实际步数、物理时钟与dt，便于将FP32时间累计漂移和解的差异分开。

证据：native_fp32_fp8_runtime_controls_conservation_20261021.xml、
native_fp32_fp8_m12_output_controls_20261021/report.json、
native_fp32_fp8_no_grid_smoke32_20261021/summary.json及native_fp32_fp8_pure_plate32_20261021/summary.json。
准备/配对/计时汇总工具32项通过，见native_fp32_fp8_tools_20261021.xml。
正式32配置三轮性能矩阵及4个40000步TGV长窗尚未运行。

## 23. 独立资源观测与长窗工具

FP32双卡32^3实际原位程序的standard-device/direct-device均完成开关资源观测及独立Nsight捕获。
20 ms采样下，原位开启相对关闭的设备峰值差约290/272 MiB每卡，主机RSS峰值差约1.36/1.42 GiB
（两显示管线分别测量；峰值差不是阶段内部任意瞬态的分配证明）。设备空闲量始终高于已批准下限。
产品提取、流线和显示范围内未见大块D2H流场/几何载荷；存在小控制量、设备内拷贝和固定查表上传。
pinned halo及OpenGL颜色/深度图像读取另列，不把它们混称为零传输。
证据：native_fp32_fp7_observe32_20261021/report.json及各rank SQLite。

新增run_native_precision_tgv_long.py复用ASTR推进，仅负责四组配对启动、统计实读与离线比较。
32^3、12步、4步采样的显式/紧致及两种精度试跑完成，检查了相同步号、原生时钟及无场输出。
长时正式矩阵将额外逐完整步启用既有只读有限/正值检查，处于RK计时之外；
该长算例不作为纯性能测量。曲线分析保留实际统计相位，最后样本为39960步、名义t*=9.99，
40000步终止时钟另报，不将它冒充为末步后处理样本。分析新增的-dEk/dt仅为离线诊断。
plot_native_precision_tgv.py沿用项目科学作图风格，分别保存EPS/JPEG；
短试跑图仅检验绘图流程，不用于长时物理判断。

配对与时钟分析工具检查通过，见native_fp32_long_and_performance_tools_20261021.xml。
正式中等规模32配置预跑全部通过，所记录最大CFL为0.3262191，单卡采样显存峰值最大13.15625 GiB。
资源汇总为native_fp32_fp8_medium_20261021/preflight_resources.json；
带资源观察的RK时间不作为正式性能结果。补全逐步只读检查后的四组长窗短试跑也通过，
见native_fp32_tgv_long_statecheck_smoke_20261021/accuracy.json，明确标为非正式长时矩阵。
40000步长时矩阵尚未启动。

## 24. M12与只读预览范围

两种精度的64x32x24、NP=2 x分解、10步、dt=0.005 M12检查使用原随机吹吸及
11/22/41/52/1/1边界。standard-device下Q、物理切片和瞬时流线有效，开启处理不改变权威场，
5+5精确续算的状态与连续推进逐值一致，续接JPEG逐字节一致，源检查点不变。
证据为native_fp32_fp7_m12_insitu64_20261021，以及
native_fp32_fp7_m12_insitu32_v3_20261021中已完成的standard-device部分。
后者在随后尝试direct-device时被既有boundary_layer准入拒绝，不能将整个目录宣称为全通过。
本轮保留该预设仅支持标准入口的边界，不为凑齐测试组合放宽原范围。
两显示入口的匹配精度已由TGV/CURVE的对应产品验证。

M12测试前两轮分别误要求纯显示色标1.1变成FP32舍入值、误将原有“非图例几何非空”
检查改成至少500个几何像素；已恢复既有显示元数据与图像判据，未改数值或相机。
Q在这个短时层流夹具上仅占少量像素，另由全局非零单元数检查几何存在。

两种精度32^3、NP=2的冻结状态只读预览完成6个候选（瞬时、Reynolds及Favre）检查，
零推进、场/统计不变、源目录不变，改变相机/色标复用轨迹缓存，改变种子/速度类型重新积分。
数值几何无主机回读，默认仅JPEG。证据：native_fp32_fp7_preview32_20261021及
native_fp32_fp7_preview_observe64_20261021，其中FP64也补齐了两管线资源/传输独立观测。

## 25. 后续执行顺序

1. FP8完成本地256^3 TGV、450x130x96 Ma12的NP=1/2独立通信报告，再执行交错纯RK计时。
2. 性能矩阵沿用CP9的零吹吸幅值计时夹具，原随机吹吸功能由FP3及本轮M12原位短测单独覆盖。
3. 按新批准范围运行4个NP=2、256^3、40000步TGV精度对照，报告完整统计曲线及峰值/时钟漂移。
4. 核对受影响FP64回归与用户说明。当前小组件或含观测的短测耗时不作为正式性能结论。

完整FP0-FP8 goal继续有效，本记录没有将其缩为已完成的小网格矩阵。
当前无需要人工决定的物理/离散修订；未操作Git、其他worktree、子代理或远程任务。

## 26. 正式计算批次已启动

2026-10-21完成8组NP=2、滤波开启的独立Nsight捕获，覆盖TGV/Ma12、显式/紧致、FP32/FP64。
在所选公共halo标签21001/21002和紧致接口标签24001/24002中，记录的API消息/接收缓冲
字节数FP32均为FP64的一半。各API、标签和rank分开保留，不能将发送和接收相加称为唯一网络流量，
也不据此推断通信时间减半。整个捕获含初始化，CUDA/MPI/API耗时不能相加当作墙钟时间。
汇总：native_fp32_fp8_medium_20261021/communication_pairs.json，逐rank原始记录见profile/。
分析器已适配Nsight在无非阻塞等待事件时不导出MPI_START_WAIT_EVENTS表的行为；
首组既有捕获只重新分析，没有重跑。

当前源码的CPU主程序两种精度重建成功。16^3两步最小回归中，当前FP64与实施前冻结基线的
三个post_update阶段物理q及flowstate逐值一致，最大绝对差0；FP32实际载荷为float32且状态有限、正值。
证据：native_fp32_final_cpu_regression_20261021/report.json。

本地用户服务astr-native-fp32-validation-20261021.service已经启动，依次执行：

1. 32配置、每配置3轮，共96次纯RK测量，10步预热加30步计时；不运行profiler或资源采样。
2. 四个256^3、NP=2、40000步TGV统计对照；无检查点或场输出，逐步只读有限/正值检查开启。
3. 离线统计汇总及EPS/JPEG曲线。任一命令失败即停止后续阶段，不自动更改配置重试。

启动脚本为tests/gpu_validation/run_native_precision_campaign.sh。
性能日志为native_fp32_fp8_medium_20261021/timing.log，长窗启动/结果日志为同目录long_tgv.log。
长算例目录为native_fp32_tgv_long_20261021；在正式计时结束前不创建该目录并行运行。
当前只确认首个正式计时配置正常结束，不将单次结果当成三轮性能结论。
goal保持active，尚未完成FP8和长时精度评估。

首个双卡40步计算正常结束，但合并stdout时两rank的末尾批量计时行发生字节片段交错，
计时汇总因缺行而失败。已仅修改测试启动与读取：buffered模式用Open MPI的
--output-filename保留各rank独立stdout/stderr，计时和长窗终止时钟从各rank文件读取。
不修改CUDA/Fortran求解器、二进制、原计时窗口或数值状态。
17项工具检查及四组32^3长窗短试跑通过；新的双卡40步记录已完整通过汇总。
原缺行目录保留在timing_incomplete/，不猜测缺失时长、不纳入结果。
同一服务已恢复，显式复用两组有效单卡记录，继续其余配置；未重跑全部预检或通信捕获。
最终状态以服务和各summary.json为准，以上初轮失败不计为数值失效。

## 27. 待决定的旧主机兼容入口

按D9b补查发现，scripts/insitu/tgv_streamlines.py的旧主机兼容路径用
ProgrammableSource创建double种子，并调用VTK StreamTracerWithCustomSource。
本地ParaView 6.1.1依赖的vtkStreamTracer.cxx使用double积分位置、速度、权重及误差，
不是仅图形投影使用double。因此不能把已有设备产品的FP32通过证据扩展到该旧数值路径。
该问题不影响已验证的standard-device/direct-device私有FP32轨迹或当前纯计算批次。

已向用户提交两种处理范围，尚待答复：本轮明确拒绝旧FP64数值路径，仅认可已验证的设备
数值处理；或者补做旧主机入口的原生FP32提取/积分。FP64原有路径不修改，不静默放行FP64
数值回退。没有据此将完整goal标为完成，也没有取消已批准的性能/长窗计算。

另核查product_dispatch.py的4个ctypes.c_double时间字段：其Fortran绑定
adaptive_info_c明确输出real(c_double)，只是把原生调度状态复制给输出元数据，未用于更新时钟或判据，
不存在float指针被按double读取的ABI错误。数值调度仍使用real_kind。

新增report_native_precision_fields.py负责复用冻结短测，统一输出守恒量、rho/u/v/w/p/T、
Ma12瞬时展向平均剖面及原有壁面标量差值；只作离线FP64报告，不运行CFD或反馈到求解器。
共享节点和物理/周期末端所有权6项合成检查通过，见native_fp32_field_report_tools_20261021.xml。
为避免主机内存/磁盘处理干扰纯RK计时，真实冻结数据报告待正式计时结束后执行。

## 28. FP8中等规模正式性能结果

本地两张RTX 4000 Ada，各rank绑定一张GPU。TGV为256^3区间、dt=2.5e-4；
Ma12为450x130x96区间、dt=0.005，743e MP7-LD对流和NSCBC边界，计时夹具关闭吹吸幅值。
显式/紧致分别为TGV的643e/643e、643c/643c，以及Ma12的743e/643e、743e/643c。
各组滤波开启时使用相应十阶滤波。scalar工作区、explicit同步、公共及私有pinned通信均相同。
关闭渲染、场/检查点及只服务输出的统计，保留真实边界、通信、滤波和修复。

三轮交错共96次已完成。每次10步预热、30步采样，以每rank累计RK窗口耗时的最大值除以30；
不含启动、完整步间输出或分析工具。下表为三轮中位数，单位ms/完整RK步。
末列是同格式、同NP下FP64耗时/FP32耗时，不是相对CPU的加速比。

| 算例 | 格式 | 滤波 | NP | FP64 ms | FP32 ms | FP64/FP32 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TGV | 显式 | 关 | 1 | 380.876 | 194.706 | 1.956 |
| TGV | 显式 | 关 | 2 | 394.136 | 194.647 | 2.025 |
| TGV | 显式 | 开 | 1 | 456.024 | 238.401 | 1.913 |
| TGV | 显式 | 开 | 2 | 489.397 | 248.834 | 1.967 |
| TGV | 紧致 | 关 | 1 | 2462.334 | 1394.045 | 1.766 |
| TGV | 紧致 | 关 | 2 | 1519.598 | 823.253 | 1.846 |
| TGV | 紧致 | 开 | 1 | 2973.754 | 1669.519 | 1.781 |
| TGV | 紧致 | 开 | 2 | 1867.584 | 995.933 | 1.875 |
| Ma12 | 显式黏性 | 关 | 1 | 883.665 | 161.498 | 5.472 |
| Ma12 | 显式黏性 | 关 | 2 | 558.217 | 128.690 | 4.338 |
| Ma12 | 显式黏性 | 开 | 1 | 934.331 | 182.305 | 5.125 |
| Ma12 | 显式黏性 | 开 | 2 | 605.593 | 148.878 | 4.068 |
| Ma12 | 紧致黏性 | 关 | 1 | 1548.669 | 563.408 | 2.749 |
| Ma12 | 紧致黏性 | 关 | 2 | 899.331 | 335.525 | 2.680 |
| Ma12 | 紧致黏性 | 开 | 1 | 1760.769 | 654.968 | 2.688 |
| Ma12 | 紧致黏性 | 开 | 2 | 1021.485 | 394.744 | 2.588 |

TGV显式在双卡上未获得中位数扩展收益；滤波开启时FP64/FP32的NP=1至2加速分别为0.932/0.958。
紧致对应1.592/1.676。Ma12滤波开启时，显式黏性的扩展加速为1.543/1.225，紧致为1.724/1.659。
消息载荷减半并不等于通信时间减半或扩展效率提高。这些是本机结果，不外推A800。
同NP=2、滤波开启时，紧致/显式耗时比在TGV为3.816/4.002（FP64/FP32），Ma12为1.687/2.651。

每组的3个原值、最小值/最大值均保留在performance.json/csv及96个summary.json。
FP64最大三轮跨度/中位数为0.736%；FP32最大为TGV显式滤波NP=2的10.851%，
原样本为222.015/249.017/248.834 ms，没有剔除最快样本或用它替代中位数。

Ma12显式滤波组的两种精度、两种NP及全部三轮均报告3次非零修复调用、291次节点修复访问。
节点访问含存储副本，不是291个唯一物理点；计数覆盖完整40步，不能说成只发生于后30步。
逐分量点值最大改变量另列，不能把它当成积分质量或能量缺陷。其余受测组合未报告非零修复。

独立20ms资源采样中，NP=2、滤波开启的最大单GPU峰值如下，含运行库和可控工作区：

| 算例/格式 | FP64 GiB | FP32 GiB |
| --- | ---: | ---: |
| TGV显式 | 4.758 | 2.490 |
| TGV紧致 | 7.791 | 4.023 |
| Ma12显式黏性 | 1.934 | 1.039 |
| Ma12紧致黏性 | 3.668 | 1.930 |

这是采样进程显存峰值，不是理论最小显存或所有瞬态的分配证明。全32配置的主机RSS、
设备峰值和紧致受控数组字节保留在preflight_resources.json及performance.json，
主机量为进程树RSS之和，不声称等于去除共享页后的物理占用。
证据根目录：tests/gpu_validation/out/native_fp32_fp8_medium_20261021/。

## 29. 冻结短测的跨精度场与剖面汇总

report_native_precision_fields.py复用既有不可变运行，未重跑CFD。比较step9/RK3的pre_rhs
守恒量及同相位rho/u/v/w/p/T，排除存储重复节点但保留物理壁面端点，按相同逻辑节点配对。
离线计算使用FP64，不反馈求解器；没有跨精度误差上限。下表为各自短测矩阵的最大绝对差，
不是长时结果，也不是所有阶段最大值：

| 短测组 | q最大差 | u最大差 | T最大差 |
| --- | ---: | ---: | ---: |
| 32^3 TGV显式 | 4.298005e-4 | 3.937850e-6 | 2.734889e-6 |
| 32^3 TGV紧致 | 1.109881e-2 | 3.349726e-5 | 2.010527e-5 |
| 96x48x32 Ma12显式黏性 | 9.361247e-6 | 9.456364e-6 | 3.386879e-4 |
| 96x48x32 Ma12紧致黏性 | 1.608052e-4 | 2.163270e-4 | 1.016485e-2 |

完整报告保留五个守恒分量、六个原始变量、峰值逻辑位置、max(1,参考分量全域最大值)缩放差及统计序列差。
Ma12另给x/L约0.25/0.5/0.75的瞬时展向平均剖面及两种精度的实际坐标，不把它们称为时间平均。
fbcx沿用旧底壁面积平均mu(T) du/dy，wallheatflux沿用面积平均k(T) dT/dy的原符号，
不重命名为“流入气体的热流”。这不是壁面格式修改或外部物理验收。
Ma12显式/紧致组的fbcx最大绝对差分别1.248897e-10/1.090224e-7，
wallheatflux分别1.307217e-10/8.054702e-8。相同时钟步号下的原生时间差单独保留。

证据目录为native_fp32_fp8_fields_tgv_explicit_20261021、tgv_compact、m12_explicit、m12_compact
（后三项沿用相同native_fp32_fp8_fields_前缀和_20261021后缀），各有report.json；
Ma12两组还各有12个剖面CSV。汇总工具24项检查通过：native_fp32_fp8_final_reporting_tools_20261021.xml。

## 30. 长时TGV推进历史

本节保留2026-10-21至22日的阶段记录；四组均已结束，最终结论以§31为准。

三轮计时结束后服务已自动启动显式FP64的256^3、NP=2、40000步TGV。
初始统计正常，40步一次采样、完整步逐次有限/正值检查；没有场或检查点输出。
其后仍依次执行显式FP32、紧致FP64和紧致FP32，不并行混跑。
按本次NP=2滤波组中位数估算，四组纯RK合计约40.0小时；实际墙钟还含主机循环、
检查和轻量统计，且长时流场下成本可能变化。该估算不是承诺完成时刻。
未缩短已批准窗口，长时精度、峰值和守恒结论仍待4组真实结果。

在等待首组推进时修正了离线共同物理时钟对齐：只在两组实际时钟覆盖的交集中插值，
记录重叠区间和参考样本数，不使用np.interp的区间外端点延拓。
同一步号/名义时间比较不变。当前长任务在启动时已加载旧分析函数，
因此4组结束后须用现行run_native_precision_tgv_long.py --mode analyze重新汇总，
再更新图像/结论；不重跑CFD，不据进程内旧报告宣称完成。

### 30.1 显式FP64长窗完成

643e_fp64已完成40000步，逐完整步状态检查及实读门槛通过，墙钟20455.678 s（5.682 h），
累计RK最大rank窗口19455.523 s。该长任务含额外检查/统计，不替代§28的独立纯计时结果。
终止时钟9.99999999999937，名义终点10；诊断采样窗仍为0.01至9.99。
拟涡能离散采样峰值8.163957623，位于step35160、名义t*=8.79；
黏性耗散峰值0.01024583230位于同一采样步。质量均值最大漂移2.241873354e-12，
总能量均值最大绝对漂移2.870024218e-10，末次相对漂移-1.606089300e-12；
这些漂移包含原生归约舍入。crashfix关闭，未报告修复。

证据：native_fp32_tgv_long_20261021/643e_fp64/summary.json及
643e_fp64_diagnostics.json。随后显式FP32已自动启动。
上述峰值不等于DLR 512^3背景峰值，也不是FP32精度损失；后者须由同格式配对确定。
本轮没有为了接近外部曲线而改变网格、滤波、物性或dt。

### 30.2 显式FP32长窗及配对结果

643e_fp32也已完成40000步并通过逐步状态和统计实读检查，墙钟10780.932 s（2.995 h），
最大rank累计RK窗口9907.489 s；没有激活crashfix。随后服务自动进入643c_fp64。
完整四组尚未结束，以下报告明确命名为explicit_pair_partial.json，不作为总矩阵完成标志。

| 量 | FP64 | FP32 |
| --- | ---: | ---: |
| 拟涡能采样峰值 | 8.163957623 | 8.178306580 |
| 峰值步号/名义时间 | 35160 / 8.79 | 35160 / 8.79 |
| 黏性耗散采样峰值 | 0.01024583230 | 0.01025063731 |
| 末次密度均值相对初值变化 | -2.241873e-12 | +1.183510e-3 |
| 末次总能量均值相对初值变化 | -1.606089e-12 | +8.385243e-4 |
| 40000步终止时钟 | 9.99999999999937 | 9.99741268157959 |

拟涡能峰值相对同格式FP64增加0.175760%，两个离散峰值均在同一步，采样名义间隔0.01。
按相同步号比较整个0.01至9.99窗口，动能/拟涡能/黏性耗散最大绝对差分别为
1.008234e-4、1.510086e-2、5.362085e-6；分别以FP64对应曲线峰值缩放后为
0.080662%、0.184970%、0.052334%。这些是已完成显式配对的报告值，不是新设通过阈值。

FP32密度均值与总能量均值出现可测累计漂移，不能用主曲线视觉重合宣称守恒无损。
这些数值包含原生归约舍入；本轮没有增加补偿、重归一化或修复来压低结果。
守恒量的末次采样仍在step39960，终止时钟另在step40000记录。
FP32时钟相对名义10的偏差为-0.00258731842，按步数乘目标类型dt的累计偏差为
-0.00258779339；不能把这项记录时钟漂移混入拟涡能峰值的步号变化。
共同原生时钟的插值比较仅使用重叠区间，实际采用998个参考样本，没有区间外延拓。

已生成并检查主曲线、FP32-FP64有符号差值、密度/总能量漂移和单独时钟漂移图，
合计9组EPS/JPEG（18个图像文件），位于native_fp32_tgv_long_20261021/figures_explicit/。
两个源曲线CSV、explicit_pair_partial.json及figure_sources.json保留定义和来源。
DLR灰色背景曲线与本轮FP64已有明显峰值差别，不归为FP32损失，也不宣称严格复现外部DNS。
本段不替代仍在运行的两组紧致长窗或§27的兼容入口决策。

## 31. 四组长窗及性能后处理完成（2026-10-22）

四个256^3、NP=2、40000步TGV均正常结束，逐完整步状态检查通过，crashfix关闭。
长任务服务最终exit=1发生在绘图而非CFD：Matplotlib外部TeX缺少ptmr7t.tfm。
本轮只将该绘图脚本改用MathText，沿用科学样式和红蓝配色，不安装/指定新字体。
使用现行run_native_precision_tgv_long.py --mode analyze重新读取四组原始统计与日志，
共同原生时钟只取交集，共998个参考样本，没有使用进程内旧分析的端点延拓。

| 格式 | 拟涡能峰值FP64 / FP32 | 峰值名义时间FP64 / FP32 | 峰值相对差 | FP32密度 / 总能量末次相对漂移 |
| --- | --- | --- | ---: | --- |
| 显式 | 8.163957623 / 8.178306580 | 8.79 / 8.79 | +0.175760% | +0.118351% / +0.083852% |
| 紧致 | 9.549039793 / 12.513916016 | 8.83 / 8.98 | +31.048946% | +22.280437% / +22.445731% |

紧致FP32不是只出现峰值时刻偏移：其动能采样最大值在t*=4.12升至0.131150544，
高于本轮初始值约0.125。质量和总能量均值同步持续上升。共同原生时钟插值后，
紧致拟涡能最大绝对差仍为3.281445，不能用约0.00259的累计时钟漂移解释。
质量/能量漂移包含原生归约误差，但不能在未拆分来源时将22%量级称为统计噪声。
本轮不修改紧致系数、PCR、滤波、RK或精度合同，也不认定某一项已经是唯一根因。
严格FP32紧致路径应保留为待诊断候选，不推荐直接用于长时生产。

96次既有正式计时已重新汇总，完整16组配对值见§28和新报告。
FP64/FP32耗时比在TGV为1.766至2.025，Ma12为2.588至5.472；
不是CPU加速比，也不是同等物理误差下的效率比较。滤波开启时TGV显式双卡仍无扩展收益。

完整后处理报告：[ASTR_NATIVE_FP32_VALIDATION_REPORT.md](ASTR_NATIVE_FP32_VALIDATION_REPORT.md)。
新报告包含定义、性能矩阵、扩展/容量、四组长时统计、冻结短测场差和待解决问题。
最终长时图为native_fp32_tgv_long_20261021/figures/的18组EPS/JPEG；
性能/容量/扩展及Ma12近壁剖面图为native_fp32_postprocessing_20261022/的12组EPS/JPEG。
两个目录的figure_sources.json或postprocessing_summary.json记录来源；原始CFD未重跑。
§27兼容入口决策仍未解决，不能将本次后处理完成写成原生FP32整体goal完成。
本次27项离线分析测试通过，30份JPEG可读且非空白、30份EPS经Ghostscript解析通过。

## 32. 四组完整检查点补跑已启动（2026-10-22）

用户要求补齐长窗三维场证据，四组从头重跑，不覆盖§31统计量结果。
网格256^3、NP=2、2x1x1、dt=2.5e-4、每组40000步、对应十阶滤波开启，
顺序为643e FP64、643e FP32、643c FP64、643c FP32。
数学与编译二进制沿用首轮，只新增完成步checkpoint；crashfix仍关闭。

每5000步写一次，exact_restart=t、keep=2；完整q/物性缓存/halo及共享几何保留。
最终目标是每组仅保留step000000035000和step000000040000，当前尚未达到首个5000步事件。
主机/设备预算字段提高为64 GiB，不预分配该容量；无额外资源采样硬停、无任务时限、无CRC全场重读。
volume/slices/原位渲染关闭，原有轻量统计继续每40步输出。
ASTR_BENCHMARK_BUFFER_OUTPUT继续只缓冲日志；启用checkpoint时不再设置ASTR_GPU_BENCHMARK_NO_FIELD_IO。

启动前16^3、NP=2、四配置均完成8步，真实写入2/4/6/8四次，最终仅留下6/8两个批次。
八份保留state.h5均含11个原生类型的全局场分量及正确维度，几何资源存在。
首轮夹具遗漏了由调用方建立的空输出根目录，已补接现有目录约定；成功证据为
native_fp32_tgv_checkpoint_smoke_20261022_v2/，没有为此修改求解器或放宽文件有效性语义。
相关离线测试29项通过，shell语法与git diff --check通过。

正式服务astr-native-fp32-tgv-checkpoints-20261022.service于2026-10-22 20:58:10启动，
使用systemd用户服务保活，Linger=yes。运行根为
tests/gpu_validation/out/native_fp32_tgv_checkpoints_20261022/。
首组已完成约2.0 GiB的geometry.h5初始化写盘并进入RK推进，nvitop确认两卡均在计算。
原程序已复制到新目录bin/astr_fp64和bin/astr_fp32，避免后续重编影响串行后续组。
campaign.log记录各组进度，specification.json记录5000间隔/keep=2/exact配置。
本节记录的是启动，不是四组补跑完成；服务结束后再核对35000/40000的实际留存。
