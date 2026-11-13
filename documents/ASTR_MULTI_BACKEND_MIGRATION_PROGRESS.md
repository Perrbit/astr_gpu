# ASTR 多后端迁移进度

更新日期：2026-11-13。主计划：`ASTR_MULTI_BACKEND_MIGRATION_PLAN.md`。
本轮 goal：完成 MB0--MB3，不包含完整输出/恢复、KEEP/ME6、紧致格式、
复杂物理边界、AIR5 或原位渲染。不能据本轮完成宣称完整 ASTR 已适配 DCU。
2026-11-13：MB0--MB3的工程与短窗数值门槛已完成，DCU多卡证据见第7.2节。

## 1. 当前状态

| 阶段 | 状态 | 已有证据或缺项 |
|---|---|---|
| 参考冻结 | 已完成 | 独立源码归档、CPU/CUDA Fortran 二进制及构建记录；周期 TGV 新鲜交叉检查通过 |
| MB0 本地工具链 | 已完成组件检查 | Fortran/C ABI、多文件设备调用、512线程归约、非默认流/事件、HDF5摘要往返；内存与共享内存竞争检查通过 |
| MB0 DTK 26.04 | GNU7及新GNU12组合组件通过 | 新作业127800728完成，NP=2不同DCU，归约/D6及私有HDF5往返通过 |
| MB1 工程与状态 | 普通astr已在本地及DCU接通 | 根构建明确none/cuda/gpufusion；旧CUDA Fortran独立；use_gpu=f的同二进制CPU路径逐值一致；关闭测试/探针仍能构建运行 |
| MB2 单rank TGV | 本地及DCU普通主程序通过128³/20步代表检查 | 固定/自动步长、TGV标量与CFL通过；本地另核对同步两模式及内存工具，不扩大为所有算例支持 |
| MB3 多rank TGV | 本地NP=1/2及DCU NP=1/2/4通过 | NP=2 x/y/z、DCU NP=4 2×2×1；本地315/DCU495个交换相位；场、缓存、节点协调和观测/同步隔离通过 |

## 2. 冻结参考

- 起点：`0d0f367f13d01221279f18779063e64720ec2ea5`。
- 独立归档：`/home/dell/workspace/astr_gpu_reference_0d0f367.tYkRm3`，
  全目录设为只读；`reference.json` 记录源码树、二进制来源和离线散列。
- `source/` 不与当前工作区共享可变源码；`bin/` 保存参考可执行文件，
  `build_records/` 保存原缓存、编译/链接设置和库解析记录。
- 参考范围仅为 FP64、standard 643e/643e、十阶显式滤波的周期五变量 TGV。
- 新鲜参考检查：16³ 区间、4个完整步、固定 dt=1e-3，CPU/GPU NP=1及
  NP=2 x/y/z 分解。最大缩放场差为 `1.1147314445468803e-15`；
  现有短测同时覆盖同拓扑精确续接及变更滤波配置的拒绝行为。
- 证据：归档内 `evidence/standard_tgv_explicit10/`。这些结果只证明参考对
  在该范围的一致性，不是新 C++ 后端的实现或物理验证。

## 3. MB0 本地组件证据

目标 `astr_backend_mb0` 由根 CMake 的
`ASTR_BUILD_GPU_BACKEND_PROBES=ON` 启用，不编译旧 `.cuf`。
源码与构建说明位于 `tests/gpu_backend/`。

| 项目 | 实际检查 | 结果 |
|---|---|---|
| Fortran/C ABI及索引 | 1031/65537个FP64值，输入不变，二进制可精确表示的变换 | 逐值相等 |
| 跨设备翻译单元 | 主核调用另一个 `.cu` 中定义的非内联设备函数，执行设备链接 | 本地编译、链接、运行通过 |
| 六阶周期导数 | 独立Fortran参考，逐值比较并按全场最大值缩放 | 最大 `5.5511151231257827e-17` |
| 512线程归约 | 共享内存树形归约，长度不整除512，无32线程宽度假设 | 本地两种长度结果与参考相等 |
| 流/事件与复制 | 两个非默认非阻塞流、事件依赖、固定页主机缓冲 | 数值检查通过 |
| MPI/HDF5 | NP=1/2归约；摘要关闭后重新打开读取 | 通过；HDF5逐值相等 |
| Compute Sanitizer memcheck | 显式主机中转MPI环境、NP=1 | 0 errors，0 bytes leaked |
| Compute Sanitizer racecheck | 同一探针、NP=1 | 0 hazards，0 errors，0 warnings |

本地环境：NVHPC26.1、CUDA13.1、HPC-X、HDF5 1.14.6、双 RTX4000 Ada，
驱动595.99.02。证据目录：`results/multi_backend_mb0_20261112/native/`。
这里的NP=2只验证探针调用和标量MPI，不是多rank TGV或halo通信完成。

实际遇到并处理的构建/环境问题：

1. 直接使用SDK内底层 `cuda/13.1/bin/nvcc` 时，编译器识别遇到glibc
   `rsqrt`声明冲突。改用现有ASTR同样使用的SDK `compilers/bin/nvcc` 驱动；
   没有修改系统头文件或SDK。
2. CUDA链接语言下，NVHPC Fortran主程序需要位置无关对象及 `f90main.o`；
   只在探针目标中补齐，不改变原ASTR链接方式。
3. 独立启动曾混入系统MPI库；探针加入所选Fortran工具链的运行库路径，
   实际测试用匹配的MPI启动器。固定页中转测试显式使用 `osc=pt2pt`；
   未设置时UCX在MPI初始化的 `cuCtxSetFlags` 中触发Sanitizer错误，
   发生于首个计算核前。保留原记录，不把它算作内核检查通过。

## 4. DCU 环境与当前障碍

工作根目录：远端 `~/weiph/backend_mb0_20261112_1/`。
全部源码、构建、HOME替代目录、临时目录及缓存均在该范围内。

- DTK：`/public/software/compiler/rocm/dtk-26.04`。
- GPUfusion前端：`cuda/cuda-12/bin/nvcc`；不是旧版 `cuda/bin/nvcc` 布局。
- 公共MPI：HPC-X2.11.0 / GNU7.3.1。
- 公共HDF5：1.12.2 `gnu_fortran_parallel`，其构建记录使用上述MPI/GNU组合。
- 公共CMake：3.20.4；默认系统2.8.12.2不用于本轮构建。
- GPUfusion的设备编译、最终链接、512线程实际运行、设备/驱动及流依赖
  仍必须以新版平台实测成立，不借用DTK22.10.1旧探针或26.10手册作结论。

login04上CMake先记录CUDA编译器识别命令超时，后续DCC进程停在
`lock_blkgrp_sync`。已按本次确切PID结束编译及其设备编译子进程，
原会话返回143。此为主动结束停滞构建，不是探针数值失败。
具体文件系统锁根因尚未证明；没有调整公共环境或添加自动重试。
原日志为 `build.log`、`build_2.log`。`login09/`独立构建已通过，完成真正的
跨翻译单元设备链接和Fortran主程序最终链接，CUDA编译器身份为12.6.77。
动态库解析到已选择的GPUfusion、HPC-X与HDF5。该结果缩小了问题范围，
但不能据此断定login04具体锁故障，也不代表计算节点驱动或数值结果通过。

GPUfusion报告不支持CMake自动加入的 `-forward-unknown-to-host-compiler`；
当前未依靠该选项传入额外数学设置，不能据NVIDIA版本号推断所有选项受支持。
自定义最终链接模板不再主动添加该选项。设备核补显式 `return` 以消除其
“void function is missing a return statement”前端警告，不改变数学运算。
最终根CMake重建通过，返回警告已消失；本地留存构建记录为
`results/multi_backend_mb0_20261112/dtk2604/`。
探针分别记录编译头版本、CUDA运行时API版本和CUDA驱动API版本；
GPUfusion的这些API值不等同于实际DCU内核驱动版本，后者仍需作业内核对。

### 4.1 已批准的单卡运行

用户已单独批准1节点、1rank、1DCU、2CPU核、5分钟的MB0任务及该任务查询。
仅提交作业 `127785220`，状态为 `COMPLETED`，退出码 `0:0`，运行6秒，
节点为 `b15r1n00`，应用日志时间为18:34:15--18:34:19。

- 实际设备：`Device 66a1 gfx906:sramecc-:xnack-`，只可见已分配的设备0。
- API与kernel的执行宽度均为64，512线程共享内存归约通过。
- 跨文件设备变换、输入保持、归约结果与参考逐值相等；D6最大缩放差为
  `5.5511151231257827e-17`；HDF5摘要关闭/重开后逐值相等。
- HDF5实际运行版本1.12.2。摘要下载后直接读取六个误差值为
  `[0,0,0,0,5.5511151231257827e-17,0]`。
- SMI报告实际驱动 `5.6.20a`。DTK环境仍提示推荐较新驱动；没有升级驱动。
  当前结论只覆盖实际通过的组件组合，不据此声称全部26.04功能受支持。
- 编译头版本12060、运行时API原值60326113、驱动API原值12060，三者分开
  记录；不把GPUfusion模拟的CUDA版本号解释成实际内核驱动版本。
- 本地证据：`results/multi_backend_mb0_20261112/dtk2604/mb0-127785220.log`
  和同名HDF5摘要。提交脚本：`tests/gpu_backend/run_mb0.slurm`。

没有运行登录节点GPU程序，没有操作A800或其他DCU任务，没有Git写操作。
此授权不包含后续MB1/MB2/MB3任务提交。

## 5. MB1 状态基础

新增 `src_gpu/device_state.h`、`device_state.cu`、`device_state.F90`。
每个上下文持有自己的非默认流和三个FP64缓冲，Fortran只持有不透明句柄。
视图记录类型字节数、实际下界、长度及元素步幅，设备地址不在主机解引用。
上传/回读是显式完成的初始化和诊断接口，不是每步整场传输策略。

创建状态时仅分配q、qrhs、qsave。MB2的fluid_prepare另按需分配原始量、
度量及可选梯度；没有为化学或原位统计预分配工作区。数值核同时检查debug=0
的流依赖模式与debug=1的逐核完成模式。该基础组件批次时普通astr尚未接入；
之后的普通主程序与多rank结果见第6、7节。

`astr_backend_mb1`通过独立Fortran编码数组和CUDA视图消费者，检查7/5/3个
区间、hm=3/5、全部分量/halo反向索引、物理q到RHS映射、qsave隔离、
q不变和销毁句柄失效，比较FP64位模式，包括负零。NP=2为每rank一个上下文，
不是跨rank halo通信。证据位于 `results/multi_backend_mb1_20261112/native/`。
该批次根CMake使用显式探针目标构建，未引入 `.cuf` 依赖，未作为正式推进证据。

本地NP=1的Compute Sanitizer memcheck与initcheck均为0 errors，memcheck
报告0 bytes leaked。状态库与MB1探针也已在DTK26.04下编译、设备链接和
最终链接通过，记录为 `results/multi_backend_mb1_20261112/dtk2604/build_state.log`。
后续单独批准的MB0/MB1双卡任务及其启动上限问题见第5.3节，未借用MB0
单卡授权提交MB1。

### 5.1 完整Fortran宿主构建的工具链缺口

使用同一GNU7.3.1/HPC-X/HDF5组合，在登录节点构建根目标 `astr` 已实际失败：
`src/adaptive_output.F90` 的 `FINDLOC(...,dim=1)` 无法识别，首批报告包含
339/346/469/473行。不是GPU核数值失败，也不是FINDLOC算法本身的缺陷。
不修改这些既有功能来兼容旧编译器，不把组件探针通过当作宿主已可构建。
原日志：`results/multi_backend_mb1_20261112/dtk2604/build_fortran_host.log`。

公共环境只读检查发现：

- 已安装GNU12.2.0。
- 已安装Open MPI4.1.5，其 `mpi.mod` 报告GFORTRAN模块格式15；这是候选
  信息，不是GNU12/MPI完整运行ABI已验证的证据。
- 查到的GNU并行HDF5仍为GNU7/HPC-X组合；GNU串行HDF5不能据此代替并行库。
  另有Intel2020的并行HDF5，需要不同Fortran/MPI组合，尚未尝试或选用。

用户已批准在 `~/weiph/` 使用现有GNU12.2.0和候选公共Open MPI4.1.5，
准备私有并行HDF5 1.14.6并核对新宿主组合。不升级DTK或驱动，不改变
FINDLOC或数值源代码；新组合失败则报告，不自动重建MPI、切换Intel或添加
兼容层。新组合的DCU作业验证仍须另行取得具体Slurm授权。

新工作目录：`~/weiph/backend_mb1_gnu12_20261112/`。
构建脚本：`tests/gpu_backend/build_private_hdf5.sh`；仅安装在该目录的
`host_env_clean/hdf5-1.14.6/`前缀。MPI包装器通过OMPI_CC/CXX/FC明确使用公共GNU12，
不依赖登录环境默认GNU7。测试、示例、工具、静态库和外部依赖自动下载关闭，
并行共享C/Fortran及高层库已实际安装成功，`libhdf5.settings`确认这些设置。
该安装成功尚不能代替完整宿主构建和新组合的计算节点验证。

过程中的两个问题分别保留记录：

1. 首次构建的登录环境带有旧HPC-X的OPAL_PREFIX及编译器搜索路径，导致
   调用公共Open MPI4.1.5包装器时仍引用GNU7模块。仅在构建子进程中清理
   CPATH/LIBRARY_PATH/C/C++头搜索路径、明确设置OPAL_PREFIX为所选MPI，
   然后在新目录构建；不修改登录配置，不重建MPI。修正后showme、编译器
   识别及实际Fortran模块编译均使用指定的新组合。
2. 初稿多构建了静态库和非必需工具，h5copy的构建make子进程收到外部
   SIGKILL。没有把它解释成数值或MPI错误，也没有证实外部终止的来源。
   去掉本轮不需要的工具和重复静态库目标后，所需共享库完整构建和安装
   返回0。旧日志为`host_env_clean/build_hdf5_interrupted_tools.log`，
   最终日志为`host_env_clean/build_hdf5_libraries.log`，均保留。

安装产物包括h5pfc、h5fc、h5hlfc及GFortran格式15的模块。指定运行库路径后，
Fortran库解析到私有HDF5、公共Open MPI4.1.5和GNU12的libgfortran.so.5，
不是旧HPC-X/GNU7。构建脚本同时显式提供公共zlib1.2.8路径，避免配置和
运行库搜索路径不同。

### 5.2 新宿主构建结果及最小源码类型修订

已用新组合从根CMake完成 `astr`、`astr_backend_mb0`、`astr_backend_mb1`
编译和链接。宿主目标的实际产物为
`host_env_clean/astr_cpp/build/bin/astr`；构建记录包含明确的
`[100%] Built target astr`。它仍是现有CPU宿主，不是已接入C++流体核的新GPU
主程序，不能据构建成功宣称MB1/MB2整体完成。

新HDF5的h5pfc只暴露基础Fortran库，h5hlfc才暴露本项目要求的高层Fortran库。
测试构建脚本第四个参数现为显式包装器路径，而不是假定每个HDF5的h5pfc都
包含高层库。没有添加探测循环、包装器别名或回退；本组合使用h5hlfc。

GNU12还实际指出8处非标准的混合integer-kind构造器，位于
`src/insitu_air5_statistics.F90`、`src/insitu_wall_statistics.F90`和
`src/insitu_session.F90`。已仅给这些构造器显式指定integer(int64)，保留原有
最外层int(...,int32)、文件标识及字段顺序。统计运算和浮点数组未改动；
没有扩展int32记录的取值范围或升级文件格式。GNU12完整宿主及本地NVHPC
CPU目标均编译链接通过；本次未以该类型修订为由重跑完整AIR5物理矩阵。

本地旧构建目录中的CPU目标可以重建，但冻结目录内的源码和参考二进制
保持只读，未用上述新二进制替代冻结参考。

设备绑定补充了cudaGetDevice核对，MB0日志新增可见设备数和PCI域/总线/设备
信息。本地NP=2仍通过，两个rank分别报告17:00与73:00；状态层hm=3/5的逐位
检查和memcheck仍通过。新GNU12组合后续独立设备运行见下一节。

本地新组合构建证据保存在
`results/multi_backend_mb1_20261112/toolchain/gnu12/`。
单独批准的组件任务合并MB0 NP=2及MB1 NP=1/2，仅运行组件，不启动TGV推进。
该授权不覆盖后续新任务。

源码来自[HDF Group 1.14.6发布页](https://support.hdfgroup.org/downloads/hdf5/hdf5_1_14_6.html)，
实际下载地址为官方GitHub发行资产
`https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.6/hdf5-1.14.6.tar.gz`。
离线源码散列为 `e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b`；
它用于构建来源追溯，不加入程序运行或检查点I/O流程。

### 5.3 已登记的敏感编译设置

| 现有源码及条件 | 当前设置 | 新后端处理边界 |
|---|---|---|
| chemistry_filter_math/operator，AIR5启用，NVHPC | `-Kieee -Mnofma`；设备另加`-gpu=nofma,noflushz` | 保护分离IEEE乘加及误差补偿；MB8实际迁移时核对DCC等效语义，当前不移植、不假装继承 |
| 同一AIR5源码，GNU | `-ffp-contract=off` | 保留现有宿主设置；当前AIR5关闭，不改变其源文件属性 |
| solver_gpu及AIR5高寄存器设备例程 | NVHPC私有`-gpu=maxregcount:128`等 | 资源合法性设置，不是通用浮点语义；不照抄给DCC或对普通核全局套用 |
| 原位涡识别/产品及相应探针 | GNU `-ffp-contract=off`、CUDA `--fmad=false` | 当前原位迁移不在MB0--MB3内；继续保留旧实现的独立设置 |

状态层仅复制FP64位模式，不使用浮点归约或补偿；MB0导数探针的
该阶段的`--fmad=false`只用于C++探针，不改变原Fortran编译设置；
后续新C++流体核的明确浮点选项见第6节。

### 5.3 新GNU组合双卡组件运行

经单独授权提交作业127800728：1节点、2rank、2DCU、每rank 2核、5分钟。
实际完成状态为COMPLETED、退出0:0，耗时18秒，节点b14r4n12。
任务分配物理设备0、2，运行时映射为逻辑0、1；两个rank分别报告PCI
0000:04:00与0000:43:00，不是两rank误用同一卡。

- MB0 NP=2：512线程、执行宽度64、复制/变换精确，D6最大缩放差
  5.5511151231257827e-17，归约与HDF5 1.14.6摘要往返通过。
- MB1 NP=1/2：hm=3/5的索引、全halo/分量、负零及字段隔离逐位检查通过。
- 运行时同时警告MB1 reverse_kernel的默认launch bound为256，而启动为512。
  因此不能仅用数值PASS关闭此问题。测试消费者两核已补显式
  __launch_bounds__(512)，本地重建通过；当批尚待DCU复核，之后已在
  第6.1节的127808864组件批次确认通过且未再报告该启动上限警告。
- 实际驱动仍为5.6.20a，未升级。DTK推荐新驱动的提示继续保留为环境边界。

脚本为tests/gpu_backend/run_mb01_gnu12.slurm；日志、MB0 HDF5摘要及分项日志
保存在results/multi_backend_mb1_20261112/toolchain/gnu12/。
此作业已结束，其授权不包含提交后续流体组件任务。

## 6. MB2 组件实施

新增fluid_explicit.h/.cu/.F90及内部device_state_internal.cuh。状态仍由
不透明句柄持有，字段保持Fortran顺序。原始量和度量按需准备；梯度、
Jacobian、应力和能量黏性通量只在相应模块启用时分配。

已完成本地组件检查：

| 项目 | 对照与结果 |
|---|---|
| q到原始量 | 冻结CPU和旧CUDA Fortran各3个RK阶段，逐变量缩放差小于2e-10 |
| 原始量到q | 使用原温度型能量定义，不以压力重算替换；同上6个冻结状态通过 |
| D6及度量投影 | 独立修正波数制造解、全部9个速度梯度和3个温度梯度，NP=1/2通过 |
| CFL | 编译原cfl_spectrum作为独立参考，检查3方向极大值及MPI后的C_upper之和 |
| 对流/完整RHS | CPU/旧GPU共6个冻结状态，5分量逐项比较，最大缩放差2.6644539439374393e-14 |
| 本地周期 | 端点协调、最终halo-only刷新、未使用角区保持，全部位模式一致 |
| 十阶滤波 | x/y/z顺序及中间halo刷新，初始滤波状态最大缩放差4.749911492783413e-16 |
| RK3更新 | 3阶段分别对照原始相位，最大缩放差3.3306690738754696e-16 |
| 4完整步耦合 | 同一设备状态组合滤波、原始量刷新、梯度、对流/黏性及RK；对照两种冻结参考第2/4步，最大缩放差1.2666454347843286e-15 |
| 执行与失败 | debug=0/1；非正密度、非法声学谱拒绝，失败状态不被后续有效调用清除 |
| 内存/竞争 | 本地组件memcheck为0 errors/0 bytes leaked，racecheck为0 hazards |

最新基础组件证据为results/multi_backend_mb2_20261112/native/components_final/；
当时最大缩放差1.6971225417643732e-15。memcheck_final.log、
memcheck_frozen.log和racecheck_final.log分别保存对应工具输出。
新增算子的最新完整矩阵为native/coupled_4steps/report.json，包含14个测试进程
配置，每个配置覆盖debug=0/1。CUDA C++库显式使用--fmad=false，未启用
fast-math。新耦合路径memcheck_coupled_4steps.log为0 errors、0 bytes leaked；
initcheck_coupled_4steps.log为0 errors。

对流按原standard通量做六阶散度；黏性先构造物理节点应力/能量通量，
刷新通量halo后再求散度，不在halo外继续套用嵌套梯度模板。
新device_halo组件区分单rank周期端点平均和halo-only刷新，不把本地周期
操作误用于分布式方向。fluid_step.F90通过显式交换回调组织RK阶段；
步内只在阶段检查处回读状态标志，不作全场主机往返。测试在观测点回读
场，仅用于与冻结参考对照。当前滤波为按需分配的完整5变量ping-pong；
尚未迁移现有单分量工作区选项，不宣称新后端已覆盖全部滤波存储模式。

首个RHS测试曾出现能量分量缩放差3.4685881109604993e-10，触发首失败停止。
原因是测试夹具用了110.4 K，而冻结参考perfect_gas_transport.F90默认
Sutherland常数为110.3 K；更正测试参数后通过，没有修改设备算子、冻结
参考或容差。失败记录保留在native/rhs_initial/。

这些组件通过时尚未接入普通astr；随后单rank主程序接线见第6.2节，不能把
该批组件等同于DCU多卡闭环。用户曾另行批准1节点、2DCU、2rank、每rank 2核、
5分钟的新组件批次及仅该任务查询，脚本为run_mb2_gnu12.slurm；已提交作业
127808864，结果见第6.1节。
登录节点只编译，不运行GPU。

最新耦合版本的首轮链接出现refresh_primitives符号缺失。检查确认远端源码
已含该定义，但保留的源码时间戳21:38:24早于已有对象21:38:32，静态库只有
该符号的Fortran引用、没有C++定义。只更新该候选源码的时间戳使其重编译，
没有添加兼容入口、自动重试或更换数学实现。失败日志
host_env_clean/build_coupled_4steps.log保留，修正构建另写
host_env_clean/build_coupled_refreshed.log，不用旧二进制提交新任务。
修正后的根构建已返回0，astr、MB0/MB1及两个流体目标均完成链接；
原失败与修正日志另存本地results/multi_backend_mb2_20261112/dtk2604/。

### 6.1 DCU组件与四完整步结果

作业127808864状态COMPLETED、退出0:0，耗时13秒，节点b16r4n03。
分配物理设备1、2，运行时可见逻辑设备0、1。实际脚本完成9次MPI程序启动：
MB1 NP=2、制造解NP=1/2，以及CPU/旧GPU各3个冻结阶段RHS对照。
第一阶段对照另包含滤波与4个完整RK步，两个执行模式均运行。

- MB1的4个hm/debug/rank组合全部逐位通过，不再出现256/512启动上限警告。
- 342个误差比较全部取得输出并满足2e-10，没有因只读最后一条PASS漏掉阶段。
- 全部算子最大缩放差2.4867315238236154e-14。
- 两种冻结参考第2/4步的最大轨迹缩放差1.1102229888203125e-15。
- 阶段内原始量/halo刷新、滤波、梯度、对流、黏性及RK都使用同一C++设备
  状态；运行中没有把这些算子换回旧CUDA Fortran实现。

总日志中一次PMIX ERROR: NOT-FOUND来自dstore_base.c:217，位于两个MPI
程序的输出之间，未进入对应应用分项日志。批次及所有程序返回0，后续
阶段输出完整；它不构成本次数值门槛失败，但根因尚未证明，不宣称公共
MPI运行时无异常，不擅自修改公共MPI或加入重试。后续真实多rank TGV
仍需其自身的通信证据，不能由这一批替代。

本地证据：dtk2604/mb2-127808864.log及同名目录；位于
results/multi_backend_mb2_20261112/。远端只操作上述批准的唯一作业。
该批次结束时，普通astr后端接线、标量统计、自动步长短窗及MB3尚未完成；
后续状态分别见第6.2节和第7节。

### 6.2 普通主程序及新增诊断

根CMake新增ASTR_GPU_CPP_BACKEND=none/cuda/gpufusion，与ASTR_WITH_CUDA
的旧CUDA Fortran实现互斥。device_state库已移入src_gpu/CMakeLists.txt，
普通astr直接链接，不依赖先构建探针。Fortran保留输入、拓扑、步长控制、
完整步时钟与统计输出；fluid_runtime.F90持有C++上下文，fluid_step.F90
分开阶段准备与推进，第一RK阶段滤波后采样原TGV标量，不重复滤波。

新增fluid_diagnostics.cu，设备归约原有三项TGV未归一化积分及q极值；
MPI及原归一化继续在Fortran执行。完整CFL诊断保留方向最大值、节点谱和
最大值、全局位置键和可选y剖面；不拿方向最大值之和冒充节点最大谱和。
波收集器直接接原select_run_timestep，未重写自动步长算法。

新主机中转路径的设备绑定在读取use_gpu后、建立上下文前完成；当前测试
显式关闭MPI设备指针路径。MB0的预MPI绑定探针仍独立保留；新普通主程序
尚未接入GPU-aware MPI，不据本地host-staged路径宣称MB5b通过。

本地普通astr结果：

| 范围 | 数值结果 |
|---|---|
| 16³、4固定步、dependency/explicit | 最大场差1.2666454347843286e-15；CPU/GPU冻结参考均核对 |
| 16³、4自动步 | 最大场差1.1102229691411208e-15；时间记录一致，CFL差≤4.17e-17 |
| 同二进制CPU路径 | 16³及128³终态相对冻结CPU逐值相等 |
| 128³、20固定步、dt=2e-4 | 最大场差5.953570969552402e-15；两种同步模式一致 |
| 128³、20自动步 | 最大场差9.109005867299902e-15；CFL差≤2.23e-16，dt差≤2.17e-19 |
| TGV标量输出 | 相对CPU最大差≤2.01e-16；相对旧GPU≤4.98e-14，含原文本输出精度差 |
| 新增归约制造解 | 含非均匀度量、全局位置键、确定性平值键与关闭y剖面不写入的检查 |
| 工具 | 普通主程序memcheck为0 errors/0 bytes leaked；新增归约racecheck为0 hazards |

128³采用冻结二进制重新生成参考，不拿16³文件充当放大后的参考。每个
模式先检查CPU与旧GPU参考间一致性，再对照新实现；固定/自动模式不因
步数相同而混比不同终止物理时刻。固定参考及自动参考末态的CPU/GPU差
分别≤4.89e-15及7.19e-15。

证据在results/multi_backend_mb2_20261112/native/下的main_fixed_final/、
main_automatic/、main_memcheck/、main_128_20/及diagnostics_integrated/。
当前C++普通主程序未实现正式场、检查点、恢复、均值场、crashfix及原位
产品提供者；请求实际消费这些未接字段时明确报告，不借用过期CPU数组。
静态几何仍可由原CPU初始化提供者输出，不因关闭检查点而禁用无关选项。

用户已批准新DCU任务：128³固定/自动各20步，固定dt=2e-4；自动目标CFL0.5、
initial_dt=2e-4、max_dt=1e-3、增长1.1；1节点/1DCU/1rank/2核/10分钟，
另含新增归约组件。仅终态测试快照，不启用正式场或检查点。已提交唯一
作业127818756，结果见第6.3节，不操作其他作业。
平台预检发现MPI包装C编译器使FindMPI把include视为隐式，CUDA单元因而
缺少mpi.h。构建改用已批准GNU12的gcc/g++及显式MPI_C_COMPILER识别；
不换MPI/SDK。新工作目录为host_env_clean/astr_cpp_main，根构建已实际完成
普通astr及探针链接；CUDA编译命令明确含所选MPI的include，日志为
build_cpp_main_128_plain_gcc.log和build_cpp_main_128_final.log，原失败日志保留。
三份日志另存本地dtk2604/main128/。固定20步终止物理时间0.004，自动模式
本地终止时间约0.01018628；本地自动步后CFL峰值0.500000948，按步前谱
预测，与冻结参考一致，没有新增裁剪或硬性步后阈值。

### 6.3 DCU普通astr的128³放大检查

作业127818756已COMPLETED、退出0:0，耗时26秒，节点b15r3n06；分配物理
设备2，运行时逻辑设备0。固定与自动两个案例均由普通astr报告
backend=gpufusion、sync=dependency，分别完成20个完整步。另完成新增
诊断组件的80个误差比较，未把登录节点编译当作GPU运行证据。

终态快照、flowstate.dat和逐步CFL/时间记录已下载到本地，分别与该规模
由冻结CPU及CUDA Fortran二进制生成的参考对照：

| 比较量 | 最大缩放差或结果 |
|---|---|
| 固定步长5变量场 | 5.509481759702339e-15 |
| 自动步长5变量场 | 7.721471032007265e-15 |
| CFL记录 | 2.7755575615628914e-16 |
| 自动时间记录 | 8.673617379884035e-19，步号和采样序列一致 |
| TGV标量相对CPU | ≤2.01e-16 |
| TGV标量相对旧GPU | ≤4.98e-14，包含原文本输出精度 |
| 状态可容许性 | 全部物理节点有限、rho/p/T严格正；推进未启用crashfix |

当前作业日志未见上一组件任务的PMIx诊断或256/512启动上限警告；
这不等于所有MPI配置均已验证。26秒包含初始化、组件检查、两个案例、
标量/诊断及终态快照I/O，不作为纯RK性能或加速比。

证据：results/multi_backend_mb2_20261112/dtk2604/main128/job127818756/，
comparison.json记录每个分量相对两种参考的误差；分析脚本为
tests/gpu_backend/compare_cpp_main_results.py。
该作业执行时普通astr只接单rank周期回调，不构成分布式TGV或GPU-aware
MPI通过。其后的本地多rank接线见第7.1节，DCU NP=2/4仍待执行。

## 7. MB3 传输组件

halo_transport.cu/.F90/.h接收Fortran拓扑与通信器。只有分布式轴才分配
固定设备/固定页主机send/recv槽；创建一次，循环复用。C++管理请求和
MPI完成；没有应用通信线程或MPI_THREAD_MULTIPLE要求。使用复制的私有
通信器及固定tag 0/1，不占用原Fortran通信器的tag空间。

halo-only只传hm层，reconcile才附加共享端点信息并平均端点；
最终halo-only调用不再修改物理节点。主线程在MPI读取发送槽前完成打包
及D2H；下次重用接收槽前等待同流已有H2D/解包完成。单rank轴执行明确的
本地周期操作，不分配MPI传输槽，未加入设备直接通信或自动降级。

halo_probe使用独立Fortran数组切片、原升序端面包及MPI_Sendrecv作参考，
不是复制C++索引计算。NP=1及NP=2 x/y/z，共28个hm/debug/rank配置，
每个轴的平均与最终刷新相位均逐位相等，包括未消费角区保持；q指针
在所有交换前后不变。双rank memcheck分别0 errors/0 bytes leaked。
原组件证据为results/multi_backend_mb3_20261112/native/halo_initial/；
收紧halo-only包为恰好hm层后的同矩阵与memcheck在halo_exact_width/。

### 7.1 普通主程序的本地多rank闭环

fluid_runtime现已接入device_transport_prepare和真实交换回调；设备按节点内
local rank绑定，不超额复用。本地普通astr运行16³、dt=1e-3、4完整步，
覆盖NP=1、NP=2 x/y/z，以及交换观测开/关、dependency/explicit同步对照。
数值公式、物性常量与冻结参考保持一致。

| 检查 | 结果 |
|---|---|
| 8个实际运行的终态守恒场 | 相对冻结CPU/旧GPU最大缩放差1.2666454347843286e-15 |
| 三个RK阶段的q与物性缓存 | 最大缩放差1.2148680556009271e-15；物理节点有限且rho/p/T正 |
| 315个rank-交换相位 | 核对全部hm层、共享端点、字段/方向/平均标识；复制逐位一致 |
| 交换观测与同步模式隔离 | 三方向观测开/关及x方向显式/依赖模式的终态物理场逐位一致 |
| 普通主程序NP=2 x内存检查 | memcheck、initcheck每rank均0 errors；memcheck均0 bytes leaked |
| 六分量观测器独立检查 | Fortran构造已知值、负零及填充区，三个方向、两种同步模式均逐位一致 |

观测只在BUILD_TESTING及显式测试步号/前缀选中时执行；它会回读测试字段，
不属于正式每步路径。未选中观测时只有原有标量与halo包传输，没有整场往返。
测试中曾出现NVHPC回调内将C指针再次按VALUE传入内部观测子程序后句柄为空；
仅把该非C内部形参改为intent(in)，保留C ABI入口的VALUE，未改流体算法。
原失败记录保留在tgv_initial、tgv_view_debug、tgv_callback_debug。

证据：results/multi_backend_mb3_20261113/native/tgv_wired/下report.json及
reanalysis_observation.json；tgv_memcheck/、tgv_initcheck/和observation_copy.log。
脚本run_cpp_multi_tgv_gate.py读取求解器发布的实际拓扑和起点；逐阶段先交叉
核对冻结CPU与旧GPU，再分别比较新实现，Python不推进流场或提供halo。
最后一次halo观测另与独立整q下载逐位比较，避免只依靠观测器自身判定。

另以BUILD_TESTING=OFF、ASTR_BUILD_GPU_BACKEND_PROBES=OFF通过根CMake独立
构建普通astr，在NP=2 x运行同一4步输入；未生成验证快照。TGV标量及步长
记录相对冻结CPU文本逐值一致，CFL差≤1.39e-17。该项检查普通运行不依赖
探针或测试观察代码，不替代前述逐场数值门槛。证据在native/runtime_*。

DCU多卡输入已准备为NP=1/2/4同一16³短窗，存于已限定工作目录中的
host_env_clean/astr_cpp_main/dcu_multi_20261113。2026-11-13登录节点构建
已完成普通astr及MB0/MB1/fluid/RHS探针，没有编译旧.cuf；所用28份相关
源码/构建文件与本地内容一致。普通astr的ldd未发现缺失库，解析到指定的
GNU12、Open MPI4.1.5、私有HDF5及DTK26.04运行库。这不是设备运行证据。
日志与离线源码散列见results/multi_backend_mb3_20261113/下
build_cpp_multi_dcu.log、ldd_cpp_multi_dcu.log及source_{local,dcu}.sha256。
用户分别批准四卡批次及修正后新批次，范围均为1节点/4DCU/4rank/8核/
10分钟、16³/4步、仅查询对应任务。首批127826437退出0:0，但只运行NP=1：
mpiexec继承循环标准输入，消耗了清单其余条目。只计其45个单rank交换相位，
最大终态缩放差1.1102229888203125e-15，不能计为多卡通过。该日志仍有
PMIx dstore NOT-FOUND诊断，保留记录，未添加MPI绕行策略。
本地最小复现原写法只访问第一条，加</dev/null后全部三条执行；证据为
manifest_stdin_reproduction.json。仅修正脚本标准输入隔离，并使用全新
dcu_multi_20261113_b2目录，保留首批记录。新任务127827004的完整检查见下节。

### 7.2 DCU普通主程序多rank结果

任务127827004在i02r3n19实际分配4张DCU和8个CPU核，COMPLETED、退出0:0，
批次27秒。10个案例全部执行，逐案例核对实际拓扑、rank数、gpufusion后端、
local-rank到设备0..3的绑定，不以资源申请数推断设备使用数。
同一16³、dt=1e-3、4完整步，开启原standard 643e对流/黏性与十阶显式滤波。

| 实际分解 | 终态最大缩放场差 | 交换rank-相位数 |
|---|---:|---:|
| NP=1，1×1×1 | 1.1102229888203125e-15 | 45 |
| NP=2，2×1×1 | 1.1102229888203125e-15 | 90 |
| NP=2，1×2×1 | 1.1102230246251565e-15 | 90 |
| NP=2，1×1×2 | 1.1102229888203125e-15 | 90 |
| NP=4，2×2×1 | 1.1102230246251565e-15 | 180 |

每个变量分别比较冻结CPU与CUDA Fortran参考，表中为最大值。全部495个交换
rank-相位的hm层与协调端点逐位检查通过；三个RK阶段q及物性缓存的最大
缩放差1.3253106061101023e-15。物理节点有限且rho/p/T正。CFL记录最大差
2.7755575615628914e-17，步长/时刻记录一致；TGV标量最大缩放差
3.58046925441613e-14，包含旧GPU文本精度。均沿用2e-10门槛，未放宽。

NP=2三个方向及NP=4双方向的观测开/关，加上NP=2 x的explicit/dependency，
共5组终态物理场逐位一致。没有正式场、检查点或渲染；测试快照包含有意的
整场诊断回读，27秒不作为纯RK性能数据或扩展效率。
日志在两个MPI启动结束处仍有PMIx dstore NOT-FOUND，应用均退出0且后续
案例完成、数值门槛通过；根因未确定，作为平台诊断保留，不宣称MPI栈无问题。

本地证据：results/multi_backend_mb3_20261113/dtk2604/job127827004/，包括
原始批次日志、逐案例run.log/快照、comparison.json及汇总report.json。
脚本为tests/gpu_backend/run_cpp_multi_gnu12.slurm，分析使用
run_cpp_multi_tgv_gate.py中的analyze/layouts/physical/bits_equal。

### 7.3 本轮完成核对

| 目标 | 证明完成的来源 |
|---|---|
| 独立冻结参考且CPU/旧GPU先互校 | 第2节归档；二进制离线散列本次重核一致；逐阶段参考互校在分析中执行 |
| 新环境、跨设备翻译单元、512线程与流/事件 | 第3、4、5.3节本地及DTK26.04实际组件日志，不借用22.10.1 |
| 根构建、C ABI、上下文和x连续视图 | 第5节已知位模式/下界/分量/隔离测试；第6节普通astr；第7.1节无探针构建 |
| 原643e/643e、explicit10、RK3和物性刷新 | 第6节冻结分项及轨迹；第7.1、7.2节三个RK阶段q/缓存与物理状态检查 |
| 固定和自动时间控制、TGV标量 | 第6.2、6.3节128³/20步；分别比较dt/CFL/时刻和原统计定义 |
| 真实本地/DCU多卡、全部消费halo与共享节点 | 第7.1、7.2节实际NP和495个DCU交换rank-相位，覆盖三轴与双方向分解 |
| 依赖同步、调试同步及观测不改变结果 | 本地4组、DCU5组逐位隔离；本地普通main的memcheck/initcheck零错误 |
| 常驻及可选工作区边界 | device_state按需拥有缓冲；halo固定设备/固定页槽；除halo外，正常诊断回读仅标量/可选一维CFL剖面；测试字段回读显式选中 |

本轮没有改变原数值公式、参考归一化或CPU物理算法，没有新增自动回退、重试、
回滚或应用通信线程。goal执行阶段没有Git写操作；收口后的提交另按用户授权执行。
没有操作A800作业、扩展远端目录权限或安装SDK。

## 8. 后续顺序及终止条件

1. MB0本轮组件已取得实际DCU证据；设备工具的深入调试/计数器能力仍按
   主计划独立推进，不能把本次组件通过等同于工具全部可用。
2. 本地及DCU的pinned-host普通主程序门槛已完成；MB0--MB3本轮收口，不再
   为文档变更重复运行短测，也不据此扩展到未实现格式、边界或物理模型。
3. 后续按主计划推进MB4的正式输出与恢复接口；当前C++后端这些消费者明确
   未接入，不从陈旧CPU数组写场或伪装续算成功。具体Slurm作业仍分别授权。

已知KEEP/ME6边界与几何问题继续按原记录处理，不纳入本轮修复或移植。
MB4输出/恢复、MB5a性能及MB5b设备直接通信仍是后续独立任务。
用户追加：当前goal完成后，按既有A800显式格式的完整RK口径单独安排
1/2/4卡性能与扩展测试；关闭正式场/检查点I/O，保留标量统计，预热及
多轮计时。同硬件比较新旧实现，DCU/A800硬件结果分开说明；平台提交前
另审阅具体参数。本轮数值验收的20步总耗时不直接作为性能结论。
