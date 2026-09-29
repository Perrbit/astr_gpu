# ASTR GPU 部署能力边界与 release/prod 裁剪方案

日期：2026-09-29。状态：部署审阅稿，不是发布验收证书。

本文依据当前工作目录源码、构建定义和已有验证记录编写。后续已获准在
feature/gpu_dev 修改测试构建依赖，保留原 CPU 的 examples 配置行为。
未操作 Git、创建分支、删除文件或干预本地/平台任务，生产数值路径不变。
最终发布基线的提交号、工作区差异和构建环境应在获准执行发布时冻结；
不能将当前目录内容直接称为某个已提交版本。

## 1. 对外交付定位

ASTR GPU 是面向 NVIDIA GPU 的结构化高阶可压缩流动求解器，采用
CUDA Fortran 和 MPI，实现主要时间推进变量设备常驻及多 GPU 子域通信。
当前可交付的基础范围是已验证配置下的非反应流显式数值路径。
固定 AIR5 双温反应流、混合精度及部分性能候选保留源码，但不作为
无条件生产能力承诺。

“支持”必须同时限定物理模型、网格、边界方向、数值格式、精度模式、
通信后端和验证窗口。单项通过不代表其任意组合均通过。
部署成功、CPU/GPU 数值一致、物理验证和性能验收是不同结论。

| 交付等级 | 含义 | 部署策略 |
|---|---|---|
| 基础能力 | 有既有数值验证记录的受限配置 | 纳入基础交付，目标平台仍需启动检查 |
| 条件能力 | 仅指定网格、边界、后端或短窗组合验证 | 明确配置和限制后按需启用 |
| 研发能力 | 已实现但长期物理或生产门槛未关闭 | 源码可保留，不默认启用，不作生产承诺 |
| 不在范围 | 未移植、暂缓或缺乏当前准入证据 | 不对外交付为受支持能力 |

## 2. 基础求解能力

| 项目 | 当前范围 | 不应外推的范围 |
|---|---|---|
| 平台 | NVIDIA GPU、NVHPC CUDA Fortran、MPI | AMD/HIP、国产 DCU 尚不是已交付后端 |
| CPU/GPU 选择 | 编译加入 CUDA 支持，输入文件运行时选择 usegpu | 不是按算例编译两个独立求解器 |
| 基础物理 | 单组分理想气体，受支持无黏/黏性流，Sutherland 变黏性 | 任意真实气体、任意化学机制 |
| 时间推进 | GPU 三阶段 RK3 | CPU 遗留 RK4 不代表 GPU 支持 |
| 中心离散 | 六阶显式中心差分及显式黏性扩散 | 紧致差分、三对角/五对角求解不在 GPU 范围 |
| 激波离散 | 已验证的一阶 Steger-Warming、WENO7、MP7、Ducros 和选择性 Roe 组合 | 任意格式、传感器和边界的自由组合 |
| 滤波 | 十阶显式中心滤波，已实现物理边界闭合 | 紧致滤波、FP32 滤波不支持 |
| 网格 | Cartesian；静态、单块、三维结构化曲线网格 | 多块、非共形、动网格、ALE/GCL、GPU 浸入边界 |
| 源项 | 已验证的槽道驱动、RTI 重力等专用路径 | 任意用户源项自动 GPU 化 |
| 变量常驻 | 主要求解状态、工作区和 halo 计算在 GPU | 不承诺文件输出和所有诊断零主机参与 |
| MPI | x/y/z slab 及已测多维分解；生产建议一 rank 一 GPU | 共享两卡的多 rank 正确性不等于多节点性能验证 |

几何法向投影是曲线壁面处理原则。网格需合法，子域尺寸需满足实际模板和
halo 要求；不是只要能读入网格文件就属于受支持算例。

## 3. 非反应流边界范围

下表是五方程路径的能力摘要，不作为 AIR5 十一变量路径的边界白名单。
实际准入以源码中的配置检查为准。

| bctype | 已有范围与限制 |
|---:|---|
| 1 | 三方向周期 |
| 11 | x-min 给定入口；受控静态剖面和动态时序入口 |
| 12 | 受限 x-min 特征入口，不是任意曲面通用入口 |
| 21 | 主要为轴对齐 x-max 出口；非轴对齐曲面不在此合同 |
| 22 | 受限 Cartesian NSCBC 出口；非正交曲面不在此合同 |
| 31 | 已验证 RTI 固定边界路径 |
| 41 | 曲线六面零吹吸等温无滑移壁面；另有受限 y-min 法向吹吸 |
| 42 | 曲线 x/y 绝热无滑移壁面；z 面缺少对应 CPU 分支，不承诺支持 |
| 411/421 | 受限 y 向组合，按局部几何法向处理速度 |
| 50 | 曲线六面零外推 |
| 51 | 受限轴对齐远场组合 |
| 52 | 受限 upper-y NSCBC；静态曲面 upper-y 非反射及黏性特征源项组合 |
| 60 | 曲线六面对称边界，局部切平面投影 |

CURVE-C22/C23 的非反射与黏性源项结果只覆盖静态单块、单组分、五方程、
upper-y 面。不能宣称通用全方向 NSCBC/GCBC 已完成。

## 4. 已有算例证据的含义

- TGV、三维挤出涡输运、HIT、Channel、LDC、RTI：已有显式路径与相应
  CPU/GPU、多 rank 或边界检查；不代表各算例均完成长期物理统计验收。
- Sod、Shu-Osher、OpenShock：已有激波格式及受控开放边界验证。
- 平板/高速边界层：已有 Cartesian/静态曲线网格、Sutherland 和受控入口验证。
- 受控 SBLI：已有激波和边界耦合数值验证；不能替代真实湍流反应 SBLI 验收。
- OpenSBLI 层流 SBLI：长跑及重启已接通，外部物理验证未完整关闭。
  不宣称壁面热流峰值已与参考严格匹配。
- 动态时序入口：五变量切片插值、换片和部分重启/拓扑检查通过。
  接口可运行不等于真实湍流前驱统计已验收，也不等于 AIR5 动态入口已覆盖。

## 5. AIR5 双温反应流边界

### 已实现的受限能力

- 固定 N2/O2/N/O/NO 五组分模型、双温状态、固定机制的热力学/输运/反应内核。
- 总密度方程与 N-1 独立组分架构，十一变量 GPU 输运及能量相容性处理。
- 自适应 ROS-2、固定 6x6 线性求解，以及受控化学与输运耦合路径。
- 组分保正和一致性限制、有限精度补偿、共享节点/halo 刷新。
- 受控壁面、入口、特征顶面、补偿状态重启和多拓扑同相位检查。
- 显式十阶滤波支持 full/scalar 两种工作区；补偿状态与特征顶面组合
  已通过短矩阵和指定重启检查，但仍需验证开关明确准入。

### 不得作为已完成生产验收的事项

- 任意温压、任意电离/解离机制、催化壁面和任意化学机制导入。
- AIR5 与五方程 CURVE、混合精度、CUDA-aware MPI 等能力的任意组合。
- 已完成长时真实湍流反应 SBLI、已得到收敛的湍流入口或数据库级验证。
- 在所有工况下保正限制不激活、维持名义高阶精度或保持相同时间步成本。

本次核查的补偿滤波矩阵明确记录
`status=short-matrix-pass-not-production-admission`、`production_ready=false`。
当前 28 us 前驱续算记录同样为 `production_ready=false`。此前读到的均值
漂移下降和极小展向速度，只支持准稳态发展趋势，不证明三维湍流建立。

**已批准方案 A：源码保留 AIR5，基础部署默认关闭其构建选项。需要交付 AIR5 时，
单独列为受限验证版，附明确工况与配置，不通过改名 release/prod 升级其可信等级。**
根 CMake 和 GPU_Quickstart/build.sh 均默认关闭 AIR5，后者通过 AIR5=ON
显式启用。执行状态见 [发布执行清单](ASTR_RELEASE_READINESS_CHECKLIST.md)。

## 6. 精度、通信与输出

| 选项 | 部署建议 |
|---|---|
| FP64 | 权威默认 |
| mixed_workspace | 可选 FP32 临时工作区，受限候选；不是整个求解器低精度化，不默认启用 |
| full 滤波工作区 | 显式可选；五方程采用完整五变量工作区，AIR5 按其布局处理 |
| scalar 滤波工作区 | 2026-09-29 起为默认，逐分量复用三维工作数组；不自动等价于所有平台性能提升 |
| explicit 同步 | 基础正确性配置 |
| selective/dependency 同步 | 仅受控配置，不能作为任意物理边界和化学默认 |
| pageable/pinned | 主机中转通信；pinned 在目标平台检查后可选 |
| device-aware | 需编译及运行时 CUDA-aware MPI 支持，并对目标软件栈准入 |
| pinned-overlap/pinned-pipeline | 受限重叠/流水线选项，不保证所有工况更快 |
| fused 上风通量 | 受限性能候选，不替代通用默认 split |

性能基线：历史 A800 单节点 512^3 TGV、FP64、full filter、explicit 同步下，
1/2/4 卡完整 RK 时间分别为 1.75580/0.90395/0.46482 s；四卡强扩展效率
94.43%。它是历史特定软件栈和输入下的结果，不是当前所有工况的性能保证。

HDF5、checkpoint 和完整场输出仍由主机参与。关闭场输出不消除二进制的
HDF5 动态库依赖。统计量通路与完整场输出通路不同，不应将其统称零 I/O。
ParaView/Catalyst 原位后处理仍属计划范围，不纳入本版已实现功能。

## 7. release/prod 文件范围建议

目标是仅交付源代码与构建工具，不交付测试脚本和历史运行产物。
许可证、最小构建说明和本能力边界建议作为必要发布元数据保留，
这是对“只有代码”的少量必要补充，待用户确认。

| 路径 | 建议处理 |
|---|---|
| src/、src_gpu/ | 保留生产编译依赖闭包；不要按文件名删除 chemistry 或 conservative 模块 |
| user_define_module/userdefine.F90 | 保留，主程序直接依赖；该目录其他内容逐项确认 |
| 根及 src/CMakeLists.txt | 保留；用 BUILD_TESTING 解耦新增测试和探针，保留原 examples 行为 |
| LICENSE、必要版权声明 | 保留 |
| README.md、本说明 | 精简后保留最小操作和能力说明，不携带全部研发历史 |
| .gitignore、.gitattributes | 审核后保留必要仓库配置 |
| examples/ | 按用户确认保留，维持原 CPU CMake 行为；示例存在不代表均可 GPU 运行 |
| tests/、miniapps/ | 生产树不携带；配置时显式设置 BUILD_TESTING=OFF |
| scripts/、script/ | 不整体复制；只有确认必需的生产构建工具才列入白名单 |
| build_nvhpc*.sh、旧 Makefile* | 不直接当作发布构建入口，使用根 CMake 的可配置包装工具替代 |
| chem/、chemMech/ | 不默认携带；固定 AIR5 生成源码已在 src/，遗留 Cantera 不纳入基础配置 |
| pastr/ | 不作为本次主求解器发布内容；离线后处理工具另行交付 |
| docs/、documents/ | 历史材料留开发分支；生产版只提取最小能力/构建文档 |
| build*、tmp/、输出、profile、缓存、PPT/PDF | 不交付 |
| .agents/、.codex/、.superpowers/、.vscode/、AGENTS.md | 不属于运行依赖，默认不交付 |
| .github/ | 不直接保留引用已删除测试的 workflow；按发布需要另行审定 |

算例输入、网格、入口剖面、时序切片、checkpoint 属于外部运行数据，
不属于测试脚本，也不能因源码树精简而认为运行时无需它们。
部署时单独提供匹配格式的运行包。当前依赖测试脚本生成的配置，需形成
明确输入和环境变量清单，不能仅复制可执行文件就启动。

## 8. 构建解耦与保留项

1. 根 CMake 无条件执行 `add_subdirectory(examples)`，按用户确认保留。
   因此生产源码包也需保留 examples。
2. `BUILD_TESTING` 在进入 src 前定义，默认 ON 保持开发行为。OFF 时
   主程序不再编译 `boundary_rhs_manufactured.F90`；对应五个新增诊断命令
   明确报错，原 CPU 内置测试及 bcfg 配置检查保留。
3. 独立 probe/test 目标由 BUILD_TESTING 控制，OFF 时不再引用 tests 源文件。
   MPI completion trace 显式要求 BUILD_TESTING=ON，避免静默忽略用户配置。
4. 生产需要的 `MPIX_Query_cuda_support` 检测和 AIR5 编译宏保持独立。
   测试开关不会关闭 CUDA-aware MPI 的生产能力检测。
5. 旧 `build_nvhpc.sh` 硬编码本机 SDK/HDF5 路径、维护独立源列表，并删除
   构建/输出目录；不适合未经改造直接作为部署工具。
6. `chemistry_air5_data.F90` 是已生成源码，运行不需要重新生成机制。
   生成器可留开发分支，但生成来源和版本应在发布清单保留，避免无法追溯。

本次仅修改构建依赖及诊断入口条件，不删除 tests 或 examples。
构建隔离检查见 `tests/gpu_validation/test_cmake_production_decoupling.py`。

2026-09-29 本地验证结果：

- 使用 NVHPC 26.1、配套 HPC-X MPI 和本机 HDF5 1.14.6，在不含 tests 的
  临时源码副本中，CPU/CUDA 与 AIR5 OFF/ON 四种配置均完成 astr 编译和链接。
- 生产模式拒绝启用依赖 tests 的 MPI completion trace；连同四种构建，
  集成检查为 5 passed。CUDA-aware 查询和 AIR5 编译定义均经检查保留。
- 开发模式 CPU/CUDA 配置检查为 2 passed，确认原探针目标及诊断宏存在。
  CPU 边界代数探针另外完成编译及 mode=1 运行，返回状态 0。
- 生产 CPU 二进制经 MPI 启动调用 bcrh，以非零状态退出并明确要求
  BUILD_TESTING=ON，未静默跳过检查。
- 未运行新的流场计算、全量物理回归或平台任务。未执行 Git 操作。

## 9. 待批准的发布步骤

1. 冻结被选定的源码版本与未提交变更归属，保存开发分支及验证证据。
2. 在授权后建立 release/prod，按白名单裁剪文件，不改写历史，不另建 worktree。
3. 采用已解耦的根 CMake，发布配置显式关闭 BUILD_TESTING；保持生产数值路径。
4. 从精简树独立配置、编译 CPU 和 CUDA 基础目标，确保不引用被删除目录。
   AIR5 若纳入交付，再单独验证对应构建配置。
5. 在目标计算环境核查 NVHPC、驱动、MPI、HDF5 Fortran/HL 及动态库解析，
   保持编译器模块 ABI 和 MPI 依赖一致。CUDA-aware 不可用应明确拒绝或显式选
   主机通信，不能无提示伪装成功。
6. 使用开发分支保留的外部验收脚本检查精简构建：短周期、一个生产壁面、
   NP1/多 rank 通信，以及实际需要的输出/重启。脚本不必放进生产树。
7. 建立发布清单：版本、编译配置、依赖版本、默认选项、允许组合和已知限制。
   通过后再执行经用户授权的提交/推送。

部署包不带测试，不等于发布无需测试。构建隔离修改应验证精简树独立编译，
不需要因此重跑昂贵生产算例。

普通 release 分支的文件删除只影响该分支最新文件树，不会从共享 Git 历史
或仓库体积中抹掉旧测试。若希望交付物不含任何历史，后续应另行导出干净
源码包；不通过擅自改写历史实现。

## 10. 核查依据

- `CMakeLists.txt`、`src/CMakeLists.txt`：实际编译、MPI/HDF5、测试和示例依赖。
- `src_gpu/case_capability_gpu.cuf`、`src_gpu/gpu_runtime.cuf`：GPU 组合准入。
- `src/readwrite.F90`：运行时 usegpu、AIR5 补偿及滤波显式验证门禁。
- `src_gpu/halo_transport_gpu.cuf`、`src_gpu/precision_mode_gpu.cuf`、
  `src_gpu/gpu_check.cuf`：通信、精度和同步选项。
- `ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`：非反应流、CURVE、化学与性能历史；
  该文件含多期进展，旧段落不能覆盖新实现或被直接当作当前统一白名单。
- `tests/gpu_validation/out/air5_compensated_filter_20260928/result.json`：
  补偿滤波仍为短矩阵准入，不是生产放行。
- `tests/gpu_validation/out/air5_filtered_precursor_28us_20260928/result.json`：
  长窗发展和重启记录，本次读取时仍为 running / production_ready=false。

本文件为当前目录的能力盘点及发布建议，不代表全量数值回归或目标平台
重新验收。发布状态将在上述步骤实际完成后更新。
