# ASTR OpenSBLI 层流 SBLI 完整验证 Goal

2026-09-07，用户要求设定并执行完整 OpenSBLI 仿真 goal。当前工作状态：
参考边界已接入 CPU/GPU 生产主循环，restart 验证通过，step 60000 长时间续算
正在运行；时间、网格和外部物理验证尚未完成。
这里指用 ASTR 复现 OpenSBLI 数据，不是直接运行 OpenSBLI 代替移植。
前置资料：`ASTR_OPENSBLI_KATZER_STARTUP_AUDIT.md`。

## 范围和边界

- 保留当前工作树、CPU/GPU 默认行为、FP64、MPI halo 合同及同步政策。
- 显式格式，不引入 compact、化学或 RANS/LES；不使用远程平台，不自行提交 Git。
- 测试输入生成器的法向速度修复已获用户批准并验证。
- 新参考边界模式须通用命名、显式启用、校验支持范围，不能以作者名字硬编码主算法。
- 发现已有 CPU 明确 BUG 时先请用户决策，不静默复制或更改。

## 完成门槛

| 阶段 | 验收 | 状态 |
|---|---|---|
| G0 参考合同 | 归档状态、物性与边界源码逐项核对，Pr 冲突有明确处理 | 进行中 |
| G1 输入 | 参考网格、位移厚度归一化、沿 x 均匀初值、EOS/连续性/文件测试 | 候选输入已通过 |
| G2 边界 | CPU/GPU 可选参考边界，顶层 CMake 构建和独立边界/halo/RK 测试 | 通过：生产接线、角点、MPI 准入、restart halo 和 memcheck 已验证 |
| G3 一致性 | NP=1、NP=2/4，同相位逐场/统计 atol=rtol=1e-10，z均匀性 | 进行中：短程 restart 等价通过；完整 NP=1/2/4 同相位矩阵和长期 z 均匀性待闭合 |
| G4 时间 | 参考网格积分至 t=13000，继续到 t=26000检查 | 进行中：609x255x9、NP=2 从 step 60000 续算 |
| G5 网格 | 至少三档网格，固定物理域与物性，时间收敛后比较 | 待做 |
| G6 外部验证 | 原始参考壁压/Cf/分离区/场结构对比，报告与可复现脚本 | 待做 |

时间收敛初定门槛：分离长度、壁压及 Cf 的归一化变化不超过 1%。
Cf 在零点附近不使用逐点相对误差，采用整个比较区间最大绝对 Cf 作为尺度。
若指标失败则继续诊断，不能放宽阈值使其通过。
外部物理验收指标与插值/比较区间须在首次生产运行前冻结；不以 CPU/GPU 一致证明物理正确。
参考 t=13000 的归档不一定是严格定常解，必须分别报告“同参考时刻差”和“更长时间收敛差”。
goal 只有在完整证据齐备后才完成，不在短步或输入检查通过时完成。

## 当前实现

新增 `tests/gpu_validation/prepare_opensbli_reference_inputs.py`，只生成数据资产，不生成可运行旧边界输入。
输出：`grid.h5`、`initial.h5`、`profile.dat`、`manifest.json`。
默认参考网格为 609x255 物理点，周期挤出 9 点，对应 ASTR 的 IM/JM/KM=608/254/8。
展向长度 1 是 ASTR 二维挤出选择，不是原二维参考参数，后续需检查其不敏感性。
生成器拒绝覆盖已有目录。

候选已生成于：

```
tests/gpu_validation/out/opensbli_input_gate_pr072
tests/gpu_validation/out/opensbli_input_gate_pr071
```

两者都使用 Sutherland 温度 110.4 K；现有其他驱动仍保持 110.3 K 默认。
`solve_profile` 新增可选 keyword 参数，未改变现有调用默认值。
分析积分 delta_star=1；目标网格梯形积分约 1.00009726。
Pr=0.72 外缘 v=0.00564966272，Pr=0.71 外缘 v=0.00566301646。
较接近归档不能单独用来确定归档实际 Pr，只作为诊断证据。

验证命令：

```
python3 -m unittest discover -s tests/gpu_validation -p test_compressible_blasius_profile.py
python3 -m unittest discover -s tests/gpu_validation -p test_opensbli_reference_inputs.py
python3 tests/gpu_validation/check_blasius_mass_continuity.py
```

输入相关共 7 项测试通过。OpenSBLI CFD 尚未启动；物性配置已另行完成 TGV 回归，见下节。
本机检查：两张 RTX 4000 Ada，各 19195 MiB 显存，识别正常；不将硬件识别视为运行占用证据。

## 入口源码补充核对

固定版本 `e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d` 的
`opensbli/core/boundary_conditions/inlet_pressure_extrapolate.py::apply` 实际行为：

1. 用边界当前 rho、动量、总能量计算速度绝对值、压力及声速。
2. abs(u)>=a 时，从第一个入口 halo 复制全部守恒量到边界。
3. 亚声速时保留边界当前守恒量，只将边界总能量复制到入口各层 halo。
4. 初始 halo 的密度/动量来自规定剖面，亚声速阶段不强制覆盖为内部值。

这比“压力外推入口”文字描述更具体。不能自行改为从 i=1 复制压力或固定温度入口。
该版本尚未证明为2018归档生成版本，边界时序/差分闭合仍需核对后冻结。

## G2 前置：可选物性配置回归

2026-09-07，新增 `src/perfect_gas_transport.F90`，由 `solver::refcal` 在 rank 0
读取环境参数并广播给所有 rank，CPU 与 GPU 使用相同的 `commvar` 参数。
未设置时保持 Pr=0.72、S=110.3 K，不改变原有默认物性。

```bash
export ASTR_PERFECT_GAS_PRANDTL=0.71d0
export ASTR_SUTHERLAND_TEMPERATURE_K=110.4d0
```

只允许非 COMB 的无量纲模式显式覆盖，参数必须是有限正实数。
错误参数所有 rank 非零退出，不静默回退。Pr=0.71 在此是配置测试值，
不是解决了归档 Pr=0.71/0.72 的来源冲突。
GPU 黏性通量和统计核沿用传入的 Pr、tempconst、tempconst1，无需另设 GPU 常量。

顶层 CMake 的 `build_cpu_probe` 和 `build_gpu_probe` 已构建通过。
参数解析 3 项单元测试通过，输入相关 7 项测试复查通过。
可复现集成命令：

```bash
python3 tests/gpu_validation/run_transport_config_compare.py \
  --out tests/gpu_validation/out/transport_config_gate
```

输出目录必须不存在。脚本保留二进制 SHA256、日志、逐场和统计报告及退出码，
使用本地已确认的 `OMPI_MCA_sharedfp=individual`。
本次 128^3 TGV、2 steps、643e、显式滤波/黏性开启，组合容差 atol=rtol=1e-10：

| 物性 | NP / 拓扑 | 原始量与重构 q 的最大绝对差 | 逐场 / 统计 |
|---|---|---|---|
| 默认 0.72 / 110.3 K | 1 / 1,1,1 | 2.8422e-13 | PASS / PASS |
| 默认 0.72 / 110.3 K | 2 / 2,1,1 | 2.8422e-13 | PASS / PASS |
| 配置 0.71 / 110.4 K | 1 / 1,1,1 | 3.1264e-13 | PASS / PASS |
| 配置 0.71 / 110.4 K | 2 / 2,1,1 | 3.1264e-13 | PASS / PASS |

另有 CPU/GPU 各两个非法参数 NP=2 测试，均检测到指定诊断并以退出码 1 结束。
共 12 项集成检查通过，证据为 `out/transport_config_gate/summary.json`。
这些检查验证配置通路，不代表 OpenSBLI 边界或物理结果通过。

初次手工测试遗漏 sharedfp 配置，GDB 定位到
`mca_sharedfp_sm_file_open` 打开 `datin/grid.h5` 时的信号量等待。
该次已终止，后续使用 individual 成功，不改求解器或 HDF5 源码。
一次非法输入探针遗漏强制拓扑，提前在域分解退出，未计入上述通过项。
NP=2 重跑期间 `nvitop --once` 观察到两个计算 PID，各约 898 MiB、进程 SM 采样
14%/10%；采样已接近进程退出，不能用作持续利用率或 OpenSBLI 占用证据。

## 下一实现批次：边界、halo 与 RK 合同

参考固定版本的
[ExtrapolationBC](https://raw.githubusercontent.com/opensbli/opensbli/e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d/opensbli/core/boundary_conditions/extrapolation.py)
在 order=0 时，将第一个内部点的守恒量复制到出口物理点及全部外侧 halo。
参考
[IsothermalWallBC](https://raw.githubusercontent.com/opensbli/opensbli/e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d/opensbli/core/boundary_conditions/isothermal_wall.py)
将壁面动量置零、总能量由当前密度和 Tw 指定，密度本身保留演化。
第 h 层壁面 halo 的温度取 `(h+1)*Tw-h*T1`，压力取壁面压力，
速度取第 h 个内部点速度的反号，再通过 EOS 构造密度与能量。
不能使用该文件中另一种未被算例选用的 `IsothermalWall_ZeroPressureGradBC`。

ASTR `bc::noslip` 当前通过内部压力外推和 Tw 重构壁面密度，
与上述合同不同。这是两种边界离散策略的差别，本轮不将旧模式判作 BUG 或直接替换。
新增模式需检查 halo 温度正性，失败时停止，不能裁剪负温度继续计算。

下一批工作顺序：

1. 核对上游生成算法的 BC/RK 调用顺序及角点覆盖顺序。
2. 新模式独立配置、守恒量更新、物理 halo 与 EOS 单元测试。
3. 接入 CPU/GPU 边界调度、RHS 范围、qsave 与完整 RK 后比较入口。
4. 小网格逐 stage 和 NP=1/2/4 对比通过后，才生成可运行参考输入。

当前仅完成物性前置，不更新 G3-G6 为通过，不启动不匹配边界的长时间运行。

## G2 更新：基础算子 CPU/GPU 验证

2026-09-07 后续批次已实现基础算子，但未接入 `boucon` 或 GPU 主循环。
上一节“仅完成物性前置”对应前一批次的状态。现有算例默认行为仍不变。

- `src/perfect_gas_boundary_body.inc`：独立数学定义，无 MPI、文件读取、全局流场访问。
- `src/perfect_gas_boundary.F90`：CPU pure 模块。
- `src_gpu/perfect_gas_boundary_gpu.cuf`：独立 device 模块，共用数学定义，不插入数组传输。
- `constant_conservative_boundary`：物理点和 hm 层 halo 都复制指定守恒状态，
  出口适配器将传入第一个内部点的 q。
- `piecewise_conservative_boundary`：严格 `x>split` 选择右侧，否则选择左侧状态。
- `pressure_extrapolation_inlet`：保留此前核对的 abs(u)>=a 分支及亚声速能量 halo 规则。
- `density_evolved_isothermal_wall`：保留壁面密度，强制零动量和 Tw，构造全部壁面 halo。

算子采用无量纲五方程理想气体定义。状态顺序为 rho、rhou、rhov、rhow、rhoE。
压力由 `(gamma-1)*(rhoE-sum(momentum**2)/(2*rho))` 得到，
温度由 `p*gamma*Mach**2/rho` 得到。
参数或状态不合法时返回非零 status，整列校验完成前不写回 face/halo。
这不是全边界事务回滚，后续适配器必须检查 status 并停止，不能忽略错误继续推进。

### 参考执行顺序

已核对固定版本的
[TraditionalAlgorithmRK.generate_solution](https://raw.githubusercontent.com/opensbli/opensbli/e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d/opensbli/code_generation/algorithm/algorithm.py)：
每个时间步先施加边界，再进入子步循环；子步内按空间算子、RK 更新、边界的顺序执行。
最后一个子步之后仍执行边界。ASTR 新模式需要在 RK 更新后恢复边界约束，
不能仅依赖下一子步开始的 `boucon`。

[SimulationBlock.apply_boundary_conditions](https://raw.githubusercontent.com/opensbli/opensbli/e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d/opensbli/core/block.py)
按方向及 side=0,1 顺序排列。因此该二维参考的覆盖顺序是入口、出口、壁面、顶面。
壁面在下方角点晚于入口/出口执行，顶面在上方角点最后执行。
新模式的 MPI halo 与角点适配必须保持该依赖次序，不能用四个同时覆盖角点的 kernel。
此结论来自固定版本源码，仍不声称已找到了 2018 数据归档的确切生成提交。

### 验证证据

```bash
cmake --build build_gpu_probe --target astr boundary_contract_cpu_probe boundary_contract_gpu_probe -j4
cmake --build build_cpu_probe -j4
ASTR_BOUNDARY_PROBE_EXE=build_gpu_probe/bin/boundary_contract_cpu_probe \
  python3 -m unittest discover -s tests/gpu_validation -p test_perfect_gas_boundary.py
ASTR_BOUNDARY_PROBE_EXE=build_gpu_probe/bin/boundary_contract_gpu_probe \
  python3 -m unittest discover -s tests/gpu_validation -p test_perfect_gas_boundary.py
```

CPU/GPU 各 5 项测试通过，包含 11 个探针输入：亚声速、超声速、反向超声速、
恰好声速、六层壁面 halo、首层/外层负温度拒绝、分段点左侧/恰好/右侧、零阶复制。
Python 使用独立 EOS 和逐层公式检查结果，不只比较 CPU 与 GPU 彼此相同。
额外 gfortran bounds/FPE 检查版本也通过。CUDA 每次探针 kernel 后显式同步并检查返回码。

Compute Sanitizer memcheck 对亚声速入口、有效壁面及外层负温度拒绝三个探针均为
`ERROR SUMMARY: 0 errors`。日志：
`tests/gpu_validation/out/opensbli_boundary_contract_gate/{inlet,wall,wall_reject}_memcheck.log`。
这些是单列算子测试，不是全网格、RK、MPI 或 OpenSBLI 物理验证。

下一批次将配置解析、初始入口 halo 缓存、物理边界适配器与新模式的 RHS/RK 范围一并接入。
必须先验证各 stage、角点和内部 rank 交界，再启动完整参考网格。

## G2 更新：配置和按面适配

2026-09-07，本批新增以下实现，仍未启用主循环中的参考模式：

- `conservative_boundary_config.F90`：Fortran namelist 解析器，读取 schema=1、split_x 和左右五个守恒量。
  空路径表示关闭；指定文件缺失、字段缺项、未知字段、非有限数、非正密度或内能均拒绝。
  文件读取失败不回退到旧边界，也不会将 enabled 留为真。
- `conservative_boundary_faces_body.inc` 和 CPU/device 包装模块：将单列算子映射到流场面。
  x 面处理物理 j/k，y 面扩展到 x 方向 halo，按入口、出口、壁面、顶面顺序调用。
- `apply_conservative_face_kernel`：一个线程处理一列，以 256 线程块启动边界探针。
  错误通过原子操作返回 side/status，不做温度截断或状态修补。
- `prepare_opensbli_reference_inputs.py` 额外生成 `conservative_boundary.nml`。
  最新完整输入资产位于 `out/opensbli_input_gate_pr072_v2`，旧目录不覆盖。
  仍然标记 `INPUT_ONLY_NOT_RUNNABLE`。

### 当前验证结果

顶层 CMake CPU/GPU 主程序及配置/面探针均构建通过。
配置解析 3 项测试在 gfortran 和 NVHPC 上均通过，覆盖 LF/CRLF、缺项和非法输入。
生成器输出的 609x255x9 配置由 NVHPC 探针实际读入，split_x=40、左右状态与 manifest 一致。

`test_boundary_faces.py` 的 3 项测试在 gfortran、NVHPC CPU 和 GPU 探针上均通过：
检查内部状态不变、展向均匀、入口能量 halo、出口复制、壁面密度/EOS、上下角点优先级，
以及单面掩码不写入无关区域。
另外以掩码 0、1、2、4、8、12、13、14、15 比较 CPU/GPU 完整探针数组，
本组最大绝对差均为 0。这里只覆盖探针输入，不声称任意场逐位相同。
四面 GPU memcheck 为零错误，日志为
`out/opensbli_boundary_contract_gate/faces_memcheck.log`。

```bash
cmake --build build_gpu_probe --target conservative_boundary_config_probe boundary_faces_cpu_probe boundary_faces_gpu_probe -j4
ASTR_BOUNDARY_CONFIG_PROBE_EXE=build_gpu_probe/bin/conservative_boundary_config_probe \
  python3 -m unittest discover -s tests/gpu_validation -p test_conservative_boundary_config.py
ASTR_BOUNDARY_FACES_PROBE_EXE=build_gpu_probe/bin/boundary_faces_cpu_probe \
  python3 -m unittest discover -s tests/gpu_validation -p test_boundary_faces.py
ASTR_BOUNDARY_FACES_PROBE_EXE=build_gpu_probe/bin/boundary_faces_gpu_probe \
  python3 -m unittest discover -s tests/gpu_validation -p test_boundary_faces.py
```

这些面掩码模拟不同物理面所有权，没有启动 MPI，不是 NP=2/4 验收。
适配器假定输入 halo 已初始化且满足调用阶段要求，尚未实现全程序的初始化/通信调度。

### RHS 接入的必要改动

源码检查确认 `parallelini` 对非周期方向默认令物理边界外的更新范围为
`is=1/ie=im-1`、`js=1/je=jm-1`。`solver::convrsduwd` 和中心对流路径使用这些范围。
参考模式的亚声速入口 q 和等温壁面 rho 必须演化，不能仅挂接 `boucon` 后沿用旧范围。

也不能直接把全局 is/js 改为零：现有迎风物理边界通量分裂按 npdc 将取值限制在物理域内，
Roe 分解、近边界重构闭合和 halo 度量必须一并核对。
这是新增边界模式与旧离散范围的接口差异，不在本批修改旧 CPU 边界合同。
下一批应完成新模式独立的边界 RHS 范围/离散合同、初始化缓存和 RK 后恢复，再接通主程序。

## G2 更新：物理面迎风 RHS 制造解

2026-09-07，新增 `solver::convrsduwd(physical_halo_rhs=.true.)` 可选路径。
默认调用不变，新路径仅在函数局部将 RHS 范围扩展到全部物理点，
并令重构使用调用者提供的完整 halo。全局 `npdc`、网格度量和 MPI 所有权不改动。
目前限制为三维无量纲 MP7，尚无生产主循环调用启用此选项。

该路径复用 ASTR 的 MP7/Steger-Warming 分裂和已有 Roe 特征重构。
固定版本参考脚本则使用 WENO5-Z/LLF、四阶中心黏性离散和 SSP RK3。
两者不是同一离散算法；本 goal 是参考物理问题的 ASTR 显式格式复现，
后续必须通过独立网格和时间验证评估离散差异，不能要求与归档数组逐位相同。

### 本轮证据

`boundary_rhs_manufactured.F90` 使用恒定速度、恒定温度及三方向仿射密度，
从 Euler 通量直接求解析散度。检查所有物理点、边界与角点的五个分量，
包括有限数检查和旧模式不更新物理面的检查。CPU 算子返回正通量散度，
GPU 算子已带 RHS 负号，比较时明确处理该符号差异。

| 路径 | 最大绝对误差 |
|---|---:|
| CPU 物理空间 | 3.6591823321385775e-16 |
| CPU Roe 特征空间 | 3.2797115717686509e-16 |
| GPU 物理空间 | 3.6591823321385775e-16 |
| GPU Roe 特征空间 | 2.1694885471434944e-16 |

GPU 探针实际调用现有 x/y/z flux/RHS kernel，每次启动后显式同步。
此处采用单位度量且 halo 人工填入解析值，未验证非均匀网格的度量 halo、
shock mask 生成、传感器耦合、扩散、RK 时序或多 rank 新边界模式。

可复现驱动及验收：

```bash
python3 -m unittest discover -s tests/gpu_validation -p test_boundary_rhs_gate.py
python3 tests/gpu_validation/run_boundary_rhs_gate.py \
  --out tests/gpu_validation/out/opensbli_boundary_rhs_gate_v3
```

驱动保存二进制 SHA256、命令、计时、完整日志和 JSON，拒绝覆盖输出目录。
必须同时满足退出码、唯一成功标记、有限误差和绝对误差 1e-10。
CLI 子命令是 `test bcrh`，不是 `bcrhs`，因为旧 `testmode` 长度为四。
不以打印帮助后退出零作为测试通过。驱动自身四项校验测试通过。

初次 sanitizer 在 `MPI_Init` 的 UCX CUDA context 探测中报告 11 个错误，
失败证据保留在 `out/opensbli_boundary_rhs_gate`。
仅对不交换数据的 NP=1 sanitizer 子测试指定
`OMPI_MCA_pml=ob1`、`OMPI_MCA_osc=pt2pt`、`OMPI_MCA_btl=self,vader,tcp`，
不屏蔽 CUDA API 错误，不改生产通信配置。隔离后 memcheck 为零错误，
普通 CPU/GPU 子测试仍使用原 MPI 设置。
加入逐元素有限数检查后的最终重编译版本也通过全部三个执行检查，
证据为 `out/opensbli_boundary_rhs_gate_v3/summary.json` 及同目录日志。

默认路径回归：`out/opensbli_rhs_legacy_tgv_np2`，TGV、NP=2、2x1x1、
128^3、2 steps、滤波/扩散均开启，CPU/GPU 逐场组合容差 1e-10 通过，
最大绝对差为 q5 的 2.8421709430404007e-13。
同目录 `flowstate_compare.txt` 的动能、涡量平方统计量与耗散比较通过，
三者最大绝对差分别为 2.9198865547641617e-14、8.4376949871511897e-15、
5.7354294924483185e-17。
该项只说明本轮可选接口未破坏此默认用例，不是新 SBLI 多 rank 验收。

### 接入顺序

1. 扩散 RHS：现有 `diffrsdcal6` 在各方向累加时仍用全局 `is:ie/js:je/ks:ke`。
   新模式需独立扩展累加范围，并保留现有显式物理边界导数闭合。
   先通过黏性制造解，不将迎风重构的局部 `npdc=3` 传播给中心导数。
2. 处理物理面几何 halo 与入口初始守恒 halo，再验证 primitive 更新。
   `dataswap` 不为物理边界生成几何外推值，不能假设通信后即已有效。
3. 配置广播、CPU/GPU 按面调度和每个 RK 更新后的边界恢复一起接通。
   保持入口、出口、壁面、顶面的角点覆盖次序，保留完整 RK 场输出语义。
4. 新模式 NP=1 短步，然后 NP=2/4 同相位逐场/统计与展向均匀性。
   这些检查通过后再启动参考网格与长时间物理验证。

G2 仍为进行中，G3--G6 未验收，完整仿真 goal 保持 active。

## G2 更新：CPU 黏性物理面 RHS

新增 `diffrsdcal6(physical_boundary_rhs=.true.)`，只扩展三个方向的 RHS
累加范围到全部物理点，不改 `fds`、全局范围或 `npdc`。
默认调用保持旧行为，新选项限制为三维无量纲单组分、无湍流模型的 `643e`。
当前尚未由生产主循环启用。

`boundary_rhs_manufactured.F90::check_boundary_diffusion_rhs` 采用恒温、
三个速度分量分别沿对应坐标二次变化的制造场。解析应力为
`tau_dd=2*mu*(du_d/dx_d-div(u)/3)`，动量 RHS 为 `8*mu*a_d/3`，
能量 RHS 为 `sum(tau_dd*du_d/dx_d+u_d*8*mu*a_d/3)`。
这里 `u_d=0.1+a_d*x_d^2`、`a=(0.001,0.002,0.003)`、`mu=1/950`，
单位度量，计算坐标步长为一。

测试初次失败源于制造解预期遗漏边界截断误差，并非已确认的 CPU 程序 bug。
实际 `derivative::diff6ec` 最外点和相邻点使用二阶闭合，即使输入标签是 `643e`。
对三次项 `c*x^3`，外侧三点单边导数误差为 `-2*c`，相邻中心导数为 `+c`。
黏性功中该系数是 `c=8*mu*a_d^2/3`。探针单独计算这一解析截断项，
并保留相对连续方程的原始误差，不改生产系数，不放宽 1e-10 门槛。

- 相对上述解析离散结果最大绝对误差：3.2187251995663413e-20。
- 相对连续方程最大绝对误差：7.8596491228078503e-8。
- 检查所有物理点五分量有限，旧模式角点不累加，全局闭合策略不变。
- 顶层 CMake CPU/GPU 构建通过，完整 RHS 驱动与 GPU memcheck 通过。
  证据为 `out/opensbli_boundary_rhs_diffusion_gate/summary.json`。

此新增黏性探针在 CPU 和 CUDA 可执行文件中均执行 **CPU** 黏性函数。
CUDA 可执行文件中的迎风探针确实执行 GPU kernel，但不能据此声称 GPU 黏性项已验收。
下一步需要实际调用 GPU 黏性 flux/RHS kernel 验证 x/y 物理面、z 周期薄层组合，
然后再接通几何 halo、边界初始化和 RK 调度。

### 同批后续：GPU 黏性 kernel 验证

上述 CPU 门槛之后，已增加并运行 `check_boundary_diffusion_gpu`：
使用 x/y 二次速度、z 恒定速度的周期薄层制造场，实际调用
`diffusion_flux_xyphysical_global_kernel` 以及三个方向的 stored RHS kernel。
所有物理点均参与累加，x/y 保留真实物理闭合，z 周期通量 halo 在探针中
通过 host staging 填充。这里没有调用生产 MPI transport，不是通信验证。

GPU 相对解析离散结果的最大绝对误差为 1.2493735972000930e-19，
Compute Sanitizer memcheck 零错误。
完整证据：`out/opensbli_boundary_rhs_gpu_diffusion_gate/summary.json`。
该探针包含 GPU 梯度计算、Sutherland 黏度（恒温时为常数）、应力、黏性功
和 RHS 累加，但尚不验证变温热传导、非单位度量、MPI 分区或 RK 更新。
CPU 三方向物理面的制造场与 GPU 周期薄层制造场不同，分别对解析解验收，
此处不把两者的误差数字解释为逐场 CPU/GPU 差值。

默认 NP=2 TGV 两步逐场回归再次通过，最大绝对差 2.8421709430404007e-13，
证据为 `out/opensbli_diffusion_rhs_legacy_tgv_np2/flowfield_compare.txt`。
下一批进入物理几何 halo、入口缓存、配置广播和主循环的分阶段边界调度。

## G2 更新：运行时配置 MPI 分发

新增 `conservative_boundary_runtime.F90`。`load_conservative_boundary_environment`
仅由 rank 0 读取 `ASTR_CONSERVATIVE_BOUNDARY_FILE`，复用已有 namelist 解析器，
先广播解析状态再处理失败，然后广播 enabled、split_x 和两组五分量守恒目标。
未设置或空字符串表示关闭，纯空白路径、缺文件和非法配置报错退出。
公开状态为 protected，外部不能绕过加载器任意改写。

当前由 `astr test bcfg` 调用此加载器，尚未在 `astr run` 主初始化中启用。
因此仅设置该环境变量还不能启动参考边界模式，不能把配置层验收当作生产接入完成。

```bash
python3 tests/gpu_validation/run_boundary_config_mpi_gate.py \
  --out tests/gpu_validation/out/opensbli_boundary_config_mpi_gate
```

顶层 CMake CPU/CUDA 构建通过，真实 MPI NP=1/2/4 共 40 项测试全部通过。
包括 unset、空字符串、合法、非法、缺文件、纯空白及 root-only 配置。
root-only 测试通过 MPI 多应用启动给非根 rank 设置不存在的文件路径，
确认非根 rank 不读自己的路径，仍收到完整且相同的 rank-0 状态。
拒绝项要求非零退出、错误标记且无成功配置行，超时不算通过。
证据在 `out/opensbli_boundary_config_mpi_gate/summary.json`，含二进制 SHA256、
命令和逐项日志。该测试不交换流场、不调用 GPU kernel，不能证明多卡物理计算正确。

### 几何初始化审计补充

`geom::gridgeom` 在 `dataswap` 与 `datasync` 后还调用 `bc::geombc`。
因此准确结论不是所有物理度量 halo 都未填，而是现有初始化不完整覆盖新模式所需的面及层数：
`geombc` 只处理 y 向 bc=41，并固定外推四层。当前 hm=5。

发现 `src/bc.F90:434` 的上壁循环将 Jacobian 写入 `jacob(1,jm+1:jm+4,k)`，
而右侧使用当前 i 列。它会反复覆盖 i=1，其他列未由本分支正确写入。
这是明确索引问题，已向用户报告，未经批准不修复、不复制到新 GPU 实现。
本目标顶面采用分段守恒状态，不需要启用原上壁 bc=41 分支。
新模式的几何 halo 需独立验证，并保留原物理点度量和 MPI 所有权。

后续顺序仍为几何 halo、入口守恒缓存、CPU/GPU RK 调度、短步多 rank、
参考网格和长时间物理验证。完整 goal 保持 active。

## G2 更新：几何 halo 与自由流门槛失败

新增 `rectilinear_metric_halo.F90`，针对本目标的可分离直角拉伸网格：
检查正有限 Jacobian、正对角逆度量、近零非对角项，以及法向余因子
`J*dxi(d,d)` 沿对应方向不变，然后沿拥有的物理面常数延伸全部 hm 层。
只填法向 halo、横向索引位于物理域的面条带，不填多方向都在 halo 内的边角，
也不覆盖未拥有的面或 MPI halo。物理点不改动，尚未接入生产初始化。

顶层 CMake `rectilinear_metric_halo_probe` 对 64 种面所有权组合、hm=5
逐元素检查通过，包含物理点、未拥有区域与边角保持原值，以及非法度量拒绝。
这只能证明填充合同，不能证明与现有数值格式组合后自由流保持。

追加的 `check_stretched_metric_rhs` 实际调用 CPU `convrsduwd`，
采用均匀流、x/y 拉伸且 z 周期的单位状态。这个门槛 **失败**：
物理空间路径最大 RHS 为 5.1500469574805585e-5，出现在连续性方程，
Fortran `maxloc` 返回 (8,8,1,1)，对应物理索引 (7,7,0)、第一个守恒量。
测试立即非零退出，未执行后续 Roe/GPU 拉伸检查，不能声称这两项已通过。
失败证据为 `out/opensbli_stretched_metric_rhs_gate`；加入定位输出的重跑为
`out/opensbli_stretched_metric_rhs_gate_diagnostic`。旧均匀度量门槛仍已通过。

源码与独立代数检查发现一个需要人工决策的离散问题：
`riemann::flux_steger_warming` 固定 `eps=0.04`，GPU 分裂也使用相同固定值。
特征速度随逆度量缩放，固定平滑参数却不随之缩放，使均匀流的正/负分裂通量
分别沿拉伸方向变化。它们在同一点相加仍正确，但不同方向迎风重构不保证完全抵消。
例如 gamma=1.4、u=0.1、c=0.5、法向间距 (1,1.02,1.16)、单位横向余因子时，
正质量分裂通量变化幅度为 5.67235849959824e-4；仅在独立代数试算中令
eps 随法向逆度量缩放，该幅度降至 5.55e-17。
此结果说明固定 eps 破坏该缩放性质，但尚未证明它解释完整 RHS 残差的全部来源。

未经用户批准，未修改原 CPU/GPU 分裂算法，也未放宽自由流门槛。
建议批准后在新参考模式下引入可选的度量一致平滑参数，保留原默认值，
重新检查物理空间/Roe、拉伸自由流、激波算例和 CPU/GPU 一致性。
在此门槛通过前不启动参考生产仿真。完整 goal 未完成，保持 active。

### 待批准期间的 Roe 交叉诊断

未收到修改平滑参数的明确批准。本轮只调整失败探针的诊断顺序：
分别测量物理空间和 Roe 路径，然后统一非零退出；不改变任何生产分裂公式。
源码确认 `convrsduwd` 的 Roe 分支投影同一份 `flux_steger_warming` 分裂结果，
因此两条路径共同继承固定 eps 的度量依赖。

实测结果：

| CPU 路径 | 最大 RHS 绝对值 | 物理索引/分量 |
|---|---:|---|
| 物理空间 MP7 | 5.1500469574805585e-5 | (7,7,0), rho |
| Roe MP7 | 3.4544124696098133e-5 | (8,8,0), rho |

证据：`out/opensbli_stretched_metric_roe_diagnostic/cpu.log` 与 `summary.json`，
状态为 fail。GPU 拉伸测试没有执行，不推断其数值误差。
这排除了“仅切换 Roe 即可通过自由流门槛”的替代路径，仍不证明 eps 是全部误差来源。
下一项改变分裂平滑方式的试验需要用户明确批准，自动 goal 续行不视为批准。

### 阻塞审计

同一待批准事项已连续三轮存在。最新失败报告仍为非零退出，未发现仍运行的
mpirun、Compute Sanitizer 或 CMake 构建进程。本轮未重跑同一失败测试。
将 goal 标记为 blocked，而不是完成或缩减验收范围。
恢复条件为用户对新参考模式下度量一致 eps 试验的明确决策。
批准后的第一步仍是根因隔离及自由流验收，不直接启动长时间仿真，
也不把修复 eps 预先当作已经解决全部残差。

## 已批准：度量一致 eps 试验

用户明确回复“继续，确认实验度量一致eps”。此次只引入可选路径，未更换旧默认算法。
在第 d 方向取 `eps=0.04*sqrt(sum(dxi(d,:)**2))`，与该方向特征速度按同一逆度量缩放。
CPU `convrsduwd`/`flux_steger_warming` 新增可选参数 `metric_consistent_eps`，
不传参或传 false 时保留固定 0.04。三个方向均传递该选项。
GPU `commvar_gpu` 新增开关，由 `copy_commvar_to_gpu` 默认关闭，
覆盖物理空间标量、物理空间向量和 Roe 共用标量分裂函数。
试验探针显式开启并在结束时关闭；在本小节记录的阶段，生产主循环尚未开启
该选项。后续生产接线状态以文末更新为准。

首次 GPU 试验漏改 Roe 共用标量分裂函数，物理空间通过但 Roe 失败，
日志保留在 `out/opensbli_metric_eps_gate`。补齐同一选项后重新构建并验证：

| 路径 | 拉伸自由流最大 RHS |
|---|---:|
| CPU 固定 eps，物理空间对照 | 5.1500469574777830e-5 |
| CPU 固定 eps，Roe 对照 | 3.4544124696153644e-5 |
| CPU 度量一致 eps，物理空间 | 2.4980018054066022e-16 |
| CPU 度量一致 eps，Roe | 2.4980018054066022e-16 |
| GPU 度量一致 eps，物理空间 | 2.4980018054066022e-16 |
| GPU 度量一致 eps，Roe | 2.2204460492503131e-16 |

同一可执行文件还检查默认调用与显式 false 的 RHS 完全一致。
完整探针及 Compute Sanitizer memcheck 通过，证据为
`out/opensbli_metric_eps_gate_v2/summary.json`，保留最终二进制哈希、命令、日志。
此结果说明该选项消除了本制造例中观察到的自由流残差，不代表一般曲线网格、
激波捕捉或完整 SBLI 物理问题已经验收。

后续需对开启新选项的非均匀状态与激波进一步验收，再接通几何 halo、
入口缓存和 RK 调度。现有无新选项的激波回归只验证旧默认路径未被破坏。

最终二进制旧路径回归：
- `out/metric_eps_final_legacy_shuosher`：NP=2、2x1x1、400x8x8、3 steps，
  传感器 mask 无差异，逐场最大绝对差 7.1054273576010019e-15，统计组合容差通过。
- `out/metric_eps_legacy_tgv_np2`：NP=2、2x1x1、128^3、2 steps，
  滤波和扩散开启，逐场最大绝对差 2.8421709430404007e-13，组合容差通过。
- `test_boundary_rhs_gate.py` 四项驱动校验通过，`git diff --check` 无错误。

这里的 Shu-Osher 和 TGV 均未开启度量一致 eps，不能用来宣称新选项的激波验收完成。

### 开启新选项的非均匀 RHS 比较

新增 `check_metric_nonuniform_rhs`，在同一拉伸网格和已初始化的度量 halo 上，
分别构造平滑密度/压力/速度扰动、x 向以及 y 向密度压力间断。
间断左右状态为 rho=(1,0.125)、p=(1,0.1)、速度为零，温度由同一理想气体
无量纲 EOS 重构。z 向均匀。每组先计算 CPU 物理空间/Roe 正通量散度，
再调用实际 GPU kernel 比较负 RHS，覆盖全部物理点和五个分量。
检查有限数且参考 RHS 非平凡，避免全零结果伪通过。

| 状态 | 物理空间 max_abs | Roe max_abs |
|---|---:|---:|
| 平滑非均匀场 | 1.9428902930940239e-16 | 2.9143354396410359e-16 |
| x 向间断 | 0 | 1.9984014443252818e-15 |
| y 向间断 | 6.6613381477509392e-16 | 6.6613381477509392e-16 |

本轮 CPU/GPU 均开启度量一致 eps。完整驱动与 GPU memcheck 通过，
证据为 `out/opensbli_metric_eps_nonuniform_gate/summary.json` 及同目录日志。
这些是初始间断上的算子一致性，不是已经形成激波后的物理准确性、时间推进、
传感器选择性重构或 MPI 通信验收。Roe 探针将所有界面标记为特征重构。
生产默认开关仍关闭，下一步接通新模式初始化与 RK 边界调度，随后开展短步演化测试。

### 保守边界阶段的连续调用验证

`conservative_boundary_runtime` 新增一次性初始化和重复阶段入口，尚未接入
`astr run`。初始化扩展矩形拉伸网格的物理面度量 halo，并从入口物理面
播种 x 向初始 halo。该操作要求上游提供已验证的 x 均匀初始剖面，
当前阶段入口本身尚未验证全域 x 均匀性，不能作为一般入口初始化器使用。

重复阶段顺序为：刷新物理点物性、qswap、入口、出口、qswap、壁面、
顶面、qswap、刷新物理点和物理边界 ghost 物性。这里复用的 CPU qswap
还包含共享节点平均，不是单纯复制 halo。顶面最后覆盖两端上角点。
原有生产边界调度未改动。

新增实际调用探针 `astr test bcst`，在 NP=1 下初始化后扰动壁面密度和
入口能量，再执行下一次边界阶段。验证壁面密度保留、无滑移和等温条件、
入口 ghost 能量跟随物理面更新、压力匹配，以及分段顶面的角点优先级。
CPU 构建与 CUDA 构建的最大绝对误差均为 2.2204460492503131e-16。
两种构建此次均执行 CPU 阶段，不涉及 GPU 边界调度或 RHS/RK 时间推进。

证据：`tests/gpu_validation/out/opensbli_boundary_stage_gate/summary.json`，
记录实际命令、日志、配置和二进制 SHA256。
`run_boundary_stage_gate.py` 可独立生成配置并复跑，拒绝覆盖既有输出目录。
`test_boundary_stage_gate.py` 五项测试通过，拒绝帮助信息、非零退出、重复
PASS 标记、非有限误差和超限误差造成的伪通过。

本小节记录阶段的下一验收关口为生产模式准入、GPU 阶段调度、初始化/RK
接线和 NP=1/2/4 短步同相位比较。当时尚未启动完整参考网格长时间仿真；
后续生产接线与长跑状态以文末更新为准。

### GPU 保守边界阶段接通与 NP=1 对照

新增 `src_gpu/conservative_boundary_stage_gpu.cuf`，顶层 CMake 纳入 CUDA
构建。初始化上传 x 坐标线和左右顶面目标，此后 q 保留在设备上。
入口、出口、壁面、顶面顺序调用已有共享公式的 GPU 面 kernel，每次
kernel 后保持同步。交换位置与 CPU 阶段一致，复用现有 GPU halo
交换及共享节点平均。每个面的错误标志在 MPI 全局归约后决定是否退出。
目前模块由探针调用，尚未连接生产主循环，也没有替代原有边界模式。

扩展 `astr test bcst`：CPU 初始化后，对壁面密度和入口能量施加扰动，
将同一状态交给 CPU 与实际 GPU 阶段。完整 q 数组最大绝对差为 0，
已比较的 rho/pressure x 面条带、temperature/velocity y 面条带差值为 0。
这是一次 GPU 阶段对照，不是完整 RK、多 rank 或任意初始 halo 验证。
初值在整个探针数组内均为有限数，不能据此证明生产未初始化角点安全。

初次 memcheck 返回 84 个错误，日志栈位于 HPC-X 的主机逻辑量
MPI_Allreduce 缓冲区探测 `cuMemRetainAllocationHandle`，不是面 kernel
越界报告。失败证据保留在 `out/opensbli_boundary_stage_gpu_gate`。
经所链接 HPC-X 的 ompi_info 核查，NP=1 memcheck 专用环境设置
`OMPI_MCA_opal_cuda_support=false`，同时保留此前的 ob1/pt2pt/host BTL
隔离设置。未关闭应用 CUDA 检查或 sanitizer 错误报告，未更改生产 MPI。

重新验证 `out/opensbli_boundary_stage_gpu_gate_v2/summary.json`：
CPU、实际 GPU、GPU memcheck 均通过，memcheck 为 0 errors，GPU 比较为 0。
测试驱动现在要求独立 GPU PASS 标记，六项解析单元测试通过。
后续仍须完成生产准入与初始化/RK 调度接线，随后 NP=1/2/4 短步比较。

### 完整 RHS 入口参数传递

`src/solver.F90::rhscal` 新增可选 `physical_halo_rhs` 和
`metric_consistent_eps`，分别传给显式对流及扩散算子。未传入时保留
旧调用语义，原有 `qrhs=-qrhs`、扩散叠加和算例源项顺序不变。
生产调用目前仍未传入新参数。

实际探针新增两项检查：
- 在既有制造状态上，直接调用对流、取负、叠加扩散，与完整 rhscal
  比较。关闭和开启新选项两条路径均零差异；默认调用与显式关闭也零差异。
  此项只检查算子组合，沿用的 q 与物性来自不同制造探针，不作为新的
  自洽流动解或物理验证证据。
- 在自洽的拉伸自由流状态上，经完整 rhscal 开启度量一致 eps，验证
  参数实际传递至通量分裂，未因调度遗漏重新引入固定 eps 残差。

证据为 `out/opensbli_rhscal_composition_gate/summary.json` 及 CPU/GPU/
memcheck 日志。完整探针通过，memcheck 为 0 errors；四项驱动单元测试通过。
本关口验证 CPU RHS 总入口与既有 GPU 算子回归，不包含 GPU 主循环接线。
本小节记录阶段的下一步仍为生产模式准入和初始化/RK 集成，不能仅据此启动
无准入保护的长跑。后续准入、集成与 restart 状态以文末更新为准。

### OpenSBLI restart 支持与 step 60000 续算

OpenSBLI 专用模式现已区分 fresh 与 restart 初始化。fresh 路径继续要求
三维初值沿 x 均匀，并从初始物理面播种 x 向 halo。restart 路径允许演化后
的非均匀物理场，但仍逐点拒绝非有限或不可接受的理想气体守恒状态。
checkpoint 不存储 ghost。入口压力外推算子又要求入口 ghost 的密度与动量
保持规定剖面，因此 restart 从每次启动均读取的 `inlet.prof` 重建入口
ghost 的前四个守恒分量，再由边界 stage 更新能量并重建其余物理边界和
MPI halo。不能用已演化的入口物理面替代该规定剖面。

通用 checkpoint 读取同时修复两项问题：HDF5 中的 nstep 必须先与
`auxiliary.txt` 比较，比较通过后才能写回全局 nstep；序列 checkpoint
路径使用调用者传入的 folder，而不是硬编码 `outdat`。GPU 边界层
`flowstate.dat` 改用 CPU 已有的 `listinit` restart 语义，在 checkpoint
步覆盖续写，保留此前记录并去除 checkpoint 之后的失效尾段。

`run_opensbli_restart_equivalence.sh` 验证 CPU NP=1 和 GPU NP=2：连续四步
与两步加 restart 两步的六个原始场及五个重构守恒量最大差为
1.3322676295501878e-15；不一致的 auxiliary/HDF nstep 均被拒绝；统计
步号均为 1,2,3,4。fresh/restart 初值准入矩阵覆盖 NP=1/2/4，restart
空 halo 的 CPU/GPU stage 和 GPU Compute Sanitizer memcheck 通过。

指定的 609x255x9、NP=2 checkpoint 已从 nstep=60000、time=2400.0
隔离复制并单步推进到 nstep=60001、time=2400.04，六个输出场全部有限。
正式续算由 `astr-opensbli-np2.service` 从 step 60000 启动，使用两张 GPU，
每个 rank 约 938 MiB，启动检查时 nvitop 报告 82% 和 85% SM。user linger
与服务开机启动已启用，服务只引用隔离续算目录，不修改原 step 60000
证据目录。长时间 OpenSBLI 外部物理对比仍在进行，完整物理 goal 未完成。

### Restart 前后统计连续性检查

原连续计算与 step 60000 restart 计算在 `60000--62211` 共 2212 个重叠步上
完成比较。重启后第一步的时间、质量流量和 `fbcx` 增量与连续计算一致；
壁面热流和 `wrms` 增量差分别约为 `7.0e-19` 和 `6.1e-21`。前 100 步的
统计差保持在舍入误差量级，因此 restart 接缝没有可辨识的统计跳变。

重叠区间内的最大差异为：

| 统计量 | 最大绝对差 | 相对连续计算最大尺度 | 判断 |
|---|---:|---:|---|
| `massflux` | `2.8497204596078518e-11` | `2.6689130596453071e-11` | 无明显变化 |
| `fbcx` | `1.3341951068251978e-10` | `3.7014441845971177e-7` | 无明显变化 |
| `wallheatflux` | `5.5144718938542487e-9` | `6.6488814794166704e-4` | 小于 `0.067%` |
| `wrms` | `3.8914281162951948e-6` | `8.0715315069486293e-1` | 长期轨迹明显分离 |

`wrms` 的绝对差约在 step 60329、60920 和 61303 先后超过 `1e-14`、`1e-7`
和 `1e-6`。现有证据支持“restart 局部连续”，但不支持“长期逐点轨迹严格
相同”。后续 checkpoint 必须比较同相位完整场、z 向非均匀度、Ducros 传感器
和选择性 Roe 区域，判断该分离是非线性放大舍入扰动，还是存在未纳入
checkpoint 的演化状态。该问题不影响继续积累长时间物理统计，但必须在 G3
完成前给出归因或明确的统计验收合同。
