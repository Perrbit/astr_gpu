# KEEP and ME6-Opti Integration Progress

日期：2026-11-10。实施基点：`e992dc7`，`feature/gpu_dev`。
周期算子、G1/G1a、接线与验证工具已保存于源码提交 `64f48fc`。
总计划：`documents/ASTR_GPU_KEEP_INTEGRATION_PLAN.md`。

## 当前状态

用户已授权启动总计划 goal。当前已完成 KP1 显式 CPU/GPU 数学组件及周期
主程序首轮接线、三方向双 rank 短测和精确恢复。周期紧致 KEEP 已接入
CPU/GPU 全局线求解并通过分项、冻结 RHS 和代表长窗，周期主滤波也已
独立选择。KP1/KP2 周期基线已具备下列代表验证证据；KP0 的完整几何
职责、KP0-A 具体候选及后续物理／联合范围仍未完成。ME6 已完成均匀
周期主程序接入、私有边 halo、代表滤波与短恢复检查。
G1/G1a 已完成周期静态 KEEP 几何的三层收敛、代表 CPU/GPU/MPI、
交叉格式滤波及短恢复检查；ME6 曲线扩展和物理边界仍开放，详见文末。
本记录与总计划分开，不用组件结果替代完整阶段验收。

| 阶段 | 状态及剩余工作 |
| --- | --- |
| KP0 | 周期 KEEP 的几何／对流／黏性梯度／滤波职责已分开并有代表结果；全项收支和物理边界专用滤波仍开放 |
| KP0-A | G1/G1a 首轮检查通过；ME6 G2 范数比方案已批准、待实施，物理端闭合仍待定稿 |
| KP1 | 组件、同状态三维 RHS、三方向周期短测、代表滤波、精确恢复及 32^3 至 t≈10 的无滤波长窗已取得结果；联合阶段和物理范围仍未全部完成 |
| KP2 | 分裂等价、内部六阶、周期收支、同状态三维 RHS、三方向双 rank 短测、精确恢复及 32^3 至 t≈10 的无滤波长窗已取得结果；几何／边界联合项仍未完成 |
| KP3 | mixed 修订、频谱／制造解、组件、均匀周期主程序与同状态黏性 RHS、NP=1/2/4/8、两个 halo 后端、代表滤波和精确恢复已取得证据；非严格守恒仅记录，不外推长窗稳定性 |
| KP4 | 周期静态 KEEP 曲线／拉伸几何首轮通过；ME6 几何扩展与物理端范围未完成 |
| KP5--KP8 | 未完成，不据周期 KEEP 结果宣称壁面、开放边界或 ME6 曲线路径已验收 |

## 已核对的主程序接口

- `src/fludyna.F90:fvar2q_sca`：`q=(rho,rho*u,rho*v,rho*w,rho*E)`，
  不含几何 Jacobian。
- `src/solver.F90:convrsdcal6`：映射通量含 `jacob*dxi`；`rhscal` 在
  对流和高阶面修正后统一 `qrhs=-qrhs`。新增组件返回正的通量散度，
  不自行重复取负或除以 Jacobian。
- `src/mainloop.F90` 的 RK 更新将 `qrhs` 除以 `jacob`。
  KEEP 主循环接入须保留这一约定和既有非对流源项顺序。
- 接线前 CPU `comsolver:filterq` 与设备滤波通过导数／全局紧致状态选择
  滤波；本轮已分离周期主滤波的选择与资源初始化，见下节。
- `compact_runtime` 替换共享 `fds`，`geom:gridgeom` 使用它生成度量；
  这是接线依赖，不是本轮发现并擅自修复的 CPU 缺陷。
- 原刊 Kuya--Kawai 2021 附录 A/B 已直接阅读：紧致 KEEP 将分裂形式的
  各项导数分别应用 Lele 六阶紧致算子。不能只对整个 Euler 通量求一次
  紧致导数。论文的 J 定义须与 ASTR 的 `jacob` 分别映射，不能按名称照搬。

## 本轮组件与数学检查

实现：`src/keep_operator.F90`，`src_gpu/keep_operator_gpu.cuf`。
测试：`tests/gpu_validation/keep_operator_probe.F90`。
构建：根 CMake，`keep_operator_probe` / `keep_operator_gpu_probe` 目标。
两份构建缓存均为 FP64、NVHPC 26.1。

二点通量使用算术平均密度、速度及映射度量构造同一质量通量；动量共用
该通量，比内能取两点算术平均，动能项取两点速度内积的一半，压力功取
交叉乘积。六阶线散度为两点通量差的 `2*(3/4,-3/20,1/60)` 组合。
组件接收已有效的半径 3 数据，不拥有 halo 交换、边界闭合或 RK 调度。

周期解析状态为 x 属于 [0,2*pi)，rho=1+0.1*sin(x)，
p=1+0.1*cos(2*x)，速度为
`(0.2+0.1*sin(2*x),0.1*cos(x),0.05*sin(3*x))`，gamma=1.4，
映射方向向量为 `(1,0,0)`。解析参考为普通 Euler 物理通量的精确导数，
不是另一份同式 KEEP 实现；测试比较全部五个分量。

| 检查 | 本轮观测 |
| --- | --- |
| 二点交换对称性 | 指定非同态两状态通量逐值一致 |
| 同态一致性 | 与独立 Euler 物理通量比较，误差在 2e-14 内 |
| 32→64 节点的五分量观测阶 | 5.956003 / 5.888284 / 5.861203 / 5.821462 / 5.905439 |
| 64→128 节点的五分量观测阶 | 5.988968 / 5.971854 / 5.980180 / 5.955049 / 5.960868 |
| 周期五分量散度总和 | 满足组件中的缩放舍入误差检查 |
| 动能—压力功交换残差 | 32/64/128 节点分别为 -9.29e-19 / -1.23e-17 / -5.76e-18 |
| 同状态 CPU/GPU 最大缩放差 | 三层均为 0；各分量尺度 S=1，门槛 2e-10 |
| Compute Sanitizer memcheck | 退出 0，ERROR SUMMARY: 0 errors |

动能收支比较的是周期和
`u dot D(momentum) - 0.5*|u|^2*D(mass) - u_x*D6(p)`，
以网格步长积分；不是要求可压缩动能本身恒定。该结论仅针对本轮恒定
度量、无物理边界的半离散组件，不覆盖熵、CURVE、滤波或完整 RK。

运行命令：

```sh
cmake -S /home/dell/workspace/astr_gpu -B build_cpu_probe
cmake --build build_cpu_probe --target keep_operator_probe -j 8
build_cpu_probe/bin/keep_operator_probe
cmake -S /home/dell/workspace/astr_gpu -B build_gpu_probe
cmake --build build_gpu_probe --target keep_operator_gpu_probe -j 8
build_gpu_probe/bin/keep_operator_gpu_probe
compute-sanitizer --tool memcheck --error-exitcode 99 build_gpu_probe/bin/keep_operator_gpu_probe
```

设备实现最初将非连续行切片传给设备子程序，NVHPC 拒绝该参数重排。
已改为线程内显式收集三个分量后调用；公式和主数组布局不变。
GPU 回读仅出现在组件对照测试中，未新增生产推进的主机往返。

## 周期主程序首轮接线

新增 `src/numerical_forms.F90`，在 `&numerics` 使用
`convection_form='keep'`；未指定时仍为 `standard`。配置在读入后通过
既有 MPI 广播传播，CPU `rhscal` 和设备 RK 对流阶段分别分派。
首轮接入 `643e`，后续已扩展到 `643c` 的无组分／无反应／无湍流模型
周期方程。物理边界仍未连接并明确报未实现，不按算例名决定是否使用 KEEP。
输入示例只说明当前可用路径，不缩小总计划后续范围：

```fortran
&numerics
  conschm='643e', difschm='643e', rkscheme='rk3',
  convection_form='keep', lfilter=f
/
```

CPU 体散度返回正的映射散度，由 `rhscal` 统一取负。GPU 三方向核直接
返回负的映射散度，使用已有 x=(512,1,1)、y=(32,16,1)、z=(64,1,8)
线程块，各方向后显式同步；读取已有设备原始量、Jacobian 和度量，不在
每步新增流场主机往返。halo 沿用原有交换顺序，本轮未改变通信后端。

首轮控制记录为显式 KEEP 写入 `astr-original-keep6-v1`，不依赖全局
紧致启用。本轮独立滤波接线后身份另包含 `/filter-none`、
`/filter-explicit10` 或 `/filter-compact10`，按实际启用方式记录。
没有新增附属文件。

真实运行脚本：`tests/gpu_validation/run_keep_periodic_gate.py`，复用既有
算例生成和恢复工具。`named_fixture` 新增可选 numerics 字典供测试显式
写入该字段；原调用结果保持不变，相关 pytest 三项通过。

本轮为 16^3 区间 TGV，FP64、dt=1e-3、4 个完整步、无滤波、无 crashfix、
原有黏性开启；体场序列和切片关闭，仅写检查点用于场比较。没有使用
这些短窗结果推断长期湍流统计、稳定性或性能。

| 检查 | 结果 |
| --- | --- |
| CPU/GPU NP=1、NP=2 x/y/z | 五个守恒场最大缩放差均不超过 6.662e-16；最大绝对差 8.527e-14 |
| 同后端 NP=2 x/y/z 与 NP=1 | 最大缩放差不超过 4.441e-16 |
| CPU/GPU 同拓扑 NP=2 x，连续 4 步与 2+2 恢复 | state.h5 全部数据集逐值一致，三个控制文件逐字节一致 |
| KEEP→standard 精确恢复 | 因算子身份不一致被拒绝 |
| KEEP→standard 显式非精确恢复 | 完成，场严格有限，rho 和内能均为正；不声称与 KEEP 轨迹一致 |
| 三维 GPU NP=1 memcheck | 4 个完整步完成，ERROR SUMMARY 为 0 |

逐分量误差和位置：`results/keep_kp1_periodic_20261110/report.json`。
三维内存检查：`results/keep_kp1_memcheck_20261110/`。
结果仅支持本轮实际覆盖的配置，不表示一般 CURVE、物理边界或独立滤波
选择已经通过。测试未设置新的资源或运行时间预算。

```sh
python tests/gpu_validation/run_keep_periodic_gate.py \
  --cpu build_cpu_probe/bin/astr --gpu build_gpu_probe/bin/astr \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --output results/keep_kp1_periodic_20261110
```

## 独立周期主滤波接线与验证

`&numerics filter_scheme='explicit10'/'compact10'` 选择主滤波类型，
仍由 `lfilter` 控制是否启用。未指定该字段时解析为既有输入默认值：
对流／黏性均为显式时取显式，否则取紧致；显式指定后不再与导数类型
联动。滤波内部系数和 RK 内的调用相位没有改变。

`compact_runtime` 将紧致通信资源是否已初始化与紧致导数是否启用分开。
只有紧致滤波的周期路径也会初始化分区和线通信器，但不替换 `fds`。
GPU 紧致滤波不申请普通显式滤波工作数组；紧致导数＋显式滤波则按
已选 scalar/full 工作模式分配显式滤波所需数组。物理边界子程序中的
专用横向滤波仍保留原描述符初始化，尚未据本轮结果宣称其类型可独立
切换，须在 KP5/KP6 连同闭合验证。

新增 `numerical_forms_probe` 检查未指定时的两个默认值、显式／紧致覆盖
选择及禁用结果，输出 `FILTER_SELECTION_PASS`。代表真实短测均为
16^3、4 个完整步、NP=1/2 x；CPU/GPU 字段尺度 S=1，门槛 2e-10。

| 对流／黏性 | 主滤波 | 最大缩放场差（含同后端分解对照） | 最大绝对场差 |
| --- | --- | --- | --- |
| 显式 KEEP／原显式黏性 | explicit10 | 1.116e-15 | 1.990e-13 |
| 显式 KEEP／原显式黏性 | compact10 | 2.334e-14 | 4.178e-12 |
| 原紧致对流／原紧致黏性 | explicit10 | 1.113e-15 | 1.990e-13 |

三项的 CPU、GPU 同拓扑连续 4 步与 2+2 精确恢复，全部状态数据集逐值
一致，三个控制文件逐字节一致。KEEP 两项的对流形式切换检查也通过。
另用 `results/keep_kp0_filter_identity_20261110/` 检查保持 KEEP 而切换
滤波类型：精确恢复因有效滤波身份不同而被拒绝。

原始报告分别位于以下目录中的 `report.json`，逐分量误差和位置保留：

- `results/keep_kp0_explicit_filter_20261110/`
- `results/keep_kp0_compact_filter_20261110/`
- `results/keep_kp0_compact_derivative_explicit_filter_20261110/`

这些是数值接线与恢复短测，不是滤波对长期湍流统计影响或性能结论。
紧致滤波独立路径还通过 NP=1、4 步 Compute Sanitizer memcheck，
ERROR SUMMARY 为 0；证据位于
`results/keep_kp0_compact_filter_memcheck_20261110/`。

## KP0-A 先行接口分析

先行分析不选定新闭合系数，也不将既有局部闭合当成 KEEP 稳定性证明。

1. 现有共享 `fds` 由 `difschm` 初始化，普通中心对流复用它，因此当前
   `solvrinit` 要求中心 `conschm` 与 `difschm` 一致。新显式 KEEP 已自行
   构造对流散度，不依赖该共享对象。紧致 KEEP 后续需要独立请求分裂
   导数计划，不能借修改 `difschm` 同时更换几何、黏性和滤波。
2. 设映射面积分量为 `A(axis,c)=jacob*dxi(axis,c)`，实际度量应满足与
   对流算子相配套的 `sum_axis D_axis A(axis,c)=0`。周期显式采用 D6；
   周期紧致候选采用全局六阶 Pade 导数，生成度量也需核对该算子和
   affine-periodic 坐标增量。仅紧致滤波不应改变这一几何选择。
3. 周期 D6 在唯一节点上是反对称矩阵；周期紧致 `D=A^-1 B` 中，循环
   对称 A 与循环反对称 B 可交换，故 D 也反对称。这为周期分裂收支
   提供基础；物理端截断后不能沿用循环矩阵的可交换性。
4. 现有 `compact_operator:compact_derivative_core/rhs` 的物理端行是
   `d0+2*d1=(-2.5*f0+2*f1+0.5*f2)/h`，首个相邻行是
   `0.25*d0+d1+0.25*d2=0.75*(f2-f0)/h`。代入精确导数，两个方程
   分别具有三阶、四阶局部一致性残差；这不是最终联立解全域收敛阶的
   结论。它们也不同于捐赠显式实现的二／四阶拼接，不能混称六阶物理
   边界或直接作为新 KEEP 的边界收支依据。

因此，周期线计划可以先复用；物理端必须连同积分权重、守恒／能量边界
项和 RK 响应提出候选后审阅。以上分析不构成原 CPU 数值缺陷的判定，
本轮没有据此修改任何物理端系数或已有边界默认值。

ME6 接线前已核查混合项数据范围：例如 x 方向通量在 i±4 上计算 j±3
或 k±3 的速度导数，需要双方向越界的速度值，但原始直角网格公式不
同时读取三个方向越界点。当前 `refresh_reconciled_qswap_halos` 的物性
重建覆盖面区，不能仅凭它推断原始量边区已有效。KP3 应在私有通路中
提供并验证所需边 halo，保持只交换 halo、不再次平均共享节点；这是一项
新模板依赖，不据此宣布原 ASTR 面交换存在错误。本轮未改通信代码。

## KP2 周期紧致 KEEP

原刊附录 A/B 的逐项分裂导数现已通过 `src/keep_operator.F90` 组织。
在扩展状态 `(rho,u,v,w,e,p,A_x,A_y,A_z)` 上，将二点算术平均的乘积
展开成“本点乘积 × 另一乘积的导数”。294 个代数项去除重复导数后，
需要 85 个导数基项。空导数项 D(1) 为零；该关系首先针对已检查的
周期线性导数使用，不构成物理端稳定性结论。

`tests/gpu_validation/keep_split_probe.F90` 以实际全局紧致 CPU 线求解器
构造周期导数矩阵，再将展开结果与 `2*sum_j D(i,j)*F_pair(i,j)` 直接比较。
非恒定映射向量用于代数检查，不当作一个已满足 GCL 的物理曲线网格。
紧致矩阵残差、反对称性、常数导数、五分量周期守恒及内部精度均有检查。

| 组件检查 | 结果 |
| --- | --- |
| D6 展开与原二点通量比较 | 最大缩放差 2.998e-15 |
| 全局紧致展开与原二点通量比较 | 最大缩放差 3.775e-15 |
| 32→64 节点五分量观测阶 | 6.026800 / 6.069657 / 5.990455 / 6.112321 / 6.058653 |
| 64→128 节点五分量观测阶 | 6.006654 / 6.017056 / 6.011947 / 6.027300 / 5.998748 |
| 周期动能—压力功残差 | 绝对值不超过 1.017e-15 |
| 内能对流与 p*D(u) 关系 | 满足 2e-10 缩放门槛 |

CPU 在 `compact_runtime:global_keep_convection_cpu` 中按一条局部线打包
85 个基项，复用既有分区及全局 CPU 线接口。GPU 在
`compact_operator_gpu:compact_keep_convection_gpu` 中使用原有导数批次
容量和两层 PCR，设备打包、求解、累加依次显式同步。没有把整个 Euler
通量只求一次紧致导数，没有独立 rank 端点闭合，也不新增 85 份完整
三维工作场。小型代数表首次使用时上传一次，推进中不回读完整场。

当前周期接入配置为 `convection_form='keep'`、`conschm='643c'`、
`difschm='643c'`，尚未解除原共享梯度架构的中心对流／黏性匹配要求。
紧致 KEEP 的身份为 `astr-original-keep6c-v1` 加有效滤波标识，与普通紧致
对流及显式 KEEP 区分。格式默认仍为 `standard`，未修改物理端闭合。

真实验证仍为 16^3、4 个完整步、dt=1e-3、修复关闭，结果如下：

| 真实运行检查 | 结果 |
| --- | --- |
| 无滤波，CPU/GPU NP=1、NP=2 x/y/z 及同后端分解比较 | 最大缩放差 7.952e-16，最大绝对差 1.422e-13 |
| 显式滤波，CPU/GPU NP=1、NP=2 x 及同后端分解比较 | 最大缩放差 1.111e-15，最大绝对差 1.706e-13 |
| 紧致滤波，CPU/GPU NP=1、NP=2 x 及同后端分解比较 | 最大缩放差 2.524e-14，最大绝对差 4.520e-12 |
| 三项 CPU/GPU NP=2 x，连续 4 步与 2+2 步精确恢复 | 全部状态数据集逐值一致，三个控制文件逐字节一致 |
| 紧致 KEEP→普通紧致对流 | 精确恢复正确拒绝，显式非精确恢复完成且状态有限、密度／内能为正 |
| 无滤波 GPU NP=1 memcheck | ERROR SUMMARY: 0 errors |

证据目录：`results/keep_kp2_compact_periodic_20261110/`、
`results/keep_kp2_compact_explicit_filter_20261110/`、
`results/keep_kp2_compact_compact_filter_20261110/`、
`results/keep_kp2_compact_memcheck_20261110/`。首项覆盖 x/y/z，但精确
恢复只选 x 方向；不将短步场一致性当成长时稳定性验收。

组件命令须用本机 HPCX 启动器；直接执行在 MPI_Init 前引用打包路径而
失败，未产生数学检查结果，也未为此增加程序重试或修改 MPI 配置：

```sh
cmake --build build_cpu_probe --target keep_split_probe -j 8
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec -n 1 build_cpu_probe/bin/keep_split_probe
```

## 三维冻结状态 RHS

`run_keep_periodic_gate.py --rhs --axes` 在 NP=1 记录内部 step=0、rk=1
的初始 RHS，不使用已分离的后续轨迹作严格同状态参考。权威 q 在物理
节点及半径 3 的面 halo 上要求逐值一致；对流模板不读取双／三方向越界
角区。另以既有 `compare_fields` 比较 geometry.h5 全部 17 个数据集，
包括坐标、Jacobian、度量和附属 halo 数据，要求逐值一致。
CPU/GPU 由同一 q 得到的原始量仍有约 4.438e-16 的缩放舍入差，单独报告。

| 路径 | 权威 q／几何 | 对流 RHS 最大绝对差 | 最大缩放差，S=1 |
| --- | --- | --- | --- |
| 显式 KEEP | q 模板区域逐值一致，几何全数据集逐值一致 | 1.972e-14 | 1.972e-14 |
| 紧致 KEEP | q 模板区域逐值一致，几何全数据集逐值一致 | 1.821e-14 | 1.821e-14 |

报告：`results/keep_kp1_frozen_rhs_initial_20261110/report.json`、
`results/keep_kp2_frozen_rhs_initial_20261110/report.json`。
几何比较在不重复 CFD 的后处理中补入，单独记录于
`results/keep_frozen_geometry_20261110.json`，没有改写原始结果。

首次探针误取内部 step=1，此时已完成一轮推进，q 最大差为 2.842e-14；
比较因此停止。`mainloop` 在 RK 后才增加 nstep，故初始相位是 step=0。
该次失败保留于 `results/keep_kp1_frozen_rhs_20261110/`；修订的是测试
相位，不是程序、误差门槛或输入状态，也未据此判定 CPU 数值缺陷。

## 代表性较长时间窗

本轮沿用原 TGV 模型：Re=1600、Ma=0.1、32^3 区间、GPU NP=1、
dt=1e-3、10000 完整步至 t≈10，关闭主滤波与 crashfix。显式路径用
643e/643e，紧致路径用 643c/643c，因此不能将二者差异仅归因于对流。
这不是生产 DNS、性能比较或严格熵守恒检验。

运行脚本 `tests/gpu_validation/run_keep_long_window.py` 复用现有原生
flowstate.dat 和 tgv_conservation.dat，采样每 100 步。CFL 每步检查
上界≤0.5；仅保存初、末两个完整检查点，不写中间三维场序列或切片。
末态必须有限且密度／压力为正；质量、总能漂移和熵变化作为观测值报告，
没有把 CPU/GPU 2e-10 门槛改用作长期轨迹或守恒漂移容差。

后处理在均匀周期盒唯一节点上使用等权积分。原生参考 rho_inf=u_inf=1。
平均动能为 mean(|rho*u|^2/(2*rho))，熵密度指标为
mean(rho*log(p/rho^gamma)/(gamma-1))，对应 rho*s/R 的空间积分除以盒体积，
熵参考值相差一个固定常数；
压力由 p=(gamma-1)*(rhoE-|rho*u|^2/(2*rho)) 重建，gamma=1.4。
初末值从检查点提取，过程曲线保留原程序定义，不把空间平均当时间平均。

显式路径已完成：最大 CFL=0.1733951469；质量相对漂移 5.618e-13，
总能相对漂移 4.036e-13；末态最小 rho=0.9961583499，p=71.14926312。
平均动能从 0.125 降至 0.09156854612；熵密度指标增加 4.698e-4。
结果位于 `results/keep_kp1_long_window_20261110/report.json`。
紧致路径也已完成：最大 CFL=0.1727739935；质量相对漂移 5.618e-13，
总能相对漂移 4.037e-13；末态最小 rho=0.9946290157，p=71.04528122。
平均动能从 0.125 降至 0.08229345656；熵密度指标增加 5.999e-4。
结果位于 `results/keep_kp2_long_window_20261110/report.json`。
两项各含 10000 个 CFL 样本、99 条流动统计和 100 条守恒均值记录。
本轮只对既有未改变的黏性模型作组合稳健性观察，不据末态动能差直接
评价两种 KEEP 对流离散的精度优劣。

## KP3：ME6 mixed penalty 审计与待决定项

参考 [Chandravamsi--Frankel, arXiv:2312.12069v2](https://arxiv.org/abs/2312.12069v2)
的表 2、表 5、式(41)、式(43)、式(48)。本轮只移入审计探针的原始有理数
系数并作 real_kind 类型转换，未建立或修改生产 ME6 算子。

供给包 `me6opti_minimal/source/src/solver.F90:me6opti_tgv3d_div_x/y/z`
在负半点对 straight 导数系数作负反射，对 penalty 只作索引反射而不
变号。表 5 的 penalty 是偶对称系数，因此同一半距的两个 penalty
相同。常物性时，外层半点通量差会将它们抵消。对 k_x=k_y=k、
网格间距为 1 的 Fourier 模式，截止 k=pi 时中央横向一阶导数也为零。
这与式(43)的非零截止波数阻尼不一致；不能按原包照搬后称为该频谱。

独立原生探针：`tests/gpu_validation/me6opti_symbol_audit.F90`，使用
1025 个 k∈[0,pi]，直接组合供给包系数和正负半点规则，与原文闭式比较。
候选仅在诊断中将负半点 penalty 变号；没有改变插值、straight 系数或
外层散度权重，也没有执行完整 CFD 或生产默认变更。

| 对照 | 最大符号绝对差 |
| --- | --- |
| 供给 straight 与原文式(48) | 2.665e-15 |
| 供给 mixed 与原文式(43) | 7.7845333333 |
| 候选 mixed 与原文式(43) | 2.665e-15 |

k=pi 时，供给 mixed=0，候选=-7.784533333333333，
式(43)=-7.784533333333334。探针最后以非零退出，明确保留原实现不符合
该参考的失败，不将候选的吻合改写成原实现通过。

原文式(41)的正负半点记法本身未清楚表达上述变号；因此本轮判断是
“供给实现与式(43)频谱不一致”，不擅自认定可以无授权改写论文或代码。

用户已于 2026-11-10 确认：以式(43)的频谱为 mixed 的参考，批准负半点
penalty 取反、系数不变的数值修订，CPU/GPU 同步；先做制造解、变物性
与完整黏性符号（含非对角波向）的验证，通过后再接入，保持原生产默认不变。
对角波向的符号吻合不是完整多维黏性算子的稳定性证明。

```sh
cmake --build build_cpu_probe --target me6opti_symbol_audit -j 8
build_cpu_probe/bin/me6opti_symbol_audit
# Expected current result: nonzero exit, donor mixed reference mismatch.
```

## 获批 ME6 候选组件与新发现

候选 CPU 数学模块为 `src/me6opti_operator.F90`，GPU 模块为
`src_gpu/me6opti_operator_gpu.cuf`。GPU 系数表由 CPU 同一入口首次初始化，
两者都已应用获批的负半点 penalty 变号。没有增加运行输入选项或接入
主程序黏性 RHS；现有默认黏性路径不变。

`me6opti_candidate_probe` 对常物性黏性子系统检查三维速度矩阵和导热
标量。波数覆盖 [-pi,pi]^3 的 25^3 个组合，含反向与斜向；四组网格
间距分别为 (1,1,1)、(0.5,1,2)、(1,0.25,0.75)、(1,0.01,2)。
在每个采样点检查动量符号 Hermitian 部分的负半定性（主子式），而非
仅凭特征值判稳。62500 项中，归一化最小主子式 -4.441e-12，导热
符号最大实部 3.331e-12，最大虚部 1.863e-14，均在 2e-10 检查门槛内。
这是常物性线性采样结果，不是任意状态、变物性或完整 NS/RK 稳定性证明。

局部制造解包括 9 个 straight/mixed 标量组合、3 个动量和能量通量，
有非恒定正黏度、导热系数和多方向速度。32/64/128 网格间距上的
最细层观测阶为 5.921--5.995。首次使用 64/128/256 时，zz 标量误差
降至 1.627e-12、阶数为 5.306；在不改连续导数的温度常数平移对照中，
该误差变为 1.113e-12，说明消减舍入已影响此窗口。没有改生产公式或
阶数门槛，而是固定使用仍由截断误差主导的 32/64/128，并保留 256
对照输出。GPU 候选探针的后续结果见本节末；主程序接线仍未完成。

### 变物性周期净散度不为零

令周期盒长度 2*pi，phi=1+0.1*cos(x)，系数 mu=1+0.2*cos(x)>0。
检查 scalar straight 项 d/dx(mu*dphi/dx) 的唯一节点平均，连续周期净量
应为零。它也可解释为正温度、变导热系数下的周期导热测试。

| 节点数 | 候选离散平均 RHS | 独立 Fourier 零模预测 | 常系数对照平均 RHS |
| --- | --- | --- | --- |
| 32 | -1.1234370404e-8 | -1.1234380579e-8 | 1.027e-14 |
| 64 | -1.7745052541e-10 | -1.7749448802e-10 | 4.276e-14 |
| 128 | -2.6100143097e-12 | -2.7778690439e-12 | 1.669e-13 |

预测由半点插值与导数符号的零模乘积得到，不使用整个流场推进结果。
该偏差随网格呈六阶缩小，但不是机器精度的离散净通量抵消。
问题已在 straight 项出现，不使用 mixed penalty；因此不是本次获批
变号造成，供给包的同一 straight/物性插值组合也具有这一性质。
半点近似在每个中心各自重构，不能仅凭正负半点写法推断相邻中心使用
同一界面通量。不能用空间收敛代替严格离散守恒。

发现时探针在最后的周期净散度检查以非零退出。用户于 2026-11-10 明确
决定“仅记录现象，跳过”：不设计或实现严格守恒扩展，不对净量做均值
扣除、限幅或调参；此项保留为诊断，不再作为接线拒绝条件。其余系数、
频谱、制造解、CPU/GPU 同状态、保正／有限性及恢复门槛不变。

此前提出的两个方向（保留讨论记录，不再待决定）：

- A：保留论文型 ME6 作为默认关闭的研究路径，明确变物性下六阶近似、
  不是严格离散守恒；将净量作为误差／长期漂移诊断而非零残差门槛。
- B：先设计严格守恒扩展，再接入；保留六阶及原常物性优化频谱为设计
  目标，重新推导非线性共用界面通量和能量相容性。它应与原论文型区分，
  不能预先承诺只改一行或自动保持全部原半点公式。

用户未批准 B 的新设计范围，本轮不执行该扩展。保留已批准 mixed 变号，
不将该修订后的论文型 ME6 描述为变物性下严格离散守恒的算子。

### CPU/GPU 组件验证继续完成

按用户决定，`record_periodic_budget` 仍输出全部净量和 Fourier 预测，
但不再因该已知性质中止；没有改系数或进行全域均值补偿。当前 CPU
探针正常退出，结尾 PASS 只列系数、常物性完整黏性符号和变物性制造解，
不包含严格守恒。

`me6opti_gpu_probe` 在同一主机初始化的速度、温度、黏度、导热系数及
Jacobian 上，对照 CPU 四个动量／能量贡献，密度贡献必须保持为零。
物理节点为 8×6×4，ghost 为 hm=5，三方向分别使用既有
(512,1,1)、(32,16,1)、(64,1,8) 线程块，方向后显式同步。
这是一块提供完整解析 halo 的局部组件夹具，不是 MPI 交换验收。

| 同状态输入 | 最大缩放差，S=1 | 最大绝对差 |
| --- | --- | --- |
| 常物性光滑场 | 7.994e-15 | 7.994e-15 |
| 变物性光滑场 | 7.661e-15 | 7.661e-15 |
| 斜向／反向波数 | 8.181e-15 | 8.882e-15 |
| 网格交替模态 | 0 | 0 |

全部满足已批准 CPU/GPU 门槛 2e-10。Compute Sanitizer memcheck
正常退出，ERROR SUMMARY: 0 errors。GPU 系数仅初始化上传一次，测试
中的字段上传和结果回读用于组件对照，不是生产每步主机往返方案。

```sh
cmake --build build_cpu_probe --target me6opti_candidate_probe -j 8
build_cpu_probe/bin/me6opti_candidate_probe
cmake --build build_gpu_probe --target me6opti_gpu_probe -j 8
build_gpu_probe/bin/me6opti_gpu_probe
compute-sanitizer --tool memcheck --error-exitcode 99 build_gpu_probe/bin/me6opti_gpu_probe
```

## KP3：均匀周期主程序、边 halo 与恢复

新增 `&numerics viscous_form='me6_opti'`，默认仍为 `standard`。
`src/me6opti_runtime.F90` 和 `src_gpu/me6opti_runtime_gpu.cuf` 分别承担
主机与设备接线；本轮仅实现 FP64、非反应五变量、三方向均匀笛卡尔周期
网格。未对算例名建立准入表。实际物理坐标在首次调用时与全域间距核对，
不将拉伸／曲线网格当作等距网格运行。

ME6 的半点导数、插值及外层散度由其自身系数决定，不由 `difschm`
末位改写。现有 `difschm` 仍承担原梯度／几何初始化；原中心格式共享
`fds` 的一致性约束尚在，不能据此声称 KP0 几何职责已经完全分离。
本轮代表显式／紧致配置分别使用 643e/643e、643c/643c。

每个 RHS 阶段先在已有有效面 halo 上更新速度和温度的私有扩展 halo，
再由同阶段温度生成 Sutherland 黏度与导热系数。无量纲系数沿用原程序
`mu=miucal(T)/Re`、`kappa=(mu/Pr)/const5`；有量纲分支沿用
`mu=miucal(T)`、`kappa=cp*mu/Pr`，本轮真实场证据来自无量纲路径，
不据代码分支存在宣称有量纲物理验收。物理黏性散度乘一次 Jacobian 后
加到动量／能量 RHS，不重复运行原黏性通量与散度链。

设备物性工作数组驻留并复用，方向核后显式同步。不新增整场主机往返；
host-staged 后端只搬运私有 halo 包，device-aware 后端直接提交设备包。
现有原梯度／通量工作区的容量优化不在本轮内，未以此宣称性能收益。

### 私有扩展 halo

`src/me6opti_halo.F90` 与 `src_gpu/me6opti_halo_gpu.cuf` 先沿 y 交换包含
x ghost 的面，再沿 z 交换包含 x/y ghost 的面。固定交换 hm 层、速度
三个分量及温度。只更新 halo，不平均共享节点、不修改 q 或内部节点。
复用已有两种 GPU 传输接口，私有固定 tags 为 21801--21804。

独立探针采用全域 17×19×13 区间、hm=5，分区大小不均匀。初始只给内部
和六面有效的精确整数编号，边角放置不同值。完成两次不同编号批次后，
逐值核对全部内部、面、边、角；另核对物理节点未改变。CPU、GPU pinned
和 GPU device-aware 均通过 NP=1、NP=2 三轴、NP=4 三个双轴以及
NP=8 的 2×2×2。NP=4/8 是本地双卡共享的正确性检查，不是扩展性测试。
NP=8 pinned 的 Compute Sanitizer memcheck 八个进程均为 0 errors。

入口为 `tests/gpu_validation/run_me6opti_halo_gate.py`；证据目录：
`results/keep_kp3_me6_halo_{cpu,pinned,aware,memcheck}`。

### 完整步场与身份验证

真实场采用 16³ 周期 TGV、dt=1e-3、4 个完整步、无 crashfix。
缩放差仍为 `abs(a-b)/max(1,abs(a),abs(b))`，门槛 2e-10。

| 组合／验证范围 | 场比较最大缩放差 |
| --- | --- |
| 显式 KEEP＋ME6，无滤波，NP=1/2(x/y/z)/4(xy)/8(xyz) | 5.552e-16 |
| 紧致 KEEP＋ME6，无滤波，NP=1/2(x/y/z) | 1.111e-15 |
| 原中心对流＋ME6，无滤波，NP=1/2(x) | 4.778e-16 |
| 显式 KEEP＋ME6＋紧致十阶滤波，NP=1/2(x) | 2.410e-14 |
| 紧致 KEEP＋ME6＋显式十阶滤波，NP=1/2(x) | 1.333e-15 |
| 显式 KEEP＋ME6，device-aware，NP=1/2(y/z)/4(xy)/8(xyz) | 5.552e-16 |

各行包含同拓扑 CPU/GPU 比较和各后端对 NP=1 的分解比较。前五行的
CPU/GPU NP=2(x) 均完成连续 4 步与 2+2 精确恢复，全部 state 数据集逐值
一致，三个控制文件逐字节一致。ME6 身份包含 `/me6opti-podd-v1`，区分
已批准的负半点 penalty 变号修订；复用 control.bin，不新增附属文件。
切换为原黏性形式、KEEP→普通对流或切换有效滤波的精确恢复按预期拒绝；
显式选择非精确数值形式迁移后，状态有限且密度／内能为正。

冻结状态采用 step=0、RK1 的同一 q 和实际几何。显式／紧致两组的完整
RHS 最大缩放差分别为 1.983e-14、1.843e-14。另从原生保存的完整 RHS
扣除同阶段对流 RHS，分离 ME6 黏性贡献，两组最大绝对／缩放差均为
2.910e-16；映射黏性 RHS 最大幅值约 3.6681e-4，密度贡献严格为零。
该扣除是同状态诊断，不是数值推进中的修正或均值补偿。

入口 `run_keep_periodic_gate.py --viscous-form me6_opti`，冻结黏性分项为
`compare_frozen_rhs(...,stage='diffusion')`。实际捕获流场及报告位于：

- `results/keep_kp3_me6_periodic_e`、`results/keep_kp3_me6_periodic_c`；
- `results/keep_kp3_me6_standard`；
- `results/keep_kp3_me6_e_compactfilter`、`results/keep_kp3_me6_c_explicitfilter`；
- `results/keep_kp3_me6_native_aware`、`results/keep_kp3_me6_native_memcheck`。

完整 GPU 真实场 memcheck 正常结束，ERROR SUMMARY: 0 errors。
这些结果验证周期接线、状态相位、通信与短恢复，不构成生产长窗稳定性
或变物性严格离散守恒证明。

## KP0-A：实际曲线几何复核与 G1 首轮失效记录

已新增 `run_keep_metric_audit.py` 和原生 `keep_geometry_probe.F90`。
在既有 32³ 周期曲线夹具上只读复核真实初始化生成的几何；不修改求解器。
显式 KEEP 的原生恒定物理场 RHS 最大值为 6.673e-14，而紧致 KEEP 使用
现有几何时为 6.863e-8，原生结果与独立 Fourier 诊断相符。
当时不能宣布紧致 KEEP 的曲线自由流保持通过，也未将该现象外推为所有
旧 CPU 算例的缺陷。该状态下没有继续曲线 GPU／长窗验证。

具体调用顺序、实际度量数据、G1 和 ME6 各向异性 penalty 尺度问题已写入
[静态几何候选](ASTR_GPU_KEEP_GEOMETRY_CANDIDATES.md)。用户随后批准 G1，
随后接入独立几何入口、黏性梯度分派及 `/g1` 身份，CPU/GPU 构建通过。
但 G1 首轮只降低自由流残差，**逆度量与物理梯度精度检查失败**。
32³／64³ 上的归一化逆度量误差约 0.615／0.619，不能归为舍入误差。
当时停止下游场比较，提出候选 G1a，待用户单独批准后才实施以下修订。

## G1/G1a：周期静态 KEEP 几何首轮通过

用户已批准 G1a。新增 `src/keep_geometry.F90`，只在周期紧致 KEEP 中
按仿射背景＋周期位移构造保守 curl；原 Jacobian 行列式保持不变。
CPU/GPU 构建通过。曲线 32³/64³/128³ 的 Jacobian、逆度量及三个物理
梯度分量观测阶为 5.973--6.033；可分离拉伸的观测阶为 6.001--6.035。
显式路径也补齐三层检查，其最细两层观测阶为 5.973--5.998。
实际度量／恒定场 RHS 均通过已批准门槛；未把零散度代替逆度量精度。

32³ 曲线覆盖 NP=1、NP=2 x/y/z、NP=4 xy、NP=8 xyz，64³ 覆盖 NP=1。
33×31×29 拉伸网格覆盖 NP=1/2 x、NP=4 xy、NP=8 xyz，包含不均匀分区。
另完成 643c/643e＋显式十阶滤波、643e/643c＋紧致十阶滤波的 NP=1/2 x。
全部场缩放差不超过 6.346e-14；关滤波同状态完整 RHS 差不超过 5.318e-15。
代表同后端 2+2 续算的状态数据集逐值一致、控制文件逐字节一致。
紧致几何改变黏性／滤波后逐值不变，显式几何与修订前参考也逐值一致。

新混合组合首次滤波检查暴露了缓存刷新漏接：内部旧原始量与 MPI 面上
的新状态混用。已只扩展 CPU/GPU 的 KEEP 刷新条件，不改滤波公式；
修订后原始量与当前 q 的重建差在 6.652e-16 内。旧失败记录保留，未放宽
容差。严格同输入的 RHS 检查与各自滤波后的轨迹检查分开执行。

身份 `/g1a` 区分新几何，混合 KEEP 开启滤波时另记录策略版本 2。
旧 `/g1`、旧混合滤波策略精确恢复均正确拒绝；显式非精确几何迁移完成。
详细数值、源文件、失败原因和证据路径见
[几何候选文件第 7 节](ASTR_GPU_KEEP_GEOMETRY_CANDIDATES.md#7-g1a-实现与首轮验证结果)，
汇总为 `results/keep_g1a_summary.json`。这些结果不是生产长窗或物理端稳定性验收。

## 下一步

mixed 修订已批准，非严格守恒现象已按用户决定记录并跳过修复。
周期静态 KEEP 几何首轮已取得上述证据；下一步补齐 KP0 收支记录，
按已批准的 ME6 G2 几何／penalty 方案推进组件，物理端闭合另行定稿、审阅。
G2 的完整度量投影与 mixed penalty 范数比候选已写入几何文件第 8 节：
保留原系数及均匀各向异性极限，先检查完整常度量符号和制造解，
通过后才接周期主程序。范数比为本项目提出的扩展，并非直接抄用论文公式；
用户于 2026-11-10 已批准，按总计划 G2-1 至 G2-4 执行；目前未实现该
新公式或修改 ME6 CPU/GPU 准入。组件全部通过后才能接周期主程序。
ME6 曲线、bc41、11/22/41/52 及联合数值交付仍未完成。
周期静态 KEEP 验证不能替代这些范围。没有修改普通路径生产默认、
新增物理边界闭合或执行 Git／平台操作。目标未完成，不能以本轮结果
宣布整个 KP0--KP8 计划完成，也不将代表周期验证外推到任意初值的稳定性。

## KP0：冻结状态收支后处理补充

新增只读脚本 `tests/gpu_validation/analyze_keep_frozen_budgets.py`，复用
既有 q、原始量缓存、对流／完整 RHS 和实际几何，不推进 CFD，不改原始场。
使用唯一周期节点和 J 权重；参数显式给定 gamma=1.4、Ma=0.1、Re=1600、
Pr=0.72、参考温度 273.15 K、Sutherland 常数 110.3 K，与所选档案一致。

完成 6 份报告、12 条 CPU/GPU 记录：均匀显式／紧致 KEEP＋原黏性，
G1a 曲线紧致 KEEP＋原黏性的 RK1/RK2，以及两种 KEEP＋ME6 的均匀 RK1。
KEEP 守恒量体积平均变化率残差最大 1.116e-15，动能／压力功残差最大
8.058e-18，局部内能乘积关系最大缩放差 6.408e-13。
原节点黏性的独立应力功恒等式残差最大 3.795e-19，离散熵分部求和参考
残差最大 4.693e-19。原始量缓存相对当前 q 的最大缩放差为 4.441e-16。
历史错误缓存档案在该前提检查处以 5.913e-8 的差值拒绝，未生成通过报告。

ME6 节点参考仅用于对照，不把不同算子的节点应力功恒等式强加给半点
ME6。所选 16³ TGV 初始状态的 ME6 黏性总能体积平均变化率约为
-4.404e-8／无量纲时间，属于此前决定仅记录的非严格净量；没有修正或
重新列为待审批缺陷。熵链式变化率与另行差分的节点熵通量残差分别记录，
不要求原算术平均 KEEP 严格零熵残差，也不据正熵产率宣布熵稳定。
RK2 使用各后端自身状态检查恒等式，不冒充逐值相同输入的后端对照。

证据为 `results/keep_kp0_budget_{explicit_uniform,compact_uniform,compact_curve,
compact_curve_rk2,me6_explicit_uniform,me6_compact_uniform}.json`。
这补充了周期半离散收支，不关闭物理边界通量或完整 RK／滤波分项收支。
