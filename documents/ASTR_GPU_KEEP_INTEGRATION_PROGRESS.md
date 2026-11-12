# KEEP and ME6-Opti Integration Progress

更新日期：2026-11-12。实施基点：`e992dc7`，`feature/gpu_dev`。
周期算子、G1/G1a、接线与验证工具已保存于源码提交 `64f48fc`。
总计划：`documents/ASTR_GPU_KEEP_INTEGRATION_PLAN.md`。

## 当前状态

当前活动goal已更换为方案A完整低阶边界兼容，见主计划开头LA0--LA6。
下文KP记录保留历史证据；新增高阶研究不再是本轮执行主线。LA1单物理
方向匹配双面的KEEP／标准黏性及ME6近端带已通过下述代表矩阵，LDC的
现有双物理面范围也已接入。其他开放端／多物理面组合尚未完成，不能
宣布全部边界适配完成。

### LA1：现有简单物理面与KEEP接线

`simple_physical_boundary_axis`按真实边界实现分类：41/50/60为x/y/z，
42为x/y，31/411/421为y。接通KEEP输入、全局紧致资源初始化及检查点
能力／外部网格资源登记，不改变各边界的物理处理。恢复测试发现网格
虽已冻结但路径仍落到外部输入文件，已补齐`output_resource_path`选择。

`run_keep_simple_boundary_gate.py`覆盖13种方向组合，每种16区间、4个完整
步、dt=1e-5、NP=1及沿物理方向NP=2。分散选择显式／紧致对流、显式／
紧致标准节点黏性，滤波关闭／explicit10／compact10与scalar/full工作区，
不做全排列。三阶段有效面q、物性缓存及对流／完整RHS均比较；物性缓存
另由其当前q核对。各双rank后端完成2+2精确恢复，不再提供外部网格和
初场文件；全部state数据集和control/archives/insitu_control逐值一致。

终场CPU/GPU及分解比较最大缩放差4.049e-14，三阶段RHS／缓存比较最大
缩放差7.570e-14，物理节点rho/p/T严格有限正值。411夹具将xslip放在周期
x范围之外，仅验证其下壁滑移段，混合段及吹吸仍归LA3。31沿用原固定
状态，短轨迹不作RTI物理验证。首个资源接线失败记录保留；本矩阵通过
不表示低阶边界长窗稳定或全部CURVE几何已准入。

证据：`results/keep_low_all_20261111/simple_matrix_resources/report.json`。
### LA2：ME6固定近端带与独立对流选择

CPU按全局节点到真实物理端的距离判定近端带，GPU按同一几何范围独立
构造内部盒；距任一物理端0--3个节点累加原节点黏性，其余用ME6。
私有边halo不读取x物理外部区域，物性计算也排除三方向的物理外部；
GPU复用各方向原应力／热通量核和物理端导数。全周期分支不使用该带。

KEEP＋ME6＋643e近端梯度已通过LA1同样13种实际方向的4步矩阵、NP=1/2
及2+2精确恢复。终场／分解最大缩放差4.097e-14，三阶段RHS／缓存最大
6.670e-14。原显式对流＋ME6另通过41x、42y、60z的NP=1/2/4（xy/yz/xz）
及精确恢复，分别最大3.958e-14／4.290e-14。原y壁CPU制造解16区间NP=1
和三个方向NP=2的误差与拼接净量保持原值。GPU首跑的逻辑数组FINDLOC
运行时不支持已改为直接扫描方向，失败日志保留。

证据分别为同根`me6_simple_matrix/report.json`、
`standard_me6_axes/report.json`和`me6_wall_component_cpu/report.json`。
近端643c梯度／散度复用全局线算子，GPU仅在物理近端带提交该散度。
KEEP＋643c近端带的42x/60z、NP=1/2及2+2恢复通过，最大终场／分解差
3.784e-14、三阶段RHS／缓存差4.409e-14；相同冻结状态下与标准节点
黏性对照，近端带CPU逐值一致，GPU差不超过2.157e-18。证据为
`me6_compact_nodal/report.json`。

原中央对流与节点带的独立选择已补接：CPU对流不再借用所选黏性的
`fds`。拉伸网格检查发现GPU紧致对流原先未排除物理端面的体算子写入，
而CPU按`is:ie/js:je/ks:ke`排除；首阶段内部差1.140e-15、端面差
4.482e-4。已对齐GPU三维写入范围。随后新C/E组合的滤波后刷新条件
遗漏ME6，CPU/GPU均出现缓存与当前q不一致；已将ME6显式接入两个
后端的刷新，不改变原先已支持组合的条件。

原紧致对流＋ME6、643e近端带的拉伸42y/41x，在NP=1/2/4及精确恢复
通过，见`standard_compact_me6_stretched_fixed/report.json`。修正缓存后
60z的同样矩阵也通过，见`standard_compact_me6_cache_fixed/report.json`。
上述失败原始记录保留，不将最后失败的整组报告改成通过。
该60z拉伸C/E路径另完成单rank GPU Compute Sanitizer：4步、零内存错误，
三阶段RHS／缓存最大缩放差6.166e-14，见`me6_endpoint_memcheck/report.json`
及其GPU运行日志。本轮没有Git写入或平台操作。

### ME6紧致辅助梯度几何：发现、批准修订与复验

原显式对流＋ME6、643c近端带的拉伸41x完成NP=1/2及精确恢复，60z的
NP=1 CPU/GPU及三阶段缓存检查也通过。但60z的CPU自身NP=1/2场差
2.672e-10超出2e-10，矩阵已停止，没有继续GPU NP=2或续算。
证据为`standard_explicit_me6_compact_stretched`目录；其报告保持未完成。

只读核对两份`resources/geometry.h5`：q0001--q0003坐标逐值一致；
q0004 Jacobian最大差1.038e-9，q0013度量最大绝对差1.151e-7、缩放差
3.760e-8。当前`begin_compact_geometry`仅对KEEP或旧743e/643c剖面路径
提前初始化；其他标准对流＋ME6组合在几何阶段仍使用原rank局部紧致
算子，到`steploop`才配置全局线耦合。至少度量已随分解变化，不能把
该场差称为纯机器噪声；尚未通过反事实重算断言它是唯一误差来源。

该最小修订随后获用户批准：ME6且`difschm='643c'`时，在`geomcal`前配置
已有全局紧致资源，复用原`geometry_phase`的仿射周期坐标处理；CPU/GPU
共用该几何。不改边界公式、内部ME6系数、滤波或2e-10门槛。

修订后CPU独立NP=1/2先验检查通过：度量最大缩放差降至1.919e-15，
末场差降至3.461e-14，见`geometry_cpu_first/report.json`。随后60z的
CPU/GPU、NP=1/2/4、多轴及2+2精确恢复通过：报告中的度量最大缩放差
1.919e-15，场差3.695e-14，三阶段RHS／缓存差5.586e-14，见
`me6_geometry_fixed_gate/report.json`。拉伸41x/42y的NP=1/2及精确恢复
也通过，度量差1.919e-15、场差1.433e-15、RHS／缓存差8.672e-16，见
`me6_geometry_fixed_xy/report.json`。

这一几何变更在原对流＋ME6紧致辅助路径的算子身份追加`gc1`，避免把
原有不同度量的批次称为精确续算。旧批次精确恢复在推进前按算子身份
拒绝，见`geometry_old_exact/run.log`；不是增加兼容层或自动切换。
新路径单rank GPU memcheck为零错误，见
`me6_compact_endpoint_memcheck/report.json`及GPU日志。
受影响的周期原紧致对流＋ME6／643c辅助路径也完成16区间、4步、NP=1/2 x
短复验：终场及分解差最大4.441e-16，冻结完整RHS差最大1.782e-14；两个
后端2+2精确恢复的全部state数据集及控制文件逐值一致。原黏性切换的
精确恢复拒绝与显式非精确恢复仍按原合同执行。证据为
`me6_global_geometry_periodic/report.json`。
此结果关闭已复现的简单物理端紧致几何分解差，未推广为一般CURVE或
长期稳定性结论。开放端、多物理面和新增组合的统计切换仍未关闭。

### LA2-P：周期MP5＋ME6的实际对流分派修订

周期原MP5选择器原来同时要求无黏、无主滤波且difschm=643e。加入ME6
与643c辅助导数后，全局紧致配置提前放行，但MP5选择器为假，设备循环
落入最终的中央显式核分支。这不是舍入误差：16区间TGV、4步、dt=1e-3、
Ma=.1、Re=1600、无主滤波／无crashfix，终场密度最大缩放差
3.07654e-4，冻结对流密度RHS差4.710834e-3。原始失败保留于
`results/keep_low_all_20261111/periodic_mp5_me6_before`。

另用相同原生CPU初场／几何、相同ME6／643c辅助，只改对流为643e，
得到`periodic_mp5_central_counterfactual`。`compare_frozen_rhs`确认该CPU
中央参考与错误GPU输入q及全部几何数据集逐值一致，物性缩放差
4.438e-16，五分量对流RHS最大差8.882e-15，证实实际误入中央核。

现将已有周期TGV的543e、recon=1或3、无特征重构核选择与ME6／辅助
导数／主滤波独立连接；本轮代表使用recon=3。非ME6旧选择条件不变，
ME6内部公式和CPU离散不变。最终中央分支现在只接受conschm=643e，
没有实际分派的其他格式明确报告缺少核，不再静默替代成中央计算。

以下均为16区间、4步、dt=1e-3、无crashfix，目录均在同根下：

| 目录 | 周期MP5＋ME6选择 | MPI及检查 |
| --- | --- | --- |
| periodic_mp5_me6_wired | 643c辅助，主滤波关 | NP=1、2 x/y/z、4 xy、8 xyz；冻结q／几何／对流与完整RHS、2+2精确恢复 |
| periodic_mp5_me6_explicit_filter | 643e辅助、explicit10 scalar | NP=1/2 x、精确恢复、全部GPU rank的memcheck |
| periodic_mp5_me6_compact_filter_aware | 643c辅助、compact10 scalar | NP=1/2 x、精确恢复、实际device-aware主halo |

终场／分解最大缩放差2.431e-14；无主滤波时，冻结对流／完整RHS
最大差3.840e-13，独立提取的黏性RHS差2.637e-16，密度黏性项为零。
滤波后物性按当前q核对，最大差4.441e-16。GPU单／双rank memcheck
均为零错误；device-aware双rank均记录active=T，不据此声称硬件P2P。
精确恢复比较全部state数据集及三个控制文件。原中央剖面动态／模态
示例另复核为`central_after_dispatch_fix`，CPU/GPU全部state数据集与
旧`central_me6_nscbc_shipped`各自逐值一致。

旧错误GPU批次的输入／身份也曾声明543e，不能当成修正后MP5的精确
源。CPU/GPU共用的新算子记录追加短标签`mp5v1`，不改变64字节身份
字段或检查点格式；连同最长现有滤波名称仍不截断。旧未带此标签的
CPU/GPU批次在推进前按算子身份拒绝精确恢复，见`periodic_mp5_old_identity`
两份日志；没有自动修补、转换或从错误GPU源运行非精确续算。

仅身份序列化修改后的`periodic_mp5_me6_identity_gate`重新检查NP=1/2 x、
643c辅助、compact10、device-aware及精确恢复，包含完整标签与滤波身份
断言，通过。先前无标签矩阵继续作为该相同数值实现的证据，不作为
当前版本的精确源；不是因身份改动重跑整套矩阵。
`periodic_mp5_long_filter_identity`另以643c辅助＋explicit10检查最长
64字节身份，完整mp5v1／滤波标签均在记录中，没有扩充旧字段。

独立`examples/KEEP_Taylor_Green_Vortex/datin/input.mp5_me6.case`配既有
input.run/input.output，无外部网格或初场。`run_keep_example_gate.py`
新增显式mp5-me6变体，实际复制这份文件运行CPU/GPU NP=2 x，并核对
2+2精确恢复、mp5v1标签及CPU/GPU场。现有KEEP默认示例未改为MP5。
当前带版本批次见`periodic_mp5_me6_shipped_versioned/report.json`，
终场CPU/GPU最大缩放差6.662e-16；先前无标签示例记录单独保留。
周期其他原迎风／特征分支及非周期边界不据此宣告完成；没有性能、
强激波、长期稳定性或生产DNS验收结论。

### LA4：现有LDC双物理面低阶接线

本单元完成于2026-11-12，证据目录沿用20261111启动的批次根目录。

范围为`flowtype='ldcavity'`、`bctype=41,41,41,0,1,1`，x/y物理、z周期，
保留现有轴对齐tensor几何（含正交拉伸）。默认CPU UDF与GPU专用核均
由顶盖最后覆盖顶角：顶面及两个顶角u=1、v=w=0、T=1；其余三个物理
面静止，温度读取`twall(1:3)`。不修改这项所有权，不接任意其他UDF。

已补输入、KEEP物理端／全局紧致资源、输出及冻结网格恢复接线。ME6
内部盒按各物理方向距边界至少4节点确定；其补集使用原节点黏性，
x/y近端带交集仅累加一次。GPU显式带复用xy物理通量核与各方向端部
模板，紧致带复用全局节点梯度／散度；内部ME6不变。普通GPU梯度入口
也按LDC选择现有xy物理模板，不再将其当作全周期方向。原单物理方向
的x/y/z代表NP=1/2复核通过，见`me6_layout_axis_recheck/report.json`。

16区间、4步、dt=1e-5、无crashfix的代表矩阵如下，证据目录均在
`results/keep_low_all_20261111/`下；没有对各选项做全排列。

| 目录 | 对流／黏性 | 滤波／网格 | MPI |
| --- | --- | --- | --- |
| ldc_keep_explicit | KEEP E／标准E | 关闭，均匀 | NP=1、2 x、4 xy |
| ldc_keep_compact_filter | KEEP C／标准C | compact10 scalar，均匀 | NP=1、2 y |
| ldc_me6_explicit | KEEP E／ME6＋节点E带 | 关闭，均匀 | NP=1、2 x、4 xy |
| ldc_me6_compact_filter | KEEP C／ME6＋节点C带 | compact10 scalar，均匀 | NP=1、2 y |
| ldc_standard_me6_stretched_filter | 原E对流／ME6＋节点E带 | explicit10 full，x/y拉伸 | NP=1、2 z、4 xz |
| ldc_standard_compact_me6_stretched | 原C对流／ME6＋节点C带 | compact10 scalar，x/y拉伸 | NP=1、2 x、4 xy |

上述终场／分解最大缩放差4.709e-14，三阶段q／缓存／对流与完整RHS
最大差4.885e-14。每个stage另按已知边界目标检查壁面速度与温度，最大
差4.441e-16，包括顶盖优先的顶角。所有物理节点rho/p/T有限正值。
同一冻结状态的近端黏性对照：E带CPU/GPU均逐值一致，C带CPU逐值一致、
GPU差不超过1.673e-19，证明交角带没有重复累加。各并行配置的2+2恢复
中全部state数据集及control/archives/insitu_control逐值一致，恢复不依赖
外部网格／初场文件。

LDC边界身份在原控制字段记录顶盖规则和三个实际壁温的精确位值，不
新增附属文件。改变壁温的精确恢复在推进前被拒绝，见
`ldc_changed_wall_exact/run.log`。KEEP＋标准E切换到KEEP＋ME6的非精确
续算在CPU/GPU NP=2 x通过，恢复探针的五分量物理q与源批次逐值一致，
控制记录从step=2建立新段，随后两步物理状态有限正值；此夹具关闭
统计，不替代非零时均窗口迁移验证，见`ldc_me6_nonexact/report.json`。

拉伸LDC原E＋ME6、explicit10 full的GPU memcheck为零错误，见
`ldc_me6_memory/report.json`。独立`examples/KEEP_Low_Order_Boundaries`
新增`--case ldc`及原生输入模板，不依赖tests目录；实际生成后在CPU/GPU
NP=2 y运行4步，场差4.440e-16，见`ldc_shipped_example/report.json`。
以上关闭现有LDC低阶范围的代表兼容检查，不代表稳态LDC收敛、一般
曲面或高阶壁面闭合。开放端LA3及一般物理CURVE LA5仍未完成。

LA3最初只读核查发现的普通单x入口保留当前rho问题，随后在下述单元
中复现并修复；不再作为尚未处理的问题。

### LA3-1：单x静态11/21入口／出口

已有GPU实际双面布局为`11/21/1/1/1/1`或`12/22/1/1/1/1`，其余方向
周期，不能据边界编号推定任意组合均可用。本单元先关闭前者的静态
入口范围，后者的CPU基线问题另列。

一步非均匀入口rho夹具复现：CPU `complete_state`的rho与x动量目标误差
均为0，旧GPU均为1.000e-3，见`x_inflow_before_named/report.json`。
GPU原核仅改速度／压力、保留当前rho，与CPU目标定义不同。现将CPU
原有目标构造抽成`initialize_x_inflow_target`共享接口，CPU仍在原时机
调用；GPU初始化仅构造目标、不修改流场，并将入口面6字段一次上传。
原三维场／RK推进没有新增整场主机往返。入口面数据在设备驻留并在
finalize释放，未新增用户数值选项。单轴时间变化模式尚未连接时明确
不冒充静态目标；现有双物理面平板动态路径没有因此改写。

新核按实际目标rho/u/v/w/p/T施加完整状态，或按原`mach_pressure`策略
处理：目标法向亚声速时固定速度和温度、压力由原两点模板外推、密度
按EOS重构；超声速目标分支使用完整状态。近端／内部KEEP与ME6公式、
原outflow21处理不变。输入、紧致资源、冻结网格和恢复身份已接通。

以下16区间4步、dt=1e-5、无crashfix检查均通过；目录位于同批次根下。

| 目录 | 组合／入口策略 | MPI |
| --- | --- | --- |
| x_inflow_keep_explicit | KEEP E＋标准E，完整状态 | NP=1、2 x、4 xy |
| x_inflow_mach_pressure_compact | KEEP C＋标准E，explicit10 scalar；Ma=.3压力策略 | NP=1、2 y |
| x_inflow_mach_pressure_supersonic_me6 | KEEP E＋ME6节点E带，compact10；Ma=1.5压力策略 | NP=1、2 z |
| x_inflow_stretched_compact_viscosity | KEEP E＋标准C，compact10；x拉伸，完整状态 | NP=1、2 x、4 xy |

终场／分解最大缩放差1.128e-13，三阶段q／缓存／对流／完整RHS差最大
1.286e-13；每个stage另按完整状态或压力外推目标核对入口rho/u/p/T，
代表误差在7e-16以内。各并行运行2+2精确恢复的所有state数据集和三个
控制文件逐值一致，物理节点rho/p/T有限正值。

普通入口策略已编码到原控制字符串，改变策略的精确恢复在推进前
拒绝，见`x_inflow_changed_mode_exact/run.log`。CPU/GPU NP=2 x的
KEEP标准E→ME6非精确恢复通过，恢复前后物理q最大缩放差2.522e-19，
新段从step=2建立，见`x_inflow_nonexact_me6/report.json`；统计关闭，
不据此关闭非零时均窗口迁移。NP=1/2 GPU与精确恢复各rank memcheck
均为0 errors，见`x_inflow_memory`。原11/22/41/52 nonreflecting剖面
路径短复核也通过，见`shared_inflow_profile_recheck/report.json`。
独立生成器新增`--case inflow`及原生模板，以`flowtype='uniform'`、
`ninit=3`读取预设弱扰动场，不调用TGV初始化、不依赖测试目录或新增UDF。
实际CPU/GPU NP=2 y四步运行通过，场差2.880e-15，见
`x_inflow_shipped_example/report.json`。

### LA5-1：ME6方法切换的非零统计窗口

本单元不依赖下述CPU12/22决定。原统计重建条件仅识别源／目标KEEP，
遗漏原对流＋ME6：16区间周期场在step=4从原E对流切到原C对流、黏性
保持ME6后，CPU旧统计仍从step=0累计7个样本、覆盖0.006，而目标窗口
应从step=4累计3个样本、覆盖0.002。原失败证据保留在
`me6_statistics_before/report.json`，不是数值通过记录。

现复用已有算子身份，源或目标含ME6且身份不匹配的非精确恢复也重建
统计段。不增加检查点字段／附属文件，不修改采样、时间积分或流场。
同方法精确及非精确恢复保留历史；精确续算的state/statistics全部数据集
和control/archives/insitu_control逐值一致。日志保留原有
`ASTR_KEEP_RESTART statistics_new_segment_step`前缀，此项也用于ME6。

已执行16区间周期场、6步、dt=.001、无滤波／crashfix的代表检查：

| 证据目录 | 方法切换 | 统计通道／MPI |
| --- | --- | --- |
| me6_statistics_mean_fixed | 原E→原C，均为ME6 | CPU旧时均，NP=1 |
| me6_statistics_insitu_fixed | 原E→原C，均为ME6 | CPU/GPU原位统计，NP=1/2 x |
| me6_statistics_enter | 原E＋标准黏性→原E＋ME6 | CPU/GPU原位统计，NP=1 |
| me6_statistics_leave | 原E＋ME6→原E＋标准黏性 | CPU/GPU原位统计，NP=1 |
| keep_me6_statistics_enter | KEEP C＋标准黏性→KEEP C＋ME6 | CPU/GPU原位统计，NP=2 x |

各变更窗口均从step=4/time=.004开始，覆盖.002；同方法窗口仍覆盖.006。
终场CPU/GPU最大缩放场差6.662e-16，归一化均值／协方差／密度加权量
绝对差不超过5.552e-16。NP=1以恢复点及其后两个完整步的实际场独立
重构统计积分，最大差6.662e-16。原失败与修订后CPU终场state.h5的全部
数据集逐值一致，确认修订只改变统计历史处理。

这是周期恢复生命周期的代表验证，不关闭物理边界组合的非零统计联测、
一般CURVE或长期统计收敛。此前关闭统计的LDC／单x迁移仍只按各自
原证据解释。

### LA3-3：静态剖面11/22/41/52与ME6固定近端带

范围仍为原`flowtype='bl'`、`turbinf='prof'`、x/y物理与z周期的轴对齐
tensor网格，不含入射激波或动态入口。本单元将原剖面边界阶段合同
接到KEEP＋ME6：CPU/GPU复用已有壁面／出口预处理、出口滤波后刷新、
顶面分类和交线所有权；距任一真实x/y物理端0--3节点用原节点黏性，
内部保持ME6。没有为ME6另造边界目标或局部紧致接口闭合。
原对流＋ME6的混合剖面组合尚未随本单元接入。

以下使用16区间、4步、dt=1e-5、原剖面弱三维扰动和y向拉伸，全部关闭
crashfix。各证据目录在同一批次根下。

| 目录 | 对流／节点带／滤波 | 顶面模式与MPI |
| --- | --- | --- |
| profile_me6_nonreflecting | KEEP E／643e／无主滤波 | nonreflecting；NP=1、2 x/y/z、4 xy |
| profile_me6_compact_incoming | KEEP C／643c／compact10 scalar | incoming_only；NP=1、2 y/z、4 xy |
| profile_me6_full_compatibility | KEEP E／643e／explicit10 full | compatibility；NP=1、2 x |

终场／分解最大缩放差3.170e-14，冻结边界／对流／完整RHS最大缩放差
2.054e-14，物理节点有限且rho/p/T正值。每组首个NP=2拓扑做2+2精确
恢复，全部state数据集和三个控制文件逐值一致；incoming_only变更
远场目标的精确恢复在推进前拒绝。nonreflecting交线保留独立对流
投影参考，最大参考差1.056e-16，不称完整黏性无反射验证。
该组GPU NP=1 memcheck为0 errors。

同一首阶段q与原`shared_inflow_profile_recheck`冻结数据逐值一致；
在x/y固定近端带比较完整RHS减对流RHS，CPU差0、GPU缩放差1.695e-21。
这验证本夹具近端带沿用原节点黏性、交角不重复累加，不证明拼接区
严格守恒或一般ME6壁面理论。

`profile_me6_statistics/report.json`另覆盖NP=2 x、6步非零统计：源为
KEEP E＋ME6，非精确目标为KEEP C＋ME6／643e节点带。同方法精确续算的
state/statistics数据集和三个控制文件逐值一致，同方法非精确保留历史，
统计开关不改变五个守恒场。变更后从step=4/time=4e-5累计3个样本，
覆盖2e-5；恢复探针与源物理q缩放差2.433e-21，终场CPU/GPU差
1.222e-15。CPU mean44与GPU展向统计先对应字段，再按真实z长度积分，
前12量归一化最大缩放差4.477e-16。没有将未采样的旧窗口并入新段。

独立生成器新增`--viscosity me6_opti`，默认仍为standard；不依赖tests。
open示例CPU/GPU NP=2 y实际运行4步，最大缩放场差9.993e-16，见
`profile_me6_shipped_example/report.json`。以上仅关闭这套静态剖面组合
的代表低阶兼容，不关闭其他开放端／动态入口、一般物理CURVE或长窗。

### LA3-4：静态剖面11/21/41/50和11/21/41/51

新增范围为已有静态prof入口、原压力出口21、下壁41、顶面50或51，
x/y物理、z周期，读入轴对齐tensor网格（含正交拉伸）。未改CPU边界
公式。`profile_state_boundary_layout/enabled`共同连接输入、实际剖面
状态核、x/y物理端体算子、全局紧致资源、几何检查及输出能力；不是
仅通过输入检查。KEEP、标准节点黏性和ME6固定节点带独立选择。
原中央对流＋ME6也使用同一现有剖面边界／x/y物理通量核。

以下目录均为16区间、4步、dt=1e-5、三维弱扰动与y向拉伸，关闭
crashfix；每组首个NP=2拓扑另做2+2精确续算。

| 证据目录 | 顶面／对流／黏性 | 主滤波／MPI |
| --- | --- | --- |
| profile_state_51_keep | 51／KEEP E／标准E | 无；NP=1、2 x/y/z、4 xy |
| profile_state_51_keep_me6_compact | 51／KEEP C／ME6＋节点C带 | compact10 scalar；NP=1、2 y/z、4 xy |
| profile_state_50_original_me6 | 50／原中央E／ME6＋节点E带 | explicit10 full；NP=1、2 x/z、4 xy |
| profile_state_51_original_compact_me6 | 51／原中央C／ME6＋节点E带 | 无；NP=1、2 x/y、4 xy |
| profile_state_50_keep_compact | 50／KEEP C／标准E | compact10 scalar；NP=1、2 z |

终场及同后端分解比较最大缩放差1.166e-13；三个RK阶段的q、缓存及
RHS最大缩放差9.138e-14，其中对流／完整RHS差最大2.318e-14。缓存
另由其当前q独立核对EOS，物理节点rho/p/T有限正值。精确续算的全部
state数据集及control/archives/insitu_control逐值一致。首组GPU NP=1
memcheck为0 errors，其日志和原始快照均保留。

`profile_state_targets`按实际inlet.prof核对入口内部节点、壁面速度／
温度及亚声速出口压力。顶面50另核对原二阶零法向导数条件：原始量
对rho/u/v/w/p采用(4*f(jm-1)-f(jm-2))/3，再按EOS重构温度，并非
复制相邻节点，也不独立约束T的离散外推。后续动态入口检查揭示原
检查器误将T代替rho作第五个独立外推量；原6.771e-12描述的是该错误
附加约束，不作为bc50验收量。原生CPU/GPU场及RHS比较不受此检查器
修订影响，保留原始快照并按正确合同复核。目标检查覆盖三个RK阶段，
不另写推进器、不改变程序边界公式或2e-10门槛。

`profile_state_51_statistics/report.json`为NP=2 x、6步非零统计：源为
KEEP E＋标准E，非精确目标为原中央C＋ME6／节点E带。统计开关不改变
五个守恒场；同方法精确恢复的state/statistics数据集和三个控制文件
逐值一致，同方法非精确保留原历史。改方法后从step=4/time=4e-5建窗，
3个样本覆盖2e-5。恢复探针与源q最大缩放差2.482e-23，终场CPU/GPU
差3.997e-15；按真实展向长度积分的前12项统计差6.107e-16。
继续沿用既有数值身份和边界配置记录，不新增检查点附属文件。

独立输入生成器新增`--case plate`，默认11/21/41/51，可在新运行前将
顶面改为50。其ME6版本实际在CPU/GPU NP=2 y推进4步，场差
3.331e-15，见`profile_state_shipped_example/report.json`。输入生成不
依赖tests目录。此单元不包括动态入口、吹吸、原迎风对流＋ME6、
非正交物理CURVE或长期稳定性；这些不能由本矩阵推定完成。

### LA3-5：读入tensor网格的动态入口与三种吹吸

在上述剖面布局中接通现有`turbinf='intp'`四平面缓存与原三次插值，
不改入口时间定义、内部KEEP/ME6或物理边界公式。单x周期横向入口
仍不由此取得动态准入。测试使用12个非多项式时间变化的原生HDF5
扰动平面，间隔1e-5，首轮4步实际从初始0--3窗口推进到第4号切片。
GPU仍只加载必要入口平面；没有新增整场主机往返。

吹吸沿用已有三种实现：固定相位模态、按全局节点确定的扰动，以及
原rank-local intrinsic RNG。夹具均A=1e-3、beta=2、xa/xb/xc=2/5/8、
展向模态数2；固定相位另用2个时间模态和15个已记录相位。原随机
模式不改种子或采样顺序，GPU沿用已有主机壁面面数组供给，不是
把RNG改成另一种设备随机数发生器。

已补资源生命周期：新格式剖面在启动时由root确定吹吸资源清单，
`wallbs.dat`及实际使用的`wallbs_phase.dat`随批次冻结，恢复前绑定
冻结路径；相位路径使用与其他资源一致的长路径容量。原随机模式
复用现有控制文件中的每rank RNG记录，动态入口复用`inflow.h5`，
不新增全场快照、重试或兼容层。恢复短测没有本地wallbs输入，精确
结果仍一致；旧未冻结相位／参数的批次不据此追认为精确续算来源。

本轮16区间、4步、dt=1e-5、y拉伸、无crashfix的代表矩阵如下。

| 目录 | 入口／吹吸 | 格式／边界／MPI |
| --- | --- | --- |
| profile_dynamic_keep | intp，无吹吸 | KEEP E＋标准E，21/51；NP=1、2 xyz、4 xy；NP=1 memcheck |
| profile_dynamic_compact_me6 | intp，无吹吸 | KEEP C＋ME6节点C带，compact10 scalar，21/51；NP=1、2 yz、4 xy |
| profile_dynamic_original_me6_counter_checked | intp，坐标确定扰动 | 原中央C＋ME6节点E带，explicit10 full，21/50；NP=1、2 xz、4 xy |
| profile_modal_keep_me6_nscbc | prof，固定相位模态 | KEEP E＋ME6节点E带，22/52 nonreflecting；NP=1、2 xy、4 xy |
| profile_legacy_random_same_topology | prof，原随机 | KEEP E＋标准E，21/51；NP=1、2 xyz、4 xy |
| profile_legacy_random_compact_nscbc | prof，原随机 | KEEP C＋ME6节点C带，compact10 scalar，22/52 incoming_only；NP=1、2 xz、4 xy |
| profile_dynamic_compact_nscbc | intp，无吹吸 | KEEP C＋标准E，explicit10 scalar，22/52 nonreflecting；NP=1、2 xy、4 xy |
| profile_dynamic_mach_pressure | intp，Ma=.3压力策略 | KEEP C＋ME6节点C带，explicit10 scalar，22/52 nonreflecting；NP=1、2 xz |
| profile_dynamic_modal_device_aware | intp，固定相位模态 | KEEP E＋ME6节点E带，22/52 nonreflecting，CUDA-aware halo；NP=1、2 xy |

除最后一行外本轮使用pinned halo；最后一行检查每个GPU rank的
device-aware接口实际激活，并完成同状态、分解与精确恢复比较，不由此
断言MPI内部的数据通路一定为硬件P2P或GPUDirect。

各动态源实际入口游标均为4（首组在只读检查中核对，后续报告直接记录）。
完整目标与亚声速`mach_pressure`分别核对规定rho/T或内部压力外推下
的EOS密度；不把NSCBC22的松弛压力当作强制压力。三个RK阶段的q、
缓存、对流／完整RHS最大缩放差8.449e-14，其中RHS差2.318e-14。
终场／允许比较的同后端分解场差最大1.413e-13。所有物理节点有限且
rho/p/T正值。目标检查最大残差6.242e-11，来自原随机模式保存状态的
壁温，仍在2e-10内，不将其描述为机器精度。首组memcheck为0 errors。

非随机模式的代表NP=2做2+2精确恢复；原随机模式对所有所列NP=2/4
拓扑分别恢复。state数据集、启用时的inflow缓存及三个控制文件均逐值
一致，RNG记录也在控制文件比较内。原随机强迫随rank数／分解而变，
因此不把其NP=1/多rank当作相同输入；仍严格比较每个同拓扑CPU/GPU。
坐标确定扰动和固定相位模态保留同后端分解比较，不放宽门槛。

`profile_dynamic_random_statistics`另做NP=2 x、6步：动态入口＋原随机
吹吸、KEEP E/C＋ME6的统计开关、精确恢复及非精确方法切换。开关
不改变五个守恒场；精确state/statistics/inflow与控制文件逐值一致。
变方法从step=4/time=4e-5开始3样本、覆盖2e-5的新窗；恢复q差
2.647e-23，终场CPU/GPU差4.219e-15，前12项展向统计差4.997e-16。

独立open/plate生成器增加`--inlet-mode`和`--blowing`，不依赖tests。
`profile_dynamic_forced_shipped`实际运行open的intp＋模态吹吸＋ME6，
CPU/GPU NP=2 y四步场差9.993e-16。12个短序列平面不是生产湍流入口。

保留的两项测试失败及定位：`profile_dynamic_original_me6_counter`的
9.464e-9来自检查器错误地要求T也独立外推；rho/u/v/w/p在CPU/GPU均
满足原bc50合同，修订检查器后在新目录完整重验。`profile_legacy_random_keep`
四rank恢复曾漏传2x2x1而请求4x1x1，程序正确报RNG拓扑不符；已修
脚本并在新目录保持原拓扑重验，不删除拓扑约束。不是CPU数学修订。

本单元不关闭非正交物理CURVE、二维网格挤出／其他初始化资源路线、
原迎风对流＋ME6、411真实混合段和其他现有专用组合的剩余工作。
CPU12/22两项待批准修订未动，旧紧致长窗负壁压记录保持未解决。

### LA3-6：既有HBL二维网格挤出与初始化资源解耦

范围是已有`user_define_module/examples/userdefine_hbl.F90`网格生产者，
不新增或修改UDF数学。它从`datin/grid.2d`读x/y并挤出固定展向长度90；
默认空UDF构建不能代替此构建。复用同一checkout下已配置该UDF的
`build_native_fp3_fp64_cpu/gpu`，均由根CMake更新，FP64、测试构建，
本批次不启用AIR5或Catalyst。默认probe构建的UDF选择没有改动。

原生CPU反例位于`extruded_resources_before`：连续4步完成，但冻结
资源只含生成的grid.h5，不含生产者恢复时需要的grid.2d；删除本地
输入后，恢复在HDF5读grid.2d时报文件不存在。不是缺HDF5库，也不是
CPU边界公式错误。现按网格生成方式选择资源：新格式剖面且
lreadgrid=f时冻结原grid.2d，而不以ninit决定网格类型。已有HDF5层
负责重定向UDF读取，未改UDF、增加三维网格反推二维的兼容层或重试。
原MP7专用分支的旧限制不因本单元整体删除；新格式使用其已接通的
吹吸资源／RNG生命周期。旧缺grid.2d的批次不修补为有效恢复源。

`extruded_resources_cpu_fixed`先以CPU NP=2 y关闭反例：恢复无本地
grid.2d/初场/剖面，全部state数据集及三个控制文件逐值一致。
随后完成下列16区间、4步、dt=1e-5、无crashfix代表矩阵：

| 目录 | 网格／初场 | 格式、滤波与MPI |
| --- | --- | --- |
| extruded_keep_basic | 既有挤出／ninit=2 | KEEP E＋标准E，无主滤波；NP=1、2 xyz、4 xy；NP=1 memcheck |
| extruded_dynamic_modal_compact | 挤出／ninit=2、intp＋模态吹吸 | KEEP C＋ME6节点C带，compact10 scalar，22/52 nonreflecting；NP=1、2 yz、4 xy |
| extruded_original_me6_random | 挤出／ninit=2、原随机吹吸 | 原中央C＋ME6节点E带，explicit10 full，21/50；NP=1、2 xz、4 xy |
| extruded_native_profile_init | 挤出／原blini，ninit=0 | KEEP C＋标准E，explicit10 scalar，21/51；NP=1、2 x |
| extruded_3d_initial_me6 | 挤出／三维弱扰动，ninit=3 | 原中央E＋ME6节点C带，无主滤波，21/51；NP=1、2 z、4 xy |
| read_grid_2d_initial | 读入三维网格／二维初场，ninit=2 | KEEP E＋ME6节点E带，explicit10，21/51；NP=1、2 y |

前五组逐值核对生成坐标与原二维x/y，并检查展向长度仍为90；冻结
资源只有原grid.2d而非生成grid.h5。最后一组反向检查仍正确冻结
grid.h5，未把二维初场等同于二维网格。各代表恢复均没有本地几何、
初场及剖面输入；state数据集和控制文件逐值一致，启用的入口缓存
也一致。原随机模式对所列每个并行拓扑分别恢复，保留其rank依赖，
不作不同强迫之间的NP=1分解场比较。

前五组终场／允许的分解比较最大缩放差3.367e-14，三阶段q、缓存及
对流／完整RHS最大缩放差6.199e-12，仍在2e-10内；物理rho/p/T有限
正值。memcheck为0 errors。源UDF和原始KEEP／ME6公式均未改变。

`extruded_dynamic_modal_statistics`另做NP=2 x、6步：intp＋模态吹吸、
KEEP E/C＋ME6、CPU旧时均／GPU展向统计。统计开关不改守恒场，同
方法精确续接的state/statistics/inflow和控制文件逐值一致；变方法
从step=4/time=4e-5建立3样本、覆盖2e-5的新窗。恢复q差3.309e-23，
终场CPU/GPU差3.775e-15，按实际展向长度90积分的前12项统计差
4.363e-15。说明恢复保留已发展出的三维状态，不是重读二维初场续算。

独立open/plate输入生成器新增`--extruded`，必须配上述既有HBL UDF
构建。`extruded_shipped_example`实际CPU/GPU NP=2 y、intp＋模态吹吸
＋ME6四步运行通过，场差5.373e-16，不依赖tests生成数据。
本单元不是任意UDF兼容、ninit=1单独验证、非正交物理CURVE、原迎风
对流＋ME6或长窗物理验证；完整goal仍保留这些适用的剩余范围。

### LA3-7：原中央对流＋ME6的剖面NSCBC接线

现有11/22/41/52、x/y物理且z周期的剖面端合同不再绑定KEEP名称。
共享选择函数改名为`profile_low_order_nscbc_enabled`，原nonreflecting
交线装配也改为对应的profile名称；KEEP已有算子／边界身份字符串不变。
原中央643e/643c对流＋ME6复用相同的状态准备、出口滤波halo刷新、
壁面／交线所有权及输出资源接口，不改边界公式。体对流仍排除物理
端面，不与边界RHS重复累加；几何仍要求轴对齐tensor网格。

16区间、4步、dt=1e-5、无crashfix的代表检查如下，目录均位于
`results/keep_low_all_20261111/`：

| 目录 | 原中央对流／ME6辅助节点带 | 顶面／主滤波 | MPI |
| --- | --- | --- | --- |
| central_me6_nscbc_checked | E／E | nonreflecting／关 | NP=1、2 x/y/z、4 xy |
| central_me6_compact_incoming | C／C | incoming_only／compact10 scalar | NP=1、2 y |
| central_me6_full_compatibility | E／C | compatibility／explicit10 full | NP=1、2 z |

终场／分解最大缩放差2.794e-14，三阶段q／缓存／RHS最大差2.883e-14，
首阶段边界／体对流／完整RHS最大差2.056e-14。物性另按当前q独立检查，
物理节点rho/p/T严格有限正值。各代表并行配置的2+2精确恢复state数据集
与三个控制文件逐值一致；incoming_only改变目标的精确恢复被拒绝。
nonreflecting交线另通过独立对流投影参考，不据此声称完整黏性无反射。
首个测试脚本的几何轴变量覆盖错误保留于`central_me6_nscbc_first`；
修正仅改变测试变量名。修正后的GPU单rankmemcheck为零错误。

`central_me6_nscbc_statistics`另做NP=2 x、6步：统计开关下五分量守恒场
逐值一致；同方法精确恢复state/statistics及控制文件逐值一致，非精确
同方法恢复保留统计历史。E→C非精确切换从step=4/time=4e-5建立3样本、
覆盖2e-5的新窗；恢复q最大差2.433e-21，终场CPU/GPU差1.075e-15，
按真实z坐标积分的前12项统计最大差4.477e-16。

本单元不修订普通单x12/22，不将原中央对流称为KEEP，也不据此放行
迎风＋ME6、入射激波、一般物理CURVE或全部选项排列。

原中央C＋ME6、explicit10 scalar、intp＋模态吹吸的nonreflecting读入
tensor短测另通过NP=1/2 x、精确恢复及实际device-aware halo，见
`central_me6_dynamic_modal_aware/report.json`。CPU/GPU入口游标均前移至4，
终场／三阶段最大缩放差分别为1.612e-15／1.612e-15；不据此声称设备
P2P硬件传输。独立示例生成器增加`--convection standard`并配ME6，
`central_me6_nscbc_shipped`实际运行open/intp/modal、NP=2 y四步，
场差1.223e-15。相同共享接口改名后的KEEP C＋ME6、compact10、NP=1/2 y
复核通过，最大场差2.620e-14，见`keep_profile_shared_trait_recheck`。

### LA3-8：已有迎风对流＋ME6路径的代表复核

没有扩大原迎风能力选择范围，没有修改重构、传感器或特征公式。
测试工具增加543e/743e选择及对应已有recon_schem=3/5配置；GPU在
测试显式请求RHS快照时，补记ME6常规顶面路径的boundary快照。其初次
缺快照记录`mp5_me6_profile_first`保留，不将缺少证据误报为数值失败。

| 目录 | 现有数值／入口／边界 | MPI及主滤波 |
| --- | --- | --- |
| mp5_me6_profile_checked | 543e、recon=3、无特征分解、ME6＋节点E；读三维初场，prof，11/21/41/51/1/1 | NP=1、2 x/y/z、4 xy；滤波关 |
| mp7_me6_profile_random | 743e、recon=5、特征分解开、ME6＋节点E；读网格／二维初场、prof、原随机吹吸，11/21/41/50/1/1 | NP=1、2 x/y/z、4 xy；滤波关 |
| mp7_me6_compact_random | 同上，节点C辅助 | NP=1、2 y；compact10 scalar |

均为16区间、4步、dt=1e-5、无crashfix。终场／分解最大缩放差8.927e-14，
三阶段q／缓存／RHS最大差1.345e-13，首阶段边界／完整RHS最大差
2.067e-14。CPU/GPU同拓扑2+2精确恢复的全部state及控制文件逐值一致；
原随机吹吸随rank分解变化，不作不同拓扑的同输入场比较。MP5及紧致
辅助MP7的单rankGPU memcheck均为零错误。现有平滑弱扰动可检查这些
核的接线，不能代替强激波特征空间或生产SBLI稳定性验证。
MP5＋节点C、迎风的其他顶面／资源组合及全部参数排列不据此放行。

既有MP7-LD＋ME6的22/52 incoming_only路径另完成节点C、compact10、
原随机吹吸、二维初场、NP=1/2 x/y/z/4 xy与各并行拓扑的2+2精确恢复，
见`mp7_me6_nscbc_incoming_checked/report.json`。最大缩放场差3.740e-14，
三阶段q／缓存／RHS差1.225e-13，首阶段RHS差2.107e-14，单rankGPU
memcheck零错误。原随机壁面温度目标的重构残差最大6.242e-11，低于
2e-10但不描述为机器epsilon。最初脚本错误要求这一旧路径携带KEEP
专用`wall-z0-v1`标记，已按实际合同限定该断言；原始记录保留于
`mp7_me6_nscbc_incoming`。没有为通过测试给旧算法添加错误身份。

### LA3-9：411域内滑移／无滑移分界的夹具矛盾

测试工具增加显式`--xslip`及逐阶段边界目标检查。首轮仍用原channel
单y物理布局，x/z周期、16区间；把xslip从域外10改为pi，试图同时
覆盖两段。实际首阶段CPU/GPU均在周期x接缝不满足目标，矩阵已停于
NP=1，未运行后续MPI／续算，不作为411混合段通过。

原因是同一周期节点的x=0副本被分类为滑移，x=2pi副本却为无滑移。
原qswap平均两份守恒状态后，两个端点速度相同，但既不为完整滑移
目标也不为零。CPU/GPU最大速度目标差分别为3.0311393020828374e-3／
3.0311393020828387e-3；去掉x两个周期端点后，最大差分别为8.674e-19／0。
原始状态和日志见`wall411_mixed_keep`，此处不是CPU/GPU移植偏差，
也不通过放宽2e-10或删除周期平均来关闭检查。

现有单y物理／周期x布局不能无矛盾地承载x<=xslip这种单分界混合目标。
后续需选择有物理x入口／出口的真实411组合并明确交角所有权，或者
另行批准周期相容的壁面分段定义；后者会改变边界定义，本轮未修改。
此前xslip在域外的全滑移段证据继续有效，不代表混合段已验收。

### LA3-10：迎风＋ME6的独立辅助导数／入口／滤波接线

在原MP5剖面21/51路径选择ME6＋643c辅助导数时，首次CPU启动即在
全局紧致配置阶段被拒绝，见`mp5_me6_compact_before`，尚未推进。
这不是改边界公式的依据，而是原中央／KEEP剖面资源选择尚未包含已有
迎风核。现将非反应五变量tensor剖面状态合同与其实际重构类型共同
连接：543e/recon3/无特征分解接已有物理空间重构；743e/recon5/特征
分解接已有xy物理特征重构。两者均独立选择ME6近端节点E/C与主滤波。
共享`profile_state_upwind_enabled`供设备分派使用，不把它们交给中央核。
全局辅助导数、几何、物理边界、动态缓存、吹吸资源及输出均复用既有
实现；没有新格式系数、CPU边界公式、MPI案例白名单或运行失败回退。
原非ME6路径及22/52的旧MP7模式选择不随该状态边界接线改写。

根CMake CPU/GPU重编完成。GPU首次构建发现新分派漏导入lchardecomp，
已补显式导入，修正后才执行以下原生检查。仍为16区间、4步、dt=1e-5、
无crashfix，证据均位于`results/keep_low_all_20261111/`：

| 目录 | 实际选择 | MPI |
| --- | --- | --- |
| mp5_me6_compact_wired | MP5、ME6＋节点C，21/51、prof，无主滤波 | NP=1、2 x/y/z、4 xy |
| mp5_me6_dynamic_filter_full | 同上，intp＋模态吹吸，explicit10 full | NP=1、2 y |
| mp7_me6_dynamic_farfield | MP7-LD、ME6＋节点C，21/51、intp＋模态吹吸，compact10 scalar | NP=1、2 z |
| mp7_me6_compact_rewire_recheck | 原MP7、21/50、二维初场、原随机吹吸、compact10 scalar | NP=1、2 y |

前三组终场／分解最大缩放差7.816e-14，三阶段q／缓存／RHS最大差
1.326e-13；物理节点rho/p/T有限正值，动态入口缓存均实际前移到4。
各代表并行配置2+2精确恢复的state、动态inflow缓存及三个控制文件
逐值一致。MP5节点C及MP7动态组的单rankGPU memcheck零错误。
最后一组另与改造前不可变`mp7_me6_compact_random`对应的CPU/GPU
NP=1、NP=2 y全部state数据集直接比较，四组均逐值一致。

`me6_upwind_statistics_migration`进一步在NP=2 x、6步、无主滤波下，
将ME6／643c辅助的MP5非精确切换到中央643c对流，其他物理目标不变。
统计开关不改变五个守恒数组；同方法精确state/statistics/control逐值
一致，非精确同方法保留历史。变方法恢复q逐值一致，step=4建立3样本、
覆盖2e-5的新窗；最终CPU/GPU场差3.997e-15，前12项真实展向积分
统计差5.596e-16。测试工具新增source/target scheme及共同diffusion
选择，默认仍为原E→C窗口；没有新增生产统计开关。

以上是已有物理端／重构／ME6算子的接线及短轨迹证据，不是KEEP激波
耗散或强激波稳定性证明。原NSCBC12/22的CPU待决策、411有效混合
布局、421入射专用连接、其他GPU多面组合及非正交物理CURVE仍未关闭。

### LA5-2：物理CURVE零推进几何诊断与待批准G1m

新增`run_keep_physical_metric_audit.py`，只运行CPU max_steps=0，生成
真实几何并读取解析／矩阵参考，不推进CFD。采用既有y-wavy、幅值.15、
y双bc41及x/z周期。物理方向导数矩阵复用不可变
`keep_wall_low_20261110/high_order_diagonal_sbp_audit/`下的
`matrices_n16.txt`和`matrices_n32.txt`；相应生产导数与
紧致矩阵源码最后修改于6f3046e，当前无差异。这里复用的是实际导数
矩阵，不开展SBP高阶研究。通用短测准备函数仅增加intervals参数，
默认仍16，用于本项32区间几何，没有改变旧夹具默认。

`physical_curve_metric_diagnosis/report.json`记录16区间NP=1及NP=2 x/y/z，
另做32区间NP=1。归一到连续计算坐标的逆度量误差，显式为
1.70449e-5→2.73763e-7（约5.960阶），紧致为.510818→.522812（约-0.033阶）。
紧致Jacobian与解析值误差约1e-14，度量恒等式／静止恒压几何RHS残差
也小于2e-10；这些不能证明几何正确，报告明确不接受解析几何精度，
不作为曲壁准入通过。NP=1/2度量差最大9.318e-15。
随后只读这些几何，以phi=y核对精确物理梯度(0,1,0)：紧致投影误差
16/32为.510818／.522812，物理体节点同样受影响；固定原生Jacobian的
候选对应误差不超过2.021e-14，见同目录`candidate_fixed_J_analysis.json`。
这不是端点或单位尺度显示问题。未重新推进，也未写回几何。

`physical_curve_standard_me6_geometry`另证实原中央643e＋ME6／643c辅助
的Jacobian及九项度量与紧致KEEP逐值一致，同样受影响。没有继续这些
路径的流动计算，也没有修改任何生产CPU几何。

只读候选把已有全周期G1a的仿射背景／位移curl扩展到混合方向：周期
方向沿用循环D，物理方向用原非周期端行。16/32逆度量误差降为
2.66734e-7→4.11156e-9，约6.020阶；尚是诊断候选，不是原生接线结果。
具体原因、公式、受影响路径、身份和后续门槛置于几何候选文件开头
“物理CURVE紧致几何修订G1m”，待人工批准后才实施。首次
`physical_curve_metric_audit`只检查度量恒等式和MPI，不能由其阶段状态
推断几何正确；全部原始诊断保留，未放宽门槛。

### LA3-11：显式曲壁41/52的现有完整源项接线

本单元不依赖G1m：采用现有643e几何／对流及643e节点辅助，主滤波关。
原GPU已有两种y非反射布局：周期x/z的TGV盒`1/1/41/52/1/1`，以及
静态prof入口／压力出口的`11/21/41/52/1/1`。新增共享边界类型和
`low_order_y_nonreflecting_enabled`，将KEEP／ME6接到原几何法向、
非反射RHS和完整黏性增量投影；不改CPU/GPU边界公式。输入选择、
几何分派、网格／初场／剖面资源及边界身份共同登记，不是删除tensor
检查后仅以程序完成为依据。C几何、主滤波、其他远场模式及intp尚不
在这条原有y-NSCBC合同内；22/52的已验路径保持原范围。

根CMake CPU/GPU重编及numerical_forms_probe通过。原负测试使用的
41/52如今成为已接类型，负项改为仍未实现的42/52，并保留独立正项
及拒绝643c的检查，没有删除失败测试或放宽数值门槛。

`run_keep_y_farfield_gate.py`使用16区间y-wavy、幅值.15，参考Mach=.3、Re=3000、
Tw=1、4步dt=1e-5，无主滤波／crashfix，固定x/z周期。以下各组均完成
NP=1、2 x/y/z及4 xy，同后端各并行拓扑的2+2恢复state数据集和三个
控制文件逐值一致；所有目录位于`results/keep_low_all_20261111/`。

| 目录 | 对流／黏性及初态 | 额外检查 |
| --- | --- | --- |
| y_farfield_keep_uniform | KEEP E／原节点E，静止常量 | 常量状态最大变化1.066e-14；单rankGPU memcheck零错误 |
| y_farfield_keep_nodal | KEEP E／原节点E，非均匀速度／温度 | 三阶段q／物性缓存／完整RHS及独立传入波投影 |
| y_farfield_keep_me6 | KEEP E／ME6＋节点E带，同一非均匀场 | 同上，近端带对照与单rankGPU memcheck零错误 |
| y_farfield_original_me6 | 原中央E／ME6＋节点E带，同一场 | 三阶段／独立投影／MPI及恢复 |

CPU/GPU与分解最大缩放场差3.582e-16，三阶段q／缓存／RHS最大差
4.228e-15，物理节点rho/p/T严格有限正值。非均匀场顶面保留的映射守恒RHS黏性增量
约8.413e-6，独立Euler法向谱投影的传入分量残差小于1e-18，覆盖实际
传入波数1和4两类，投影幂等误差约1.777e-15。该投影由当前q与真实
曲面单位法向计算，不调用CPU/GPU pmatrix作为参考，不证明声学反射率。

同一初始q的ME6近端带与原节点带对照，CPU逐值一致、GPU差8.471e-22，
见`y_farfield_keep_me6/nodal_band_comparison.json`。同时内部黏性RHS差
9.529e-10，说明并非整域偷偷改用节点黏性。独立示例生成器增加
`--case y-farfield`，实际NP=2 y、KEEP＋ME6四步通过，场差3.582e-16，
见`y_farfield_shipped/report.json`；该输入不依赖tests目录。

#### 同步发现：21/52剖面交线的CPU旧halo

`y_farfield_profile_keep`首个单rank冻结检查失败：物理q差1.333e-15，
边界RHS最大差1.220437e-6，仅在x+／y+交线的周期z端点。去掉z重复
端点后全域差2.272e-16。两后端四步程序已结束，但矩阵未通过，未
继续MPI／恢复或宣称剖面21/52验收。

CPU在boucon中先执行bc21强状态更新，随后装配上边界52 RHS，期间
未刷新出口面的z halo；GPU已有边界状态后的halo交换。只读初态与
冻结状态，构造旧／新z通量差，再经独立的出射特征投影，预测两端
RHS差；最大剩余残差7.365e-17，见该目录`halo_diagnosis.json`和
`analyze_keep_y_farfield_halo.py`。这是时序缺陷，不是需要替换21/52
交线物理定义，也不应让GPU重新读取旧halo以匹配CPU。

拟复用`parallel:refresh_reconciled_qswap_halos`，在CPU强状态完成后、
顶面RHS之前，仅刷新halo并重建相应物性，不再平均物理共有节点；保持
原边界公式、角点所有权和门槛，并更新受影响的精确续算身份。该项
CPU修订待人工批准，当前未实施，详见NSCBC物理修复计划。新剖面类型
的资源接线已准备，但不作数值准入通过；周期x的已验盒式范围不依赖
该CPU缺陷，不据此推广全部CURVE、滤波或长期物理结论。

### LA3-12：无黏421/GRCBC显式KEEP代表短测通过

已审计现有`11/22/421/52/1/1`、无黏、静态prof、sbli_grcbc路径。
ME6要求黏性项开启，不适用于这个无黏参考；该事实不作为遗漏ME6的
理由替代真实黏性壁面范围。首轮仅连接显式KEEP，紧致仍未接。
共享选择连接现有外部参考、NSCBC出口滤波后刷新及完整GRCBC RHS，
GPU首stage和后续stage都走原prepared-profile流程，不把新421调用
交给中央或特征重构核。未修改CPU/GPU边界公式、传感器或增加KEEP耗散。
新KEEP数值身份含inc1；边界目标沿用原GRCBC记录，输出使用现有联合
SBLI资源登记。未修复待批准的普通CPU22或21/52时序。

测试复用已有Euler斜激波夹具，16区间、参考Mach=2、beta=40度、4步
dt=1e-5、无主滤波／crashfix。第一次将入射点放在x=2、计算域x∈[0,1]，
使实际边界参考均为波前常量；并非激波物理验证。原出口自身六阶横向
滤波保留，不能把“无主滤波”解释为没有任何边界滤波。
`incident_keep_cpu_checked`原生CPU完成，常量状态最大变化4.441e-16。
初次`incident_keep_cpu_first`因测试误配冷启动restart_output=override
在推进前拒绝；已修正测试配置，不修改求解器来接受无效输入。

`incident_keep_uniform`单rankGPU memcheck在
`nscbc_incident_complete_rhs_kernel`报告CUDA非法地址，矩阵停止，没有
执行MPI／恢复或实际入射响应。以下均为对故障的有目的诊断，不作放行：

- `incident_plain_with_dumps`普通GPU保留快照完成四步；与CPU冻结边界／
  完整RHS差不超过3.253e-19，三阶段q／缓存最大差4.441e-16。
- `incident_memcheck_without_dumps`关闭快照仍失败，未支持快照导致故障。
- `incident_mp7_memcheck_control`同输入、原MP7-LD、memcheck零错误。
- `incident_keep_cuda_debug`在MPI启动的cuda-gdb下程序完成；最初直接
  singleton启动因MPI环境失败，未当成GPU证据。debugger后续bt命令
  因程序正常退出返回非零，也不误报为求解器失败。
- `incident_memcheck_coredump`工具在生成GPU core阶段持续运行但没有
  产生文件；已明确终止这一辅助诊断进程，未重启或中断其他作业。

cuobjdump记录该核196个寄存器、1272字节栈帧。KEEP进入前设备上下文
栈为1024字节，同输入原MP7路径为2672字节。先前尝试按核属性将栈
提高至1280字节，`incident_keep_stack_fixed`仍报非法地址；因此不能把
核报告的局部资源量当作检查工具下的充分栈需求。该生产初始化尝试
已撤除，不保留一个没有消除故障的修复。

`incident_stack_2672_solver_only`仅在测试进程进入前显式设置2672字节，
同一KEEP输入完成四步且memcheck为零错误。该对照支持检查工具下的
栈空间问题，不将其表述为已找到具体数组越界。最初未限定进程的
预加载探针还干扰了子进程初始化，已终止；其失败不作为CFD结果。
测试版现在提供显式`ASTR_VALIDATION_GRCBC_STACK_BYTES`钩子，只在
`ASTR_BUILD_TESTING`下编译且未设置时不做任何栈API调用。正式版本
不增加栈默认值或用户选项；独立探针保留在results诊断目录，不入Git。
随后的`incident_keep_uniform_checked`和`incident_keep_response_checked`
均完成NP=1、NP=2 x/y/z及NP=4 xy；所有并行拓扑各做2+2步同配置精确
恢复，state各数据集及三份控制文件逐值一致。均匀参考的常量最大变化
4.441e-16。域内x=.25入射参考在顶面产生1.388732e-4密度响应，各后端
及分解一致；所有终场CPU/GPU与分解比较最大缩放差1.222e-15。
响应场首步三stage的q／物性缓存差不超过6.662e-16，完整／对流RHS差
不超过3.253e-19，物理节点保持有限且密度／内能为正。
两组单rank memcheck均使用显式2672字节诊断栈且为零错误；其他普通
MPI／恢复运行不启用此设置。旧MP7同输入在最终二进制上另做
`incident_mp7_final_control`，无诊断栈输入，memcheck仍为零错误。

独立`examples/KEEP_Low_Order_Boundaries --case incident`使用相同域内
参考与无黏421布局，不依赖测试目录。首轮`incident_keep_shipped`因新
示例误用整数形式剖面行，被原实数字段识别器读至EOF；改成既有规范
科学计数格式后，`incident_keep_shipped_correct_profile`的NP=2 y四步
CPU/GPU最大缩放差1.222e-15。未改CPU读取规则或加入兼容层。
上述证据只覆盖显式643e/643e、静态prof、无主滤波／crashfix的既有
GRCBC边界接线；不覆盖紧致、ME6黏性入射、动态吹吸、非正交几何或
强激波长期稳定性，也不表示KEEP新增了激波捕捉耗散。

### LA3-13：无黏421/GRCBC的全局紧致KEEP接线

本项不依赖待批准G1m：网格保持同一笛卡尔tensor网格，只将已存在的
KEEP全局紧致体算子／低阶端行连接到既有GRCBC完整RHS。修改前
`incident_compact_before`在输入阶段以90退出，明确拒绝该组合。
现共享数值选择与`low_order_incident_enabled`允许独立643e/c选择，
`configure_global_compact`在几何阶段按该真实边界路径建立全局线资源；
GPU复用已有two-level PCR及原staged profile准备。没有增加独立rank
端行、边界重构核、KEEP耗散、主滤波或失败时的显式替代。旧GRCBC
法向模板、出口自身滤波、入口顶角及出口顶角所有权均不变。
新紧致身份使用既有keep6c/gw/w34并追加inc1，不与显式身份混用。

CPU先验`incident_compact_cpu_first`完成4步，随后
`incident_compact_cpu_decomposition_first`的NP=2 x/y/z和NP=4 xy与
单rank场差不超过1.111e-15。与之前显式CPU结果的五个守恒量最大
绝对差5.334e-9，表明没有仍然调用原显式体算子。

正式`incident_compact_response_checked`采用conschm=643c、difschm=643e，
保留Ma=2、beta=40度、入射点x=.25、16区间、4步和dt=1e-5。
NP=1/2 x/y/z/4 xy的CPU/GPU及分解最大缩放场差1.444e-15；首步
三stage的q／物性缓存差不超过6.662e-16，边界／完整RHS差不超过
5.964e-19。每个并行拓扑的2+2精确恢复state和控制记录逐值一致；
单rank memcheck使用已记录2672字节诊断栈，零错误。日志同时确认
全局物理端资源与GPU two-level PCR，不只检查输入准入。

独立辅助选项另测`incident_compact_auxiliary_uniform`的643c/643c常量
保持及NP=1/2 y、精确恢复和memcheck，以及
`incident_explicit_compact_auxiliary`的643e/643c入射响应及NP=1/2 x。
这些仍为diffterm=f，不能称为643c黏性算子验证。
`incident_compact_shipped`从独立`--case incident-compact`输入完成
NP=2 y四步，CPU/GPU最大缩放差1.444e-15，不依赖测试目录造输入。

`incident_explicit_after_compact`复验原643e/643e响应的NP=1/2 y及精确
恢复；与LA3-12的CPU/GPU旧参考比较，全部state数据集、control.bin、
archives.bin和insitu_control.bin逐值一致，见before_after.json。
这关闭无黏421/GRCBC显式／紧致的代表低阶接线，不关闭黏性入射、
动态入口／吹吸、主滤波、非正交CURVE或长期强激波稳定性。

### LA3-14：黏性GRCBC接线与新发现的CPU交线漏项，待人工批准

本轮连接既有`11/22/41/52/1/1`的静态prof、GRCBC完整RHS与KEEP E/C、
标准643e/c节点黏性、ME6固定近端带及原中央E/C＋ME6。仍只用笛卡尔
tensor网格，不依赖G1m；未改变原边界、内部KEEP或ME6公式。ME6原
迎风体算子与该入射边界的联合、旧sbli_shock目标均未据此完成。
`low_order_incident_enabled`参与共享选择、几何准入、全局紧致资源及
分stage准备，原中央＋ME6也登记inc1。上述是实施接线，不是数值放行。

首轮输入为16³区间、1×1×0.125域、Ma=2、Re=1000、参考温度288 K、
下壁Tw=1，4步固定dt=1e-5，无主滤波／crashfix／统计。初场
u=1-exp[-(y/0.15)^2]配1e-4三维扰动，rho=1/T；静态入口取相同基剖面。
出口压力比取1，使未受激波影响的近壁亚声速段保持波前目标；外部
入射参考仍为x=.25、beta=40度。此短窗不是真实SBLI或相似解验证。
改动前`incident_viscous_before`在输出准入处以71退出，未推进。

#### 先修正诊断相位，不改变计算

新增测试快照`incident_pre`位于黏性／源项后、GRCBC闭合前，已有`full`
应位于闭合后。CPU首次原中央＋ME6的独立传入声学残差约3.908，检查
发现旧`full`只为KEEP或H1延后写出，原中央分支仍在GRCBC前写出。
该两份快照逐字节相同，而下一stage顶面密度已从1变为1.0000347296。
现仅将新低阶入射路径的`full`快照延后，未改变任何q或RHS操作。
`incident_central_cpu_corrected_phase`的显式前后全部state数据集及三份
控制文件逐值一致；E/C两种CPU声学残差分别不超过5.774e-15／6.218e-15。
原相位失败保存在`incident_viscous_cpu_variants/central_me6_e`，不删除。

CPU先验覆盖KEEP节点E/C、KEEP＋ME6 E/C及原中央E/C＋ME6。
独立Euler声学投影核对当前两态参考与原始变量差：传入声学残差≤
6.218e-15，GRCBC对传出部分的改变量≤1.220e-19（后者为映射RHS量），
两类入流波数1／4均实际出现。黏性映射RHS非零，最大约1.257e-6／
1.441e-6。`band_comparison.json`中，ME6固定近端带与对应节点算子的
CPU贡献逐值相同，内部差分别4.456e-8／3.097e-8，不能解释为整域节点
替代。上述CPU组件检查不等于新CPU/GPU轨迹通过。

#### 首个CPU/GPU门槛失败并停止后续矩阵

`incident_viscous_nodal_explicit`仅执行到NP=1，两个后端均完成四步，
GPU memcheck显式使用2672字节诊断栈且零错误。但第一stage边界RHS
最大差1.063046e-7，超过2e-10；后续MPI、精确恢复和其他GPU格式矩阵
均未执行，不将“能跑完／内存零错误”登记为数值通过。
q／物性输入缩放差分别4.441e-16／6.662e-16，几何数据逐值一致。

冻结诊断`analyze_keep_incident_corner.py`只读复用该失败状态，结果见
`corner_diagnosis.json`：差异仅在x+／y+交线上，其余边界差≤5.961e-19。
CPU `nscbc_outlet_top_spanwise_enabled`未纳入GRCBC，故出口z项的j循环
仍截止je=jm-1；顶面自身的z循环又排除物理x端点。GPU出口函数已经
包含一次该z项。这是CPU交线漏装配，而不是要让GPU复制的旧语义。

用当前物理状态独立构造中心z通量差，最大1.063046e-7，与原始边界
差的预测残差2.711e-19。完整GRCBC后仍有5.315329e-8差异；独立声学
投影预测残差5.578e-19。此处应仅扣除传入声学投影，因为GRCBC的
对流相容项保留横向熵／剪切运输，不能简单扣掉全部传入子空间。
两个后端黏性增量`incident_pre-conv`之差仅3.885e-19，排除了黏性
离散导致该首stage差异的解释。全程未再次推进该失败算例。

**待批准最小修订：** 只把低阶GRCBC纳入CPU出口／顶面展向项所有权，
由出口在交线上唯一装配；GPU已有一次正确贡献，不改其公式。保持
原法向模板、松弛、滤波及其他模式。低阶GRCBC恢复身份v2同步升为v3，
不增加兼容转换，因此旧v2批次不能直接作为新v3恢复来源；H1批次
不受影响。批准后先复核这个同状态反事实，再推进原计划MPI／恢复
及其他新格式。尚未修改该CPU分类条件或恢复身份。

此前无黏421/GRCBC的常量／二维参考响应证据保持其原窗口范围，但
它们没有非零展向运输，不能据此宣称一般三维低阶GRCBC已经验收。

### LA3-2：单x12/22候选与待决定的CPU修订

候选已接输入／紧致／输出接口。GPU沿用现有特征核，使用现有独立
出口面缓冲，不依赖完整qwork；出口自身六阶y滤波在首阶段统计前
准备一次，后续RK阶段各准备一次，再同步其当前物性与halo，避免重复
滤波。该自带边界滤波与可选的全域十阶主滤波不同，后者关闭不等于
出口面没有滤波。没有改变NSCBC模板／系数。

首个非均匀状态短测的最终密度缩放差7.454e-10，超过2e-10，矩阵已停止，
不作数值通过／生产准入结论。首阶段物理q差仅7.106e-15，而入口边界
RHS差1.792e-5、出口差7.204e-8，二者是不同原因：

1. CPU `inflow_nscbc`仍固定覆盖LODi(1:4)，LODi(1)=0，使用
   `kin=.25*(1-gmachmax2)*a/L`。GPU已按局部法向传入波选择，以正的
   `.25*a/L`投影目标状态差。全域Mmax>1还可令旧CPU系数为负；这里是
   边界合同不一致，不是机器误差，也不应把GPU退回旧规则来通过对照。
2. CPU普通 `outflow_nscbc`在滤波后仅当`compact_plate_outlet`为真才
   刷新qswap；单轴路径先用旧横向halo计算RHS，到外部boucon之后再刷新
   已无法修正先前RHS。出口面内部差4.689e-15，误差集中在周期接缝。

独立Euler切向通量分析将滤波后的物理值与旧周期halo配对，预测出口差
7.203610525e-8，观测7.203610569e-8，扣除后最大残差4.689e-15。
证据为`x_nscbc_first/diagnosis.json`及只读
`analyze_keep_x_nscbc_rhs.py`，不另写Python推进器、不修改任何流场。

已请求用户决定两项最小修订，当前未执行：非反应五变量低阶CPU12
对齐现有GPU传入波／正松弛／目标投影合同，保留现有目标构造及法向
前向／切向二阶模板；CPU22出口面滤波后、计算RHS前补现有halo／物性
刷新，不改滤波系数或门槛。其他模型与高阶分支不随此项修改。依赖
该决定的12/22检查已停，完整goal仍active，其他未完成范围未删减。

用户已授权启动总计划 goal。当前已完成 KP1 显式 CPU/GPU 数学组件及周期
主程序首轮接线、三方向双 rank 短测和精确恢复。周期紧致 KEEP 已接入
CPU/GPU 全局线求解并通过分项、冻结 RHS 和代表长窗，周期主滤波也已
独立选择。KP1/KP2 周期基线已具备下列代表验证证据；KP0 的完整几何
职责、KP0-A 具体候选及后续物理／联合范围仍未完成。ME6 已完成均匀
周期主程序接入、私有边 halo、代表滤波与短恢复检查。
G1/G1a 已完成周期静态 KEEP 几何的三层收敛、代表 CPU/GPU/MPI、
交叉格式滤波及短恢复检查。G2-1--G2-4 已完成 ME6 周期拉伸／曲线扩展、
组件与代表主程序检查，详见文末；物理边界仍开放。
用户已新增“先适配现有低阶边界”的推进顺序，见总计划 KP5-L／KP6-L。
KP5-L0模板核对及KP5-L1显式KEEP＋原黏性、平直bc41首轮接线已完成组件、
代表MPI／恢复及10000步GPU中窗检查，详见文末。随后完成L2紧致物理端
组件、原黏性的显式／紧致独立选择及L4代表滤波／恢复检查。
ME6低阶壁面及KEEP出口候选已获批，CPU通用x+ bc22修订的解析对照已通过；
ME6壁面组件、私有halo、代表主程序／滤波／恢复及无修复窗口已通过。
KEEP开放端已修复CPU交线漏展向项，compatibility／incoming_only代表三维、
MPI与精确恢复通过。紧致滤波联测暴露原compatibility的掠流符号分支放大，
compatibility最小分类修订已通过代表紧致滤波、MPI、恢复及实际device-aware检查。
incoming_only的同类掠流修订也已通过解析、代表滤波／MPI／恢复与device-aware检查。
KEEP开放端的目标值精确恢复身份遗漏及出口底壁展向项所有权已修复。
nonreflecting交线已获批接入，显式／紧致KEEP的4步、代表MPI及精确恢复通过。
其紧致黏性／两种滤波、pinned与device-aware halo联合代表检查随后通过。
交线传入投影与唯一装配有独立公式对照；显式KEEP的OPEN-P1弱法向声学
窗口通过。紧致／斜向声学及完整黏性特征源项仍未验收。
固定时间壁面网格自收敛已测量，尚不能认定渐近全域阶数；紧致KEEP低阶壁面
无滤波和compact10长窗出现负壁压，稳定性未关闭，见文末失败归因与对照。
现有平直bc41与11/22/41/52范围的低阶工程兼容可分项交付；这不关闭
已记录的紧致壁面长窗失败，也不扩展到ME6开放端、曲壁、入射激波或AIR5。
高阶闭合仍开放，
全域解的固定物理时刻精度收敛尚未关闭。
用户已批准高阶壁面路线A的具体首轮公式：显式内部D6不变，SBP6/3＋SAT
弱壁面及配套原节点黏性已接入CPU/GPU；16³ NP=1/2 y、20³ NP=1/4 y
四步同态、独立RHS／功热收支、精确恢复及GPU memcheck通过。
长窗、解误差网格收敛和紧致／ME6／开放端高阶耦合仍未完成。
本记录与总计划分开，不用组件结果替代完整阶段验收。

| 阶段 | 状态及剩余工作 |
| --- | --- |
| KP0 | 周期半离散及代表完整RK／滤波收支已有证据；低阶开放端原节点黏性的应力功、热流、熵及端部掩码分项已独立重构。ME6物理端、高阶及完整边界专用处理联合收支仍开放 |
| KP0-A | G1/G1a 与 ME6 G2 周期候选已实现并取得代表证据；物理端闭合仍待定稿 |
| KP1 | 组件、同状态三维 RHS、三方向周期短测、代表滤波、精确恢复及 32^3 至 t≈10 的无滤波长窗已取得结果；联合阶段和物理范围仍未全部完成 |
| KP2 | 分裂等价、内部六阶、周期收支、同状态三维 RHS、三方向双 rank 短测、精确恢复及 32^3 至 t≈10 的无滤波长窗已取得结果；几何／边界联合项仍未完成 |
| KP3 | mixed 修订、频谱／制造解、组件、均匀周期主程序与同状态黏性 RHS、NP=1/2/4/8、两个 halo 后端、代表滤波和精确恢复已取得证据；非严格守恒仅记录，不外推长窗稳定性 |
| KP4 | 周期静态 KEEP 和 ME6 G2 曲线／拉伸几何首轮通过；物理端范围未完成 |
| KP5-L／KP6-L | 原计划两组低阶边界的接线、代表滤波／MPI／恢复及可运行示例已补齐；nonreflecting显式法向声学窗口通过。紧致壁面无滤波／compact10长窗失败仍保留；不将工程兼容等同于所有配置长期稳定或全域高阶 |
| KP5--KP8 | 未完成，不据周期 KEEP 结果宣称壁面、开放边界或一般物理端已验收；ME6周期曲线范围见G2记录 |

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
周期静态 KEEP 几何及下列 ME6 G2 周期扩展已取得代表证据。
下一步按 KP5-L2 核对并接入紧致KEEP的既有物理端方程，不等待新的高阶
边界推导。显式L1首轮结果见文末；后续分项适配ME6半点壁面和代表滤波，
再推进已有开放端。ME6若需要新的半点闭合，仍须具体方案审阅。
KP0-A 的高阶相容闭合设计仍保留，连同积分权重、守恒／能量边界项、
近壁黏性及边界专用滤波另行审阅；不把低阶实测或周期符号称为一般证明。
KP0 的周期冻结状态收支已补充，但完整 RK／滤波分项收支仍开放。
bc41、11/22/41/52 及联合数值交付仍未完成。
周期静态 KEEP 验证不能替代这些范围。没有修改普通路径生产默认、
新增物理边界闭合或执行 Git写入／平台操作。总计划未完成，不能以本轮结果
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

## G2-1--G2-4：ME6 周期映射扩展

本轮实施基点为 `2567532`，只执行已批准的几何候选第 8 节，不更改
系数、mixed 负半点符号或已记录的变物性非严格净量。新增 CPU
`me6_mapped_direction` 和设备 `me6_mapped_direction_kernel`，分别位于
`src/me6opti_operator.F90`、`src_gpu/me6opti_operator_gpu.cuf`。

每个半点分别插值逆度量 G 和映射面积 JG，先形成计算坐标梯度，再通过
完整 G 投影得到物理梯度。mixed penalty 使用插值后逆度量行范数之比。
应力和完整温度梯度生成映射动量／能量通量，外层取差后直接加入 RHS；
不重复乘 J，RK 仍只除一次 J。静态几何的正 J 和半点非零有限范数在
准备时检查一次，不在每个核后回读几何。速度／温度仍使用原私有边 halo，
没有新增传输后端、整场主机往返或生产选项。

### 常度量与制造解

`me6opti_candidate_probe --mapped-symbol` 检查四组原各向异性间距、对应
刚性旋转和三组斜切，共 11 组几何、171875 个波数。完整黏性符号和
导热符号通过原归一化非增长门槛；最小主子式为 -5.992e-12，最大导热
正实部为 4.494e-12，均小于 2e-10 的舍入比较阈值。该结果不等于一般
变度量耗散性的证明。

对角常度量符号相对原算子的最大缩放差为 1.552e-12，实际 CPU 算子在
平滑变物性场上的均匀各向异性极限差为 3.572e-14。每组常度量另以
实际 CPU 算子的正弦／余弦响应核对完整符号，不只检查另一份解析表达式。

组件 `me6opti_mapped_probe` / `me6opti_mapped_gpu_probe` 只读复用实际
G1a 初始化生成的 32³、64³、128³ 几何，使用两组物理波数和 16 个固定
比例节点，计算变物性动量、导热与黏性功的采样最大误差。解析物理导数
作为独立参考，不用解析度量替换真实 G、J。

| 实际几何 | 64³→128³ 动量三分量、导热、黏性功观测阶 | CPU/GPU 最大缩放差 |
| --- | --- | --- |
| G1a 周期曲线 | 5.855 / 5.867 / 5.914 / 5.972 / 5.884 | 8.418e-15 |
| 可分离周期拉伸 | 5.868 / 5.834 / 5.895 / 5.869 / 5.849 | 5.434e-15 |

GPU 首次组件执行因寄存器资源超过 512 线程块上限而启动失败（CUDA 701），
并非获得了全零黏性结果。已对新核设置 `launch_bounds(512)`，保持原
x=(512,1,1)、y=(32,16,1)、z=(64,1,8)，不调公式或改门槛。
组件 memcheck 为 0 errors。设备探针逐次检查启动错误，并分别记录
CPU/GPU 差和 GPU 自身相对解析解的误差。

### 完整步、MPI、滤波与恢复

CPU/GPU 构建均通过根 CMake；运行均为 FP64、周期五变量 TGV、固定
dt=1e-3、4 个完整步、关闭 crashfix。主程序使用当前阶段 Sutherland
物性。NP=4/8 在本地双卡共享设备，只作通信正确性检查，不计扩展性能。

| 组合 | 代表拓扑 | 最大场缩放差，含分解对照 |
| --- | --- | --- |
| 均匀16³，显式KEEP＋G2，无滤波 | NP=1/2 x | 5.552e-16 |
| 周期曲线32³，紧致KEEP＋G2，无滤波 | NP=1/2 x/y/z/4 xy/8 xyz | 1.444e-15 |
| 拉伸33×31×29，显式KEEP＋G2，无滤波 | NP=1/2 x/4 xy/8 xyz | 1.113e-15 |
| 周期曲线32³，643c/643e＋G2＋显式滤波 | NP=1/2 x | 1.555e-15 |
| 拉伸32³，643e/643c＋G2＋紧致滤波 | NP=1/2 x | 2.353e-14 |
| 周期曲线32³，显式KEEP＋G2，设备直接通信 | NP=1/2 y | 2.366e-15 |

前三项同状态完整 RHS 最大缩放差不超过 1.985e-14，独立提取的黏性
RHS 差不超过 2.637e-16。滤波后物理节点缓存相对当前 q 重建值的差
不超过 4.441e-16。前三项与两项滤波组合均完成 CPU/GPU 同拓扑 NP=2 x
连续 4 步及 2+2 精确恢复，全部状态数据集逐值一致，三个控制文件逐字节
一致。设备直接通信组的单、双 rank 主程序 memcheck 均为 0 errors。

新身份为 `/me6opti-g2-v2`，复用原 control.bin，不新增附属文件。
冻结 v1 检查点的精确恢复被拒绝，显式非精确恢复完成且状态有限、密度
与内能为正；关闭 ME6 或改变对流／滤波身份的恢复检查沿用现有协议。
普通黏性路径仍为默认，ME6 物理边界尚未实现。

另外复用只读 `analyze_keep_frozen_budgets.py` 对新原生 RK1 档案记录收支。
按唯一周期节点计算 `sum(mapped_RHS)/sum(J)`，曲线32³与拉伸33×31×29的
ME6 总能平均变化率分别约为 -1.65021e-9、-1.32486e-8／无量纲时间。
对应 CPU/GPU 差约 4.29e-17、3.39e-17。该已知非严格净量不作零残差
验收或全域补偿；KEEP 对流的守恒／能量交换恒等式仍由各自门槛检查。

### 证据与剩余范围

证据根目录为 `results/keep_g2_mapped_20261110/`：

- `final_symbol.log`：完整常度量符号、实际组件响应及笛卡尔极限。
- `components_final/report.json`：三层实际几何组件误差与 CPU/GPU 对照；
  脚本为 `tests/gpu_validation/run_me6_mapped_component_gate.py`。
- `uniform`、`curve`、`stretched`、`curve_explicit_filter`、
  `stretched_compact_filter`、`aware_memcheck32` 下的 `report.json`：
  原生场、拓扑、恢复、缓存及同状态 RHS；入口为现有
  `tests/gpu_validation/run_keep_periodic_gate.py`。
- `component_memcheck_final.log` 及 `aware_memcheck32/*/memcheck.*.log`：内存检查。
- `budget_curve.json`、`budget_stretched.json`：周期冻结状态收支诊断。

G2-1--G2-4 在上述周期代表范围内完成。物理端闭合、非周期网格、壁面和
开放边界、一般变度量稳定性、生产长窗及性能计时均不由本次结果关闭。
原已知非严格净量继续只记录，不作全域均值扣除或守恒性改称。

## 低阶适配前的独立 CPU 壁面索引修复

KP5-L0 源模板核对中发现的 CPU x+ 壁面组分索引问题已单独复现，随后获
用户批准修复。`src/bc.F90:noslip/noslip_adibatic` 两处第一组分外推的
第二个样本由 `spc(i,j-2,k,1)` 改为 `spc(i-2,j,k,1)`，不修改其余边界公式。
原错误把切向行／halo 混入法向外推，甚至在组分沿 x 恒定时产生约
2.357e-2 的壁面偏差，并传入 q6/rho；均匀组分与无组分输入会掩盖它。

回归入口为 `tests/gpu_validation/run_wall_species_boundary_probe.py`，
驱动为 `wall_species_boundary_probe.F90`。它只替换根 CMake 构建的主程序
对象，直接调用实际 CPU `boucon`／GPU 壁面分派，不复制待修公式。
CPU 的 bc41/42、NP=1/2 共20项检查通过，组分误差最大5.552e-17，
法向零梯度残差最大8.327e-17。GPU五变量壁面8项同状态对照通过，
最大守恒状态缩放差2.221e-16；GPU压力／温度的法向索引正确，无需改核。
另静态核对AIR5专用壁面：CPU/GPU均由法向第一内点取组分，并非此旧分支。

证据位于 `results/wall_species_index_audit_20261110/`，原始失败记录保留，
修复后报告为 `fixed_cpu/report.json` 和 `fixed_gpu_final/report.json`。
这是实际生产对象的边界级回归，不是完整CFD轨迹或新AIR5物理验收。
该独立缺陷本身不构成KEEP物理端接线交付；后续L1首轮结果另见下节。

## KP5-L0/L1：显式低阶 bc41 首轮接线

### 公式、接口与范围

法向第一、第二内点分别复用节点Euler映射通量的中心二阶、四阶导数，
索引步长为1：`D1(F)=(F2-F0)/2`、
`D2(F)=2*(F3-F1)/3-(F4-F0)/12`；正端按相应索引反向排列。
第三内点起及周期方向保持原六阶KEEP两点通量。近壁节点通量直接读取
当前q，不从原始量重建另一份q。CPU/GPU均采用现有有效推进范围，
对流模块不向壁面固定面写入体散度，壁面状态仍由原bc41流程处理。

实现为 `keep_operator:keep_explicit_volume` 及
`keep_operator_gpu:keep_explicit_volume_kernel`；无新增生产工作数组或
整场主机往返。几何显式导数在物理端使用原类型对应的闭合，不再硬编码
周期类型3。本节首轮接入643e/643e、原黏性、y向双bc41、平直轴向tensor网格、
滤波关闭；没有按flowtype名称放行。随后紧致及滤波结果另见下节，
独立Hermite7耦合与ME6壁面尚未接入。普通路径的默认选择不变。

`keep_wall_probe` / `keep_wall_gpu_probe` 的常数和多项式检查给出非零
首项截断系数，确认理论2/4/6阶；n=16的最高多项式一致性残差约2.34e-15。
壁外输入明确填NaN，结果保持有限；GPU同输入RHS最大缩放差4.771e-18，
组件memcheck为0 errors。三层16/32/64的壁面相容制造场为
`psi=0.1*sin(x)*sin(y)^2`，`u=dpsi/dy`、`v=-dpsi/dx`、`w=0`，rho=p=1，
y属于[0,pi]。最细两层的两类近壁及内部RHS误差观测阶为
2.991/4.946/5.963；近壁超收敛是该场的抵消，不能改称一般3/5阶模板。
这些是半离散RHS误差，不是固定物理时刻完整解的收敛证据。

### 主程序与中窗

`run_keep_wall_gate.py` 采用16³槽道、共有确定性三维初场、Re=3000、
Ma=0.3、Tw=1、Sutherland物性、固定驱动力1e-4，dt=1e-4，关闭滤波与
crashfix。4步覆盖NP=1、NP=2 x/y/z和NP=4 xy，场最大缩放差4.441e-16。
共有初场不依赖各rank随机流。CPU/GPU物理面有效输入q及缓存存在
4.441e-16的舍入差，因此原位档案按同相位而非逐位同输入描述；对流和
加入源项后的完整RHS差均不超过2.036e-16，含壁面物理节点。

诊断接线曾暴露两个缺口，均仅在显式请求快照时修订：channel的KEEP
漏写原始量快照；CPU旧`full`在驱动力之前、GPU在之后。后者的差值
1.746e-6恰为映射驱动力，扣除解析源项后残差2.036e-16。现KEEP的
CPU/GPU完整RHS均在源项后记录，未更改实际推进公式或默认文件输出。

CPU/GPU NP=2 x的连续4步与2+2精确恢复，全部状态数据集逐值一致，
三个控制文件逐字节一致。`/w24`记录低阶闭合身份；改变对流形式的
精确恢复被拒绝，显式非精确迁移完成且有限、正值。主程序单rank
memcheck为0 errors；受API改动影响的周期NP=1/2 x冒烟对照和短恢复也通过。

随后GPU NP=2 y完成10000步至t约1，仅末步写一次检查点。最大C_upper
为0.01492783，密度范围约[0.9968291,1.0031400]，最小体积内能19.7650。
唯一周期x/z节点、y端点半权的J加权密度均值变化为-3.2712e-7。
该存储态诊断还包含原有壁面投影，不作为零质量／能量变化证明，不作
全域均值补偿。检查点保留完整RK状态；下一次边界投影前，存储壁面法向
动量最大约1.737e-9，不把该相位宣称为重新投影后的严格壁面状态。

### 证据与未关闭项

证据目录：`results/keep_wall_low_20261110/`。`component_cpu.log`、
`component_gpu.log`、`component_memcheck.log`记录模板与组件；
`native_v3/report.json`记录场、MPI和恢复，`rhs_recheck.json`进一步覆盖
全部物理RHS节点；`native_memcheck`、`periodic_smoke`记录受影响短测；
`pilot`与`budgets.json`记录中窗及明确权重的存储态诊断。
`native`和`native_v2`保留快照接线／相位不匹配的原始失败，不删除证据。
全域解误差收敛、严格边界收支、高阶相容闭合、ME6壁面、一般CURVE物理端
以及生产统计稳定性未由这些结果关闭。完整KP0--KP8 goal继续，不标记完成。

## KP5-L2/L4：紧致物理端及代表滤波联合检查

2026-11-11更新。紧致KEEP仍对85个导数基分别求解同一全局紧致系统，
物理端使用原端点三阶、邻点四阶方程，内部六阶；没有把显式D2/D4拼到
紧致端点行。MPI内部接口保持全局耦合。物理壁面外、全局系统不会读取的
辅助打包槽置零，避免从无定义的壁外p/rho构造单项式；这不是为壁外状态
施加新的物理条件。辅助导数包含端点，最终对流提交仅覆盖原有效推进范围。
CPU/GPU分别由 `compact_runtime` 和 `compact_operator_gpu` 实现。

`keep_split_probe --wall` 在33/65/129节点上比较分裂和双点通量矩阵，
最大差3.664e-15，并用独立带主元LU检查实际全局解及残差。最细两层
五分量最大误差的观测阶为3.508/3.108/3.866/3.039/3.539。这是该物理端
联立算子的组件结果，不能称为全域六阶。周期模式复查仍给出约六阶，
分裂／双点通量差小于3.78e-15。物理端未加权动能压力和只作为诊断，
不套用周期零和门槛，不据此称边界严格守恒。

复用上节16³共有槽道初场、dt=1e-4、4步及关闭修复的设置，得到：

| KEEP对流／原黏性／滤波 | 代表分解 | 场最大缩放差 | 同相位对流／完整RHS最大缩放差 |
| --- | --- | --- | --- |
| 紧致／显式／关 | NP=1、NP=2 x/y/z、NP=4 xy | 1.333e-15 | 2.516e-16 |
| 紧致／紧致／关 | NP=1、NP=2 x/y | 1.111e-15 | 2.516e-16 |
| 显式／紧致／关 | NP=1、NP=2 y | 4.441e-16 | 2.036e-16 |
| 显式／显式／十阶显式 | NP=1、NP=2 x/y | 1.970e-15 | 6.588e-16 |
| 紧致／紧致／十阶紧致 | NP=1、NP=2 x/y | 3.686e-14 | 3.275e-14 |
| 显式／显式／独立十阶紧致 | NP=1、NP=2 x | 3.259e-14 | 1.594e-14 |

表中场差同时包含CPU/GPU及同后端相对NP=1的比较，不是不同离散格式
之间的差。原有显式／紧致滤波公式未变；滤波后有效q与物性缓存的最大
缩放不一致小于4.48e-16。含NP=2 x的组合均检查CPU/GPU连续4步与2+2步
精确续算，状态数据集逐值一致，控制文件逐字节一致。改变对流身份或
滤波身份的精确恢复被拒绝；显式选择非精确迁移到标准显式无滤波路径
可完成且保持有限、正值。普通启动先重算目标几何再恢复流场，不通过
读取被省略的轻量检查点geometry资源来冒充重建。

紧致／紧致无滤波主程序在NP=1、NP=2 x/y的5份Compute Sanitizer日志
均为0 errors。受打包／提交接口改动影响的周期紧致对流＋显式黏性，
NP=1/2 x同状态RHS、短轨迹和精确续算复查通过。没有新增生产全q或全RHS
工作数组，也没有新增逐核全场主机往返。

证据仍在 `results/keep_wall_low_20261110/`，子目录依次为 `compact_ce`、
`compact_cc`、`mixed_ec`、`explicit_filter`、`compact_filter_cc`、
`independent_compact_filter` 和 `periodic_compact_smoke`，各有 `report.json`。
物理端组件记录为 `compact_component_wall_mpi.log`；原单进程直接MPI_Init
失败仅是HPC-X重定位启动问题，使用已配置mpiexec后完成组件检查。

### 下一步与未关闭项

ME6物理壁面尚未实现。几何候选文件第9节提出d=0..3复用原节点黏性、
d>=4使用ME6 G2的固定空间拼接；尚待用户批准具体算法，不把总范围授权
视为该公式已获批准。开放11/22/52的特征RHS保留准备已完成，见下节，
仍未开放这些边界组合。固定物理时刻完整解收敛、紧致壁面联合长窗、
高阶相容闭合与最终KP7/KP8联合交付仍开放。可选修复的紧致准入结果另见
末节。上述组件和4步检查不关闭其余项目。

## KP6-L接线准备：保留已有RHS；KP7自动步长代表检查

普通CPU对流在`boucon`形成的RHS上累加；GPU也是先装配边界RHS再减去
体散度。此前KEEP的CPU显式／紧致入口会清零整场RHS，GPU显式轴1／
紧致首批提交也会清零或覆盖已有贡献。固定bc41面初始RHS为零，因此
壁面检查不辨别这个缺口。若只放开NSCBC准入，会擦掉非零特征RHS。

本轮先用非零RHS复现，原显式组件立即报`KEEP overwrote boundary RHS`。
随后将四个入口改为累加：CPU继续返回正映射散度，由rhscal统一取负；
GPU在现有RHS上减体散度。无效推进范围完全不写，由调用方初始化RHS。
不是在KEEP体算子中重算或补偿特征边界，没有新增工作数组或边界公式。
该修订只补齐所有权合同，不表示11/22/52的物理接线或验收已完成。

显式组件的有效区非零基线累加检查通过，物理端非零RHS逐值保留；
GPU组件同输入差不超过2.78e-17，memcheck为0 errors。独立生产对象探针
`keep_rhs_production_probe.F90`针对紧致入口，覆盖NP=1及NP=2 x/y/z，
CPU累加绝对残差最大2.78e-17，GPU与独立CPU基线比较最大3.61e-16，
排除的物理面始终精确保留原值。它不模拟NSCBC波动或证明边界物理正确。

改动后重跑显式无滤波与紧致对流／紧致黏性／紧致滤波，NP=1/2 x的
CPU/GPU短场及2+2步恢复通过；逐数据集核对修订前后的8组最终状态，
全部逐值一致。因已准入范围的实际数值轨迹未变，本轮未更改这些格式
的恢复身份。证据为`rhs_preservation_before.log`、`rhs_preservation_cpu.log`、
`rhs_preservation_gpu_memcheck.log`、`compact_rhs_cpu/report.json`、
`compact_rhs_gpu/report.json`、`additive_explicit_native/report.json`和
`additive_compact_native/report.json`，均位于前述低阶证据根目录。

同一16³显式bc41代表场另以自动步长推进4步，CFL目标0.5、增长系数1.1，
无初始dt或最大dt人为截断。首步dt为3.349664993556197e-3，第四步使用
3.349661522618497e-3；CPU/GPU四步日志的dt和累计时间相同。NP=1/2 x
场及拓扑对照最大缩放差8.96e-16，CPU/GPU各自2+2步精确续算一致。
证据为`adaptive_explicit_wall/report.json`及对应原始日志。该短测补齐
默认自动步长的代表集成，不外推为CFL0.5对所有KEEP组合的稳定上限。

## KP7可选修复：紧致KEEP接入原crashfix

计划K-D06已批准保留统一可选crashfix。本轮复现：紧致KEEP开启修复后，
CPU代表运行完成，但GPU被`crashfix_support_status_gpu`旧格式条件拒绝，
报`GPU crashfix admits central 643e or MP7-LD 743e convection`。该检查与
`compact_runtime`静态资源配置均未纳入KEEP 643c；修复核实际只处理
五变量状态和物性，不调用对流导数或紧致线求解。

本轮仅补齐显式选择KEEP 643c的准入及错误文字，未修改原修复平均、
阈值、扩大邻域规则、halo刷新、历史存储或默认开关。普通紧致格式的
修复准入未扩大，AIR5不纳入。测试仍默认`crashfix=f`，使用驱动的
`--crashfix`才开启，不把修复功能当成KEEP保正限制器。

`keep_rhs_production_probe`新增`--repair`测试入口：在每个rank的有效
内部放入一个负密度点和一个负体积内能点，调用实际CPU/GPU修复函数。
NP=1及NP=2 x/y/z均修复了预期的2或4个点；另检查普通643c仍被原状态查询
拒绝，避免意外扩大其准入。修复后的物理节点有限，
密度及体积内能为正，CPU/GPU状态逐值一致，全部critical-node标记一致。
探针GPU版本最初在四维误差归约处触发NVHPC内部错误，停止该测试编译后
改为等价标量循环，未修改生产算法或比较门槛；失败目录仍保留。

真实16³槽道以紧致KEEP／紧致原黏性／十阶紧致滤波、dt=1e-4、4步、
修复显式开启，检查NP=1及NP=2 x。场最大缩放差3.31e-14；CPU/GPU各自
连续4步与2+2步恢复的状态数据集逐值一致，控制文件逐字节一致。
3份主程序memcheck日志均为0 errors。该健康轨迹没有实际修复点；
它证明开启后的接线和零历史精确续算，非零历史的实际状态修复由前述
探针单独核对，不能将两者合称本轮完成了非零历史的整轨迹重启测试。

证据仍在低阶证据根目录：`compact_repair_before`保留原准入失败；
`compact_actual_repair_cpu/report.json`及`compact_actual_repair_gpu_final/report.json`
记录激活修复；`compact_repair_native/report.json`和对应memcheck日志记录
主程序。GPU首次测试编译失败目录为`compact_actual_repair_gpu`，后续
等价循环版本及最终准入反例完整日志为`compact_actual_repair_gpu_v2.log`、
`compact_actual_repair_gpu_final.log`。

## KP6-L审计：当前待人工确认的出口语义

2026-11-11核对`bc:outflow_nscbc`、`profile_nscbc_outlet_enabled`及GPU实际
出口核，发现KEEP若直接沿用CPU旧通用分支，会无条件覆盖第5声学特征，
而GPU已有修订按局部法向波速分类。CPU已有平板修订只由743e等条件
选择，KEEP尚未进入。旧问题及不能作GPU参考的约束已记录于
`ASTR_GPU_NSCBC_PHYSICS_REPAIR_PLAN.md`，不能以扩大KEEP准入重新移植它。

已提出独立候选：新CPU/GPU KEEP统一复用已修订的bc22及其已有常数
倒流背景，其他CPU路径不变；低阶导数和bc52三种模式的定义不变。
具体边界顺序、快照相位、源项所有权及最小检查写入几何候选第10节。
该选择尚待人工批准；本轮仅审计与形成可执行接线合同，没有开放11/22/52，
也没有修改CPU或GPU边界核。第9节ME6壁面候选也仍待批准。

## 2026-11-11追加批准与CPU右出口修订

用户已批准第9节ME6低阶壁面、第10节KEEP出口，并要求一并修订CPU
旧通用bc22。本次CPU修订具体覆盖x+、非反应五变量；其他法向面、
多组分及反应流不由此项扩大。旧通用x+分支现在复用已有局部特征分类、
非负压力松弛及其归一化，并使用既有完整常数背景处理倒流。内侧导数、
横向项和滤波公式没有改变，GPU边界公式不需要重复修订。

新探针`nscbc_cpu_outlet_probe.F90`链接根构建的实际边界对象。首先复现
均匀超声速出流仍产生非零RHS，误差约2.486e-2。随后18项NP=1/2解析
检查通过，覆盖亚／超声速出流、强切向速度、掠流、目标一致状态、非零
传出压力梯度、亚／超声速倒流和pout不等于1。CPU最大绝对误差4.80e-14，
GPU最大绝对误差7.11e-15，CPU/GPU差最大4.09e-14。该结果验证实际边界，
不是KEEP完整开放域轨迹或声学反射率验收。

证据为`cpu_outlet_red/report.json`、`cpu_outlet_fixed/report.json`及
`cpu_outlet_gpu_fixed/report.json`，驱动为`run_nscbc_cpu_outlet_probe.py`。
首个探针因漏调既有显式滤波系数初始化而退出，记录保留在`cpu_outlet_before`；
修正测试初始化后才得到上述真实数值反例，没有改生产滤波以配合探针。
CPU新通用出口也记录现有`bc22-backflow-v1`目标／算法身份，原有GPU及
已修订CPU平板身份不变，不新增文件。缺少当前身份的旧CPU检查点不能
冒称同算法精确续算，非精确恢复仍按既有选择执行。

## KP5-L3：已批准ME6低阶壁面过渡的代表交付

### 实现和范围

当前支持非反应FP64、平直y向双bc41、x/z周期、原643e节点梯度。
距物理壁面的全局节点距离d<4使用原节点黏性RHS，d>=4使用ME6 G2，
三个动量及总能量采用同一区域选择，密度黏性RHS严格为零。MPI局部
端点不是壁面。壁面面值仍由原流程处理，不额外改变完整RK存储相位。

CPU通过`diffrsdcal6(wall_band_only=.true.)`只提交近壁节点贡献，再由
`me6_diffusion_cpu`提交内部G2。GPU复用原节点应力／热通量计算及其
halo交换，由`me6_nodal_wall_kernel`只提交近壁贡献，原G2核按全局
物理壁面距离限制提交范围。标准节点通量目前仍全域构造，不称薄层
性能优化。复用原工作区，不新增全q或全RHS来拼接结果；普通黏性默认
不变。新组合使用`/me6opti-g2w4-v1`身份，周期`/me6opti-g2-v2`不变。

私有边halo接口显式接收周期属性，避免把单rank的MPI_PROC_NULL误当成
周期或物理边界。物理端不回绕、不从空MPI接收端解包，z向扩展也不读取
物理y端之外的无定义行。无效通信槽置零只为打包，不写作壁外物理状态。
物性计算和G2模板同样不读取这些物理端之外的字段。

### 制造解与实际阶数

`me6opti_wall_probe.F90`调用实际生产节点黏性和ME6，采用
`psi=0.1*sin(x)*sin(y)^2*(1+0.1*cos(z))`、`u=dpsi/dy`、`v=-dpsi/dx`、
w=0、rho=1及`T=1+0.02*sin(y)^2`。y属于[0,pi]，x/z属于[0,2pi]。
Sutherland黏度、温度依赖导热、黏性功均纳入解析RHS；壁外速度／温度
填NaN。制造解只比较有效体积内点；固定壁面上的约束RHS不冒充无约束
Navier--Stokes黏性散度。

16/32/64三层和NP=2 x/y/z组件通过，近壁区域逐值等于原节点参考，
CPU/GPU映射RHS最大缩放差1.53e-20。32到64最细两层，d=1/2/3/4及
d>=5区域最大误差观测阶为1.329/0.979/0.947/5.507/5.983。
因此“原梯度使用二／四阶模板”不等于组合黏性RHS也有该阶数。
边界梯度误差经第二次求导，会使该节点组合的近壁动量误差降至约一阶。

采用唯一周期节点、共享MPI节点只计一次、真实J测度归一化，有效体积
内点的四分量(x/y/z动量、能量)L1观测阶约为
1.983/3.006/4.775/3.006，L2为1.475/2.502/3.592/2.499，Linf为
0.952/1.993/2.996/1.986。它是已批准的低阶过渡，不是全域六阶闭合。
相同离散测度下64³原节点／混合算子的映射能量RHS和分别约
-1.0133810e-3与-1.0134334e-3，差约-5.24e-8；这些受壁面约束的
冻结状态和不是连续边界通量定理，不强制为零，也不作均值扣除。

### 原生场、滤波、MPI与恢复

复用共有16³槽道初场、dt=1e-4、4步，crashfix关闭：

| 组合 | 分解／后端 | 最大场缩放差 | 对流／完整RHS最大缩放差 |
| --- | --- | --- | --- |
| 显式KEEP＋ME6，无滤波 | NP=1、NP=2 x/y/z、NP=4 xy，pinned | 4.45e-16 | 2.04e-16 |
| 紧致KEEP＋ME6，十阶显式滤波 | NP=1、NP=2 x/y，pinned | 1.97e-15 | 4.88e-16 |
| 显式KEEP＋ME6，独立十阶紧致滤波 | NP=1、NP=2 x，pinned | 3.23e-14 | 1.60e-14 |
| 显式KEEP＋ME6，无滤波 | NP=1、NP=2 y，device-aware实际启用 | 4.45e-16 | 2.04e-16 |

含NP=2 x的组合均检查CPU/GPU各自2+2步精确恢复，状态数据集逐值一致、
三个控制文件逐字节一致；改变方法／滤波身份的精确恢复被拒绝，显式
非精确切换到原标准格式可完成且有限、正值。紧致KEEP＋ME6＋显式滤波
5份主程序memcheck日志均为0 errors。普通原节点黏性NP=1的CPU/GPU场
与本项修改前逐数据集一致。周期私有halo四rank及原曲线G2组件复查通过，
没有用物理壁面闭合取代周期／MPI内部的原算子。

随后GPU NP=2 y、无滤波、无修复以dt=1e-3完成1000步至t约1，仅末步
写检查点。完成态日志最大C_upper为0.1492785，后验低于0.5；该记录不
声称在运行中逐步截停CFL。末态rho范围[0.9968774,1.0030915]，最小体积
内能19.76628。相同J加权存储态密度均值变化-3.263e-7、总能量均值变化
-5.651e-6，包含壁面投影、驱动和热交换；存储壁面法向动量最大1.710e-8，
不把完整RK存储相位说成已再次投影的壁面状态。这个窗口不是生产DNS或
统计定常验收，也不代替未来高阶闭合／一般曲壁验证。

证据仍在`results/keep_wall_low_20261110/`：
`me6_wall_gpu_component_final/report.json`记录制造解、全域范数及收支；
`me6_wall_halo_cpu_y.log`、`me6_wall_halo_cpu_yz.log`、
`me6_wall_halo_gpu_single.log`、`me6_wall_halo_gpu_y.log`及
`me6_wall_halo_gpu_yz.log`记录物理端halo；`me6_wall_native`、
`me6_wall_compact_filter`、`me6_wall_independent_compact_filter`、
`me6_wall_aware`各有`report.json`；`me6_wall_window_v2/report.json`为
稳定窗口。`me6_periodic_halo_unchanged.log`、
`me6_periodic_mapped_unchanged.log`及`nodal_after_me6`记录受影响保护项。
编译／测试初始化的原失败目录保留，没有通过放宽门槛或改变系数补过。

### 当时的下一步（开放端接线前）

本段记录ME6壁面完成时的顺序；后续开放端接线与检查见以下各节。
当时KEEP的11/22/52主程序接线仍未实现；用户已经批准，不再作为待决策项。
下一步按第10节合同处理边界阶段顺序、最终状态快照与唯一RHS所有权，
分项覆盖三种无激波bc52模式、显式／紧致体算子、代表MPI和恢复。
高阶相容壁面／开放端、一般非周期CURVE、固定物理时刻解收敛和全计划
联合交付仍未关闭，不将本节低阶范围冒称KP0--KP8完成。

## KP6-L首版接线与三维交线缺口

2026-11-11已接入CPU/GPU KEEP的11/22/41/52组合、原黏性及平直tensor
几何。显式／紧致体算子继续各用原物理端方程，物理面由既有特征RHS
处理；未启用Hermite7或入射激波。CPU复用延后出口RHS接口，GPU也在
出口滤波、壁面／顶面处理和最终halo之后记录输入状态，不套用仅适合
已有平板路径的紧致准备程序。恢复身份区分`/o24`与`/o34`，本节尚未
取得三维、MPI和恢复交付证据，不能因配置准入已写好就称已支持生产。

驱动为`run_keep_open_gate.py`。原二维剖面、16³、4步、dt=1e-5、
compatibility、显式KEEP＋原643e黏性、无滤波／修复，NP=1同相位检查
通过，输入q／缓存及边界／对流／完整RHS缩放差均在8e-16以内。
首轮诊断曾因旧分支与新分支重复写boundary文件退出，已仅保留KEEP完成
全部边界后的写入，不修改文件覆盖规则。证据为`open_first`和
`open_first_v2`，后者通过，但它不验证展向交线项。

为辨别真实三维装配，后续改用共有3D HDF初场，保持相同网格、入口
剖面和物理参数。基准u为`1-exp(-(y/0.08)^2)`；扰动幅值1e-4，x向采用
`1-exp(-x/10)`，z向单周期sin/cos，底壁及入口扰动为零，顶面／出口
扰动非零。T与rho按rho*T=1构造。顶面法向速度含正负弱变化，原入口
目标和正温度不变。它是接口诊断场，不是湍流初值或物理生产算例。

### 已定位的失败

三维NP=1首阶段同相位输入q／缓存最大缩放差分别6.67e-16和7.78e-16，
几何逐数据集一致。但边界RHS差最大8.34961435524925e-5，位于
`(i,j,k,component)=(16,16,4,5)`，即x+出口与y+顶面的交线。

CPU右出口的展向循环采用`do j=js,je`，物理上边界上je=jm-1；顶面
自己的展向循环采用`do i=is,ie`，物理右边界上ie=im-1，因此CPU两边
都未装配这条交线的z通量差。GPU的`nscbc_tangential_x_rhs`保留了该
z项，顶面核则跳过物理x端的z项，实际有且只有一个装配者。

独立归因用冻结状态和实际J／度量重算
`Dz(Gtilde)=(Gtilde(k+1)-Gtilde(k-1))/2`。CPU边界存正散度、GPU存负
散度，故若CPU漏项，应有`RHS_cpu+RHS_gpu=-Dz(Gtilde)`。整个交线内部
z节点均符合，最大扣除残差1.262e-16；其他节点差不超过3.431e-16。
这不是舍入误差，也不是KEEP体通量、物性或快照相位造成的差。
分析脚本为`analyze_keep_open_corner.py`，原数据和逐分量结果保存于
`open_3d_first/corner_attribution.json`。三维矩阵在此首个失败停止，
没有继续MPI、恢复或其他模式的通过声明。

### 待审批的最小修复与后续范围

已提交用户确认：compatibility／incoming_only的x+22、y+52交线由CPU
出口补入一份既有二阶z通量差，与GPU所有权一致；不改系数、目标或
其他面，并更新有效闭合身份。尚未获批，不修改该CPU循环来补过。

nonreflecting另有交线耦合问题：其顶面源项函数在物理x端也计算x单边
横向项和z项，再做传入特征源项平衡；不能直接再叠加出口的全部x/z项。
它原已验证的平板配置使用x+21，不是本次x+22的联合特征边界。
需要单独冻结交线的源项所有权与特征相容处理，再检查该模式；此处是
源码依赖审计，不冒充尚未运行的nonreflecting数值失败或新公式验收。
完整三模式范围保留在总goal，不以二维首测替代或删除此项。

## nonreflecting交线的独立设计推进

当前CPU交线漏z项修复尚待批准，没有修改该循环，也没有重跑失败后的
三维矩阵。独立核对了Lodato等2008出版社摘要、2011出流角点研究摘要
及本项目顶面源项函数，形成几何候选第11节：先唯一装配已修订x法向、
原y法向和z项，再仅对交线总对流RHS施加原y传入投影闭合。它保留x面的
既有法向压力策略，但不额外承诺同时满足另一套完整x向零传入约束。
该物理合同未获批，不实施；完整黏性源项和掠流切换不由此冒称解决。

新增`review_keep_nonreflecting_corner.py`仅作独立5×5常状态代数检查，
不调用生产边界、不推进CFD。它确认x/y传入投影一般不交换，说明随意
先后执行两次投影不是同一算法；也验证候选y传入与传出投影恒等式至
2.23e-16。该证据不称数值稳定性、反射率或CPU/GPU实现验收。下一步
仍需用户批准明确的CPU修复及交线物理策略，而非自行修改规则后继续。

## 2026-11-11追加批准后的修复与复验

用户随后批准CPU交线漏z项最小修复，以及几何候选第11节的nonreflecting
组件验证。上两节的“待批准”描述为该批准前的记录，不是当前审批状态。

### 交线唯一展向装配

`bc:outflow_nscbc`仅对非反应五变量、低阶x+22/y+52、无激波的
compatibility／incoming_only组合，把右出口展向循环上限从je扩至jm。
顶面仍不装配物理x端的z项，因此该交线由出口装配一次既有D2通量差。
没有改NSCBC系数、目标、其他面或GPU通量公式。高阶边界与nonreflecting、
sbli_shock／sbli_grcbc不由此扩展。检查点复用既有driver字符串记录
`bc22-backflow-v1/yz-d2-v1`，CPU/GPU采用同一联合闭合身份。旧身份不能
当成同方法精确续算，显式非精确恢复仍沿用原入口，不增加文件或兼容层。

根CMake的CPU/GPU构建完成。实际CPU x+出口解析回查选亚／超声速、
NP=1/2四项，最大绝对误差4.45e-16。三维共有初场、16³、4步、dt=1e-5、
crashfix关闭，结果如下；每行是代表组合，不是全排列。

| KEEP对流／原黏性／主滤波 | bc52模式 | CPU/GPU及分解 | 最大场缩放差 | 首阶段边界／对流／完整RHS最大缩放差 |
| --- | --- | --- | --- | --- |
| 显式／643e／关 | compatibility | NP=1，NP=2 x/y/z，NP=4 xy | 9.53e-12 | 5.64e-16 |
| 显式／643e／关 | incoming_only | NP=1，NP=2 x/y/z，NP=4 xy | 9.53e-12 | 5.64e-16 |
| 紧致／643e／关 | compatibility | NP=1，NP=2 x/y/z，NP=4 xy | 9.53e-12 | 5.71e-16 |
| 紧致／643c／explicit10 | incoming_only | NP=1，NP=2 x/y | 1.33e-11 | 2.90e-11 |

均低于既定2e-10。各行CPU/GPU分别在NP=2 x检查2+2步精确恢复：所有
状态数据集逐值一致，control／archives／insitu_control逐字节一致。
各后端跨分解与NP=1比较的最大缩放差不超过9.47e-13。最后一行GPU
NP=1的Compute Sanitizer memcheck为0 errors。修复前三维边界差
8.35e-5降至约3.4e-16；短轨迹场差不是也降至该量级，表中另列实测值。

证据目录依次为`open_corner_fixed_compatibility`、
`open_corner_fixed_incoming`、`open_corner_compact_compatibility`、
`open_corner_compact_all_incoming_filter`；均位于
`results/keep_wall_low_20261110/`。解析出口回查为
`corner_fix_cpu_outlet_regression`。原失败证据保留。

### nonreflecting交线组件

`keep_nonreflecting_corner_probe.cuf`链接实际根构建的CPU/GPU边界组件。
测试先让出口装配原x法向及唯一z项，再借用顶面现有“保留原始法向波”
分支只提供交线Dy。该分支在测试中作为原始法向组件使用，不表示把
GRCBC选作新候选的物理边界。随后调用实际CPU/GPU传入源项投影组件，
形成`(I-Py_minus)*raw_rhs`。候选仍只在测试程序中，未接入主程序轨迹。

9³节点冻结状态覆盖恒定目标、v=-1e-4／0／+1e-4，x正向亚声速出流。
非恒定状态含x、y梯度和周期z变化。CPU/GPU原始装配绝对差<=1.76e-16，
投影后<=2.35e-16；y传入残差<=1.74e-18，y传出变化<=1.31e-18；
与独立理想气体声波投影公式差<=5.21e-18，恒定场RHS<=1.27e-16。
传入维数按原分类分别为4／1／1。此结果不证明跨零速度的连续性、
反射率或黏性非反射，也不扩大到x回流交线。

首版探针因NVHPC设备过程不接受非连续数组切片而编译失败，改为局部
连续3分量向量后重验通过；未修改生产公式。失败目录与通过目录
`nonreflecting_corner_components`、`nonreflecting_corner_components_v2`
均保留，后者`report.json`为实际组件证据。

### 新停止项：紧致滤波暴露compatibility的掠流不连续

显式KEEP＋643c黏性＋独立compact10滤波、compatibility的NP=1首阶段
失败，最大边界RHS缩放差3.098646610423591e-8，位于x+/y+交线的z=0
总能量分量。输入q／缓存缩放差仅2.76e-14／3.44e-14。矩阵在此停止，
未执行该组合的MPI、恢复或后续RHS通过声明。

冻结状态中，该节点CPU法向速度-8.802817927824877e-21，GPU为
+8.312434161897036e-20，均远小于已有波分类容差4.7371e-14。原两端
compatibility均以严格`un<0`选择声学松弛：负侧采用完整守恒状态差的
传入声波投影，非负侧采用压力差。两者在un趋零时通常不相等。于是
滤波求解的机器级符号差选择不同公式，不能据此归为KEEP通量漏项或
紧致滤波系数错误。

`analyze_keep_open_grazing.py`按该冻结状态、实际J和理想气体负声波
投影重算两分支之差。三个符号翻转节点k=0/8/16的RHS差均由该跳变
解释，扣除后最大绝对残差3.254e-15。数据保存在
`open_corner_independent_compact_filter/grazing_attribution.json`。

用户随后批准的最小修订：仅把CPU/GPU非激波compatibility的流入分支
判断统一到已有`un < -wave_tolerance`，零速容差带沿用现有压力分支；
不新增经验阈值，不改松弛系数，不剪裁速度或状态，不改紧致滤波。
这消除的是零速度舍入符号触发，不是一般有限流入跨零的连续物理模型。
incoming_only／sbli_shock的严格符号分类本轮仅记相关审计项，未修订。
当前源码已修改，复验结果在后续追加；原失败数据不覆盖。

追加冻结状态代数检查：令Pc为三个对流特征的投影，Pminus为负声波
投影，incoming_only在un趋零负／正两侧的松弛分别为
`J*sigma_in*(Pc+Pminus)*delta_q`和`J*sigma_out*Pminus*delta_q`。
对于本次相同热力学及切向状态，两者极限差最大1.027e-7，主要来自
对流子空间的恢复，而不是声学松弛系数的极小差别。它是当前公式的
代数预测，不是新增incoming_only轨迹的实测误差，不能据此撤销前述
已通过矩阵。证据另存于`grazing_one_sided_limits.json`，未覆盖原归因。
本次批准未扩展到incoming_only，不因这个审计结果自行修改其公式。

后续仍需：nonreflecting候选主程序接线及明确的物理观测窗／判据，
一般开放端滤波联合检查、固定物理时刻解收敛与高阶相容闭合。总goal
未完成；没有Git写操作、平台任务或生产默认替换。

## compatibility最小修订复验、恢复身份与周期示例

2026-11-11用户批准最小修订后，CPU/GPU compatibility已统一采用
`un < -64*epsilon*max(a,abs(un))`选择流入分支，零速带采用原压力分支。
只有该选择条件改变，没有新增阈值、松弛系数或连续权重；incoming_only
及激波模式未随此修订。这不是一般流入跨零的连续松弛模型。

### 已通过的受影响检查

- 根CMake CPU/GPU构建完成。实际边界核解析检查18项：NP=1/2，零速、
  +/-1e-20、容差带内/外、亚/超声速流入/流出；其中CPU参考也调用实际
  compatibility特征函数。没有把CPU/GPU相同代替解析结果。
- 原失败的显式KEEP＋643c黏性＋compact10滤波，在NP=1、NP=2 x/y/z、
  NP=4 xy下通过，dt=1e-5、4步，crashfix关闭。原交线RHS最大绝对差
  从3.10e-8降到3.254e-15；全域边界/对流/完整RHS缩放差最大6.839e-12，
  CPU/GPU末场最大1.426e-11，各后端跨分解最大3.009e-14。
- 同一组合的实际device-aware solution halo，NP=1、NP=2 z检查通过；
  每个GPU rank均记录`ASTR_GPU_HALO_DEVICE_AWARE ... active=T`，场差
  最大1.426e-11。没有把该结果解释成紧致PCR接口也使用同一通信后端。
- 以上pinned的NP=2 x、device-aware的NP=2 z，CPU/GPU各自2+2步精确
  恢复状态数据集逐值一致、三个控制文件逐字节一致。pinned NP=1 GPU
  的memcheck为0 errors。
- 紧致KEEP＋643e黏性、无滤波的NP=1/2 y也通过，场差最大9.529e-12，
  RHS最大5.707e-16，同拓扑精确恢复一致。

证据分别为`compat_grazing_analytic`、`open_compact_filter_grazing_fixed`、
`open_compact_filter_grazing_aware`、`open_compact_unfiltered_grazing_fixed`。
原交线的单独复核另存`open_compact_filter_grazing_fixed/postrun_rhs_recheck`，
没有覆盖原失败或首轮结果。根目录仍为`results/keep_wall_low_20261110/`。

### 恢复方法与有效目标值

compatibility修订以`bc52-compat-grazing-v2`写入既有driver控制槽，CPU/GPU
统一。原strict-sign检查点的精确恢复被拒绝；显式非精确恢复沿用既有
compatibility身份迁移，并完成到第4步、保存新身份。CPU/GPU均检查了
这两种行为，见`compat_grazing_restart_migration_v2/report.json`。第一次
测试遗漏conschm覆盖而得到其他格式准入拒绝，未误当成目标身份拒绝。

还发现本次KEEP新接线未记录incoming_only有效远场目标：实际零推进
探针从第4步恢复到第4步，把顶面目标u从1改为1.01，原程序接受精确
恢复，所有状态数据集仍逐值一致。首次探针误改输出间隔先被调度拒绝，
保持原间隔后才得到该复现。证据为`keep_open_target_identity_before_v2`。

现已在同一driver槽写入`keep-top-in-sign-v1`及rho/u/v/w/T的五个精确
位模式，不新增附属文件。当前nonreflecting面组合另有
`keep-top-nr-faces-v1`标签，不把它称为已批准的新交线闭合。没有给缺少
这些新身份的旧KEEP原型检查点新增兼容读取或猜测默认值。

复验使用此前通过的紧致KEEP＋643c黏性＋explicit10、NP=1/2 x：
CPU/GPU场差最大1.330e-11、RHS最大2.896e-11，同配置2+2精确恢复仍逐值
一致；把顶面目标u改为1.01时，CPU/GPU都明确拒绝精确恢复。
证据为`open_incoming_target_identity/report.json`。这些身份检查不代表
incoming_only的紧致滤波掠流问题已修复。

### 可运行周期输入

新增`examples/KEEP_Taylor_Green_Vortex`，含独立input.case/input.run/
input.output及README，并接入根examples的CMake安装。它使用内置TGV
初值、16³区间、Re=1600、Ma=0.1、固定dt=1e-3、4完整步，滤波/统计/
crashfix关闭，第2/4步保存检查点，不需要外部初场或网格。

`run_keep_example_gate.py`直接复制这些交付输入，不另造测试模板。
显式/紧致KEEP、原643e黏性、CPU/GPU NP=2 x检查通过，场差分别为
4.767e-16和4.778e-16；四个连续/2+2步对照的全部状态数据集与三个控制
文件均一致。测试曾发现示例必须显式给出输出缓冲配置并预建空输出根，
已修订示例与启动说明，没有改求解器文件系统行为。最终结果为
`keep_installed_example_v3/report.json`，安装文件也逐字节对照源码。
该示例不关闭KP8整阶段，也不称为已解析的TGV或生产DNS。

### 新停止项：incoming_only紧致滤波的掠流分支

compatibility修复后，按KP6-L继续检查尚未修改的incoming_only，同样的
显式KEEP＋643c黏性＋compact10、NP=1首先在边界RHS失败，最大缩放差
1.026603819166421e-7，位于(16,16,8,密度)，q/缓存输入差仍为
2.76e-14/3.44e-14。矩阵立即停止，未进入该组合MPI或恢复。

CPU/GPU对机器级负/正法向速度采用严格`lambda<0`，负侧因此额外激活
三个有限恢复率的对流特征。用该次实际冻结状态重算投影跳变，最大扣除
残差3.235e-15，量值与先前独立预测一致。证据为
`open_incoming_compact_filter_probe/grazing_attribution.json`；分析器明确
区分实测RHS与单侧极限，不把代数预测当成轨迹验收。

用户已单独批准：仅非激波incoming_only，CPU/GPU复用已有64 epsilon容差，
`lambda < -tol`才判为传入，零速带采用原出流侧松弛系数；容差带外、
sigma公式与0.25系数不变，不连续化，不改sbli_shock等模式。
这不是compatibility批准范围的自动延伸；该批准后的复验及后续接线结果如下。

## 2026-11-11：分类修订、所有权与nonreflecting主程序短测

以下路径均相对于`results/keep_wall_low_20261110/`，原失败证据不覆盖。

### incoming_only复验与出口底壁所有权

非激波incoming_only统一已有64 epsilon波分类容差；激波标志分支保留
严格符号，未改sigma或0.25系数。`incoming_grazing_analytic`的22项
NP=1/2解析检查通过，实际CPU辅助函数、GPU核及独立公式最大差
7.161e-15，包含兼容模式和激波标志回归；不是入射激波物理验证。
有效目标五个精确位模式与`bc52-incoming-grazing-v2`写入既有控制槽。

冻结RHS还定位到KEEP出口与底壁交点：GPU多加了原CPU不计算的Dz压力项，
差6.83803e-12，扣除实际映射Dz压力后残差2.982e-18。仅修GPU的KEEP
物理底壁所有权，非物理MPI接口保留z项，通用非KEEP出口仍沿用原行为。
身份追加`/wall-z0-v1`，不新增全场数组或通信。

- `open_incoming_wall_ownership_fixed`：显式KEEP＋643c黏性＋compact10，
  NP=1、NP=2 x/y/z、NP=4 xy，4步dt=1e-5，场最大绝对差3.301e-14，
  完整RHS最大缩放差1.978e-14，memcheck为0 errors。
- 同配置CPU/GPU各自2+2步精确恢复逐值一致；改变顶面目标u后拒绝精确恢复。
- `open_compatibility_wall_ownership_aware`：相同空间组合、compatibility，
  NP=1/2 y，实际device-aware每rank均active=T，场差2.865e-14，精确恢复一致。
- `open_incoming_grazing_aware`：紧致KEEP＋643c黏性＋explicit10、NP=1/2 z，
  实际device-aware及恢复通过。该结果不表示PCR接口使用相同通信后端。
- `keep_wall_corner_generic_regression`的8项通用出口解析回归通过，
  `nonreflecting_corner_components_v3`按新接口重链后组件检查通过。

另按批准在`grichan`循环前初始化`dymax=0`。`channel_dymax_fixed`中打印的
名义ymax+=41.2863742与坐标重算41.2863742016593相符；13个几何数据集和
五个守恒场与修订前逐值一致。该值使用网格生成参考Retau，不是实测壁面单位。

### nonreflecting交线接入与已批准短测

用户批准几何候选第11节主程序接线。CPU/GPU先唯一装配已分类的Dx、
原始Dy和一个Dz，再对交线总对流RHS作`(I-Py_incoming)`投影。上入口
顶角仍由剖面入口拥有，底壁沿用原CPU低阶规则。原643e黏性不进入该
投影，不改变非KEEP、默认边界模式或激波模式。控制身份为
`keep-top-nr-corner-conv-v1`，不兼容旧原型的精确身份。

实现位于`bc:apply_keep_nonreflecting_corner_rhs`、
`boundary_gpu:keep_nonreflecting_corner_rhs_kernel`及各自主循环。
只增加交线上局部五分量投影，不增加全q/RHS副本或MPI协议。

| 检查 | 显式KEEP | 紧致KEEP |
| --- | --- | --- |
| 原黏性／滤波／修复 | 643e／关／关 | 643e／关／关 |
| 规模和时间 | 16³区间，dt=1e-5，4完整步 | 相同 |
| MPI | NP=1、NP=2 x/y/z、NP=4 xy | NP=1、NP=2 y |
| CPU/GPU末场最大绝对差 | 1.4211e-14 | 2.1317e-14 |
| 完整RHS最大缩放差 | 4.5890e-16 | 4.4885e-16 |
| 交线独立公式最大差 | 1.0560e-16 | 1.5103e-16 |
| 传入残差最大绝对值 | 2.033e-20 | 1.695e-20 |
| 入口顶角对流边界RHS | 严格为0 | 严格为0 |
| 同后端同拓扑2+2恢复 | CPU/GPU各自状态及控制文件一致 | 相同 |

证据：`open_nonreflecting_corner_native`与`open_nonreflecting_corner_compact`。
显式NP=1的Compute Sanitizer为0 errors；GPU短测启用已有只读RK状态及
KEEP RHS输入检查，无修复或限制器。独立分析器
`analyze_keep_nonreflecting_corner.py`用冻结状态和实际度量重算Dx/Dy/Dz
及理想气体投影，同时检查传入为零、传出保持和入口顶角所有权。
这关闭本次批准的接线短测，不关闭反射率、倒流／斜向物理窗口、
完整黏性特征源项或生产采用。

## 固定时刻壁面自收敛与紧致壁面稳定性未关闭项

### 已测量的固定时间网格序列

`run_keep_wall_time_convergence.py`调用原生ASTR推进，不以Python求解PDE。
同一初场在N=16/32/64/128上GPU NP=2 x，dt=1e-4、100步到t≈0.01；
N=32另做CPU对照及dt减半。原643e黏性／ME6分别独立选择，滤波与修复关闭。
使用J权重、周期唯一节点和壁面端点半权，报告L1/L2/Linf相邻网格差。

最细两组相邻网格差的表观L2阶，按rho、rho*u、rho*v、rho*w、rho*E排列：

| 对流／黏性 | 表观阶 | N=32 CPU/GPU最大绝对差 |
| --- | --- | --- |
| 显式KEEP／原643e | 2.101、1.423、2.550、1.534、2.062 | 3.997e-15 |
| 紧致KEEP／原643e | 2.225、1.150、2.316、2.607、2.185 | 5.083e-15 |
| 显式KEEP／ME6低阶壁面 | 2.123、1.302、2.553、0.431、2.098 | 2.776e-15 |

证据为`wall_fixed_time_explicit_standard`和`wall_fixed_time_compact_me6`。
这些是自收敛测量，不是解析解误差。初场固定壁温且壁面剪切非零，
初始黏性发热与固定壁温的一阶时间相容性不满足；t≈0.01的热扩散层
估计约0.0018，小于N=128首层间距约0.0036。有限时间步误差也另行报告。
因此尚不能由这些表观阶唯一归因边界闭合，或判定进入渐近空间收敛区。

### 紧致KEEP低阶壁面长窗：直接失败链与限制

16³、NP=2 y、dt=0.001、原643e黏性、滤波和修复关闭时，紧致KEEP
出现近壁压力放大。实际冻结状态重算确认：底壁(4,0,8)处
`p1=7.752851222463`、`p2=31.418431120811`均为正，但原闭合
`p_wall=(4*p1-p2)/3=-0.135675410320`。CPU/GPU均复现该负壁压和负密度，
不是单独的GPU移植误差。它确定了直接触发环节，尚未确定扰动增长根因。
该计算坐标Neumann外推本身不因y拉伸就自动成为编码错误；改为限幅或
对数压力闭合属于数值方法变更，未自行实施。

`compact_wall_cpu_failure_capture/failure_analysis.json`与
`compact_wall_gpu_early_failure/failure_analysis.json`保存nstep=533/RK1状态。
随后启用已有测试专用只读阶段检查，在
`compact_wall_stage_validation_off`更早于nstep=532/RK2的KEEP RHS输入
检出非法状态，不把533称为全轨迹首次非法时刻。

两种旧停止时刻不同也已定位：GPU末步会刷新物性，并非物性缓存陈旧；
其声学CFL检查速度／温度而不单独判断rho/p，两者同时为负仍可得到正温度。
旧CPU crashcheck在GPU路径通常读未回传的主机q，诊断快照回读后才及时看到
负密度。CPU原停止流程还可能返回0并打印job done。未修改这些生产检查
或退出合同，未把零退出码当作稳定性通过。GPU已发布的581步检查点含
负状态，不可称为最后合法检查点；见`compact_wall_checkpoint_admissibility.json`。

采用同一原生夹具、原黏性及已有只读RK／RHS输入检查的隔离对照：

- `explicit_wall_standard_admissibility`：显式KEEP、无滤波，1000步到t≈1，
  rho范围[0.996880,1.003089]，状态检查通过。
- `compact_wall_explicit_filter_admissibility`：紧致KEEP＋explicit10，1000步
  通过，最大C_upper=0.165332，末场rho范围[0.174280,1.894075]。
  扰动放大仍明显，不作为已解决长期稳定性的证据。
- `compact_wall_compact_filter_admissibility`：紧致KEEP＋compact10，在
  nstep=621/RK2的RHS输入检出负密度，未通过该窗口。

没有改压力闭合、滤波系数、时间步、生产默认，或用crashfix覆盖失败。
新增测试接线仅复用已有显式启用的只读检查，默认关闭。后续先定位紧致
低阶壁面增长机制；任何新闭合／稳定化项须提交具体方案由用户决定。
高阶相容闭合、开放端物理判据及联合收口仍未完成，总goal保持未完成。

## 暂跳过低阶壁面问题后的推进：周期RK收支与开放端方法迁移

用户已要求暂跳过上述紧致低阶壁面问题，继续其他计划项。该失败保留为
未关闭项，不改压力外推、系数或滤波，不再扩大该长窗。暂停前保存的
`wall_linear_acoustics_n16`是实际导数矩阵组成的简化无黏法向声学分析，
不是完整黏性／非线性稳定性证明；本轮未据其修改数值方法。

### KP0／KP7：完整一步的分项记账

新增`analyze_keep_rk_budgets.py`只读实际三阶段快照，使用唯一周期节点与
J权重，记录五个守恒量、动能、内能和熵密度的体积平均值。源项关闭的
TGV中，`full-conv`单独代表黏性；不将该解释用于有驱动或其他源项的算例。
原气体定义gamma=1.4，动能为`sum(momentum**2)/(2*rho)`，内能为总能减动能，
熵密度采用`rho*(log(p)-gamma*log(rho))/(gamma-1)`，与既有冻结收支脚本一致。

实际RK3系数为`(a,b,c)=(1,0,1),(3/4,1/4,1/4),(1/3,2/3,2/3)`。
设Phi为上述任一平均量，B为第一阶段RHS输入，U为各阶段RHS输入，V为
各阶段更新后状态。逐阶段检查原生q是否满足
`V_s=a_s*B+b_s*U_s+c_s*dt*R_s/J`，最大缩放差门槛2e-10。
非线性时间离散余量直接测为
`D_s=Phi(V_s)-a_s*Phi(B)-b_s*Phi(U_s)-c_s*dt*Phi_dot_s`。

完整一步的对流／黏性变化率权重均为`(1/6,1/6,2/3)*dt`；三次滤波及
滤波后周期准备变化的权重为`(1,1/6,2/3)`；D_s权重为`(1/6,2/3,1)`。
这能将真实一步变化分解为对流、黏性、滤波、周期准备和RK非线性余量。
记账闭合不是新的能量稳定定理：不要求动能或熵的D_s为零，也不以定义式
闭合代替已有独立压力功／内能关系检查。

先复用`keep_kp1_frozen_rhs_initial_20261110`及
`keep_kp2_frozen_rhs_initial_20261110`，不重跑无滤波算例。显式／紧致KEEP
加各自原黏性、16³、dt=0.001的第一完整步，CPU/GPU八项重构残差均
<=1.422e-14。证据为本目录下`rk_budget_explicit_unfiltered.json`、
`rk_budget_compact_unfiltered.json`。

滤波前后快照仅补入`ASTR_BUILD_TESTING`下、显式请求RHS快照的KEEP路径。
生产构建／未请求快照时不新增回读或数组。`run_keep_rk_budget_gate.py`
仅补两组4步、dt=0.001、NP=1的原生对照，各做快照开／关：

| 组合 | CPU/GPU末场最大缩放差 | 分项平均变化最大后端差 | 八项闭合残差上限 |
| --- | --- | --- | --- |
| 16³均匀，显式KEEP＋643e黏性＋compact10 | 2.413e-14 | 1.516e-13 | 2.369e-14 |
| 32³ G1a曲线，紧致KEEP＋ME6＋explicit10 | 1.555e-15 | 1.895e-14 | 1.895e-14 |

两组CPU/GPU各自快照开／关的全部状态数据集逐值一致。人为只改只读状态
副本中一个节点的更新后能量1e-5，重构检查明确拒绝；原始数据未修改。
没有以跨构建逐位相同代替同构建的输出开关检查。

实际物理分量也分别报告。以CPU第一步为例：

- 均匀compact10组合，黏性动能变化-4.69263e-7，滤波动能变化
  -1.15267e-9；对应滤波内能增加约1.15270e-9，总能滤波变化在舍入量级。
- 曲线ME6组合，黏性总能变化-1.65020e-12，滤波总能变化-5.68434e-13，
  均未扣除或强制归零。ME6净量仍按此前批准仅记录，不能称严格总能守恒。

证据：`rk_filter_native_budgets/report.json`，另用既有独立冻结分析器检查
两组RK3的对流守恒／压力功／内能关系，原节点黏性应力功和熵SBP关系
也通过；ME6的节点应力功仅作不同算子参考，不套其零残差要求。

初次调用通用周期测试时误加`--rhs`，其逐位同输入前提被独立紧致滤波后
4.633e-12的绝对状态差拒绝，记录保留在`rk_filter_explicit_compact10.log`。
没有放宽该函数；改用独立滤波轨迹的既定缩放门槛和各自原生RK恒等式，
并以同构建快照开／关对照确认诊断不改变推进。

### KP7：新nonreflecting交线身份下显式／紧致非精确迁移

`run_keep_open_migration_gate.py`只读复用已通过的显式／紧致nonreflecting
16³、NP=2 y、step=2检查点，保持原643e黏性、dt=1e-5、无滤波／修复，
分别测试E→C、C→E，CPU/GPU各自完成：

- 有效算子变化时精确恢复明确拒绝，不将跨方法轨迹称精确续算。
- 显式选择非精确恢复后，实际恢复探针的五个守恒数组与源检查点逐值一致。
- 记录`operator identity changed; derived state rebuilt`；推进到step=4后
  控制记录写入目标o24/o34，并保留`keep-top-nr-corner-conv-v1`及wall-z0身份。
- 末场严格有限、rho／内能为正，CPU/GPU最大绝对差分别为1.777e-14、
  2.487e-14，最大缩放差不超过1.223e-15。

证据为`open_nr_operator_migration/report.json`。该短测统计关闭，不将其
当作有历史统计的新窗重置验证；此前各项同方法精确恢复证据继续保留。
本轮不改任何恢复协议、生产默认或数值公式，不操作Git或平台任务。

当前仍需具体设计／审批的主项：开放端声学观测窗与物理门槛、高阶相容
壁面／开放端闭合；物理边界分项收支、历史统计联合迁移及KP8能力交付
按相应依赖继续。低阶紧致壁面失败只是暂跳过，总goal尚未完成。

### KP7历史统计联测：方法切换未开新窗（批准修订前的复现）

沿第5.4节检查历史统计时发现公共恢复流程与计划合同不符，已停止该项
后续验收。`run_keep_statistics_migration_probe.py`使用CPU、NP=1、16³周期
TGV，原643e黏性、无滤波／修复、dt=0.001。显式KEEP先完成4步并保存
精确检查点及时间加权统计，再显式选择非精确恢复到紧致KEEP并推进2步。

| 状态 | 首采样步／时刻 | 末采样步／时刻 | 样本数 | 覆盖时长 |
| --- | --- | --- | --- | --- |
| 源检查点 | 0／0 | 4／0.004 | 5 | 0.004 |
| 实际迁移后 | 0／0 | 6／0.006 | 7 | 0.006 |
| 本计划要求 | 4／0.004 | 6／0.006 | 3 | 0.002 |

`output_runtime:compact_control_file`已判定方法改变并重建派生状态，但
没有将该事实传递给稍后的统计恢复判断。`mean_statistics_history`仍读回
原采样调度，`begin_output_runtime`在精度／统计路径等条件一致时载入旧
累计量。由此将E和C的两段轨迹混入同一时间平均，不能按本计划验收。
流场和终止时间并未因此失败；本项为统计生命周期缺口。

证据为`statistics_method_migration_same_schedule/report.json`，明确标为
`failed_new_statistics_window`，不是通过报告。首次探针额外改了检查点
间隔，被原输出调度一致性检查拒绝；已保持原间隔再独立复现，不把首次
启动拒绝误报为统计问题。原始两份记录均保留。

待批准的最小修订：仅在涉及KEEP且有效数值身份确实改变的非精确迁移中，
不读取旧累计统计／端点，重置公共采样时钟，在恢复完整步建立新统计窗。
保留权威场、物理时间和累计完整步数；同方法非精确恢复、精确恢复、
AIR5及不涉及KEEP的既有策略不顺带改动，不新增输入项或附属文件。
CPU已复现，GPU因公共入口也需代表复核，尚未声称GPU实测重现。
批准后先验证CPU/GPU方法切换新窗，再核对同方法恢复未改变。
当前未修改任何生产统计逻辑，总goal保持未完成。

## KP0-A／KP5后续：高阶相容算子的数学分析

当时统计新窗修订仍待审批，该轮未修改公共恢复逻辑或启动后续验收；
之后的批准与实际修复见本文末节。
独立推进几何候选第12节：从根构建实际导数提取矩阵，确认原低阶显式、
原紧致及已有高阶导数均不能直接配正对角H满足标准HD+D^T H=B。
已有高阶导数六阶多项式一致性仍成立，不能把此审计说成系数错误或
Hermite7整体失稳。具体排除见证及适用范围已写入候选文件。

随后按既有内部D6与精确SBP约束，推导端部六行、最多九节点的显式
6/3候选族，保留一个未选theta。16／32区间的SBP恒等式、边界三阶／
内部六阶多项式一致性、六阶积分、标量流入能量参考均有精确有理数结果。
原算术平均KEEP的五分量冻结通量在一组有理数壁端零速度状态上，
守恒量边界通量及动能／内能压力功交换残差也精确为零。

证据：`high_order_diagonal_sbp_audit/report.json`与
`high_order_sbp63_family.json`，均在`results/keep_wall_low_20261110/`。
分析入口为`audit_keep_boundary_sbp.py`及`study_keep_sbp63_family.py`。
没有选定新的生产算子、SAT系数、壁压／温度状态处理或主程序准入，
也未修改src/src_gpu。紧致高阶端部、完整NS壁面／黏性、滤波与几何
仍需分别设计；此数学候选不作为KP5或总goal完成证据。

## KP8分项交付说明同步

已将当前KEEP／ME6选项、独立主滤波、周期／低阶壁面／开放端范围及
未关闭项同步至`USER_GUIDE.md`第9.1节和发布能力说明第4.2节。
周期示例README明确方法切换的历史统计缺口，避免将场恢复通过误称
统计新窗也通过。使用说明将高阶SBP候选标为未接入、无输入选项；
没有扩大现有准入或改动冻结交付分支。文档检查复用已有数值证据，未重跑算例。

## KP7历史统计新窗：批准后的最小修复与代表验收

用户已批准涉及KEEP的方法切换新窗修订，goal恢复推进。只修改
`output_runtime`公共恢复入口：读取既有ASTRCM01算子身份时，记录非精确
请求下的身份不匹配，且源或目标涉及KEEP。此时跳过旧统计累计量／端点
的恢复，复用已有统计调度重置和原位新段初始化。源为原显式路径且没有
紧致身份尾部时，显式选择新KEEP同样建立新窗；没有新增兼容读取逻辑。

不匹配包括已记录的有效对流／黏性、滤波、几何与闭合标签。只有当前
调用涉及KEEP才改变统计生命周期；同方法精确／非精确请求及非KEEP
策略不顺带改变，AIR5由原分支控制。新增状态仅为私有逻辑标量，不改变
流场、RK、物理时间、累计步数、检查点格式、输入项或附属文件。
新窗日志明确写出`ASTR_KEEP_RESTART statistics_new_segment_step=... time=...`。

### 已通过的检查

使用16³周期TGV、dt=0.001、原643e黏性、滤波／修复关闭。原检查点带
历史统计，输出计划保持不变，不再把改输出间隔的拒绝当成统计验证。

| 通道／场景 | 拓扑 | 实际结果 |
| --- | --- | --- |
| CPU原时间加权44项统计，E→C | NP=1/2 x | step=4恢复，窗口4--6，3个样本、覆盖0.002，不再保留0--4旧积分 |
| CPU/GPU原位Reynolds／Favre时间统计，E→C | NP=1/2 x | 新段起点step=4、t=0.004，覆盖0.002 |
| 同方法精确恢复 | 上述代表通道与拓扑 | 状态及统计全部数据集与连续运行逐值一致，保留原0--6历史 |
| 同方法非精确恢复 | 同上 | 保留原0--6统计历史；沿用既有非精确场重建／共享节点平均 |
| CPU/GPU原位，C→E、KEEP→原格式 | NP=1 | 从step=6新建段，覆盖6--8、时长0.002 |
| CPU/GPU原位，原格式→KEEP | NP=1 | 从step=8新建段，覆盖8--10、时长0.002 |
| 不涉及KEEP的同方法非精确对照 | NP=1 | 不重置：保留6--10历史，时长0.004，未输出KEEP新窗标记 |

CPU原均值的前12项由真实q4/q5/q6状态独立梯形求积，除覆盖时长后的
最大缩放差6.662e-16。CPU/GPU原位均值及协方差由相同三个状态独立
两遍加权计算，NP=1最大绝对差2.777e-16；两端新统计的NP=1/2直接
比较最大绝对差4.441e-16。后续三种方法切换最大统计差6.662e-16，
末场CPU/GPU最大绝对差1.137e-13，均在原门槛内。

只读复查还确认：原错误混算轨迹与修订后CPU E→C轨迹的五个守恒数组
逐值相同，统计覆盖由0.006正确改为0.002。不是靠修改流场获得新窗结果。

### 证据与测试边界

路径均相对于`results/keep_wall_low_20261110/`：

- `statistics_keep_window_fixed_v2`内CPU NP=1已完成的原均值记录，及
  `statistics_keep_readonly_review.json`的独立重算。该早期整组曾因GPU
  TGV未准入旧CPU统计而停止，不把它标为整组通过。
- `statistics_keep_mean_np2/report.json`：CPU NP=2原统计检查。
- `statistics_keep_insitu_fixed_v2/report.json`：CPU/GPU NP=1/2原位统计、
  同方法精确／非精确对照及独立均值／协方差参考。
- `statistics_keep_switch_scope/report.json`：方法切换两侧及非KEEP对照。
- `statistics_keep_field_before_after.json`：修前／修后CPU守恒场只读比较。

新增探针最初误要求非精确恢复逐位不变。实测CPU非精确恢复已有qswap
会平均周期重复节点，差<=1.858e-16，严格内部不变，且探针写出早于统计
重置。未修改该恢复算法，改用本计划原有非精确字段缩放差2e-10合同。
GPU周期TGV不支持旧CPU mean44通道，GPU测试显式选择现有原位统计，
没有新增准入或运行时回退。旧原位provider不接受-1预算输入，测试显式
传入项目既有unlimited_budget_bytes规范值，不修改解析器或自设资源上限。

CPU/GPU根CMake构建完成。测试源码语法与差异空白检查完成；本次没有
新增核函数、通信或大数组，不为公共标量恢复条件重复整套内存／性能测试。
新窗缺口在上述批准范围内已关闭。低阶紧致壁面问题仍暂跳过，高阶相容
闭合、开放端物理门槛及完整KP0--KP8交付仍未完成，没有Git或平台操作。

### 独立发现的后续接线范围：开放端旧时均统计

为了覆盖GPU平板的专用展向归约累计量，新增
`run_keep_open_statistics_migration.py`准备在已批准16³、11/22/41/52
nonreflecting、原643e黏性、dt=1e-5、NP=2 x、无滤波／修复夹具上，
检查CPU mean44与GPU展向统计的窗口及前12项展向积分。

首次模板漏填conschm，被KEEP格式检查拒绝，已补为643e。随后在正常
配置下，CPU尚未推进即被`output_runtime:check_capability`中原有
`profile joint NSCBC admits same-topology restart; means/adaptive not validated`
拒绝。该检查同时覆盖KEEP开放端，明确禁止lavg，不是已启动后的数值失败。
原始日志分别保存于`statistics_keep_open_compact`和
`statistics_keep_open_compact_v2`，均不作为通过报告。

没有删除该条件、自动关闭统计或切换边界。周期新窗修复的代表验收已经
完成；若继续接通此联合通道，需另行批准KEEP专属范围，保留非KEEP和
自适应输出限制，并检查统计开关不改变推进、同方法精确恢复及CPU/GPU
展向归约定义。此项当前仅为待审阅的工程接线与短测，不扩展物理验收。

### KEEP开放端旧时均统计：单独批准后的接线与短测

用户随后批准上项范围。`check_capability`仅将联合NSCBC的旧时均条件
由禁止lavg改为允许KEEP；`.not.options%adaptive%enabled`不变，非KEEP
仍禁止此联合旧统计。没有修改NSCBC公式、通量、滤波、统计内核、MPI
协议或默认开关，也不新增运行时替代路径。

既有16³三维弱扰动、11/22/41/52 nonreflecting、原643e黏性、NP=2 x，
dt=1e-5、6完整步，滤波／crashfix关闭。源为显式KEEP，step=4检查点，
另做同方法精确／非精确到step=6，以及非精确E→C到step=6。

实际结果：

- CPU mean44及GPU专用展向累计量均保留同方法0--6的7样本、时长6e-5；
  改变方法后都以step=4、t=4e-5开始新窗，3样本、覆盖2e-5。
- 两端各自统计开／关的五个守恒数组逐值相同；同方法精确恢复的全部
  状态及统计数据集与连续计算逐值一致。
- E→C末场CPU/GPU最大绝对差2.132e-14。将CPU的前12项原时间积分按
  冻结物理z坐标作梯形展向积分，再与GPU平面累计量比较，除以窗口时长
  后最大缩放差5.596e-16。比较明确对应GPU的rho*T／p顺序与CPU的p／rho*T
  顺序，未把相同字段编号视为相同物理量。

全部原生运行位于`statistics_keep_open_mean_enabled/`。原汇总脚本曾在
运行完成后尝试读取非精确目标中按设计不保存的geometry.h5，因分析路径
错误停止；没有改检查点内容或重跑CFD。修正为显式读取源运行的冻结
几何，只读重核开关、精确恢复、统计窗口、场差和展向积分，结果保存在
`statistics_keep_open_mean_review.json`。原运行日志与不完整汇总均保留，
不把前者的脚本退出状态称为数值失败或整组通过记录。

本项关闭批准的接线短测，不关闭声学反射、完整黏性特征源项或长期
统计定常。其他顶面模式与新时均的联合尚无本轮同等短矩阵，不能由此
一并声称其长期统计已验。低阶紧致壁面问题仍暂跳过，高阶相容闭合及
完整KP0--KP8交付仍未完成。无Git写操作、平台任务或新资源预算。

追加只复用原step=4检查点的GPU NP=2 x紧致KEEP两步memcheck，两个rank
均为0 errors，插桩与原生运行的状态及统计数据集逐值一致，见
`statistics_keep_open_memcheck/report.json`。未重跑整组矩阵。
保留的自适应限制指`input.output`的事件自适应输出，不是外层CFL步长
控制；本次统计短测固定dt，不据此增加变步长联合验收结论。

## KP6-L物理门槛设计：OPEN-P1参数复核及批准

现有同状态、MPI及统计恢复结果不能替代声学出入检查。已核对旧曲线
声学脚本，发现其边界组合不属于当前KEEP准入，且旧黏性包装没有有限
绝对反射门槛。因此未直接执行旧脚本或将其1e30门槛套到新计划。

几何候选第13节新增具体方案：显式KEEP＋原643e黏性，保留11/22/41/52，
64×128×16、dt=1e-4、2300步到无量纲t=0.23，NP=2 y；三顶面模式分别
配无扰动基线，扩域参考隔离边界返回，nonreflecting再做y细化。提出的
R<=5%、入射一致性1%及网格差<=0.01随后已获用户明确批准，
批准后按下述顺序取得了该窗口的执行结果。

参数审阅阶段只执行了解析复核：等熵／Riemann初始化残差<=3.553e-15，入射与
返回观测窗分离，扩域返回的线性预计时刻晚于终止时间。CFL数值为估计，
在参数复核阶段没有原生ASTR物理窗口、CPU/GPU物理验收或新高阶闭合准入。具体初值、
归一化、二维输出、参考定义、文献和执行顺序均在候选文件中可查。

用户批准后接入`run_keep_open_acoustic.py`及原生索引平面分析入口。
`test_keep_open_acoustic_analysis.py`的合成信号检查返回振幅3%、各自漂移
基线／扩域扣除及时间错位拒绝，两项通过。首次预跑因测试生成器把
input.output组名与取值放在同一行，被现有输出解析器拒绝，尚未推进；
只修正测试文件写法，未改生产解析器，原记录保留在`open_acoustic_p1/`。

原生四步对照位于`open_acoustic_p1_native/short_cpu`及`short_gpu`：
FP64、64×128×16、NP=2 y、nonreflecting，选step=0及3的三个RK阶段、
两个rank，比较物理节点q及boundary/conv/full RHS，共60项，最大缩放差
7.772e-16，小于2e-10。boundary RHS明确转换CPU存储符号；未比较物理外
无定义角区，也未将逐轨迹一致性当作声学透射证明。实测最大C_upper
0.06677384。随后完成8个粗网格及4个细网格GPU窗口，无量纲终止时刻
均约0.23。粗／细窗口分别有461／921帧完整步二维探针，采用实际文件时刻
积分；物理窗口不输出检查点、三维场序列或渲染图片。

| 模式／网格 | R | 入射积分相对差 |
| --- | --- | --- |
| nonreflecting，64×128×16主域 | 0.001318135667 | 8.925072e-7 |
| compatibility，64×128×16主域 | 0.018023080726 | 8.925129e-7 |
| incoming_only，64×128×16主域 | 0.001332488566 | 8.925315e-7 |
| nonreflecting，64×256×16主域 | 0.001281578631 | 1.553178e-8 |

nonreflecting两级均R<=0.05，差3.655704e-5<=0.01；所有模式入射积分相对差
均<=0.01。粗／细最大实测C_upper分别为0.06686565／0.05480658，
GPU每rank完整步可容许域记录齐全，未启用滤波或crashfix。
`open_acoustic_p1_native/report.json`状态`passed`，各`*_analysis/metrics.json`
及`characteristic_history.csv`保存实际窗端、积分与逐时功率；未改写原始平面。

本轮没有修改生产Fortran／CUDA公式或默认值，无需重建原生二进制。
执行时HEAD为`2567532f9eea595ff88753fa51216776fbd4da07`，工作区包含此前
KEEP未提交实现，不能把结果归为该干净提交。二进制SHA256：CPU
`ca96204dedb5735f8c81d0328adfb10fc01ba5a65ec12f27f776fa44e26c4e54`，GPU
`d56fdaaaa357427c44f234e3515cd70b6a4f1abd2d87542a1a93ee26755db134`。
每运行目录保留原生输入、初场、网格、命令及所选运行环境。

OPEN-P1仅关闭现有显式KEEP、原643e黏性、11/22/41/52的指定弱法向声学
窗口。三种模式不等价，返回振幅比不等于单频边界本征反射系数。
紧致／斜向声学、交线局部脉冲、完整黏性特征源项、物理端完整离散收支
及高阶相容闭合仍开放；低阶紧致壁面失败仍按用户决定暂跳过。
全KP0--KP8目标未标记完成。未做Git写操作、平台任务或新增资源限制。

## 低阶边界工程兼容收口，2026-11-11

用户要求先完成KEEP现有低阶边界兼容，不推进新的高阶闭合。本项沿用
原计划范围：平直轴对齐tensor网格上的`1/1/41/41/1/1`槽道，及静态prof
入口的`11/22/41/52/1/1`平板。没有新增壁压外推、SAT、滤波系数或
可容许域修复，也未重新启动用户要求暂跳过的紧致壁面失稳窗口。

已有低阶壁面、ME6壁面过渡、compatibility／incoming_only、方法身份与
统计恢复证据直接复用；仅补nonreflecting尚缺的代表联合组合：

| KEEP／原黏性／主滤波 | MPI与halo | 最大场缩放差 | 完整RHS最大缩放差 |
| --- | --- | --- | --- |
| 643e／643c／compact10 | NP=1、NP=2 y、NP=4 xy；pinned | 3.653e-14 | 1.955e-14 |
| 643c／643c／explicit10 | NP=1、NP=2 z；实际device-aware | 1.612e-15 | 6.365e-16 |

均为16³、dt=1e-5、4完整步，无crashfix；场差包含同后端分解与CPU/GPU
对照，沿用2e-10门槛。CPU/GPU各自2+2步的全部状态数据集及三个控制文件
与连续推进一致；NP=1冻结交线独立投影对照通过，入口顶角RHS仍严格为0。
首组NP=1 GPU memcheck为0 errors。NP=4只验证本地两卡共享时的分解正确性。
第二组每rank明确记录solution-halo device-aware active=T，不声称PCR接口
通信由此改变。证据：`open_nr_low_filter_ec/report.json`、
`open_nr_low_filter_cc_aware/report.json`，均为`passed`。

### 原生阶段变化记账

`analyze_keep_open_budgets.py`只读NP=1的原生第一完整步快照。使用已有
J及x/y端点梯形权重、唯一周期z节点，分别记录五个守恒量、动能、内能
和熵密度的对流、其余RHS、主滤波、边界准备及RK非线性余量。
边界准备包括原边界状态覆盖、专用处理和共享／halo协调，不冒称全部是
某一种边界投影。积分权重不是新SBP范数，空间算子和推进器没有改变。

复用无滤波nonreflecting、上述两组滤波的CPU/GPU快照，三组分别得到：
分项后端最大缩放差3.553e-15、9.474e-15、4.145e-15，八项记账闭合残差
均<=1.777e-15。只在分析器内存副本中把RK2一个更新后能量改变1e-5，
重构检查明确拒绝，原始文件不变。

独立计算真实预RHS状态的x/y外向Euler通量参考，并保留体积RHS与表面积分
之差；首阶段五分量最大差在这些夹具中约5.08e-6至7.74e-5，并未归零。
它含端部离散、特征边界及约束处理，与任取求积后的连续通量参考并不
自动构成离散恒等式，不能据记账闭合宣称物理端严格守恒或能量稳定。
完整黏性壁面应力／热流收支仍属于后续专项。

证据：`open_nr_unfiltered_budgets.json`、`open_nr_low_filter_ec_budgets.json`、
`open_nr_low_filter_cc_aware_budgets.json`。此前两组旧compatibility／incoming_only
记录没有滤波前后快照，读取时明确报告缺文件；没有虚构滤波贡献，也未为
补齐相同记账重跑整组历史矩阵。本轮三组代表诊断不扩大其已有验证结论。

### 可运行输入与构建

新增`examples/KEEP_Low_Order_Boundaries/`，包含wall/open两组命名输入模板、
共享input.run/input.output、独立prepare.py和使用说明。生成器仅依赖NumPy
及h5py，不引用tests或私有结果；不在Git中放生成的三维初场或网格。
原生四步默认显式KEEP＋643e黏性、关闭主滤波／修复／统计，step=2/4精确
检查点；文档列明独立显式／紧致、滤波及壁面ME6选项，不改生产默认。

根CMake已登记该目录；CPU根构建target astr重新配置／构建成功，生成的
安装规则包含全部输入模板、生成器和说明。不是新增独立构建入口。
`run_keep_low_boundary_examples.py`实际运行交付输入，CPU/GPU均NP=2 y：

| 示例 | 最大场缩放差 | 最大C_upper |
| --- | --- | --- |
| wall | 4.441e-16 | 0.001492687 |
| open | 9.992e-16 | 0.005642536 |

wall证据位于`low_boundary_shipped_examples/report.json`中的wall条目。
该次open在读剖面时退出：生成器将首列密度写成整数`1`，而现有readprofile
用首列实数字面量定位数据起点。仅将生成器改为明确的科学计数法实数，
没有改CPU/GPU读入规则；只重验受影响open，证据为
`low_boundary_shipped_open_v2/report.json`，状态`passed_four_step_examples`。
不将保留的首次未完成整组记录改写为通过。

本项完成的是已列边界的低阶工程兼容和代表输入交付，不是将所有物理端
或所有长期配置放行。紧致低阶壁面失败、高阶相容闭合、紧致／斜向声学、
完整黏性特征源项及总计划联合物理收支仍分别登记；无Git或平台写操作。

## KP0／KP6：低阶开放端黏性应力与热流的独立归因

恢复总goal后先推进不依赖新闭合审批的物理收支。新增
`analyze_keep_open_viscous_budgets.py`，仅只读复用前三组nonreflecting原生
NP=1、第一完整步的三个RK阶段，没有再启动CFD、申请资源或修改生产代码。
显式／紧致物理端D矩阵复用此前从生产导数导出的`matrices_n16.txt`；
周期z采用对应D6／Pade的离散符号，不以连续解析导数替代生产算子。

输入气体和输运参数显式声明，并与原输入对应：gamma=1.4、Ma=0.3、
Re=1000、Pr=0.72、参考温度1 K、Sutherland常数110.3 K。这是旧组件
夹具的既有归一化，不是新物理算例建议；不把OPEN-P1的273.15 K套入旧数据。
同时由q核对rho/u/p/T缓存，物理外角区不参与重构。

从实际缓存重构物理梯度、Stokes应力、Sutherland黏度、导热与黏性功，
再次作用实际空间导数。核对原法向贡献提交范围：非周期x/y不提交端点行，
周期z保留所有唯一节点。得到的五分量映射黏性RHS与实际`full-conv`比较：

| 归档组合 | 最大重构缩放差 | 体积／功热熵分项最大CPU/GPU缩放差 | 独立逐面比较最大差 |
| --- | --- | --- | --- |
| 显式KEEP、643e黏性、无主滤波 | 9.270e-17 | 2.427e-15 | 4.392e-15 |
| 显式KEEP、643c黏性、compact10 | 1.693e-16 | 2.460e-14 | 3.858e-14 |
| 紧致KEEP、643c黏性、explicit10 | 1.613e-16 | 1.345e-15 | 3.245e-15 |

逐面比较分别覆盖x-/x+/y-/y+的应力、能量、热流、黏性功与熵扩散，
不只比较可能互相抵消的总和，均小于既定2e-10。把分析器Pr误设为0.73
时，原生RHS对照以4.025e-8差值明确拒绝；原始数据及物理参数未改变。

### 表面通量、模板缺陷和端点提交必须分开

在既有J及x/y梯形、z唯一节点测度下，将实际黏性体积变化分解为：
表面应力／功热通量，加未屏蔽空间导数的体积—表面差，再加原实现
未提交法向端点行的贡献。三个来源逐项保留，重构剩余均<=1.480e-15。
此关系核对实现与诊断，不是新的守恒或稳定性定理。

例如无滤波显式夹具CPU第一阶段的归一化总能变化率为1.24003545e-4，
由表面项-1.29461693e-5、模板体积—表面差4.46887093e-5、端点提交项
9.22610049e-5组成。不能把这一差值整体称为滤波、舍入噪声或KEEP对流耗散。
法向端点行不提交是现有逐方向边界处理，并非本轮认定并待修的CPU代码错误。

动能、内能和熵另报告连续功热关系在该离散下的残差，并继续拆为原算子
残差与端点提交贡献。它们未被强制归零。导热的符号明确区分物理向外
热流`-kappa*grad(T).n`与守恒能量RHS的导热输入`+kappa*grad(T).n`；
本夹具底壁速度为零，底壁黏性功为零，但热流并非必须为零。

证据位于`results/keep_wall_low_20261110/`：
`open_nr_viscous_budget_explicit_v2.json`、
`open_nr_viscous_budget_compact_filter_v2.json`、
`open_nr_viscous_budget_compact_keep_v2.json`及
`open_nr_viscous_face_comparison.json`。v2补充分离非线性关系中的端点贡献，
初版派生结果仍保留；没有重跑原生轨迹或改写归档。

本项完成原节点黏性在代表低阶开放域上的冻结RHS及功热归因，不包含
ME6开放端、NSCBC黏性特征投影、完整多步壁温热收支或高阶闭合证明。
接下来高阶实际接线仍须先确认具体数学路线和壁面施加方式，不能直接
把已有Hermite7导数、SBP代数族或本项小残差当作获批高阶闭合。

## KP7：开放端自动步长与精确恢复联合检查

高阶壁面路线A/B仍待用户选择。本轮只补不依赖该决策的现有时间控制
联合范围，没有选择SBP参数、SAT项或恢复暂跳过的紧致壁面长窗。

`run_keep_open_gate.py`增加测试选项`--timestep auto`，复用既有自动步长日志
格式，固定步长仍为脚本默认。采用原16³三维弱扰动开放端、nonreflecting、
紧致KEEP＋643c原黏性＋explicit10、无crashfix、pinned halo，NP=1及NP=2 y，
各推进4完整步。运行配置为CFL目标0.5、增长系数1.1，不给人为initial_dt
或max_dt上限。生产时间步与边界代码没有修改。

- CPU/GPU及同后端分解对照的场最大缩放差2.234e-15，完整RHS差6.365e-16。
- CPU/GPU实际完整步时间与dt序列最大缩放差4.337e-19。
- NP=2 y各自2+2精确恢复的全部状态数据集、三个控制文件及后两步日志
  时钟与连续推进逐值一致。
- 首步dt约8.94340763004218e-4，第四步使用约8.94301892542320e-4，
  第四步完成时刻约0.003577287866335871，无量纲。

每步都用实际已用dt与完成状态的详细C_upper核对现有选择式：
`dt_next=min(0.5*dt_used/C_upper,1.1*dt_used)`，同时核对完整步时间相加关系。
完成状态按已用dt计算的最大C_upper约0.5000087478477733，**没有将其写成
严格CFL<=0.5通过**。AT-D01已选步前目标控制、无步后拒绝／回滚，
本步内状态变化可使事后值略过目标；下一步按更新状态调整。这与固定dt
OPEN-P1的实际C_upper<=0.5门槛是不同检查，不混用两种语义或放宽原声学门槛。

证据：`results/keep_wall_low_20261110/open_nr_automatic_compact/report.json`，
状态`passed`，包含逐步clock、已用dt的C_upper、后端差及恢复结果。
本短测不支持长期稳定性、黏性扩散时间步保证或新增自适应输出准入。
总体目标仍未完成，后续新的高阶壁面施加方式仍需人工决定。

## 高阶路线A确认及具体候选推导

用户随后明确选择A，不再把弱／强施加路线作为待决定项；本节记录该
决定之后的分析。批准仅覆盖设计路线，未把第12节参数族直接投入生产。
具体候选写入`ASTR_GPU_KEEP_GEOMETRY_CANDIDATES.md`第14节。

已形成以下建议，均未改src/src_gpu：

- 保留内部显式D6，端部六行使用SBP6/3族；建议theta以H加权近壁三阶
  领先误差系数平方和最小化选择，唯一极小点约0.7049084023584565。
  此准则尚未批准，不能称稳定性／色散／性能最优。
- 无黏壁面数值通量为(0,p*n,0)，用通量差SAT而非旧壁压Neumann及速度／
  温度强覆盖。物理端节点参与体RHS，再由唯一数值边界通量施加条件。
- 原节点原始变量梯度增加目标跳量提升，再构造Sutherland应力及导热。
  黏性罚通量以SBP端点质量和材料系数定尺度；能量中的机械功用目标
  静止壁速0，温度罚项作为数值热流，不把动量和能量修订分别拼接。

`study_keep_sbp63_sat_wall.py`没有推进任何PDE。在16／32区间，对全theta族
精确核对标量变量系数扩散的SBP／Nitsche矩阵关系与负平方证书；u=x、
a=1+x的非齐次边界参考RHS精确为1。冻结静止理想气体五变量声学的
对称化及SAT对流能量抵消也精确成立。

原始算术KEEP与新无黏壁面SAT在端点速度非零的正状态上，五分量通量
及压力功残差精确为零。一维法向变化、三速度分量、变温Sutherland
材料的冻结非线性黏性参考，进一步核对了负应力耗散、非负壁面罚耗散、
总能数值热流及内能转换，代数残差均精确为零。这些结果不等同于
全三维原生实现、非线性熵稳定、保正或完整RK3稳定性。

独立文献复核发现，绝热／给定热熵流的稳定壁面不能直接当成定温壁面。
2026定温壁面预印本还报告高精度强罚项的时间刚性；本候选没有照搬其
内部熵通量或h^-p罚缩放。原算术KEEP不因此改称严格熵守恒。

证据：`high_order_sbp63_sat_route_a.json`及v2/v3派生记录，均位于
`results/keep_wall_low_20261110/`。v2补充非线性黏性功热参考，v3明确路线
已批准、具体公式未批准，并记录最大模板偏移5；旧派生结果仍保留。
没有新原生CPU/GPU算例、生产默认修改、Git操作或超算任务。

下一项需要审阅的是第14节的具体参数准则、壁面通量、提升与罚项以及
首轮组件／四步原生范围，不重复询问已经确认的路线A。紧致壁面高阶
闭合仍独立保留；暂跳过的原低阶紧致失稳窗口没有恢复。

### 路线A候选的MPI索引和固定步长预审

具体公式批准未收到。本轮继续完成第14.7节实施前需要的只读前提核对，
不接生产组件、不启动CFD。`--preflight-only`在既有数学分析脚本中独立
记录索引与线性矩阵结果，不重跑已有非线性冻结通量代数或历史轨迹。

按CPU现有五层邻居打包范围，16区间NP=1/2 y及20区间NP=4 y的全部
候选导数行所需节点均可取得。特别是全局共享节点5在两个rank上必须
取h5=43801/43200，不可按局部端点误取h0；SAT仅属于真实物理端。
该索引分析不冒称真实MPI或GPU打包实现已验。

拟定均匀盒、静止理想气体和常值材料下，组装法向SBP及切向D6 Fourier
的完整五变量声学／黏性／导热线性块。16／20区间分别核对81／121组
非负象限模式，dt=1e-5；最大dt*谱半径4.053258e-4／5.065250e-4，
线性RK3多项式的最大对称化能量范数放大率约1+8.9e-16／1+5.0e-15。
这些是含中性质量模式的浮点参考，不设置新容差，也不宣称严格放大率
<=1、非线性剪切稳定或原生验收通过。

证据：`high_order_sbp63_sat_preflight.json`，位于同一低阶证据根目录。
该预审使首轮数据布局和步长尺度更具体，但不代替尚待的候选公式批准。

## SBP6/3具体公式获批与原生数学组件

2026-11-11用户明确批准几何候选第14节的参数准则、弱无黏壁面通量、
原节点黏性梯度提升／功热罚项及第14.7节首轮范围，原目标恢复为active。
低阶w24/w34默认不变，暂跳过的低阶紧致失稳窗口不恢复。

新增`keep_sbp63.F90`、共享公式`keep_sbp63_body.inc`及GPU数学组件。
`keep_sbp63_probe.F90`经根CMake构建，在16／32／64区间实际执行导数矩、
SBP关系、无黏通量／压力功及黏性功热检查；Python只构造独立精确矩阵和
冻结通量参考，不替代原生PDE推进器。

`results/keep_sbp63_20261111/component_gate/report.json`状态为
`passed_native_component_and_independent_reference`：CPU/GPU最大缩放差
1.735e-18，精确矩阵及独立通量参考最大差4.105e-14，memcheck零错误，
插桩与普通运行的组件表逐值一致。最细两级导数观测阶为近壁2.924、
内部5.992，未称全域六阶。

主循环接线已完成：独立`high_order_wall_mode='keep_sbp63'`选项，按全局
节点选择SBP行；所有物理端节点参与体RHS，再施加唯一壁面SAT，不执行
旧bc41强覆盖。复用现有梯度／应力／热流数组，不新增整场q副本；恢复
身份单独记录为`s63sat-v1`。主程序首轮结果如下，不据此宣称整个高阶
壁面、紧致或KEEP总计划已经完成。

### 首轮四步主程序、跨边界带MPI与恢复

`results/keep_sbp63_20261111/native_wall_gate/report.json`通过CPU/GPU各自的
16³ NP=1/2 y及20³ NP=1/4 y全部矩阵。全局共享节点5切穿SBP六行端部带，
两个rank均按全局第5行计算；SAT不作用于MPI接口。固定dt=1e-5，4步至
t=4e-5，驱动0，无滤波及crashfix，物理节点rho/p/T均有限且严格为正。

- CPU/GPU终场最大缩放差2.082e-17；同后端分解对照最大8.674e-19。
- 首步对流／完整RHS的CPU/GPU最大缩放差9.094e-16；各三个RK阶段的
  实际RHS和梯度与独立三维矩阵参考最大缩放差1.411e-12。
- SBP范数下无黏壁面质量／能量积分通过既定门槛；黏性动能功关系绝对
  残差最大1.356e-20，总能与数值壁面热流积分差最大3.618e-15。
- CPU/GPU所有拓扑均完成2+2精确恢复，全部state数据集及三个控制文件
  逐值一致；GPU所有NP=1/2/4 rank的memcheck为0 errors。
- 四步后壁面trace速度最大约1.93e-10、温度偏差最大约7.96e-10，保留
  原值；不将弱壁面偏差强行归零，也不以场差门槛判定壁面物理解误差。

同根`complete_step_budgets.json`读取已有三个原生RK阶段及实际更新结果，
记录五守恒量、动能、内能、熵密度的准备项、对流／黏性贡献及非线性
RK余量，完整步收支闭合最大绝对残差1.777e-15。没有额外积分PDE。
分析初版曾重复缩放单位区间SBP矩阵，修正的只是参考程序，原生公式不变。

同根`unchanged_low_wall/unchanged_vs_frozen.json`记录旧低阶槽道CPU/GPU
各15个state数据集与改造前冻结结果逐值一致。没有Git写入或平台操作。
下一项建议窗口／网格差检查已写入几何候选第14.10节，尚未启动；原低阶
紧致长窗失败仍按用户决定暂跳过，高阶紧致及ME6新闭合仍待设计批准。

### 独立输入与弱／强壁面恢复身份交付

§14.10扩大窗口尚未获确认，本轮没有启动该窗口，只补KP7/KP8已有授权的
输入及恢复交付。新增`examples/KEEP_SBP63_Walls/`，包含独立命名模板、
四步控制、检查点配置和只生成网格／初场的prepare.py，安装接入根CMake。
示例默认16区间，支持生成20区间的已批准端部带切分夹具；不依赖测试目录。

`run_keep_sbp63_wall_gate.py`改为直接执行该示例的准备脚本。原16区间
CPU/GPU NP=1/2 y四步、冻结RHS／完整RK及2+2精确恢复全部通过，终场
CPU/GPU缩放差仍为2.082e-17。源输入和生成的场与前轮数学定义一致。
证据：`results/keep_sbp63_20261111/shipped_wall_gate/report.json`。
根CMake生成的安装规则已安装该独立示例；安装副本生成的三个命名输入
逐字节一致，网格和初场全部HDF5数组逐值一致，见同根
`installed_input_preparation/input_equivalence.json`。

新增`run_keep_sbp63_migration_gate.py`复用该不可变NP=2 y检查点，以原步长
检查SBP弱壁面与旧w24双向切换。不同算子在exact_restart=t时均在恢复
身份检查被拒绝；显式非精确模式可继续至第4完整步。目标检查点身份
正确，两种方向及后端均走已有“从第2步新建统计窗”的生命周期分支。
本夹具统计关闭，不能据标记声称非零统计历史已额外联测。

初版测试错误地对非精确恢复也要求所有重复点逐值一致。实际CPU qswap
及GPU exchange_solution_halo_gpu会重新平均共享／周期节点，因此出现
机器舍入级变化。这是既有恢复语义，不是新发现的CPU缺陷；恢复代码
未修改。修订测试沿用既有非精确缩放场差2e-10，另严格检查非共享节点
逐值一致；精确续算的逐值门槛完全不变。

`wall_method_migration_v2/report.json`通过：恢复探针全场最大缩放差
6.939e-18，所有非共享节点逐值一致；目标物性缓存与当前q的最大缩放
差4.441e-16，CPU/GPU迁移终场最大缩放差6.662e-16。初次失败目录保留。
没有改变求解器、物理模型、恢复算法或数值门槛，也没有Git／平台操作。
§14.10窗口仍待用户确认，总计划不据此关闭。

### 高阶紧致壁面的独立数学前提

未收到§14.10新窗口批准，本轮未启动CFD。继续处理KP0-A／KP5紧致壁面
的分析前提，原公式及生产默认没有修改。几何候选新增第15节，区分
周期KEEP-compact守恒文献和带状范数隐式SBP参考的适用范围。

原生导数提取结果继续复用：旧紧致物理端的内部D(1,1)非零，不能只换
一个正对角范数来获得标准SBP。新增独立质量分裂推导进一步给出：若
H非对角、HD+D^T H=B、D1=0，行和w=H1虽满足线性通量积分，但原逐点
KEEP的质量积分仍有-1/2*rho^T*(WD+D^T W-B)*u缺陷，W=diag(w)。
不能把H的线性SBP等价成这个逐点求积条件。

`study_keep_compact_norm_prerequisites.py`用四节点严格正定带状H及相容Q
作精确有理数反例，验证常数／一次导数和线性H能量恒等式均成立，
零壁面质量通量下原KEEP质量积分却为-5/11；缺陷式与直接通量装配完全
相等。它不是ASTR或高阶格式，不排除其他紧致／带状范数构造。
证据：`results/keep_sbp63_20261111/compact_norm_prerequisites_v2.json`。

该结果明确后续接线前必须核对的非线性条件：保留现有原KEEP及内部
Padé关系的新端部构造是否有正对角范数，仍是未解可行性问题；若要改用
不同带状范数非线性装配，须另行提交具体公式审批。没有选择新算子、
改动SAT、取消全局PCR耦合或将低阶失败改写为通过。

随后对“固定新显式SBP范数、只重求紧致RHS端部”作精确受限构造检查。
16区间、两端各六行、最大偏移5，原端部P和全1/3相邻耦合对称P两种
设想均得到48个未知量、221条方程，秩48／49，精确矛盾0=1。
证据为同根`compact_norm_prerequisites_v3.json`；具体前提见几何候选§15.4。
这不排除其他P／范数联合构造，也不授权更换内部公式或回退显式闭合。

上述轮次结束时没有未结束的构建或计算进程。§14.10窗口的具体范围连续三轮
等待确认，独立示例、方法迁移与可先行的数学前提检查已补充；后续CFD
不能代替人工批准直接启动。当时总goal登记为blocked而非完成，等待该范围
决定后继续；高阶紧致等未关闭项保留在完整目标中。

## 已批准§14.10：固定时刻网格／步长窗口及误差定位

用户随后明确确认§14.10，goal恢复active。原生入口
`run_keep_sbp63_wall_convergence.py`已完成CPU16及GPU16／32／64的1000步，
dt=1e-5，另完成GPU32、dt=5e-6的2000步，均到t约0.01。NP=2 y、均匀
网格、Tw=1、Re=3000、Ma=.3、零驱动，无滤波／crashfix，统计关闭，
每组只保存最终检查点及日志。未启动其他物理窗口或平台任务。

全部末场rho/p/T有限且严格为正；全组最小rho约0.9999386480、最小T约
0.9999754591。CPU/GPU16五守恒场最大缩放差2.568e-16。64区间实际
最大C_upper约0.0013642433；没有据此更改固定步长或添加控制条件。

| GPU每方向区间数 | 壁面最大速度模 | 最大壁温偏差 | 下壁平均数值黏性能量通量 |
| --- | --- | --- | --- |
| 16 | 5.3625e-6 | 2.0511e-7 | -1.7405e-10 |
| 32 | 4.8855e-6 | 1.8388e-7 | -3.4800e-10 |
| 64 | 4.1055e-6 | 3.9187e-7 | -6.8801e-10 |

热流按最终q、同一SBP提升／罚项和存储度量重构；上表不是既有梯度型
生产输出，物理向外热流符号相反。分析曾修正度量打包的Fortran列序，
只重读原检查点，没有重算流场。相应派生报告为
`results/keep_sbp63_20261111/wall_fixed_time/report_analysis_v2.json`。

本轮不能宣布全域高阶收敛：rho/p/T的L1表观阶约3，但v、w的Linf
表观阶为负。32→64时v的全域最大差约7.8004e-7，在上壁；固定内部区
3*pi/8<=y<=5*pi/8最大差仅3.0185e-12。T最大网格差2.0800e-7，也在壁面。
32区间半步长T最大差4.5741e-14、p最大差6.1817e-13，远小于相邻网格差。
不能以时间步或GPU舍入来解释全部壁面差异，也不放宽任何既有门槛。

直接对原初场连续NS方程求值，确认其壁面值相容但时间导数不相容：
下壁v_t=(16-5*cos(z))*sin(x)/9000000、w_t=cos(z)/1500000、
T_t=21*(10+cos(x)*cos(z))^2/50000000000，均不恒为零；固定目标要求
壁面这些时间导数为零。它说明该初场不能直接承担光滑相容解阶数的
验收，但尚不能单凭该事实认定所有壁面误差的唯一原因。

几何候选新增§14.11：仅测试构建的定常制造源，把相同原速度、rho=T=1
作为精确稳态解，保留非零壁面剪切，不修改当前数值格式或生产默认。
五分量源已与连续守恒通量直接求导独立核对，残差精确为零；尚未接线
或运行。该改变验证问题与意义，需要用户批准后实施。当前goal仍active，
§14.10运行已完成，KEEP全计划及全域精度／长期稳定性仍未关闭。

## 已批准§14.11：定常制造解接入与结果

用户明确批准该源项及矩阵，goal保持active。共享数学定义
`src/keep_sbp63_mms_body.inc`仅在测试构建编译，显式通过
`ASTR_TEST_KEEP_SBP63_STEADY_MMS=1`启用；CPU/GPU按固定参考与坐标求值，
不由演化q反算，不分配整场参考或源项副本。原壁面／黏性公式和生产
默认未变，检查点追加`mms-v1`。

根CMake CPU/GPU构建及16区间NP=1/2 y四步检查通过：五分量实际源与
独立守恒通量求导最大缩放差1.388e-16，CPU/GPU终场差3.470e-18；2+2
恢复的全部state数据集与三个控制文件逐值一致。关闭源的精确恢复被
身份检查拒绝，GPU所有rank的memcheck为0 errors。完整RK独立记录
制造源，闭合最大绝对残差2.961e-15。
证据：`results/keep_sbp63_20261111/mms_native_gate/report.json`。
源关闭的独立CPU/GPU各15个state数据集与此前无源冻结结果仍逐值一致，
见同根`unchanged_without_mms/unchanged_vs_frozen.json`。

随后CPU16及GPU16／32／64各1000步、GPU32半步长2000步全部到t约.01，
末场rho/p/T有限且严格正，CPU/GPU16最大缩放差4.879e-18。
下表为已知定常解的全域Linf绝对误差，不是相邻网格差：

| GPU区间数 | rho | v | w | T |
| --- | --- | --- | --- | --- |
| 16 | 4.8415e-7 | 3.5676e-7 | 1.2143e-9 | 1.9347e-7 |
| 32 | 7.0052e-8 | 1.1040e-7 | 4.3622e-10 | 2.7904e-8 |
| 64 | 6.5911e-9 | 2.5515e-8 | 1.0462e-10 | 2.5849e-9 |
| 32→64表观阶 | 3.410 | 2.113 | 2.060 | 3.432 |

细两级v/w的L1阶约3.097/2.914、L2约2.605/2.454；64区间固定内部区
误差仅约1.557e-13/8.758e-14，最大误差在壁面。32区间半步长v、T、E
最大变化约2.719e-13、8.966e-13、1.275e-11，远小于全域空间误差。
结果见同根`mms_fixed_time/report_analysis_v2.json`，原报告及检查点保留。

精确矩分析确认：常材料且参考trace满足目标时，节点黏性主项为
D63*D63，近壁仅0..3次二阶导数矩精确，有非零h^2*f''''；远离两端时
0..7次矩精确、领先项h^6*f^(8)/70。这与v/w最大误差约二阶相符，不是
CPU/GPU移植偏差，也不表示所有SBP黏性都只能二阶。复合影响延伸到
前／后九行，但仍用解变量halo供梯度、sigma/qflux halo供散度，hm=5
无需增加，不新增准入围栏。

因此该路径准确区分一阶导数6/3与复合黏性近壁2阶。提高后者需另行
设计同范数相容D2及动量／能量／热流，本轮未替换或调theta。制造解
范围完成，不替代无源长期稳定性，也不关闭高阶壁面滤波、紧致、ME6
及开放端剩余目标。没有Git写入或平台操作。
