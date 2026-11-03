# 湍流边界层原位产品实施进展

更新：2026-11-03；实施始于 2026-11-02。分支：`feature/gpu_dev`。
依据：[专项计划](ASTR_INSITU_TURBULENT_BOUNDARY_LAYER_PRODUCTS_PLAN.md)。

当前 goal 状态：active。用户已批准 §16.1 数值规格，并进一步取消本轮
人为资源预算上限，继续记录资源和保留证据。TB1–TB4 完整目标和数值/
数据驻留验收条件不变，未标记完成。旧 blocked 和预算记录仅反映当时状态。

用户已授权建立覆盖 TB1–TB4 的 goal 并开始本地实施。D01–D42 及交叉审查
补充继续有效；不操作 Git、平台任务、新 worktree 或子代理。

## 当前状态

| 阶段 | 状态 | 已有证据与剩余工作 |
| --- | --- | --- |
| TB1 | 本地定量基线已有通过证据，未关闭全范围 | 笛卡尔厚度、完整壁面量、标度/网格单位、Cf/Cq/St 及剖面矩阵通过；读取式 M12 原随机吹吸精确续算、小型 GRCBC SBLI 定量组合已有闭环。其他壁面家族、更宽单位/工况组合及生产排版仍需按证据补齐；资源政策以最新无限额授权为准 |
| TB2 | 获批首轮本地门槛通过 | 固定 j 脉动、二维壁面标量/箭头及瞬时/平均剪切线均已接通。TB2.3 有共用驱动、七拓扑、两设备后端、真实 CPU/GPU 对照、精确续算、动态缺口及成本记录；不自动扩展为生产湍流统计或一般分离验收 |
| TB3 | 未开始 | 新增 lambda2/lambda_ci、选区与设备渲染仍待实现 |
| TB4 | 未开始 | 时空图、流式 Welch 与缺口/续接状态仍待实现 |

下文为按时间保留的实施记录。历史段落中的“待确认/未实现/超额”只描述当时
状态，当前结论以本表及对应末节为准，不删除旧失败或替换原验收记录。

## 统计数学基础

实现入口：`src/insitu_spatial_statistics.F90::summarize_velocity_profile`。
输入为已完成局部时间统计的结果和唯一节点的物理长度权重，复用既有
`velocity_statistics_result` 与 `spatial_mean`，不加入展向平均场差异项。

- Reynolds 均值和局部时间协方差按物理长度汇总，RMS 从汇总方差开方。
- Favre 均值使用密度权重；先汇总密度应力，再除以平均密度得到 Favre 协方差。
- 空本地分区允许参与；不匹配的覆盖时长集体返回失败。调用方仍须验证
  请求窗口和实际采样身份，不能仅凭相同时长认定历史相同。
- 该 CPU 接口仅处理小型统计结果，不提供 GPU 采样或整场下载路径，
  也未解除平板/SBLI 现有统计入口限制。

验证入口：`tests/gpu_validation/insitu_spatial_statistics_probe.F90` 及
`test_insitu_profile_statistics.py`。通过两个真实变密度时间样本构造各点统计，
再以手工可解析的独立结果检查全部协方差分量、密度权重和局部均值不同的反例。

已观测：

1. 新探针在新增实现前因缺少公开接口构建失败，确认不是旧函数已覆盖该功能。
2. 经根 CMake 构建 `insitu_spatial_statistics_probe` 通过。
3. NP=1/2/4 直接 MPI 探针均通过，包括 NP=4 的空分区。
4. 环境变量指定真实构建探针的 pytest 三项通过，不是跳过或 Python 重写算法。
5. 根 CMake 的 CPU `astr` 目标构建通过，确认共享模块变更可以集成进主程序；
   此项仅是构建证据，不替代真实 CFD、GPU 或重启验证。

复现命令，在仓库根目录执行：

```bash
cmake --build build_cpu_probe --target insitu_spatial_statistics_probe --parallel 4
env ASTR_INSITU_PROFILE_PROBE="$PWD/build_cpu_probe/bin/insitu_spatial_statistics_probe" python3 -m pytest -x -q tests/gpu_validation/test_insitu_profile_statistics.py -k local_temporal_profile_summary
```

上述第一项基础不单独构成 GPU、真实平板/SBLI、精确重启、可视化或性能验收。

## 紧凑节点累计与设备索引采样组件

实现入口：

- `src/insitu_sample_moments.F90`：共用的选字段时间累计、均值/速度矩读取及五变量理想气体原始变量重建。
- `src_gpu/insitu_sample_moments_gpu.cuf`：设备小型字段输入累计，以及 `gather_selected_moments_gpu` 从设备 q 按索引读取并累计。
- `tests/gpu_validation/insitu_sample_moments_probe.F90`：独立原统计参考、已知压力/温度均值和 CPU/GPU 同状态探针。
- `src/CMakeLists.txt`：主程序与 CPU/CUDA 探针均从根 CMake 构建，不引入独立构建入口。

共用更新保留原时间端点权重及中心矩乘法顺序；速度对称中心矩每种平均只存六项。
窗口、实际采样身份及持久化由后续调用层负责，不把数组内容本身当成完整的恢复身份。

| 请求内容 | 每个选中节点的实数状态数 |
| --- | ---: |
| 密度均值 | 3 |
| 压力、温度均值 | 9 |
| 三分量速度均值 | 12 |
| 三分量速度均值与二阶矩 | 24 |
| 速度二阶矩加压力、温度均值 | 30 |

这些是状态布局大小，不是整段仿真的显存峰值或性能收益。组件不分配完整三维
统计体；设备调用使用外部预分配状态和工作区，只显式回读一个整数状态。
探针为比较结果另有小型矩阵回读，并从主机初始化合成 q；不能将其传输量
直接宣称为实际 CFD 的生产传输量。

按索引入口读取当前传入的设备 q，不引用旧的压力/温度缓存；只重建所需字段。
单速度字段检查允许未请求的能量、其他速度分量及 EOS 参数为 NaN，避免
因检查未使用的量拒绝该字段。该反例只证明请求依赖，不表示无效能量流场
具备物理资格。合成测试也将未选网格节点设为 NaN，检查没有扫描整个体。

已观测验证：

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| CPU 数学/索引组件及原归约探针 | 6 项通过，失败/错误均为零 | `tests/gpu_validation/out/tb1_sample_moments_20261102/cpu_indexed_rms.xml` |
| CUDA 同状态算子，NP=1/2/4 | 3 项通过，失败/错误均为零；参考、设备最大缩放误差及 RMS 原值差均为 0 | 同目录 `gpu_indexed_rms.xml` |
| 两张卡的绑定 | NP=2 为 `0:0,1:1`；NP=4 为 `0:0,1:1,2:0,3:1` | GPU XML 的 `rank_devices` 属性 |
| 纯算子 Memcheck | 分别在两张物理 GPU 上通过，均为零错误、零泄漏 | 下述 `--kernel-only` 命令，本次终端结果 |
| CPU 与 CUDA 主程序集成 | 两个 `astr` 目标构建通过 | `build_cpu_probe` 与 `build_m12_gpu` 的根 CMake 构建 |

这里的 NP 矩阵检查进程设备绑定与独立节点算子，并不包含真实求解器的
x/y/z 及混合拓扑通信验证。原归约探针另有真实 MPI 归约；两者都不能代替
平板/SBLI 的完整统计链路门槛。内存中原始状态复制的精确续接检查，也不是
checkpoint 文件序列化或新进程重启验收。

复现命令：

```bash
cmake --build build_cpu_probe --target insitu_sample_moments_probe insitu_spatial_statistics_probe --parallel 4
cmake --build build_m12_gpu --target insitu_sample_moments_probe --parallel 4
env ASTR_INSITU_PROFILE_PROBE="$PWD/build_cpu_probe/bin/insitu_spatial_statistics_probe" ASTR_INSITU_MOMENT_PROBE="$PWD/build_cpu_probe/bin/insitu_sample_moments_probe" python3 -m pytest -x -q tests/gpu_validation/test_insitu_profile_statistics.py
env ASTR_INSITU_MOMENT_PROBE="$PWD/build_m12_gpu/bin/insitu_sample_moments_probe" python3 -m pytest -x -q tests/gpu_validation/test_insitu_profile_statistics.py -k selected_primitive
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 build_m12_gpu/bin/insitu_sample_moments_probe --kernel-only
env CUDA_VISIBLE_DEVICES=1 compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 build_m12_gpu/bin/insitu_sample_moments_probe --kernel-only
```

实现时已定位并修复的组件/探针问题：

1. NVHPC 26.1 的双端例程对数组描述符及 IEEE 泛型的编译问题：共享运算改为显式形状小数组接口，有限数比较沿用既有设备核方式，未改变统计公式或容差。
2. 探针的固定设备状态标志在 GPU 绑定前分配，造成 GPU1 rank 非法写：改为绑定后分配；在原 MPI 启动方式下复测通过，没有改用单卡掩盖问题。
3. 零长度设备切片的顺序性检查及零尺寸分配问题：使用预分配容量与有效 `sample_count` 分离，检查空选择及部分有效前缀，不强迫空 rank 分配零尺寸设备数组。
4. UCX 在 MPI 初始化时的 CUDA 能力探测触发 Sanitizer API 报告：数值矩阵保留正常 MPI；纯算子内存检查明确不做 MPI 初始化，不屏蔽 CUDA 错误，不修改生产通信后端。CUDA-aware 数据通信尚须在实际接线后单独验收。

## 目录注册与真实完整步接线

新增 `src/insitu_selected_statistics.F90` 管理物理支持、字段和实际历史，
`src/insitu_selected_runtime.F90` 将它接入现有动态产品框架，而不是新建另一套目录或时钟。
`index_slice_archive` 在现有目录轮询之后、求值和输出产品之前采样当前完整步 q。
CPU 使用相同原始变量解码；GPU 使用已验证的设备索引累计核。

- 新数值短测入口为 `products='catalog_stats'`、`statistics=t`、`render=f`，
  请求显式选择 `statistics_source='selected-time'`。
- 支持 Reynolds 密度/速度/压力/温度均值及 Favre 速度/压力/温度均值。
  梯度字段展开为所需原始变量及私有模板邻域，不读取旧 RK 梯度缓存。
  不把普通密度均值标成 Favre 密度。
- 物理节点采用唯一所有权，周期末端节点为别名。按实际全局索引范围建表，
  不扫描整个三维流场来筛选少量节点。匹配字段与支持的 Reynolds/Favre 请求复用同一历史。
- 相机、色标和发布间隔不是统计身份。新字段/支持缺少历史时新建累计；
  停用后重新启用只重置区间端点，不跨缺采区间积分，也不清除其他有效历史。
- 追加组时移动既有主机数组及设备指针，不复制全部既有累计体。可控状态分配
  复用现有配置预算，并按节点、物理 PCI GPU 汇总共享设备需求，不新增固定预算上限。
- 当前均值经现有显式二维数值存档做独立参考。GPU 均值在设备端读取，
  此测试没有三维流场回读；二维存档的传输与采样核的状态回读分开记录。

配置示例与限制：`tests/gpu_validation/configs/tb1_selected_statistics/`。
首轮数值采样短测关闭 checkpoint。随后已接入同后端、同拓扑的统计序列化
与恢复，验证进度和未通过项见下节，不再保留原先的临时禁止输出检查点分支。
新入口也尚未作为均值图像/壁面/剖面产品的生产预设，不改变既有预设和默认值。

### 已观测验证

证据目录统一为 `tests/gpu_validation/out/tb1_sample_moments_20261102/`。

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 请求、唯一所有权、缺口与晚加入；CPU | 7 项通过，NP=1、NP2 x/y/z、NP4 xy/xz/yz；解析缩放差为 0 | `cpu_registry_current.xml` |
| 同上及 CPU/GPU 状态、空所有者、独立失败和分配拒绝 | 7 项通过，缩放差为 0 | `gpu_registry_current.xml` |
| 真实 TGV 16³ 十二步及原 M12 64×32×24 十步；CPU/GPU | 16 项通过，关闭渲染仍累计，均值与同一轨迹的独立端点积分比较 | `real_selected_current.xml` |
| 平板七种拓扑，各自 CPU/GPU 完整步瞬时场与均值 | 7 组对照通过，最大缩放差 `2.07557405541192e-13`，最大绝对差 `2.12940776123105e-13` | `real_selected_current_same_topology.json` |
| 独立关闭统计的 y 共享面诊断 | 两种 CPU 拓扑各做统计开/关对照，全部 11 帧请求的瞬时字段逐值一致；关闭统计后跨拓扑第 3 步仍有同一速度差 | `raw_topology_diagnosis.xml` |
| 既有数值截面精确续算回归 | 1 项通过，未启用新统计的 CPU TGV 连续/重启字段一致 | `old_slice_restart_regression.xml` |
| CPU/GPU 主程序 | 当前两个 `astr` 及独立探针构建通过 | 根 CMake 的 `build_cpu_probe`、`build_m12_gpu` |

真实平板保持原随机吹吸、dt=0.02、10 个完整步和原数值/边界，不作生产物理验收。
CPU/GPU 使用同一 CUDA-capable 构建的输入开关，保证 UDF 一致。
对每个运行，参考先读取每个完整步的显式瞬时数值截面，再独立计算窗口裁切后的
端点积分，不使用累计器自身结果作为 oracle。TGV 最大缩放差约 `8.88e-16`；
平板最大缩放差约 `5.11e-15`、最大绝对差 `7.11e-15`。

### 尚未通过的跨拓扑轨迹比较

不能将上述 16 项通过解释为不同拓扑的 CFD 轨迹已全部通过：离线交叉比较
CPU NP=1 与 NP2 y 时，均值流向速度最大差约 `3.37e-10`，超过 `2e-10`。
进一步检查同批原始瞬时数据发现：初始场逐值相同，第 1 步已出现约
`4.89e-11` 的速度差，第 3 步约 `2.84e-10`，第 7 步约 `4.41e-10`；
最大位置位于该截面的全局 `j=16,i=29,k=12`，即 y 分区共享面。

因此差异已存在于喂给累计器的主流场轨迹，不能归因于新统计均值公式，
也不能仅凭这个定位认定原 CPU 求解器有明确缺陷。没有修改 CPU 离散或放宽门槛。
进一步新建同参数的 CPU NP=1 和 NP2 y 对照，完全关闭新统计后，第 3 步
仍得到相同的 `2.8395830131699995e-10` 速度差；每种拓扑的统计开/关
共 11 帧、四个请求的瞬时字段均逐值一致。这将当前差异与新统计接线区分开，
但不代替完整权威 q 或所有时刻的产品开关不变性验收。
同状态算子门槛和真实轨迹差异仍分开记录；跨拓扑 CFD 项保留未通过，
后续需以固定真实状态回放或受控共享面诊断进一步隔离。
`analyse_selected_statistics.py` 默认仍检查完整矩阵并在首次超门槛时失败；
显式 `--scope same-topology` 的报告不包含跨拓扑放行。

### 传输与耗时记录范围

最新小型平板 NP=1 的一个均值支持含 2145 个节点：主机状态/索引计数
42900 B，设备状态/索引计数 231664 B。11 次索引采样显式回读状态标志
44 B，首次索引上传 25740 B，采样器流场回读 0 B。
这些计数不包含独立数值截面存档、MPI 内部复制、运行库隐式参数传输或完整 RSS。

`ASTR_INSITU_STAGE_TIMING selected_statistics_sample` 已记录目录更新/采样阶段。
当前 NP=1 示例普通步约 64–68 μs，初始化和个别步骤更高；这是短测原始成本记录，
不是隔离硬件负载后的性能基准或加速比。完整产品显存峰值、整段开关对照尚待交付。

## 下一实施入口

1. 隔离 y 共享面主流场轨迹差异，保持同状态/同离散门槛；未获批准不修原 CPU 数值算法。
2. 稀疏均值的精确续接矩阵已补齐；继续实际平板/SBLI 的持久化、未保存统计的新段、动态产品历史及失败恢复验证，不将 TGV 续接等同于全部工况验收。
3. 继续 TB1.2–TB1.6 的标尺、正式测站剖面、速度矩输出、完整壁面量与厚度积分，
   补真实 SBLI、产品开关不改权威场和完整生命周期验收。
4. TB2–TB4 仍按原计划推进，不把已经存在的均值数组等同于新增完整产品。

## 所选历史的检查点续接增量

实现：`checkpoint_state_io::checkpoint_rank_words`、
`insitu_selected_statistics` 的 `pack/restore_bytes/restore`、
`insitu_selected_runtime::selected_runtime_file` 与既有 `insitu_session` 入口。
仍写入正常批次的 `statistics.h5`，产品定义继续使用 `insitu_control.bin`
中的既有动态目录，不新增第二套恢复命令或附属状态文件。

数据包含原始累计矩、上一采样端点、请求窗口、实际起止/覆盖时长、连续段数、
缺采标志、字段及唯一节点支持。以整数位模式保存累计实数，不做有损转换。
HDF5 数据集为 `identity`、`numerical_kind_bytes`、`rank_word_counts`、`rank_words`，
状态角色 12；本入口不接受改变拓扑、后端或实数种类后仍宣称精确续算。
保存时只暂存所选累计状态，不下载整个 q；该显式检查点 I/O 与普通采样的
小标志回读分开，不能把普通采样的零流场回读计数解释为全部 I/O 零回读。

新增保存/恢复没有 CRC、保存后重读或全载荷校验。保留格式、尺寸、身份、
节点索引和配置预算检查，确保恢复能正确定位数组。统计没有随检查点保存时，
使用现有新统计段分支，首个实际恢复状态只重新建立端点，不伪造过去历史。
测试显式使用 `exact_restart=t`，未改变生产默认。

已观测证据（同一证据根目录）：

| 验证 | 状态 | 证据 |
| --- | --- | --- |
| CPU/GPU 原始统计状态编解码、恢复后经过停用/重加缺口 | 两套各 7 项通过，所有打包字逐值一致 | `cpu_registry_restart.xml`、`gpu_registry_restart.xml` |
| 真实 TGV 首轮 CPU/GPU NP1/NP2 x，12 对 5+7 步 | 4 项通过；完整 state/statistics 与恢复后数值帧逐值一致 | `selected_restart_matrix.xml` |
| 修复前扩展 TGV CPU/GPU NP1、NP2 x/y/z、NP4 xy/xz/yz | 首个失败停止：10 项通过、1 项失败，其余未运行 | `selected_restart_topologies.xml` |
| 已批准角区初始化后的同一矩阵 | 14 项通过，完整 state/statistics 及续算帧逐值一致；六个原始变量缓存角区均为逐位正零 | `selected_restart_corner_fixed.xml` |

### 已批准并修复：CPU 非反应缓存角区

失败为 CPU NP4 `2×1×2`。完整 q 的所有全局物理数据集均逐值相同，
`statistics.h5` 的四个数据集也全部逐值相同；只有 `state.h5/rank_extras`
的 8 个字不同，最大绝对差约 `4.47e-315`。按分区表反解得到：

- rank 1/3 的 rho 缓存，局部 `(-5,-5,-5)`、`(-4,-5,-5)`；
- rank 1/3 的 pressure 缓存，局部 `(13,18,-5)`、`(9,20,-5)`。

局部网格为 `8×16×8`、hm=5，以上全部属于至少两个方向越出物理节点的角区，
不是有效单方向面 halo。`qswap` 的物性恢复只更新物理块及面 halo；
修复前的 `commarray::allocommarray` 仅在 AIR5 分支初始化 rho 角区，非反应
rho 和其他原始变量缓存没有对应初始化。精确检查点却保存整个缓存存储。
GPU 初始缓存也从主机全数组上传，因此不能假定另一后端天然没有此问题。

用户已于本轮明确批准分配时初始化。`commarray::allocommarray` 仅在
`.not.lcomb` 分支中，将 rho/vel/prs/tmp 在至少两个方向越界的位置置为 +0；
保留 q、物理节点、单方向有效 halo 及原逐值一致门槛。原文件换行及 AIR5
分支未改变。GPU 从同一主机初始缓存上传，因此同步获得确定的角区存储。

先前失败的 CPU `2×1×2` 针对性复验通过，证据为 `cpu_corner_fix_target.xml`。
与修复前冻结的连续运行文件对照，11 个全局物理 q/缓存数据集和所有新增统计
数据集逐值一致。随后同样的 14 项完整矩阵全部通过，且新增检查按几何位置
核对六个原始变量缓存角区为逐位正零，不按数值大小过滤、截断或忽略差异。
这关闭的是未定义存储导致的精确恢复缺口，不是主流场跨拓扑误差或真实 SBLI 验收。

## TB1.2 动态产品坐标显示增量

实现：`insitu_catalog_json.c::astr_catalog_display_coordinates` 统一解析参数，
`insitu_run_config` 保存规范元数据；MPI 广播两项标志和七个实数。
既有控制文件已保存原产品 JSON，恢复时从该记录重建坐标元数据，不改变
已有渲染接口的 11 个标签、7 个整数和 31 个实数布局，不另开配置文件。
`runtime_products.py` 使用 `render_coordinates.py` 安装辅助坐标 props。

图像产品的可选 `display.coordinates`：

| 项 | 含义 |
| --- | --- |
| `visible` | 默认 true，可显式隐藏坐标轴 |
| `reference_length` | 正的固定参考长度；提供时刻度为 `(x_i-origin_i)/reference_length`，不改几何 |
| `origin` | 原生坐标中的三分量原点，默认全零；须与参考长度一起提供 |
| `display_scale` | 三个正的显示倍率，默认全 1；不缩放颜色值、求解变量或存档坐标 |

默认单位由既有原生元数据提供：SI 坐标标为 m，无量纲坐标标为 1。
显示坐标为 `origin_i + scale_i*(x_i-origin_i)`，仅设置 `vtkActor` 属性，
没有调用改变数据的 Transform filter。相机位置、焦点和向上向量采用对应
显示映射；`parallel_scale` 仍为用户显式配置的显示取景高度，不自动放大。
放大有文字说明，刻度仍对应原坐标或显式归一化后的坐标。

[VTK CubeAxesActor](https://vtk.org/doc/nightly/html/classvtkCubeAxesActor.html)
将轴线空间 Bounds 与刻度 AxisRange 分开；
[vtkActor](https://vtk.org/doc/nightly/html/classvtkActor.html) 的显示变换不修改 mapper 的输入数据。
因此归一化只改变刻度，显示放大只改变渲染属性，不进入梯度、流线积分或数值存档。
二维轴对齐截面显示两根物理坐标轴，三维等值面/流线采用实际产品包围盒。
逻辑索引截面只有在对应物理坐标确实恒定时才按二维轴处理，不能把曲面逻辑方向
直接当作固定物理坐标。斜截面当前保留 x/y/z 物理包围盒，不宣称局部二维基底已验收。

背景网格默认关闭。短投影轴使用端点刻度；屏幕上不足以容纳两个数字的轴
保留轴线，并把明确命名的数值范围写到图中说明区。这只是标签布局，不裁切
或拉伸流场，不以图像像素阈值更改任何物理诊断。辅助字形与刻度可在主机
生成，和流场等值面几何分开；没有新增后者的主机回读。

例子：`scripts/output/input.products.tgv.coordinates.example.jsonl`，
三个目录记录经原生解析器通过，渲染辅助模块及示例均进入现有 CMake 安装清单。

已观测证据目录为 `tests/gpu_validation/out/tb1_coordinates_20261102/`：

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 新坐标参数/恢复/MPI 广播及既有目录回归 | 64 通过，6 个其他可选接口测试未选中；其中新坐标 16 项均通过 | `schema.xml` |
| 最后增补的逻辑/物理轴区分与纯模型检查 | 4 项通过 | `model_final.xml` |
| 已知平面和长方体的 VTK 实际出图 | 原生/归一化/显示放大均检查原输入点不变和数值范围；发现并修订短轴标签挤压 | `vtk_reference*` 参考目录及实际终端结果 |
| 32³ TGV 两步真实设备图片 | 6 项通过：NP1/2 standard-device、NP2 direct-device、NP4 xy standard-device，以及 NP1/2 轴对齐物理截面 | `render_verified.xml` |
| 异常隔离后的并行回归 | NP2 direct-device 与 NP4 xy standard-device 两项通过 | `render_parallel_final.xml` |
| 最后物理轴识别接线复核 | NP1 standard-device 的六产品两步测试通过 | `render_axis_mapping_final.xml` |
| 有限显示倍率造成派生坐标溢出 | 只停用该产品，健康图片继续输出到末步 | `render_failure_isolation_normal.xml` |

显示溢出探针最初用了次正规参考长度，在当前进程浮点环境下先被配置检查拒绝，
没有实际覆盖运行时故障。改用正常有限浮点数 `display_scale=1e308`，才确认
运行时异常经过既有集体失败隔离；没有修改进程浮点模式、容差或物理场。

NP4 首次在既有 admission 被拒绝，随后发现统一笛卡尔流线还另有限定 NP≤2。
已仅补齐 32³、`2×2×1`、非反应周期 TGV 的本地短测入口和共享双卡识别，
以及该分区下统一网格流线的入口。共享预算仍使用显式配置及原生资源监测，
没有新增固定资源上限。未扩大 AIR5/CURVE 或生产网格准入。NP4 实际图片
通过不等同于新的解析跨分区流线轨迹门槛已经验收，后者仍须单列。
本地四个 MPI rank 共享两张 GPU，仅作为正确性和功能证据，不作为四卡性能数据。

首轮真实图片检查了切片、等值面、流线和缩放说明的布局及非空性；标准设备
和设备直连记录的流场/最终几何回读均为零。原始失败、初版布局和最终图片
分别保留，没有覆盖或删除早期证据。未将本次两步出图作为物理统计验收。

此增量尚不覆盖旧固定预设/预览出图的自动坐标接线、斜截面局部二维坐标、
高分辨率展示的统一字号检查及完整开关成本测量。TB1.2 和整个 TB1 都仍未完成。

## TB1.3 测站数值剖面首轮接线

更新于 2026-11-03。新增 `station_profile` 数值产品，沿用同一动态清单、
完整步入口、统计窗口与检查点状态，不新增并行框架。

- 固定全局 i 测站，记录实际 x；j 为剖面索引。`station_span` 显式采用
  所选 k 区间的物理长度权重，`station_line` 则仅取一个 k，保留逐点统计。
- 速度请求使用完整三分量及六个对称二阶矩；压力、温度和密度单独请求时
  不分配不需要的速度二阶矩。Favre 密度仍不准入，CSV 中的 rho_time_mean
  明确是普通时间均密度。
- 先累计各节点局部时间中心矩，再汇总 Reynolds/Favre 剖面。Favre 均值
  使用时间均密度权重，密度应力先归约，再除以展向平均密度得到协方差。
  RMS 由组合方差开方，不加入局部均值的展向变化项，保留交叉协方差符号。
- `span_node_weights` 使用非均匀笛卡尔展向节点的梯形长度权重；周期末端
  权重折回唯一节点，MPI 共享节点只贡献一次。全周期 j 剖面的末行使用
  唯一节点值，但坐标仍为实际终端坐标。
- GPU 的源矩状态和展向归约保留在设备；固定形状权重/归约工作区复用，
  每次仅回读 `nterms × local_nj` 的一维加权结果及一个整数状态。
  CPU 只处理一维合并、归一化和 CSV。诊断测试另行显式输出的二维瞬时
  截面及检查点暂存，不计入普通剖面零整场下载的表述。
- CSV 保存到 `profile_products/ID/version.../segment...`，含真实坐标、
  单位、空间测度、请求窗口、实际覆盖包络、累计时长及采样段数。
  文件关闭并原子发布后提交产品时钟，不做 CRC 或保存后重读校验。
- 产品写出失败只停用该产品。有效消费者仍可共享同一历史；没有消费者
  的组停止采样并断开区间端点，重新启用不跨缺口积分。

代码入口：

| 部分 | 实现 |
| --- | --- |
| 共用节点矩投影与一维归一化 | `insitu_sample_moments::sample_profile_terms/finish_sample_profile` |
| 展向物理长度及周期别名 | `insitu_spatial_statistics::span_node_weights` |
| CUDA 分块归约及传输计数 | `insitu_sample_moments_gpu::profile_reduce_kernel/reduce_selected_profile_gpu` |
| MPI 唯一节点、几何及小型结果合并 | `insitu_selected_runtime::selected_runtime_profile` |
| 产品解析及独立 CSV | `insitu_catalog`、`index_slice_archive::write_station_profile` |

配置说明见 `scripts/output/RUNTIME_PRODUCTS.md`，16 区间 TGV 示例为
`scripts/output/input.products.tgv.station.example.jsonl`。当前是 numerical-only
`catalog_stats` 入口，不把新均值图像、壁面单位或真实 SBLI 宣称为已交付。

### 验证证据

统一目录：`tests/gpu_validation/out/tb1_station_profiles_20261103/`。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新采样模式、非法记录隔离及原清单解析 | 51 项通过；24 项既有可选接口测试跳过，不算通过 | `catalog.xml` |
| 独立旧速度统计参考、强变密度及非均匀展向长度 | CPU NP=1/2/4 的 6 项、GPU NP=1/2/4 的 3 项均通过；含点模式、降序坐标、周期别名和空所有者 | `cpu_projection.xml`、`gpu_projection.xml` |
| 真实 16³ TGV、12 完整步；NP1、NP2 x/y/z、NP4 xy/xz/yz，CPU/GPU | 14 项通过；与同轨迹二维瞬时截面独立积分比较，最大缩放差 8.88e-16，最大绝对差 4.26e-14 | `runtime_matrix.xml` |
| TGV 同后端 NP2 z，连续 12 与 5+7 步 | 2 项通过；完整 state/statistics 及恢复后 CSV 逐值一致 | 同上 |
| M12 64×32×24、原 dt=.02、10 完整步、原随机吹吸，相同 7 种拓扑 CPU/GPU | 14 项通过；独立积分参考最大缩放差 2.66e-15，最大绝对差 1.78e-14 | `m12_matrix.xml` |
| 密度单独、压力/温度单独及输出失败隔离 | 4 项通过；独立失败不影响健康产品，开启剖面前后的权威 state 逐值相同 | `dependencies_isolation.xml` |
| CPU/GPU 同拓扑，比较所有输出帧 | TGV 最大缩放差 1.44e-15、绝对差 5.68e-14；M12 分别 9.54e-14、9.79e-14 | `tgv_same_topology.json`、`m12_same_topology.json` |
| 新归约核纯算子检查 | 两张物理卡 Memcheck 均零错误、零泄漏；Racecheck 零错误、零警告 | `memcheck_gpu0.log`、`memcheck_gpu1.log`、`racecheck_profile.log` |

RMS 验收使用平方值，原 RMS 差另行报告。TGV/M12 同轨迹独立参考的
RMS 原值最大差分别为 1.17e-16、1.35e-16；同拓扑 CPU/GPU 全帧比较
分别为 4.93e-16、7.02e-16。没有放宽 2e-10 缩放门槛。

测站新测试共 34 项真实运行检查通过。最初测试将 interval=4 的最后帧误认为
必在 step=12；已有时钟从统计 ready 的 step=1 开始，实际帧为 1/5/9。
测试改为明确逐步发布，以核对完整窗口，没有修改生产时钟语义或统计公式。
初次 CPU/GPU 构建另发现计时步号错从 `parallel` 导入，已改为 `commvar`。
原失败证据保留，不作为数值错误或已通过结果计数。

### 短测成本与剩余范围

本地 GPU NP1，包含初始化的全部产品调用：TGV 的一维剖面归约中位数
0.320 ms、CSV 发布 0.144 ms；M12 分别 0.336 ms、0.225 ms。
这些是带计时的微型样本，不是生产性能或关闭/开启总开销对照。
单次归约最大显式 D2H 为 TGV 3076 字节、M12 6340 字节，包含一个状态整数；
不是整场下载。未宣称 MPI/UCX 隐式传输或 CUDA 整进程显存峰值已全部测量。

同拓扑对照脚本 `tests/gpu_validation/analyse_station_profiles.py` 只读取已保存
证据，不重新推进；它不会把先前 M12 跨拓扑主流场差异改判为通过。
TB1.3 的首轮数值剖面已接通，但正式 SBLI、动态改清单组合生命周期仍须
继续覆盖；TB1.4 完整剪切/热流与壁面参考标度、TB1.5 厚度积分、TB1.6 输出接线以及
TB2–TB4 均未因此关闭。整个 goal 继续保持进行中。

## TB1.4/TB1.5 壁面与厚度算子基础

2026-11-03 新增 `src/insitu_boundary_layer.F90` 和隔离的 CUDA 端
`src_gpu/insitu_boundary_layer_gpu.cuf`。本节记录先完成的组件验证；后续真实
完整步壁面采样、动态统计与数值输出见下一节。壁面标度和厚度产品尚未接通，
仍不能据此关闭 TB1.4/5。

### 定义与代码

- `wall_transport` 构造完整牛顿应力
  `sigma=mu*(grad(u)+transpose(grad(u))-(2/3)*div(u)*I)`，再取
  `tau=sigma*n-n*dot(n,sigma*n)`。n 从固体指向流体。底壁 x/z 剪切分别
  包含 `mu*(du/dy+dv/dx)`、`mu*(dw/dy+dv/dz)`，不遗漏吹吸切向变化。
- 同一函数计算 `q_into_wall=kappa*dot(grad(T),n)`，只有傅里叶导热，不含
  吹吸焓、动能或黏性功。调用方提供与当前本构、量纲及无量纲化一致的
  有效 mu/kappa；组件不重新选择黏度规律，不改求解器。
- `wall_stencil_transport` 复用现有七节点六阶权重，支持中心和全部偏置
  位置，用完整度量转换为物理梯度。CUDA 两个批量入口均显式同步，输入
  输出由调用方持有，仅主动读取一个整数状态，不下载壁面场。
- `wall_reference` 接收已按同一时间/展向权重平均的剪切矢量、rho、mu、
  Tw 和导热。先对平均矢量取模，再构造 `u_tau=sqrt(|tau_mean|/rho_mean)`、
  `nu=mu_mean/rho_mean`、`ell_nu=nu/u_tau`；Cf 保留 +x 符号，Cq 使用
  固定自由来流。可选 Tr/cp_inf 仅用于批准的统计 Stanton 数。
- 零剪切或不可安全表示的壁面标度仅使 scale_available 为假；Tr-Tw 为零
  仅使 stanton_available 为假，不补分母。配置层的 Tr/r 互斥及恢复因子
  换算尚未接线，不把此函数当作完整产品。瞬时物性与通量须先算再平均。
- `boundary_layer_thickness` 只处理小型一维剖面，显式接收 rho_e、Ue
  及参考带索引。从参考带向壁面回查相连的 `U_F>=.99*Ue` 区域内边界，
  线性交点，保留 multiple_crossings。外流状态不取最大值或默认顶面。
- 位移/动量厚度节点被积量分别为 `1-rho_mean*U_F/(rho_e*Ue)` 和
  `[rho_mean*U_F/(rho_e*Ue)]*(1-U_F/Ue)`。沿真实 y 做累计梯形积分，
  分别输出至 delta99 和显式物理 y 上限的结果，包含部分网格区间。
  delta99 是距首节点壁面的距离；上限是原生 y 坐标。无交点不丢弃剖面
  或显式上限积分，不截断反向速度、不平滑、不外推。
- 不可用项有独立标志，不能把初始化的零当作结果。Cf 最后乘 2，速度
  交点用缩放量插值，避免最终结果可表示而中间乘法/差值溢出。

### 验证与范围

证据目录：`tests/gpu_validation/out/tb1_wall_thickness_20261103/`。

| 项目 | 实际结果 |
| --- | --- |
| 根 CMake CPU/GPU 主程序及新探针 | 构建通过；新增实现前探针因缺少模块构建失败，随后使用真实新模块验证 |
| CPU 算子，NP=1/2/4 | `cpu_final.xml` 三项通过 |
| CUDA 算子，NP=1/2/4，共享本地双卡 | `gpu_final.xml` 三项通过；最大缩放差约 1.20e-13、绝对差约 1.15e-12 |
| 两张卡纯算子 Memcheck | `memcheck_gpu0.log`、`memcheck_gpu1.log` 均零错误、零泄漏 |
| CUDA Racecheck | `racecheck.log` 零错误、零警告 |
| 非均匀 16/32/64 网格的动量厚度诊断 | 误差约 8.07e-4、2.01e-4、5.00e-5，呈二阶收敛，不套用 2e-10 CPU/GPU 门槛 |

解析反例含空间变化吹吸、刚体旋转、法向反转、非有限输入、零剪切和零
恢复温差、七种模板位置、二次拉伸坐标与六次多项式、内部过冲、阈值平台、
无交点、反向流及有限极值下的运算顺序。NP 矩阵仅证明多进程设备绑定下
的独立算子，不包含新壁面场的 MPI 模板采样或归约。探针显式下载 12 点
小数组比对，没有新增三维统计或真实壁面场回读，也未作生产性能结论。

### 该增量结束时的待确认和接线

D14 已冻结显式外流带，但未冻结带内归约和标量平均方式。本轮已提问：
是否在指定 j 区间按真实壁法向长度梯形加权，Ue 取 Favre 流向速度，
rho_e/pe/Te 取 Reynolds 均值，单节点区间直接取该节点，并记录定义。
确认前不固化真实参考提取；当前算子只接受显式参考，不自动补值。

后续已接通的完整壁面瞬时物性/通量、按需累计、精确续接和参考标度见下文。
厚度 CSV 及组合诊断仍待实现；原通道和旧 SBLI 壁面输出保持各自定义，
未静默替换。整个 goal 保持进行中。

## TB1.4 完整壁面采样、平均与续接

2026-11-03 已接通 FP64 非反应笛卡尔底壁的真实数值链路，不只停留在
独立算子。新增字段为 `temperature_gradient_x/y/z`、`wall_shear_x/z`、
`wall_heat_into_wall`、`dynamic_viscosity`。温度梯度复用私有七节点六阶
离散及完整度量投影，未读取旧 RK 梯度或修改求解器。

### 流程与数据驻留

1. 普通瞬时数值截面可显式保存新壁面场及梯度，供独立参考；这是一条
   主动请求的二维数据存档路径，不把其下载算成普通统计路径的零下载。
2. 新 `wall_profile` 使用 `reynolds`、`selected-time` 和独立 CSV 时钟。
   `wall_span` 取真实展向长度权重，`wall_line` 保留一个 k 节点的逐点历史。
   都选择全局 j=0 及明确的 i/k 区间，不依赖 MPI 分区编号。
3. 每个完整步先计算当前有效 mu/kappa、完整分子剪切和导热，再送入原
   selected-statistics 累计，不用平均温度/速度先求通量。密度隐式记录为
   普通时间均值。可分别只请求黏度、剪切或温度/热流，不强制输出全套字段。
4. 周期末端先映射到唯一物理节点。末端部分区间拆为实际请求的首/尾片段，
   不把中间未请求区域扩大成采样域；GPU 按索引散写同一组长期累计。
5. GPU 私有模板通信使用 CUDA-aware MPI。借用的二维设备结果在显式同步
   完成累计后释放，未回读壁面场；无未声明的主机载荷回退。源码显式传输
   与 MPI 内部可能发生的暂存区分记录，求解器 halo 后端仍独立配置。
6. 归约后只下载一维结果和状态整数，再由 CPU 合并/写 CSV。纯统计采样
   省去未使用的设备坐标缓冲及打包核，仍保留所有实际计算核的显式同步。
7. 几何、字段和实际历史仍由同一注册表管理。新壁面组使用已实现的窗口、
   唯一所有权、缺口端点及 statistics.h5 原始字序列化；派生索引缓存不持久化，
   恢复时重建。未另设统计文件、CRC 或保存后重读校验。

入口为 `index_slice_archive::sample_selected_wall_group/write_wall_profile`、
`selected_statistics::sample_external` 和
`insitu_sample_moments_gpu::push_scattered_moments_gpu`。资源预估按实际
规范片段及 MPI 全局最大私有工作区执行，沿用已配置预算，不增加硬编码
资源上限。示例为 `scripts/output/input.products.wall.statistics.example.jsonl`，
其 64×32×24 索引仅对应已批准小夹具。

### 已观测验证

证据目录：`tests/gpu_validation/out/tb1_wall_runtime_20261103/`。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新字段/壁面产品清单解析及原清单回归 | 55 项通过；24 项原可选接口测试跳过，不算验收通过 | `catalog_profiles.xml` |
| 新散写累计及规范周期区间，NP1/2/4 | 6 项通过；含任意目标顺序及非法目标不越界 | `scatter_math.xml` |
| M12 完整步瞬时梯度/分子壁面量，CPU/GPU 各 7 种拓扑 | 14 项通过，独立参考最大缩放差 6.43e-14、绝对差 2.48e-13 | `final_matrix.xml` |
| 同一夹具壁面均值，全/部分展向、末端单线及仅黏度/剪切/热量请求 | 14 项通过；与独立时间/展向积分比较，最大缩放差 8.89e-16、绝对差 8.89e-15 | 同上 |
| CPU/GPU NP2 z，连续 10 与 5+5 步精确续算；统计开关对照 | 2 项通过；完整 state/statistics 和恢复后 CSV 逐值一致；关闭壁面历史的主状态逐值相同 | 同上 |
| 壁面均值所有输出帧、CPU/GPU 同拓扑对照 | 7 对通过，最大缩放差 7.11e-16、绝对差 7.11e-15 | `wall_same_topology_final.json` |
| 原测站数值产品、标量依赖、独立失败及精确续算 | 34 项回归通过 | `station_regression.xml` |
| 双卡纯算子 Memcheck 与 Racecheck | 零错误、零泄漏；Racecheck 零错误、零警告 | `scatter_memcheck_gpu0/1.log`、`scatter_racecheck.log` |
| 与去除坐标包前已冻结的 3 种壁面产品相比 | 420 份同名 CSV 全部逐字节相同，缩放差 0 | `frozen_shared_wall_profiles.json` |

本轮正式最终壁面矩阵共 30 项真实运行检查。保持原 M12 的 dt=.02、10 个
完整步及原随机吹吸；NP=1、NP2 x/y/z、NP4 xy/xz/yz。原算例只读复用，
本地 NP4 共享双卡不是四卡性能或统计定常验收。新增 Cf/Cq/St、壁面单位
剖面和厚度输出不在这 30 项通过结论内。

### 参考错误的定位与修正

- 首个夹具遗漏检查点接口要求预建的输出根目录，失败在 RK 之前；修正
  测试准备，不改变生产 I/O 规则，原失败留在 `first_cpu.xml`。
- 独立参考最初把检查点中的周期重复末端直接用于横向模板，但现有私有
  采样明确用 modulo 规范化所有周期支持节点。差异集中在 k=N；修正参考
  时仍比较全部输出节点，没有丢弃端点或放宽门槛。
- 原 CPU 该短轨迹的重复末端与首端存在非零差：密度约 4.45e-8、温度约
  5.32e-6。缓存与同相位 q/EOS 重建一致，不是本次新增统计造成的旧缓存。
  该原始重复节点差异尚未单独完成根因审计，未擅自修改 CPU/GPU 推进。
  新诊断的材料温度与梯度统一来自相同规范节点，避免自身混用重复端点。
- 无量纲物性实际默认 Sutherland 常数为 110.3 K；独立参考误用了维度
  公式的 110.4 K，造成统一比例误差。修正参考而非改动现有物性模型。

### 成本与剩余项

最终 6 个壁面请求的本地 GPU NP1 短测，含 step=0 初始化的 11 次累计采样
阶段中位数约 5.65 ms，
单次一维归约约 0.343 ms，CSV 发布约 0.133 ms。它们分别为所有请求的
完整采样阶段和单产品调用，不能相加后当作纯 kernel 性能，也不是生产
规模或正式重复计时。NP1 长期受控状态/索引约 0.620 MB 设备、0.099 MB
主机；私有模板临时区另计，未宣称进程显存瞬时峰值已完全测得。
单次一维结果最大显式下载为 6244 字节；周期散写上传的是整数索引，不是
流场。GPU 模板阶段仅应用层状态整数回读，显式诊断截面和检查点暂存另列。

本节之后完成的参考标度、Cf/Cq、显式恢复输入 St 和网格单位见下文；
与测站剖面的关联仍待接线。厚度参考带归约方式继续等待前述确认。
动态改清单/缺口/重试的新增真实壁面组合还需补测。bc42/411/421 和维度
物性算子已有代码，但未做该新增链路的真实流动验收；读取式笛卡尔网格
目前仍被 lreadgrid 入口限制，尚未准入。真实 SBLI 也不能由本次 bc41 短测代替。
TB1 及整个 TB1–TB4 goal 均未标记完成。

## TB1.4 壁面参考标度与固定参考系数

2026-11-03 后续增量：`wall_profile` 可显式配置 `wall_reference` 对象。
空对象只增加 u_tau、nu_wall、viscous_length、Cf 和 Cq_inf；输入正的原生
恢复温度或非负恢复因子才增加统计 Stanton 数，二者互斥，不填经验值。
r=0 作为合法极限输入保留。缺少对象时，原始壁面 CSV 格式不变。

计算先使用相同窗口、节点和时间/展向权重下的壁面量均值。摩擦速度取
平均剪切矢量的模，不是瞬时模长的均值。主系数使用算例固定来流密度和
速度模长；流向 Cf 保留 +x 剪应力符号。无量纲 cp_inf=1/[(gamma-1)Mach²]，
维度路径使用算例 cp。St_stat 使用平均导热热流与平均壁温之比，温差为零
只将 St 标为不可用。标度/系数有各自可用标记，不可用值为 CSV 空单元，
不填零、不调整物理分母；固定来流速度为零也不使有效局部标度失效。

配置入口在 `insitu_catalog_json`、`insitu_run_config`、`insitu_catalog`；
自动依赖在 `selected_request_fields`；一维计算及 CSV 在
`index_slice_archive::write_wall_profile` 和 `insitu_boundary_layer::wall_reference`。
配置重新解析保存的有效 JSON，不加检查点附属文件或改渲染 ABI。
仅改变恢复输入的产品共享原采样历史，不追加三维数组或重复累计。
示例为 `scripts/output/input.products.wall.references.example.jsonl`。

证据目录：`tests/gpu_validation/out/tb1_wall_references_20261103/`。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新配置及旧清单解析 | 69 项通过、24 项可选测试跳过 | `catalog.xml` |
| 非法引用独立失败、NP1/2/4 广播及恢复输入值/形式版本更新 | 17 项通过 | `reference_schema_versions.xml` |
| 标度、恢复温度、极值浮点算术及原壁面/厚度算子 | NP1/2/4 三项通过 | `operators_fixed.xml` |
| M12 CPU/GPU 各 7 种拓扑的原始均值与标度/系数独立公式 | 14 项通过，最大缩放差及绝对差均约 6.94e-18 | `runtime_matrix.xml` |
| CPU/GPU NP2 z，完整 10 步与 5+5 精确续算及关闭派生请求对照 | 2 项通过，主状态/累计状态逐值一致，续算 CSV 逐字节相同；冲突的磁盘恢复输入不覆盖有效定义 | 同上 |
| 同拓扑 CPU/GPU 全部壁面 CSV 帧 | 7 对通过，最大缩放差 1.23e-15、绝对差 7.11e-15 | `same_topology.json` |
| 后续仅计时埋点的针对性复验 | CPU NP1、GPU NP4 xy 和 GPU NP2 z 精确续算，共 3 项通过 | `timing_final.xml` |

极值算子首轮失败原因是本次新增系数重建直接调用 NVHPC SCALE，指数为
1024 时产生 Inf，而数学结果仍可表示。通过拆成两次有界缩放修复，未改
容差、分母或求解器。原始失败保留在 `operators.xml` 和
`scale_edge_reproduction.log`，不将其记为通过。

计时复验中，CPU NP1 rank0 的参考计算中位数为 0.033 ms，GPU NP4 xy
rank0 为 0.053 ms；每次完整壁面产品发布中位数分别约 0.406/0.696 ms，
CSV 发布部分约 0.144/0.169 ms。完整发布含一维归约、参考计算及写盘，
不含共享的逐步采样；各指标不能相加作为整段开销。这些是单次短测、
混合产品及不同拓扑的成本记录，不是 CPU/GPU 加速比或正式性能验收。
开启四种派生请求仍为原六组历史；GPU NP4 xy rank0 的受控长期状态/索引
约为设备 305528 B、主机 48640 B，临时模板区另计。派生计算只操作已回读
的一维壁面曲线，无新三维场或设备最终几何回读。

本增量不代表壁面单位速度/应力剖面、网格单位或厚度产品已接通。
外流参考带加权细节仍待人工确认；TB2–TB4 尚未开始，完整 goal 继续进行。

## TB1.4 传统网格壁面单位

2026-11-03 已将网格诊断接入所有启用 `wall_reference` 的壁面数值表，
不另设默认输入或重复累计。每个流向节点均使用该行同一统计黏性长度，
不是先用瞬时尺度归一化后平均。

| 数值列 | 定义 |
| --- | --- |
| y_first_plus | j=0 到第一个物理离壁节点的距离/统计黏性长度 |
| dx_minus_plus、dx_plus_plus | 全局 i 方向左右相邻真实单元长度/同一标度；不假设均匀间距 |
| dz_min_plus、dz_max_plus | 所选展向区间内单元长度的最小/最大值/同一标度；单线选择改用该节点相邻单元 |

非周期端点不存在的左右邻格留空；周期端点采用实际末端单元的长度，
不从外推 halo 取间距。零剪切等导致标度不可用时，依赖的网格列也留空。
这些是统计参考尺度下的网格分辨率诊断，不代表全域处处相同的局部 y+。

定义与归一化入口为 `insitu_boundary_layer::wall_grid_spacings/wall_distance_plus`；
`selected_runtime_wall_grid` 复用原坐标唯一所有权规则，合并 x/z 一维坐标轴
及两个壁法向坐标；不复制三维网格，不回读设备场。内存预估包含已持有
的一维剖面、参考表、网格列和临时坐标轴，继续使用既有可配置预算。
共享的坐标提取函数从测站过程内部移至模块私有层，算法未改变。

证据仍位于 `tests/gpu_validation/out/tb1_wall_references_20261103/`：

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新产品测试在实现前运行 | 因缺少 y_first_plus 列按预期失败，原始输出保留 | `grid_red.xml` |
| CPU/GPU 构建的独立网格/壁面算子，各 NP1/2/4 | 共 6 项通过；覆盖非均匀、反向坐标、周期接缝、单节点与部分展向带 | `grid_operators_cpu.xml`、`grid_operators_gpu.xml` |
| CPU/GPU 各七拓扑及两项精确续算/开关对照 | 16 项通过；自动依赖仍共用六组历史，公式最大缩放差和绝对差均 2.09e-17 | `grid_runtime.xml` |
| 全/部分展向与末端单线，同拓扑 CPU/GPU 所有 CSV 帧 | 七对通过，最大缩放差 1.20e-14、绝对差 1.97e-12 | `grid_same_topology.json` |
| 共用坐标提取改动后的测站回归 | 34 项通过，含 TGV/平板多拓扑、标量依赖、失败隔离及精确续算 | `grid_station_regression.xml` |

最新短测 CPU NP1/GPU NP1 rank0 中位数：一维几何准备约 0.104/0.166 ms，
参考计算（包含几何准备）约 0.140/0.200 ms，单产品完整发布约 0.422/0.525 ms，
CSV 发布约 0.147/0.152 ms。不同阶段存在包含关系，不相加作为整步开销。
新增网格表没有设备分配或额外 D2H；GPU NP1 单次原一维统计结果下载仍最大
6244 B，长期受控状态/索引仍约设备 619952 B、主机 98800 B。
进程显存瞬时峰值及完整产品开关总成本尚需阶段整体验收，不能从上述受控
数组字节数推定。当前证据是小型数值与生命周期验证，不是生产 SBLI 定常验收。

后续依赖仍为：将测站速度/协方差剖面与相同实际采样身份的壁面参考关联，
不能只比较覆盖时长；按待确认的外流参考带定义接通厚度与 Reynolds 数。
读取式笛卡尔网格仍需几何识别后准入，不能把 lreadgrid 一概等同于 CURVE，
也不能直接撤掉检查而将曲壁当成法向 +y 的平板。原 HCW 小型 SBLI 使用
读取式网格，因此该项仍是新统计/壁面产品真实 SBLI 验收的前置工作。
TB2–TB4 与阶段整体验收继续保留，整个 goal 未完成。

## TB1.3/TB1.4 同历史壁面单位剖面

2026-11-03 新增 `station_profile` 的可选 `wall_units` 引用，值是清单内
另一个 `wall_profile` 的 ID。目标须开启 wall_reference、覆盖该流向测站，
并使用相同 k 范围和相同展向处理方式。输出保留原始统计列，新增 y+、
速度/RMS 壁面单位、协方差/u_tau² 及密度加权应力/(rho_wall*u_tau²)。
应力保留符号，不加负号；壁面密度不替换为剖面各点密度。压力、温度仍为
原单位的均值，不另造归一化。只选压力/温度时可只加 y+，不强制累计速度。

引用读取的是截至当前完整步的累计状态，而非最后写出的壁面 CSV；壁面
与剖面输出周期不同也可组合。没有目标、空间处理不匹配、采样历史不同或
摩擦标度不可用时，保留原始剖面，依赖的归一化列留空并写明原因。
不替换入口标度、不清空其他产品、不把不匹配的两段统计拼成一段。

### 无日志的精确采样身份

仅靠起止时间、积分时长和段数不足以判断是否能组合。本轮首先增加了
真实采样的反例：0–2/5–10 与 0–3/6–10 的单位步长序列均有相同首末时刻、
7 单位覆盖时长和两段历史，但实际样本不同。不能只比较这些标量后放行。

`selected_statistics::accept_sample` 现对每次正权重更新，按“前缀身份、
完整步/物理时间、积分权重原始位”复用同一步的更新身份，否则分配新整数。
从相同前缀归纳可知：相同身份表示该注册表内相同的实际加权样本序列。
这不是可能碰撞的散列，也不按步保存日志或新增逐步数组分配。比较只在
同一注册表内进行，不能把两个独立作业的同号整数当成相同历史。

每组新增固定 40 B 主机元数据，注册表计数器为 8 B；不新增设备状态。
只有身份和计数器需持久化，比原统计块每组增加一个 64 位字、块头增加
一个字。统计内部格式升级为 v2，仍写入原 statistics.h5 的 rank_words，
不加新文件。缺少身份的旧 v1 selected-time 块不作为 v2 的精确恢复输入。
其他统计来源的格式不变。步内临时匹配缓存不保存，完整步恢复后的重复
观察不重新更新，下一新步按已保存身份继续。

### 接线与已观测证据

入口为 `selected_statistics::matching_coverage`、
`selected_runtime_wall_scale`、`index_slice_archive::write_station_profile` 和
`insitu_boundary_layer::wall_scaled_quantity`。共享的原 GPU 累计与展向归约
未更换；归一化只用允许回读的一维结果。原始剖面、引用壁面表和临时
工作区同时存活的内存纳入既有可配置预算，不新增统一上限。
清单广播、语义版本和有效 JSON 恢复都保存引用关系，渲染 ABI 未改变。
示例为 `scripts/output/input.products.wall.units.example.jsonl`，只对应小夹具。

证据：`tests/gpu_validation/out/tb1_joint_wall_profiles_20261103/`。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 采样身份反例及原注册表生命周期，CPU/GPU 各七拓扑 | 14 项通过；含同号前缀更新、缺采、晚加入、失败隔离及逐位续接 | `registry_cpu.xml`、`registry_gpu.xml` |
| 引用键、非法产品隔离、MPI 广播/版本及有效清单恢复 | 97 项通过、12 项缺少其他可选接口的测试跳过 | `schema_final.xml` |
| 首个真实 CPU NP1 壁面单位剖面 | 1 项通过；最大缩放差 4.16e-16、绝对差 4.55e-13，RMS 原值差约 1.39e-17 | `first_cpu.xml` |
| GPU NP4 xy；CPU/GPU NP2 z 精确续算及移除归一化请求对照；缺失/错展向参考 | 5 项通过；主状态/累计状态逐值一致，续算 CSV 逐字节相同，缺失参考不丢原始剖面 | `targeted.xml` |
| 实际动态移除/重加，CPU/GPU NP2 z，无其他消费者/持续共享消费者 | 4 项通过；真实缺采不组合，持续有效采样不重置；原始统计均保留 | `gaps_fixed.xml` |
| 不可用原因文本的最终 CPU/GPU 复验 | 2 项通过，明确区分缺少目标和空间处理不匹配 | `reasons_final.xml` |
| 新速度/协方差/密度应力归一化及原算子，CPU/GPU 各 NP1/2/4 | 共 6 项通过 | `operators_cpu.xml`、`operators_gpu.xml` |

动态首轮失败保留在 `gaps.xml`：归一化列已经正确留空，失败在原因文本。
新增调用端把长度为零的可分配字符串传给 character(*) 接口，导致字符串
被截空。修正为与现有诊断接口一致的显式长度缓冲，并在发布时 trim。
未改物理量、采样身份或容差；修复后上述缺采/共享消费者及错误说明复验
通过。先前 `targeted.xml` 的有效标度数值与精确续算不经过该不可用文本分支。

上述增量结束时，完整 CPU/GPU 七拓扑矩阵尚未运行，证据约 7.86 GiB，
曾提议仅将整组磁盘提高至 12 GiB。该提议未确认。后续检查发现公式矩阵
不读取 checkpoint，却重复保存了每次约 34 MiB 的三维几何与状态；已将它
与独立精确续算测试分离，在原 8 GiB 内完成完整矩阵，见下一节。
未删除旧证据、缩小网格/步数/拓扑或取消精确续算断言；其余预算均不变。

CPU NP1/GPU NP4 xy 的引用计算短测中位数约 0.270/0.466 ms，包含额外的一维
壁面归约，不是纯标量运算时间，也不能作为不同拓扑的加速比。前者受控
主机状态/索引约 1011352 B；后者 rank0 约主机 49048 B、设备 306384 B。
GPU NP4 单次一维结果包最大 3268 B；一次产品可能有多个包，不能把这个
单包数当成整帧总传输。没有新增三维场或设备最终等值面几何回读。
完整产品开关总耗时和进程峰值仍待阶段收口。

## TB1 完整壁面单位矩阵与读取式笛卡尔几何

2026-11-03 新增证据仍在 `tb1_joint_wall_profiles_20261103/`。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 原生成式 M12，CPU/GPU 各七种拓扑 | 14 项通过；公式最大缩放差 4.16e-16、绝对差 4.55e-13，RMS 原值差 1.39e-17 | `full_matrix.xml` |
| 同拓扑全帧 CPU/GPU，包含原始列及归一化列 | 七对、280 对 CSV 帧通过；最大缩放差 9.37e-14、绝对差 4.55e-12，RMS 原值差 2.05e-14 | `full_same_topology.json` |
| 同一冻结网格改为 lreadgrid=t，CPU/GPU 七拓扑 | 14 项通过；相同公式门槛通过，每 rank 仅做一次实际几何检查 | `read_grid_matrix.xml` |
| 读取式全帧 CPU/GPU | 七对、280 对 CSV 帧通过；上述三项最大差与生成式相同 | `read_grid_same_topology.json` |
| 几何独立反例及原算子，CPU/GPU 各 NP1/2/4 | 共 6 项通过；物理点可分离、非均匀轴、未使用 halo 为 NaN；1e-12 非轴向扰动必须拒绝 | `geometry_operators_cpu.xml`、`geometry_operators_gpu.xml` |
| 新几何核的双卡 Memcheck 与 Racecheck | 两卡均零错误、零泄漏；Racecheck 零错误、零警告 | `geometry_memcheck_gpu0/1.log`、`geometry_racecheck.log` |
| 物性与壁面法向准入分离后的最终复验 | GPU NP4 xy 一项通过，原公式、字段和历史共享断言不变 | `read_grid_final_gpu.xml` |

`test_real_wall_unit_profiles` 不再写无关检查点，单目录由约 38 MiB 降为
约 5 MiB；对应精确续算测试仍显式写完整场、累计状态和有效清单，逐值一致
断言不变。读取式矩阵只读引用同一冻结 grid.h5，单目录约 3 MiB。
当前 TB1 证据约 7.98 GiB，仍在此前 8 GiB 内；没有假定磁盘扩容已获批准。

### 几何检查、相位与传输

入口为 `insitu_boundary_layer::cartesian_block_axes/cartesian_axis_direction`、
`insitu_boundary_layer_gpu::cartesian_block_axes_gpu` 和
`insitu_selected_runtime::inspect_selected_cartesian`。最后一个过程为内部调用，
不是求解器几何修复。

配置读取时不访问尚未上传的 GPU 网格。检查延后到真实完整步采样前，
只针对读取式网格执行；设备物理节点检验其各分量是否分别依赖 i/j/k，
不读角区或模板 halo。只回读本 rank 三条一维坐标轴和一个状态整数，
再用唯一所有权合并全局轴，并核对所有 rank 的局部轴与共享节点。
不以 lreadgrid=t 判定为 CURVE，也不以小于容差的非轴向项冒充笛卡尔。
各轴必须严格单调，退化单元不因该准入被放行。

原始变量、三维场、度量和最终等值面几何均未因检查而回读。64×32×24
NP1 的额外受控主机工作区为 2952 B，GPU 工作区和显式轴/状态下载各
988 B；其他拓扑更小。NP1 单次初始化记录约 CPU 0.332 ms、GPU 0.533 ms，
含预算/一致性归约和轴打包，不是纯核计时或正式加速比。设备状态字在核前初始化。
检查结果在静态网格内复用，
恢复时丢弃缓存并依据当前几何重建，未增加检查点文件。复用原预算，
不新增网格、时间或内存硬编码上限。

读取式完整壁面剪切/导热仍要求物理底壁内法向 +y；黏度不依赖壁面法向，
不受该额外条件限制。证明只准入轴对齐笛卡尔，不将旋转或曲线网格纳入
首轮新增产品。未启用 selected-time 几何检查的独立读取式新场请求仍不准入。

### 诊断记录与剩余工作

- 首个读网格短测在 RK 前找不到文件，是原 gridfile 长度 64 字符截断了
  测试绝对路径。以短相对路径链接修正夹具，不修改 CPU 接口；原失败保留
  在 `read_grid_first.xml`，后续 `read_grid_shortpath.xml` 及完整矩阵通过。
- NVHPC 不支持直接把非连续数组切片传给 host/device 共用函数；改为局部
  三分量值。其设备 64 位 MODULO 生成不支持的运行时调用，改用非负索引的
  整数商/余数恒等式；没有降低索引精度、几何精度或验收门槛。
- 此处读网格验证仍是原 M12 接口夹具，不是 HCW/SBLI 的物理验收。读取式
  几何缓存的真实连续/重启闭环、原 HCW 小型 SBLI、新壁面家族及完整开关
  成本仍待推进。D14 外流参考带的详细加权定义尚待确认，厚度产品未接通。

TB1–TB4 goal 保持进行中，TB2–TB4 未完成；未进行 Git 写操作或平台操作。

## 读取式 TGV GPU 精确续算与存储问题

2026-11-03 新增 16³ 周期 TGV、GPU NP1、dt=1e-3，连续 12 完整步与
5+7 完整步续接对照。Reynolds/Favre 两个测站产品共用同一五原始量历史，
窗口为 [0.0005,0.0115]，每步输出数值剖面。使用已冻结的正确 17³ 节点
坐标文件，配置 lreadgrid=t，不改变求解器格式、滤波或边界。
每运行目录使用已有 keep=1，保留一个完整、精确的检查点，未减少其变量。

证据：`tb1_joint_wall_profiles_20261103/read_tgv_exact_gpu.xml`。
主状态 15 个数据集和统计状态 4 个数据集逐位相同，续接 CSV 逐字节相同；
full/prefix/tail 各只执行一次实际读取式几何检查，说明恢复不复用失效的
进程内几何缓存。这里仅关闭读取式 GPU NP1 周期 TGV 的该项检查，不替代
CPU 读取式续算或壁面/SBLI 的组合生命周期门槛。

### 存储估算错误

本轮依据实际检查点尺寸估算新增约十余 MiB，但遗漏了通用 TGV 准备器的
输入模板复制。实际新目录占约 160 MiB，整组证据从约 7.98 GiB 增至约
8.14 GiB，超过此前批准的 8 GiB。已停止新增求解器运行，未自动清理文件。
这不是计算网格变为 128³，读取式物理算例和冻结输入均仍为 16³。

| 文件 | 每目录实际字节/节点形状 | 是否参与计算 |
| --- | --- | --- |
| datin/grid.h5 | 51522584 B，129×129×129 节点 | 原样例模板副本，未被读取 |
| datin/grid.cartesian.h5 | 只读链接，目标 119960 B，17×17×17 节点 | 实际显式输入 |
| outdat/new/resources/grid.h5 | 119960 B，17×17×17 节点 | 检查点冻结的正确输入 |

三个未用副本分别位于该次测试的 gpu_np1_read-full/read-prefix/read-tail
目录，合计约 147 MiB。已提问选择：批准原 12 GiB 提案并保留全部文件，
或仅授权删除这三个未用副本；未获得回答前，不删除任何已有证据。

准备器已作局部修订：`--omit-grid-resource` 现在也支持明确指定另一个
网格文件的 fresh read-grid 输入，但仍不能省略实际请求的默认模板文件。
`run_case` 的新读取式 TGV 路径在准备阶段就传入该选项及最终短网格名，
不再先复制大模板再改输入。7 项准备器单测通过，其中新增 CLI 检查以小型
临时模板验证：源文件不变、无无用副本、网格读取及 GPU 标志正确，并拒绝
遗漏正在请求的 grid.h5。此修订未启动新 CFD；原数值通过记录仍如实保留。
后续资源预估必须同时计入复制输入和产生输出，不能只算检查点与图片。

### 厚度参考的待确认细节

D14 的外流带已经获批，以下只是尚未冻结的带内估计和物性参考，不重开
D01–D42 的整体选择：

1. 候选采用真实壁法向梯形长度权重 w：rho_e/p_e/T_e 取 Reynolds 均值的
   长度平均，U_e=sum(w*rho_mean*u_Favre)/sum(w*rho_mean)，单节点直取。
   也已给出 U_e 仅对局部 Favre 均值作长度平均的另一个明确选项。
2. Re_theta=rho_e*U_e*theta/mu_e 的候选 mu_e 为算例物性关系的 mu(T_e)，
   必须标为均值状态物性，不称为 mean(mu(T))；后者需要额外采样物性历史。

[Wenzel 等（2018）](https://doi.org/10.1017/jfm.2018.179) 对比了外流与壁面
黏度定义的动量厚度雷诺数及其尺度关系，支持必须区分参考黏度的结论，
并不代表该论文采用了本项目提出的法向参考带估计。带内平均方式是工程
候选，尚未写入正式物理结果；自由来流 Cf/Cq 参考不受上述选择影响。

## TB1.2 固定预设与只读预览坐标接线

2026-11-03 继续完成独立于 D14 和新增 CFD 的坐标代码增量。前次后台检查
句柄已不存在，不能把未收取的结果当成通过；本轮重新执行无输出文件的
NP1 检查，确认正常返回。没有重启求解器或新增图片。

`render_coordinates::update_static_coordinates/static_coordinate_frame` 复用
现有坐标 actor：默认原生单位及真实比例，不另加固定预设配置项。显式显示
窗口使用原物理范围；无显式窗口则使用当前产品的 MPI 全局包围盒。轴对齐
截面保留真实截面位置，只显示变化的两根轴；斜面不猜测局部二维基底。
固定预设保留原相机和裁剪，不采用动态清单的裁剪更新。重复帧更新同一组
标尺资源，原 finalize 释放它们，不另建资源生命周期。

`device_render_pipeline::render_product` 接入固定预设及预览标注，并记录
坐标范围的来源和全局空几何状态。全局单元数采用 VTK Int64 求和；包围盒
只归约六个标量，显式窗口不重复求包围盒。全局非空但本 rank 无几何时
不隐藏公共标注。仅流线具有一维跨度时，不再误用原二维平面准入来拒绝
其坐标框；原平面产品的二维跨度检查未放宽。没有增加设备几何回读。

本轮在终端运行并检查返回码，未生成新的 XML、图片或 CFD 数据目录：

| 检查 | 观测结果 | 入口 |
| --- | --- | --- |
| 纯坐标模型及固定帧规则 | 7 项通过、13 项非本轮用例未选；含固定平面、斜面及逻辑退化面 | `test_insitu_coordinates.py` 的坐标模型选择 |
| 真实 VTK 对象、输入不变及标尺资源复用 | 正常退出；六个平面/立方体场景通过，不调用 Render | `insitu_coordinates_vtk_probe.py --metadata-only`，SDK MPI NP1 + pvpython |
| 实际包围盒函数、64 位计数及空 rank | NP1/NP2/NP4 正常退出，各 rank 均通过 | 同一 probe 的 `--collective-only`，SDK MPI + pvbatch --symmetric |
| 修改过的四个 Python 文件语法 | AST 解析通过 | 标尺模块、设备渲染模块及两个坐标测试 |

上述 NP4 也已在本轮重新执行并完整收取结果，不依赖已消失的旧句柄。
此项只是 MPI 标量元数据测试，不是四卡性能验证。真实固定预设/预览的严格设备出图、原生
单位桥接、完整主状态不变、进程资源峰值和高分辨率排版仍待验证。以上
组件检查不能关闭整个 TB1.2，也没有扩大 AIR5/CURVE/FP32 新标注准入。

### 复用已有数值证据：生成式与读取式网格

只读使用 `tb1_joint_wall_profiles_20261103/full_same_topology.json` 和
`read_grid_same_topology.json` 定位已完成的两组矩阵。对每个后端和拓扑按
产品 ID、完整步匹配全部原始 CSV 字节，而非只比较汇总误差。
CPU/GPU 各七种拓扑、每组 40 帧，共 560 对 CSV（包含头部、实际采样覆盖、
原始/壁面单位列）逐字节一致。不新跑 CFD、不生成第二份结果或报告文件。
这证明已有该夹具的两种几何来源得到相同剖面输出，不替代读取式 CPU
精确续算、新壁面家族或真实 SBLI 门槛。

完整 goal 仍进行中。已再次以三个独立问题列出磁盘处理、D14 的外流带
速度权重和边缘黏度定义；本轮未收到确认，不假定已批准，不删除证据，
也不新增 CFD/图片运行。厚度产品及 TB2–TB4 未完成，Git 和平台均未操作。

## D14 确认后的厚度产品与真实预览增量

2026-11-03 用户回复“全部采用，继续goal”，关闭上述三个待确认项：证据
预算为 12 GiB、保留已有文件；外流带的 rho/p/T 为真实壁法向梯形长度
平均的 Reynolds 均值，速度为密度与长度加权的 Favre 速度；边缘黏度为
算例关系 mu(T_e)，不作为平均瞬时黏度。固定自由来流 Cf/Cq 分母不变。
本节取代前文“尚待确认”的当前状态，保留前文作为实施记录。

### TB1.5 / TB1.6 已接入的量及数据路径

配置采用现有 Favre `station_profile` 的可选对象：
`thickness={edge_band:[j0,j1],explicit_upper:y_upper}`。省略上限时只提供
delta99 家族，给出上限则另提供明确命名的显式上限家族。索引为全局 j，
上限为物理 y 坐标；剖面从底壁 j=0 开始，参考带必须严格位于壁面之上，
且全部包含在测站范围内。真实算例仍须由用户选择外流平台，不能仅以
数值正速度判定平台。首轮要求显式请求 velocity_x/pressure/temperature，
因此不隐式改变无相关字段产品的历史依赖。

`insitu_boundary_layer::boundary_layer_edge` 在已回读的一维剖面上使用
非均匀梯形长度权重；单节点直接取值。rho/p/T 使用 Reynolds 均值，U_e
使用 sum(w*rho_mean*u_Favre)/sum(w*rho_mean)。压力和温度不因产品 source
为 Favre 就换成 Favre 均值。`nonreacting_wall_material` 使用当前算例参数
给出 mu(T_e)；无量纲黏度包含 1/Re，与原物性定义一致。

`boundary_layer_thickness` 复用已测试的外流连通分支、线性交点、非均匀
梯形与末端部分区间规则。令 f=rho_mean*u_Favre/(rho_e*U_e)，积分核为
1-f 和 f*(1-u_Favre/U_e)。输出 delta99、位移厚度、动量厚度、H 和
Re_theta，以及显式上限家族；累计积分按剖面真实 y 从壁面起算。
`thickness_reynolds` 复用指数分解乘除法，保持结果可表示时不被中间乘积
溢出破坏。没有引入高精度模拟推进器、经验截断、速度平滑或自动外流最大值。

CSV 保留原始均值/RMS/协方差列，新增累计积分与可用标志；数值边缘状态
和厚度标量使用可机读的 `# thickness_NAME=VALUE` 元数据。定义不成立的
标量不输出假数值，累计列不可用时留空。无连通交点不影响有效的显式上限
积分，非法上限不影响其他定义和原剖面。原统计窗口、覆盖身份及展向处理
继续记录在同一文件中。示例：`scripts/output/input.products.wall.thickness.example.jsonl`。

清单解析、MPI 广播、语义版本和原检查点中的有效 JSON 恢复均已接通。
不新增统计历史、三维设备数组或重启附属文件。新增一维工作区纳入原资源
预检；每个产品请求仍会执行原一维投影，不把共享累计历史误称为所有投影
都已缓存复用。GPU NP1 单次投影下载为 6340 B，多个请求需累计计数。
没有新增三维原始场、派生场或最终几何回读。

### 厚度证据与范围

证据根：`tests/gpu_validation/out/tb1_thickness_runtime_20261103/`。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新选项首个失败及壁面参考反例 | 原清单不识别新选项；其后明确发现壁面参考带未被拒绝，修正均有失败先例 | `schema_red.xml`、`wall_band_red.xml` |
| 最终清单、广播、语义变更及有效定义恢复 | 117 项通过、12 项未提供其他可选接口而跳过 | `schema_final.xml` |
| 非均匀/单节点带、参考 Reynolds 数及已有算子，CPU/GPU NP1/2/4 | 各 3 项通过 | `operators_cpu.xml`、`operators_gpu.xml` |
| M12 CPU/GPU 各七拓扑数值矩阵及两个后端的精确 5+5 续算/输出开关 | 16 项通过；公式最大缩放差 3.50e-14、绝对差 1.91e-11 | `final.xml` |
| 同拓扑 CPU/GPU，包含数值元数据和累计积分 | 七对、168 对 CSV 通过；最大缩放差 1.54e-13、绝对差 1.70e-10，RMS 原值差 6.83e-16 | `final_same_topology.json` |
| 厚度与匹配壁面单位同时输出，GPU NP2 z | 1 项通过，原始列、壁面标度与厚度均通过独立核对 | `combined.xml` |

精确续算使用生成式 64×32×24 M12 原随机吹吸夹具、dt=.02，共 10 完整步，
连续与 5+5 续接比较完整 state.h5/statistics.h5 逐值一致、续接 CSV 逐字节
一致。恢复时磁盘上的厚度配置故意冲突，仍使用保存的有效定义。关闭厚度
产品保留相同原始统计请求，主状态和累计统计逐值不变。不把该短测称为
M12 湍流定常、SBLI 物理验证或不同分解轨迹逐值一致。

初轮 `matrix.xml` 的“no-edge”测试把参考节点错设在壁面。该点真实统计 U
约 4.4e-11，并非精确零，导致无物理意义的大积分和约 3.88e12 的绝对场差，
虽然缩放差通过也不能作为外流参考物理证据。已依据 D14 的外流定义拒绝
包含 j=0 的带，最终矩阵检查该产品配置失效且原始/其他产品继续。此前
`matrix.xml`、`same_topology.json` 和旧连续/续算记录均保留，不纳入最终
有效外流带的数值结论，没有修改容差或删除大差值。

NP1 的 `station_thickness_diagnostic` 观测中位数约 CPU 0.064 ms、GPU 0.117 ms，
包含预算 MPI 归约和一维计算，不是 GPU 内核计时或 CPU/GPU 加速比。
完整产品开关的运行窗口、更多真实工况和峰值仍需阶段收口。根 CMake 的
CPU、CUDA 与 Catalyst/device 三种相关构建已通过；编译不替代以上数值门槛。

### 读取式 M12 精确续算的新能力缺口

新增读网格续算测试在首个 CPU 算例的 RK 前明确失败，保留为 `read_exact.xml`，
未继续 GPU 或弱化断言。失败为 `output_runtime::bootstrap_output_resources`
要求原随机吹吸检查点属于 `extruded_profile_output_case()`。该谓词限定
`.not.lreadgrid`，且资源命名/冻结依赖 `grid.2d`，不能只删除布尔限制后
宣称三维读取网格的随机吹吸已精确恢复。

读取式笛卡尔剖面的先前 14 项数值矩阵仍有效，但不证明此 RNG/资源组合。
已向用户提问是否纳入读取式笛卡尔 M12＋原随机吹吸的精确冻结/恢复，
不扩大 CURVE，不改变吹吸算法。尚未获该项确认，未修改原 I/O 准入或 CPU
数值程序。读取式 GPU TGV 的已有闭环不拿来替代该组合。

### TB1.2 实际可达预览入口及图片

当前正式运行只使用动态产品清单，旧固定预设正式入口已关闭。
`tb1_coordinates_20261102/static_first.xml` 中主计算正常结束而没有旧预设
图片，原因是测试入口过时，不是放开旧入口的依据。已移除本轮新加的
无效正式入口测试，改测实际支持的 `insitu-preview`；没有恢复旧出图路径。

现有 32³ TGV、step4/t=.004 检查点只读复用，无 CFD 推进、无新增统计。
标准设备管线 NP1/2，各两个候选改变相机/显示范围/线宽，第二候选复用
轨迹。真实 JPEG、原生坐标单位桥接、显式物理窗口、标尺更新和生命周期
检查通过；源文件内容/mtime 均未变化，设备对照 flow/statistics mismatches
均为零。首个测试相机的标尺有裁剪，改为更宽的显式相机范围后完成基本
800×600 目视核对；没有修改求解器、轨迹或自动改变用户相机。

- 最终证据：`tb1_coordinates_20261102/preview_margin.xml`，2 项通过。
- 图像示例：同目录 `preview_margin/test_preview_coordinate_frames0/outdat/render/`。
- 两候选图片均为 JPEG，不新开 EPS；高分辨率/细长生产域排版仍待验收。
- 资源观察点的最大增量约 RSS 705081344 B、设备 558764032 B，最小设备空闲
  18723241984 B，均满足已有配置。它们不是外部高频采样的任意瞬时峰值。

额外尝试的 direct-device 预览失败已保留在 `preview_two_rank.xml` 和
`preview_final.xml`：原 DirectMapper 不支持预览添加的显示裁剪面，上位
PV 计划本就没有授予该新入口。未修改 Mapper 围栏或扩大预览准入，最终
预览矩阵只覆盖原支持的 standard-device NP1/2。正式动态清单的设备直连
不等同于该预览入口；新构建的 NP2 direct-device 及 NP4 standard-device
动态坐标回归另有 2 项通过，见 `catalog_refresh.xml`。

本轮结束时上述八个证据目录的实占总计 9423417344 B，约 8.78 GiB，低于
已批 12 GiB；已有模板和失败证据均未删除。TB1 仍有读取式随机吹吸续算、
其他壁面家族及真实 SBLI/完整成本等收口工作，TB2–TB4 尚未完成。完整
goal 不缩减、不宣称完成；未进行 Git、平台、worktree 或子代理操作。

## 已授权并补齐：读取式 M12 原随机吹吸精确检查点

用户于 2026-11-03 确认补齐该组合。本节关闭上一节中读取式 M12 RNG/资源
支持的待决策与实施缺口，不把此前失败的 `read_exact.xml` 改成通过记录。

### 实现及生命周期

`output_runtime::profile_checkpoint_definition` 提取原平板配置条件，
`extruded_profile_output_case` 仍只处理原二维挤出路径。新增读取式候选
限于本轮 FP64 非反应三维布局；必须在 `verify_read_profile_geometry`
通过后，且实际采用 legacy_random，才成为正式读取式检查点组合。
候选识别不等于跳过验证或直接授予 CURVE 准入。

启动时 bootstrap 先将恢复所需的壁面参数路径指向冻结资源。网格读入与
原几何计算完成后、首次 RK 与检查点发布前，用已在主机上的物理坐标
复用 `cartesian_block_axes/cartesian_axis_direction`，检查三维轴可分离、
三轴严格递增及 MPI 共享轴节点一致。轴归约使用唯一节点所有权。检查
成功仅缓存于本次进程，每次冷启动或重启重建，不保存永久准入标志。
临时轴工作区计入原主机预算，函数返回即释放；GPU 不新增几何下载。

读取式资源仍按实际 `gridfile` 保存为 resources/grid.h5，二维挤出资源
仍是 grid.2d；没有把两者的文件布局混同。入口、wallbs.dat、初始场及
精确几何/状态沿用现有冻结、引用与恢复通道。新增测试删除的是新续算
目录刚生成的输入副本，原算例、冻结参考及历史证据未删除。

RNG 算法、每 rank 抽样次序、首次调用状态、编译器/运行时签名和同拓扑
检查未改变，仍使用原 control.bin 的 ASTROC09/ASTRWR02 数据。没有增加
附属文件、RNG 重播、CPU/GPU 源项修正、CRC 或主流场容差。

### 实际验证

证据根仍为 `tests/gpu_validation/out/tb1_thickness_runtime_20261103/`。
算例保持 64×32×24、dt=.02、10 完整步、743e MP7-LD/643e 黏性、不开滤波，
边界 11/21/41/50/1/1，ninit=2、prof 入口及原随机吹吸。所有 GPU 核后显式
同步保留；本地 NP4 共享两卡只验证正确性。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 先前失败的 CPU/GPU NP2 z 读取式连续/5+5 续算 | 2 项通过；完整状态、累计统计、RNG/控制文件及续算 CSV 分别逐值/逐字节一致 | `read_rng_z.xml` |
| CPU/GPU 的 NP1、NP2 x/y、NP4 xy/xz/yz | 12 项通过；同样的精确续算、撤除本地输入、输出开关和一维公式断言全部保留 | `read_rng_topologies.xml` |
| 非轴分离和反向轴网格副本，CPU/GPU | 4 项按预期在 RK/发布前拒绝，证明没有把任意 lreadgrid 放开 | `read_rng_rejections.xml` |
| 原二维挤出 CPU/GPU NP2 z 路径 | 2 项通过；仍冻结 grid.2d、不产生 grid.h5，且不执行新读网格证明 | `generated_rng_regression.xml` |
| 读取式检查点独立于时间统计，CPU/GPU NP2 z | 2 项通过；统计关时不产生 statistics.h5、不调用 selected 几何证明，主状态与统计开时逐位相同 | `read_rng_no_statistics.xml` |
| 读取式完整参考的 CPU/GPU 同拓扑比较 | 七对、168 对 CSV 通过；缩放差 1.54e-13、绝对差 1.70e-10、RMS 原值差 6.83e-16 | `read_rng_same_topology.json` |

以上 22 项功能检查没有跳过；读取式完整剖面对独立一维计算的最大缩放差
为 3.50e-14、最大绝对差 1.91e-11。NP2 z 的生成式与读取式，在 CPU/GPU
各 11 个守恒量及原始变量缓存全局数据集上也逐位相同，缩放和绝对差均为
零；该项只读比较已保存状态，不重新运行 CFD，也不忽略小值。

同一测试还比较启用/关闭厚度产品的 state/statistics/control 文件，确认
产品不改变推进或 RNG；恢复使用已保存的有效产品定义，不采用磁盘上
故意冲突的新参考带。NP2 y 与 NP4 混合分解包含没有物理底壁的 rank，
它们的“未首次抽样”状态同样包含在精确控制文件比较中。

现有分析脚本增加 `--restart-records`，只读已通过测试 XML 中的连续参考
目录，结合实际日志拓扑比较全部数值列、厚度元数据及可用状态。失败
记录会拒绝，缺失/重复拓扑也不会成为成功矩阵。原生成式报告重新以 JSON
表示核对，与冻结结果相同。初次脚本内对象比较因 tuple/list 表示不同而
失败，已查明只是测试表示层问题，未修改数值数据或容差。

### 成本、范围与剩余事项

NP2 z 的初始化证明每 rank 临时主机空间 2856 B，观测约 0.126–0.140 ms；
额外几何 D2H 为 0。该零值仅针对本次初始化证明，不抹去已有后处理轴包
和显式检查点 staging 的传输。没有逐步新增验证或把该耗时混入 RK 计时。

本轮后八个证据目录实占 12159545344 B，约 11.32 GiB，仍低于已批 12 GiB。
相比上一节新增约 2.55 GiB，高于约 2 GiB 的初估，包含完整七拓扑与旧路径、
统计开关回归的输入/检查点；实际总量已逐段复核，没有越过批准总预算。
所有已有失败记录与输入模板保留。后续新增检查点矩阵须依据剩余约 0.68 GiB
重新估算，不能默认还有原预算那么多空余。

本轮证明的是 FP64、同后端同拓扑、原随机吹吸和上述 M12 边界/格式组合。
读取式 NSCBC 22/52、紧致黏性、其他壁面家族、FP32、CURVE、改变 MPI 拓扑
或 CPU/GPU 间精确 RNG 迁移不由本轮矩阵覆盖。原有生成式及其他已支持
输出路径未削减，生产默认未改变。TB1 其他真实工况、显示与完整成本收口
及 TB2–TB4 仍未完成，完整 goal 不宣称结束。没有 Git、平台或子代理操作。

## TB1 小型 GRCBC SBLI 定量组合

2026-11-03 继续复用已有 `test_sbli_joint_nscbc::prepare` 夹具，不新增物理模型。
CPU/GPU 的实际主程序使用 64×48×16、Ma=2、Re=296000、参考温度 148.9 K、
Tw/T_ref=.8、β=32.6°，743e MP7-LD 对流与 643e 黏性，不开滤波。
边界为 11/22/41/52/1/1，顶面采用完整 RHS 的 sbli_grcbc/H1 链，不施加
顶面横向滤波。初始场已种入斜激波跳跃；dt=2.5e-4，共 12 完整步。
这是含真实边界/RK/通信的接口短测，不是发展充分的 SBLI 或湍流统计定常验收。

正式统计窗口为 [.000125,.002875]，12 步逐步输出数值产品。测站 i=24、
x=3.125，外流带 j=40..44，显式积分上限 y=.6。壁面参考的 Tr=1.72 为
测试明确指定的值，不代表自动采用恢复温度相关式。未改变这些物理/离散
参数以放过验收。新增示例为 `scripts/output/input.products.sbli.statistics.example.jsonl`。

### 独立参考及被测路径

新增 `test_insitu_sbli_statistics.py` 组合完整壁面应力/导热、Reynolds/Favre
速度均值/协方差/RMS、压力/温度均值、壁面/网格单位、Cf/Cq/St、外流参考
与两种厚度积分。每个瞬时端点的测站和壁面截面仅作为显式测试数值存档，
以独立时间梯形权重和真实展向长度重建统计，不从图片读数或重写一个 CFD
推进器。新的热流/应力仍是分子应力张量和傅里叶导热，不偷换成均值梯度
乘均值黏度。CPU/GPU NP1 的最终独立壁面参考从权威 q、EOS、实际冻结度量
与有理数七节点 Lagrange 权重构造，不调用生产壁面量函数。

模板在计算空间为六阶；本轮同状态/同离散核对不把原求解器的网格度量
误差当作已经消除，也不取代拉伸网格上的解析收敛验证。吹吸切向导数项
的非零反例继续由已有 M12/算子证据承担，未把本次无吹吸 SBLI 当作其替代。

原 `check_wall_means` 和 `check_wall_unit_profiles` 只参数化了步数、窗口、
节点数与预计历史数，M12 默认值不变。首次复用旧数据时错误假定了目录名
和今日的配方；改按冻结 input.products.jsonl 的实际定义检查后，旧 M12
结果仍通过，未删除旧量或重跑 CFD。NP2 z 连续参考由惰性模块 fixture
供矩阵、续算和输出开关共同使用，不重复保存同一完整计算。

### 暴露的依赖与真正修复

第一次 GPU NP2 x 失败并非物性或导数错误：借用的纯计算启动器明确设置
OMPI_MCA_opal_cuda_support=false，而新 GPU 壁面时间统计的私有模板交换
当前选择 device-aware。日志在核调用前报告该依赖不可用，主计算和其他
产品仍正常结束。没有删除 MPI 能力检查或把设备指针交给不支持的后端。
修正测试启动配置以启用 MPI 设备缓冲支持及 UCX，仍保留求解器 pinned
halo。每次实际命令及相关环境记录到 launch.json。修正后该 GPU NP2 x
完整检查通过。`matrix_and_restart.xml` 的初始失败保留。

第二个问题由逐帧对比发现，末帧检查不能暴露它：原新剖面 CSV 的 ES24.16
会把有限的 `1.8432683097609400e-110` 输出成 `1.8432683097609400-110`，
通用 CSV 工具读成 NaN。已仅把两处新壁面/测站原始列输出改为 ES25.16E3，
不改累计数学、精度类型或近零处理。实际 CPU NP1 首帧是回归触发器；
先误选的 GPU NP2 x 数据未出现该极小量，不能作为失败复现。最终专用
短测要求仍保留非零且小于 1e-100 的协方差，并且全部原始数值列可解析。
旧数据不注入 E、不截零、不从比较中剔除。CPU/CUDA 根 CMake 构建通过。

为完整重验该输出修订，用户批准证据总磁盘从 12 GiB 提高至 14 GiB；
显存、主机内存、网格与步数不变。所有此前失败及格式问题文件保留。

### 最终数值与生命周期证据

证据根：`tests/gpu_validation/out/tb1_sbli_statistics_20261103/`。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 极小协方差格式专用短测 | 1 项通过；原值输出为 1.8432683097609400E-110，不归零 | `exponent_fixed_probe.xml` |
| 修订后完整 CPU/GPU 七拓扑、独立参考、5+7 精确续算、统计开关及极小量回归 | 19 项通过、无跳过 | `final.xml` |
| 全帧同拓扑 CPU/GPU 测站/壁面单位/厚度 | 七对、420 对 CSV；最大缩放差 5.15e-14、绝对差 4.46e-11，RMS 原值差 1.93e-14 | `final_station_same_topology.json` |
| 全帧同拓扑 CPU/GPU 壁面及参考标度 | 七对、168 对 CSV；最大缩放差 1.74e-14、绝对差 1.65e-11 | `final_wall_same_topology.json` |
| 独立时间/展向统计及壁面单位公式 | 最大缩放差 1.23e-15、绝对差 9.10e-13；RMS 按方差判定并报告原值差 | `final.xml` 各项属性 |
| 同状态完整壁面导数/应力/导热最终参考，CPU/GPU NP1 | 最大缩放及绝对差均约 1.88e-17 | `final.xml` 的 wall_oracle 属性 |
| 无设备 MPI 能力的真实两步 GPU NP2 x 反例 | 1 项通过；仅壁面历史停用，依赖的 plus 列留空，原始剖面、其他产品及主计算继续 | `unavailable_transport.xml` |

CPU/GPU 各 NP1、NP2 x/y/z、NP4 xy/xz/yz 的原始帧均检查可解析性，不只看
末帧。CPU/GPU NP2 z 连续 12 与 5+7 续算的 state/statistics 全部数据集
逐值一致、control 及续算 CSV 逐字节一致；故意冲突的磁盘厚度定义没有
覆盖保存的生效清单。关闭全部统计和产品后，主状态仍逐值一致。输出
开关不取代不同后端/拓扑间仅按 D38 比较的数值门槛。

### 不含参考快照的开关成本

另有 1 项通过：`cost_no_oracles.xml`，完整数据见同目录
`cost_no_oracles/test_sbli_products_cost_withou0/cost.json`。本地 GPU NP2 z，
关闭二维参考截面和全部 checkpoint；outdat 中没有 HDF5。ON 每步输出
7 个小型数值产品，OFF 无时间统计及产品，二者均保持同一数值工况和日志
设置。ON 的全部 CSV 与已完成数值参考逐字节相同，不为计时放弃数值核对。

| 口径 | OFF | ON |
| --- | --- | --- |
| 12 个完整步的各 rank 窗口求和后取最大，含后处理及其小型 I/O | 0.46216 s | 0.60088 s |
| 去掉首步后的 11 步，同上口径 | 0.41707 s | 0.54277 s |
| 最大单步/rank 的完整步耗时 | 0.04510 s | 0.05812 s |

本小窗口的 ON/OFF 约 1.30，不是 CPU/GPU 加速比、五轮性能结果或生产开销
预测。采用 mainloop 的 completed_step_inclusive，而不是在后处理之前结束
的 ASTR_COMPLETE_STEP_TIMING/RK 计时；初始化另列在进程资源报告中。
现有 20 ms 外部采样记录进程树 RSS 峰值：935067648/987951104 B，差约
50.43 MiB。每个物理 GPU 的 NVML 采样峰值均为 429916160 B，未分辨到
ON/OFF 差值，不能宣称新增设备内存为零；仅统计/索引的受控状态每 rank
已明确有 197768 B。最小设备空闲大于 17 GiB，满足已有资源门槛。

按程序显式计数，12 步每 rank 一维投影回读累计 864432 B，采样状态回读
104 B、索引上传 37984 B；初始化轴包和私有 MPI 数据另有其计数。没有
为普通产品新增三维场或最终几何回读。数据量不是包含所有 MPI 协议头的
完整网络字节数，也不把测试用二维/检查点参考 I/O 算成普通产品零回读。
同帧一维投影尚未全部缓存共享，多个消费者仍重复归约/下载，属于可继续
优化项；已有累计历史已共享，不新增重复历史。

本轮补齐的是非反应笛卡尔 wall41 小型 SBLI 的 TB1 数值组合，不增加新
边界、化学、LES/RANS、CURVE 或 FP32 能力。普通产品本轮不出图，生产
高分辨率排版和更宽工况仍未以此验收。TB2–TB4 未完成，完整 goal 保持
进行中；没有 Git、远程任务、worktree 或子代理操作。

本节最终三次获批检查记录合计 21 个不同测试：final.xml 的 19 项、纯产品
成本 1 项及通信能力失效隔离 1 项，均无失败、错误或跳过。最终九个证据
目录实占 13472079872 B，约 12.55 GiB，低于新批准的 14 GiB；原失败记录
仍保留。后续按剩余约 1.45 GiB 估算，不把本地共享两卡的 NP4 当作四卡
性能证据，也不将本节单次短窗计时升格为稳定性能结论。

## TB2.1 固定网格层局部时间脉动的首轮接线

2026-11-03，保留上述全部证据，在总磁盘 14 GiB 范围内继续实施。
新增 `fluctuation=true`，缺省仍为 false。`source=reynolds/favre` 指定均值
参考，`statistics_source=selected-time` 复用正式窗口和已有累计历史。
数值定义为完整步瞬时原始量减去该节点截至当前时刻的已累计均值；速度支持
Reynolds/Favre，压力和温度只支持 Reynolds。不新增速度模长脉动的定义，
不做幅值归一化，也不把短窗均值当作收敛湍流均值。

### 代码与数据路径

- `insitu_catalog`/`insitu_run_config` 增加产品布尔选项、固定 j/原始分量
  范围检查、MPI 广播和定义身份；恢复时从已保存的原始 JSON 记录读取该
  选项，不增加检查点附属文件或改变权威场格式。
- `index_slice_source` 和 `index_slice_gpu` 在实际节点拥有者处，从完整步
  q 解码当前原始量，再减去局部选中均值，沿用现有私有截面交换。GPU 只
  增加线程局部五分量副本及两个状态关系标量，不分配新的三维场。
- `catalog_stats` 可以显式启用既有严格设备渲染，不再无条件禁止 render。
  仍不启动旧固定产品输出，不配置稠密全场统计。数值通道可保持 render=f。
- `insitu_mesh_adapter.cpp` 将新供给标为 selected-time，记录实际采样区间、
  覆盖时长和当前减均值语义，不再误记为旧等样本数统计。图例区分 u' 与
  u'' 及其 Reynolds/Favre 参考。
- 新的多产品接线发现活动均值句柄曾停在最后一个温度产品；首轮出图中
  其他分量非有限。现已在每组实际采样前重绑定，只合并统计字段一致的
  产品，避免把一个历史的指针用于另一个历史。保留 `image_first.xml` 和
  原失败目录，修复后按相同定义复验。
- 本地 NP4 共享两卡的资源计账需要节点共享 MPI 窗口。测试启动器原强制
  `osc=pt2pt` 导致 MPI_Win_allocate_shared 失败，改为 pinned 测试配置
  `osc=sm,pt2pt`。不修改求解器物理或通信算法；保留原 NP4 失败日志。

### 验收证据

新增测试：`tests/gpu_validation/test_insitu_fluctuations.py`。根证据目录为
`tests/gpu_validation/out/tb2_fluctuations_20261103/`。

| 记录 | 范围 | 结果 |
| --- | --- | --- |
| `control.xml` | 布尔/物理量定义、错误产品隔离、现有控制文件往返 | 13 passed |
| `matrix.xml` | 16³ TGV、12 步、CPU/GPU 七拓扑及 NP2 z 的 5+7 精确续算、输出开关 | 16 passed |
| `same_topology.json` | TGV 全部原始/均值/脉动分量逐帧 CPU/GPU | 7 对拓扑、427 对文件，最大缩放差 8.526512829121202e-14，最大绝对差 9.947598300641403e-14 |
| `sbli.xml` | 已有 64×48×16、Ma2、bc11/22/41/52 GRCBC 夹具，dt=2.5e-4，12 步，CPU/GPU 七拓扑 | 14 passed |
| `sbli_same_topology.json` | SBLI 全部原始/均值/脉动分量逐帧 CPU/GPU | 7 对拓扑、427 对文件，最大缩放差 1.8815560581403437e-15，最大绝对差 1.9984014443252818e-15 |
| `memcheck.xml` | GPU NP2，两个完整步，新增脉动数值路径 | 1 passed，两份 Compute Sanitizer 日志均为 0 errors |
| `image_rebind.xml`、`image_parallel.xml` 中两个通过项、`image_np4_shared.xml` | 32³ TGV，两个完整步，每步四张 800×600 JPEG | NP1/NP2 standard-device、NP2 direct-device、NP4 standard-device 均通过 |
| `image_cost.xml` | 无整场/检查点/二维参考写盘的图片开关及资源观察 | 1 passed |

TGV 逐节点“当前二维原始量减均值”独立算式最大缩放差为
8.997666943671384e-17，SBLI 为 4.440892098500626e-16。
连续/重启权威场及输出逐值一致；另只读核对已有两组 NP2 z 的
`statistics.h5`，均逐值一致。最后这个检查复用原证据，没有重跑 CFD。
均值积分本身沿用已独立验收的 TB1 端点裁剪梯形累计，不重新实现。
images 的字段、最终几何回读计数均为零；数值 HDF5 是显式测试参考通道，
不计作图像路径的零回读。所有通过图像均有完整步和实际统计覆盖记录。

### 本地短窗成本与未关闭事项

`image_cost/test_fluctuation_image_cost_wi0/cost.json` 为一组 NP2 x 的 32³、
两步记录，每步四张 800×600 JPEG。关闭图片的两步最大 rank 累计耗时
0.027738257 s，开启为 2.472356994 s；开启时首步 1.864746624 s，第二步
0.607610370 s。使用 `completed_step_inclusive`，包含同步后处理和图片 I/O，
不包含求解器启动计时，不含额外检查点或二维数值参考写盘。首步含 Catalyst
初始化；该单次极短窗口不是稳定性能基准或生产规模开销预测。

外部 20 ms 观察的节点 RSS 峰值差 900354048 B；两 GPU 的本作业进程显存
峰值差分别 108531712 B 与 95948800 B，最小设备空闲均超过 17 GiB。
满足已有附加主机 4 GiB、每卡附加显存 2 GiB 和至少空闲 1 GiB 的本地
门槛。采样不能证明捕获所有瞬时分配峰值。采样器继续报告 flow_d2h_bytes=0；
私有 MPI staging 和图片像素回读不混入该计数。

目前只关闭 TB2.1 的上述首轮有界接线，不称为全部 TB2 完成。二维壁面
足迹/矢量/剪切线、真实平板/SBLI 图像组合及其渲染续算、晚加入图像的
完整矩阵仍待补齐。800×600 图已检查非空和脉动图例，但轴刻度局部仍显
拥挤，生产高分辨率排版未以此放行。TB3/TB4 未实施；完整 goal 保持进行中。
Git、平台任务、worktree 和子代理均未操作。

本节结束时整组实占 13819576320 B，约 12.87 GiB，低于已批准 14 GiB；
旧失败记录全部保留。`catalog_regression.xml` 另有 147 passed、47 skipped，
跳过项未配置对应专用 probe，不算通过；新产品范围以本节具体通过记录为准。

## TB2.2 二维完整壁面标量足迹

2026-11-03，补接笛卡尔底壁 j=0 的二维瞬时/平均壁面标量。平均量复用
TB1 已验证的瞬时完整应力和傅里叶导热采样器，逐节点累计 Reynolds 时间
均值，不作展向归约。不以平均黏度乘平均梯度构造平均应力或热流，剪应力
分量保留正负号。这里还没有实现剪切矢量箭头或壁面剪切线。

### 责任面与启动修订

- `catalog_wall_mean` 识别显式 selected-time 的 j=0 壁面切片，允许密度、
  温度、两切向剪应力、导热和分子黏度。不把任意原始量梯度或 Favre 壁面
  应力混入该供给。原始压力均值仍可用独立原始量切片输出。
- `insitu_selected_statistics` 为该切片分配唯一二维物理节点的历史，不为
  已计算的壁面物理量再扩展三维梯度统计支撑。`sample_selected_wall_group`
  沿用先算瞬时完整张量/物性、后累计的 CPU/GPU 路径。
- `index_slice_provider`/`index_slice_gpu` 直接交换并读取平均壁面分量，
  不对平均应力再做梯度或重新计算 mu(T_mean)。GPU 图像沿用设备字段和
  几何，不引入最终几何下载。HDF5 二维数值参考是测试显式开启的另一通道。
- NP4 共享两卡首次出图失败为已有资源池的子集准入缺口，而非超额：此前
  `catalog_stats` 未接入已有四 rank/两 GPU 原子分配池。现已补接，仍要求
  原有预算、预留量和依赖生命周期补丁。三个 NP4 通过日志各有四条
  `ASTR_INSITU_SHARED_ALLOCATION`，没有关闭资源检查。
- Compute Sanitizer 首次在 MPI 初始化报 UCX cuCtxSetFlags/cuCtxGetDevice
  的 201 错误，未发现新增核的越界报告。其根因是 GPU 壁面统计独立使用
  aware 私有通信，但 render=f 的配置无法声明设备处理，MPI 前也按旧
  非原生模式预读配置。修订允许 catalog_stats 显式选择设备统计而关闭
  渲染；预读采用与实际入口一致的 native/preview 口径，复用既有 MPI 前
  CUDA 绑定。设备 catalog_stats 即使图片条带选 pinned，也会为当前的
  aware 壁面统计预绑定。没有扩大 suppression，也未改求解器 halo 后端。
- 图像夹具最初沿用了不存在的 `outdat/render`，现明确使用已存在的
  `outdat`；首轮相机包围域不足也改为显式 parallel_scale=6.5。保留旧
  配置、资源池和内存检查失败记录，不将其改写成通过。

### 当前通过证据

测试入口 `test_insitu_wall_footprints.py`，证据根目录
`tests/gpu_validation/out/tb2_wall_footprints_20261103/`。

| 记录 | 范围 | 结果 |
| --- | --- | --- |
| `parser_final.xml` | 新壁面均值范围及错误产品隔离 | 6 passed |
| `matrix.xml` | 64×48×16 GRCBC，12 步；CPU/GPU 七拓扑、NP2 z 的 5+7 续算和输出开关 | 16 passed |
| `same_topology.json` | 六个壁面分量的完整二维逐帧 CPU/GPU | 7 对拓扑、175 对文件，最大缩放差 2.2352678288684536e-14，最大绝对差 2.7977620220553945e-14 |
| `m12.xml`、`m12_same_topology.json` | 原随机吹吸、64×32×24、dt=.02、10 步、NP2 x，CPU/GPU | 2 passed，保留非零展向变化；跨后端最大缩放差 5.329076822771625e-16 |
| `memcheck_prebind.xml` | 不渲染的显式设备统计、GPU NP2 z、2 步 | 1 passed，两条 MPI 前绑定记录，两份日志均为 0 errors |
| `image_final.xml` | 七拓扑 standard-device、NP2 x direct-device；2 步，瞬时/均值剪切和热流图 | 8 passed |
| `cost_isolated.xml` | NP2 x，两步，每步两张平均壁面图，无输出 HDF5/检查点 | 1 passed |
| `derivative_regression.xml` | 原 TGV 同相位原始量/梯度/Q CPU/GPU 回归 | 1 passed |
| `example.xml` | 新两产品 JPEG 清单解析 | 1 passed |

独立端点裁剪梯形积分与二维均值的最大缩放差：GRCBC 为
1.0658176666681127e-15，M12 为 5.329071914623001e-16。每个物理节点保留
二维结果。周期输出末端采用唯一节点零的统计别名，不把末端缓存另累计一份。
连续/重启的权威场、statistics.h5 及续接二维场逐值一致；关闭产品不改变
权威场。此处不将跨拓扑主流场轨迹差异与同拓扑 CPU/GPU 门槛混淆。

瞬时图含初始 step=0 和两个完整步，共三帧；均值等待正覆盖，从 step=1
开始，共两帧。每例四产品共十张 JPEG，八例共八十张。所有图像收据的
field_d2h/geometry_d2h/geometry_host 为零，采样器 flow_d2h 为零。
显式 z 显示放大 20 倍并有标注，数值和梯度仍用真实坐标；800×600 是有界
功能示例，不放行为生产统计定常或高分辨率论文排版。

### 成本与剩余范围

采用没有其他本会话 GPU 回归同时运行的一组记录：
`cost_isolated/test_wall_footprint_image_cost0/cost.json`。更早 `cost/` 曾与
短回归重叠，保留但不用于以下时间对比。OFF 为空产品清单；ON 为每步
两张平均壁面图，包含所需统计，不将差值全归因于渲染器。

- 两步最大 rank 累计 completed_step_inclusive：OFF 0.063874053 s，
  ON 1.783673705 s；ON 首步 1.448044612 s、第二步 0.335630328 s。
  含同步后处理和图片 I/O，首步含 Catalyst 初始化，不含启动读取初始场的
  时间，也没有输出参考 HDF5 或检查点。不是稳定性能基准或生产成本预测。
- 外部 20 ms 观察的进程树 RSS 峰值差 696279040 B；两卡进程显存峰值差
  65802240 B 与 61607936 B。最小设备空闲超过 17 GiB，满足既有本地预算。
  RSS 是进程树求和，不是去重后的物理页；采样不保证抓到任意瞬时峰值。
- 两 rank 的受控统计/索引设备状态分别为 65544/67592 B。每 rank 六次
  统计核、状态回读 24 B；索引上传分别 24576/25344 B。壁面采样器还另有
  已记录的梯度有效性标记及初始化轴包回读。没有新增三维流场下载。
- 图片条带用 pinned，各图片跨 x 接缝有 136 B 的私有发送或接收 staging；
  壁面统计自己的私有交换仍为 device-aware。图片像素、MPI 协议及测试
  数值参考不冒充该 flow_d2h 计数的组成部分。

新增示例：`scripts/output/input.products.sbli.wall-footprints.example.jsonl`。
`config.xml` 有 131 passed、42 skipped，`catalog.xml` 有 150 passed、
48 skipped；缺少专用 probe 的跳过项不计验收。以上矩阵不是所有壁面家族、
AIR5、CURVE 或 FP32 的新增准入。接下来仍须实现并验证壁面剪切矢量及
剪切线、补齐图像生命周期，再推进 TB3/TB4；整个 goal 保持进行中。

当前整组证据实占 14430982144 B，约 13.44 GiB，仍在已批准 14 GiB 内。
保留所有旧证据；未操作 Git、平台任务、worktree 或子代理。

## TB2.2 设备剪切矢量箭头

2026-11-03，新增 `field=wall_shear_vector` 图像产品，复用物理字段
wall_shear_x/z，而非新增守恒量或完整三维场。可选瞬时或 Reynolds
selected-time 均值。颜色为所显示矢量的模，平均图先平均矢量再取模，
不是平均瞬时模长。它不是壁面速度流线，也不是分离线识别。

显式 `extraction.stride=[Ni,Nk]` 在全局索引零点锚定 i/k 间隔，不按 MPI
子域重新布点；`extraction.length` 指定原生物理坐标中的固定箭杆长度。
精确零矢量不产生箭头，不设经验 epsilon。有限且非零的痕量向量仍保留，
其方向可能对噪声敏感，不能据此宣称分离。箭头头部采用固定显示比例：
回退 0.25 倍杆长、两侧宽度各 0.12 倍杆长。

### 实现与两处显示/分区修订

- `src_gpu/insitu_wall_vectors.h` 提供稳定尺度化的方向/模长和六顶点、
  三线段箭头构造；`insitu_plane_products.cu` 在 GPU 分类、压缩种子、
  组装 FP64 几何和标量，沿用现有 DeviceGeometryOwner 与两个严格后端。
  同一冻结包内相同几何可复用，改变相机不重建历史。
- 类型化 stride/length 已进入定义身份、MPI 广播及既有控制文件恢复。
  从保存的 JSON 恢复，不增加附属文件或改变权威场格式。只有严格设备
  图像实现，不把 host/compatible 路径的标量面片冒充矢量图。
- 共享节点只画一次，周期末端别名不重复画。源网格坐标框仍覆盖选择的
  周期闭合端点。六个源坐标范围标量每个非空 FP64 rank 包回读 48 B，
  另有小型选择/压缩控制归约；不把它们宣称为零总 D2H。
- 原三角面片合理地要求本地至少 2×2 个节点。箭头是点产品，因此为其
  增加 point_geometry 模式，保留 MPI 接口处只有一行/列的块，再按唯一
  节点所有权去重。原面片的准入和三角化不变。新增 ROI 截止于 i=32
  接口的检查，确认末端箭头不漏、不重复。
- 首张 z 显示 x20 的图中箭头头部也被拉长。现只补偿箭头头部的显示
  形状，箭杆的物理方向和原生长度不变；显示拉伸不进入应力或统计。
  几何缓存身份包含相关显示尺度。空/稀疏几何使用源坐标框，不补造方向。

### 验证记录

入口 `test_insitu_wall_vectors.py`；证据根目录
`tests/gpu_validation/out/tb2_wall_vectors_20261103/`。

| 记录 | 范围 | 结果 |
| --- | --- | --- |
| `parser.xml`、`control_cadence.xml` | 显式参数、错误产品隔离、类型化控制恢复 | 7 个解析项及 1 个独立往返项通过；往返 probe 使用其要求的 100 步间隔 |
| `geometry_final.xml` | NP1、NP2 x/y/z、NP4 三混合拓扑；正负/斜向矢量、1e-300 痕量、局部零/全零、显示放大、确定性连接及去重 | 7 passed；与独立闭式坐标/模长参考最大绝对差为 0；全局恰好 19 根有效箭头 |
| `image_compensated.xml` | 最新显示补偿后的 NP1 出图及 MPI 接口 ROI 末点 | 2 passed |
| `image_final_parallel.xml` | 补偿后 NP2/4 的 standard-device 和 NP2 x 的 direct-device | 7 passed，与上述 NP1 合成完整八项实际图像矩阵 |
| `real_cpu_gpu.xml` | 真实 GRCBC、两个完整步，CPU/GPU 七拓扑，比较两分量均值及箭头的模长范围 | 7 passed；最大分量缩放差或模长绝对差为 6.559423143537302e-18 |
| `restart.xml` | GPU NP2 x，连续 12 步与 5+7 续算，另做空清单对照 | 1 passed；state/statistics 逐值一致，续接 JPEG 逐字节一致，输出不改权威场 |
| `memcheck_host_osc.xml` | 独立几何 probe，NP2 y，含空壁面 rank、痕量和全零场 | 1 passed，两份 Compute Sanitizer 报告均为 0 errors |
| `cost.xml` | 无输出 HDF5/检查点的 NP2 x 两步图像/统计开关 | 1 passed |
| `surface_regression_enabled.xml` | 原标量壁面图的 standard/direct-device 调用回归 | 2 passed |
| `example.xml` | 新 JPEG 示例清单解析 | 1 passed |

独立几何 probe 的坐标/标量下载是明确的测试 oracle，不作为生产零回读
证据。真实 CPU/GPU 门槛的二维 HDF5 也是显式参考；GPU 图像自己的字段和
最终几何回读仍为零。模长范围在取实际种子的平均矢量后核对，排除空 rank
对范围的干扰，不用 JPEG 像素差代替 FP64 数值比较。

原单 rank 和部分并行图在显示补偿前也通过计数，但不是最终显示记录，
保留为中间证据。初次控制 probe 失败是其“一步后不应到期”的测试前提与
示例的一步间隔不符，未改生产调度。初次独立 memcheck 报告为 MPI 的 OSC
默认装载 UCX 时的启动上下文查询，并非箭头核越界；独立 probe 只通信
主机控制整数，显式选择 host OSC 后零错误，没有新增或扩大 suppression。
初次常规面片回归因未设置数值 executable 环境变量跳过，也保留为跳过，
不计通过。当前根 CMake CPU、CUDA 数值和 Catalyst/device 构建均通过。

### 成本、产物与剩余任务

`cost/test_wall_glyph_cost_without_o0/cost.json`：64×48×16、GPU NP2 x、两步，
ON 每步一张平均矢量图，长度 .6；OFF 为空清单。两步最大 rank 累计完整步
耗时分别为 OFF 0.058232665 s、ON 1.681238653 s；ON 首步 1.499009055 s、
第二步 0.182236337 s。口径为 completed_step_inclusive，包含所需统计、
Catalyst 和 JPEG I/O，首步含首次后处理初始化；不含启动读入初始场，也
没有输出参考 HDF5 或检查点。只有一组短窗，不是稳定性能基准。

外部 20 ms 观察的进程树 RSS 峰值差为 717697024 B，两卡进程显存峰值差
44961792/40767488 B，设备空闲均超过 17 GiB，满足已批预算。受控统计/
索引设备状态为每 rank 45060/46468 B；每 rank 三次统计核、状态回读 12 B、
索引上传 12288/12672 B，flow_d2h=0。六个包围盒标量及其他控制、私有 MPI
staging、像素回读另计；不能据此宣称不存在主机传输或取得全部协议字节数。

示例 `scripts/output/input.products.sbli.wall-vectors.example.jsonl` 默认选平均
场，显式 stride=[8,4]、length=.6、JPEG only；它不是生产默认参数。已检查
实际图中箭头指向、色条及 z 显示 x20 标注；高分辨率生产排版仍需独立复核。
全零场在组件中验证为空几何，尚未增加真实 CFD 全零场的完整图片验收。

TB2.2 的首轮标量/矢量路径已有闭环，但 TB2.3 壁面剪切线仍未实现，晚加入/
缺口等更完整图像生命周期及 TB3/TB4 仍待推进，完整 goal 不宣称完成。
`parser_regression.xml` 有 155 passed、48 skipped，跳过项不纳入本轮门槛。
当前整组证据为 14742781952 B，约 13.73 GiB，低于已批 14 GiB；剩余约
0.27 GiB，后续测试须先估算，不删除原证据或自行提高预算。未进行 Git 状态
写入，也未操作平台任务、新 worktree 或子代理。

## TB2.3 接口复核与待确认实施规格

2026-11-03，继续复核现有追踪器。`trace_partition_device` 要求三个方向
extent 都为正，`crop_physical_trace_halo` 构建三维单元；不能将其原样用于
零厚度壁面。当前 `NormalizedGrid` 还直接计算三分量平方和，不能把它
不加区分地当作痕量应力的稳定归一化。共享 Cash–Karp 的最小步长接受
策略也不等于新壁面产品应承诺的严格误差受限停止。

已在专项计划 §16.1 写出可执行候选：二维双线性分量插值后再稳定归一化，
复用弧长 RK45 与现有粒子/几何/MPI 框架；显式索引种子和积分参数；真零、
stage-zero、accuracy-limited、出域/选区等原因分开记录。明确双线性插值的
空间误差不因六阶梯度或小 RK45 容差而变成六阶。未修改共享流体流线或
实现未获确认的新终止规则，未启动新算例。

该数值规格及预计完整验证所需的证据磁盘 14→16 GiB 尚待用户批准。预计
新增 0.4–0.7 GiB，超过当前约 0.27 GiB 的剩余额度；不提高其他资源范围。
本轮产出为基于当前源码的接口诊断及具体候选，而非再次宣称 TB2 已完成。
完整 goal 保持原范围和 active 状态，受影响的积分实现/完整测试先等待决策。

## TB1 显示收口：XZ/YZ 坐标文字重叠

2026-11-03，在等待 TB2.3 两项决策期间，修复已记录的平面轴文字问题。
没有把自动继续视为数值规格或磁盘扩容批准，没有修改剪切线积分，也没有
启动新 CFD 或平台任务。

本地 ParaView 6.1.1 的 `VTK/Rendering/Annotation/vtkAxisActor.cxx` 中，
`SetLabelPositions2D` 明确按“2D 没有 Z 轴”处理，只实现 world X/Y 分支。
`render_coordinates.py` 原对所有 fixed_axis 都启用 Use2DMode，因而在
XZ/YZ 图上，Z 标签和标题没有正确外移。修订仅将该条件改为 fixed_axis=2：
XY 保持原样；XZ/YZ 使用已有三维相机跟随器，同时仍只显示对应两根轴。
不修改数值刻度、归一化、显示缩放、相机或源坐标数据，不修补依赖源码。

扩展 `insitu_coordinates_vtk_probe.py` 的 XZ/YZ 原生及拉伸视图；使用独立
四点矩形和已有立方体，不读取仿真场、不推进 RK、不分配三维后处理场。
证据：`tests/gpu_validation/out/tb1_coordinate_layout_20261103/`。

- `before/` 保留原十组图及元数据；`red.xml` 精确复现不支持的 2D-mode
  选择，不能据其失败声明数值错误。
- `after.xml`：2 passed，分别检查真实 VTK 元数据和 EGL 十图输出。
  标题/刻度外移已人工查看 XZ-stretched、YZ-native；不以“非空”代替布局复核。
- `model.xml`：17 passed，覆盖原坐标表达、显示变换、清单校验与控制恢复。
- `comparison.json`：十组模型的坐标/单位/范围等元数据完全一致，仅 XZ/YZ
  的 VTK 排版模式变化；六张未受影响的 XY/立方体/静态图 JPEG 逐字节一致。
- 探针在隐藏/恢复文字的旧分支仍会输出 VTK 空字符串诊断，before/after
  均存在，未据此宣称依赖的全部显示诊断均已消除。

本轮组件证据新增约 0.6 MiB，整组实占 14743400448 B，约 13.73 GiB，
仍在已批 14 GiB 内。未提高预算、未删除旧证据。现有 CFD 图像和成本
记录保持历史版本，不自动改写为新排版的实测；高分辨率生产布局仍须在
后续获批窗口检查。TB2.3 §16.1 数值规格及 14→16 GiB 仍等待批准，
完整 goal 保持 active，未进行 Git 状态写入。

## 已批准 TB2.3 与取消本轮人为资源上限

2026-11-03 用户回复“均批准”，随后明确要求“不要自设预算上限，解除预算
上限”，且原 goal 已由用户恢复为 active。§16.1 的插值、严格壁面 RK45
终止及验证规格获批；最新无限额授权替代 14→16 GiB 提案，不再为本轮
预算增量请求重复审批。完整 TB1–TB4 及数值/驻留合同保留，不标记完成。

本次先落实资源政策：

- `insitu_run_config` 将已有的 `-1` unlimited 语义接到 catalog_stats；
  正预算仍表示显式要求的上限。该哨兵只参与计账，不按其大小预分配。
- `shared_device_fixture` 不再用旧 2 GiB/4 GiB/空闲 1 GiB 阈值拒绝
  catalog_stats 的四 rank 计账；保留实际 GPU 分组、共享原子分配池、
  生命周期依赖及真实可用容量检查。
- 外部监测器允许内存增量及墙钟限额为 None、空闲预留为零，同时仍采样
  和写报告。不是把监测整体关闭，也不隐式将真实错误改成成功。
- SBLI 验收 helper 新的 unrestricted 模式关闭外部人为限额与 180 秒
  超时，I/O host/device 总量使用 -1。insitu 配置仍由新调用方明确写成
  -1/0，不默默覆盖其输入。保留流式分块缓冲、数值容差、积分最大步数/
  弧长及有限性检查；不批量改写历史冻结算例，不改变生产默认。

证据目录 `tests/gpu_validation/out/tb_resource_policy_20261103/`：

- `unit.xml`/`regression.xml` 中，模拟 9 GiB RSS、8 GiB 设备峰值和超过
  原 180 秒的时间仍记录成功，无预算杀进程；显式正限额仍按请求拒绝。
  这是监测器单元输入，不是本机实际占用。
- 原生解析 `parser.xml` 通过；`regression.xml` 共 134 passed、42 skipped、
  1 deselected，未选择或缺少对应 probe 的项目不计通过。
- `np4_with_io.xml`：实际 64×48×16 GRCBC、NP4 (2×2×1)、两 GPU、两步、
  两类平均壁面图通过。四条共享分配池记录、四份资源 CSV；host/device
  均为已有 unlimited 哨兵 2251799813685247，reserve=0，I/O 输入为 -1。
  所有产品 active，没有因人工额度被停用。
- 实测进程树 RSS 峰值 3715940352 B；两卡作业进程显存峰值分别
  922230784/928522240 B。JSON 的增量/time 限额为 null、预留为零。
  它证明不限额模式和计账仍工作，不证明耗尽显存或主机内存后的恢复能力。
  此测试不是性能基准，也没有检查点/整场输出。

CPU 与严格 Catalyst/CUDA 根 CMake 构建通过。本次没有新化学、数值格式、
边界或积分算法改动；实际 TB2.3 数学/设备接线为下一项实现，不把资源
政策通过当作剪切线已通过。未进行 Git 状态写入或平台任务操作。

## TB2.3 二维求值与严格设备积分组件

2026-11-03，按已批准 §16.1 实施，不改变原流体 RK45 接受规则。
本阶段只交付数学及 worklet 基础，尚未开启生产壁面剪切线产品。

定义为物理弧长参数下 $d(x,y,z)/ds=(\tau_x,0,\tau_z)/\sqrt{\tau_x^2+\tau_z^2}$，
反向积分使用负步长。先在真实矩形单元内双线性插值两剪切分量，再以最大
分量缩放归一化，防止 1e-300 非零剪切因平方下溢被当成零。这里的双线性
轨迹离散不具有六阶空间精度；本阶段没有更改瞬时/平均完整壁面应力的定义。

代码与测试入口：

- `src_gpu/insitu_wall_trace.h`：二维 portal 求值、严格 Cash–Karp 步进和分段积分。
  复用现有 `cash_karp_step` 系数，不复制或修改流体积分器的最小步长行为。
  精确零、试探点零、误差受限、物理边界、选区边界及插值支持不足分别报告；
  未通过误差或最终点检查的候选不提交。最后剩余弧长短于最小步长时，
  以剩余段闭合，但仍须通过原误差控制，不接受超差末段。
- `src_gpu/insitu_streamline_worklet.h`：`WallLengthGrid` 与 `RK45WallTraceWorklet`，
  复用粒子状态、矩形所有权和 writer。输入为真正二维轴/分量数组，不创建
  虚假三维薄网格；新入口未替换普通/曲线流体 worklet。
- `insitu_wall_trace_probe.cpp/.cu`：共享 CPU/CUDA 输入与解析参考；
  `insitu_wall_trace_worklet_probe.cu` 使用现有 MPI 粒子类型、设备有效点压缩
  和分区所有权，测试侧驱动交接。它不是第二套生产调度器。
- `test_insitu_wall_trace.py`：数学、七拓扑、内存安全、资源记录与原流体回归。
  全部新增运行无人工容量/预留/超时上限，监测继续。

证据位于 `tests/gpu_validation/out/tb2_wall_trace_20261103/`：

| 检查 | 实测 |
| --- | --- |
| `final.xml` | 12 passed，含 CPU、CUDA、七种拓扑、两类 memcheck 及四 rank 资源采样 |
| 18 组轨迹输入 | 正负 x/z、斜向、反向积分、1e-300、全零、线性旋转、试探点零、误差受限、物理/选区/支持边界、步数、非有限场、短末段、非均匀轴 |
| 解析与同离散 | 最坏解析/分区点误差 1.7380541450506826e-13；CPU/CUDA 分别执行同输入，有效点、终态和接受长度差 1.3877787807814457e-17 |
| 七拓扑 | NP1、NP2 x/y/z、NP4 2×2×1、2×1×2、1×2×2；跨 x/z 移交最多 7 次，空上层 y rank 参与集体协议，全部有效点身份无丢失 |
| 最终组合零值回归 | 六个试探 stage 非零、最终候选恰为零时，CPU/CUDA 均保留旧点并报告 stage-zero，不把旧点称为精确临界点 |
| 内存安全 | 直接 CUDA 数学探针一份、NP4 混合分解四份 memcheck 报告，均 0 errors |
| `fluid_regression.xml` | 4 passed；重编原流体探针后，NP1、NP2 x/y/z 双向接续保持既有结果与零接缝误差 |

含 16 个区间、17 个节点的合成单位壁面及非均匀轴副本用于解析测试。七拓扑
探针的 halo 由合成场直接填充，尚不能证明真实求解器壁面私有交换已接通。
场数组始终检查无主机镜像；用于解析比较的紧凑几何显式标记
`oracle_geometry_readback=1`，不是生产最终几何零回读证明。

保留的初次失败：`math_initial.xml` 是 pytest 临时目录父目录未创建，未执行
数学算例；`worklet_initial.xml`/`worklet_diagnostic.xml` 定位到测试把新分段
首点的 actual-step=0 与连续参考的上一接受步长 0.03125 比较。已修复该
元数据对照：首点必须为零，坐标/弧长与其他真正接受点仍按同一严格门槛。
没有放宽场差或修改生产积分算法以迁就探针。

不限额 NP4 进程树采样：RSS 峰值 512823296 B，两卡作业进程显存峰值各
339738624 B，elapsed=1.0908361 s。此时间含 MPI/设备初始化与诊断回读，
不是纯核计时或真实 CFD 附加开销；20 ms 采样不能保证捕获所有瞬时峰值。
JSON 明确记录增量/墙钟限额为 null、人工显存预留为零。

下一项是复用现有生产粒子控制/几何框架接通真正二维壁面适配、显式种子、
真实瞬时/平均应力及 halo，然后验证产品缓存、设备图像、12 与 5+7 步精确
续算和主计算不变。完成这些前不关闭 TB2.3；TB3/TB4 与完整 goal 仍未完成。

## TB2.3 共用驱动、真实产品与本地收口

2026-11-03 继续实施 §16.1，当前代码已接通产品清单到设备图片的完整路径。
公开配置为固定全局 j=0 的 `kind=slice,field=wall_shear_lines`。种子为有序
全局 `[i,k]`，每产品显式指定 RK45 参数；沿真正二维瞬时或节点 Reynolds
平均完整壁面剪切积分，不用壁面速度或原始变量均值的梯度代替。

### 代码接线

- `insitu_device_streamlines.h::drive_device_trace`：由原流体驱动提取的共用
  粒子接续、MPI 状态交换、设备有效点压缩、连通与拼接。原普通/物理网格
  流体入口保留原接受策略；没有单独复制壁面生产调度器。
- `insitu_device_wall_trace.h`：真正二维求值和所有权适配。空壁面 rank
  参加协议；单独终止种子不伪装为可绘线单元。插值支持不足是产品失败，
  不当成物理出界。保留场驻留检查，不增加资源额度或墙钟限制。
- `index_slice_provider`/`index_slice_gpu`：复用原采样、物性及私有交换，
  在局部壁面增加三层二维插值支持。对于该新增的已证明笛卡尔路径，坐标
  和完整度量行从有效轴向面取得，保留九个度量分量与原六阶投影次序。
  不读取未填充的双方向几何角区，不更改主求解器几何或边界离散。
- `insitu_catalog*`/`insitu_run_config`/`insitu_product_schedule`：结构化解析
  种子、集体广播、定义比较及从既有 JSON 记录恢复；种子修改产生语义版本。
  不增加检查点附属文件。只保留整数/MPI 计数的可表示性要求。
- `insitu_catalog_render`/`insitu_device_products.cu`：按冻结输入组复用设备
  线几何；只换相机/线宽不重积分，换种子或积分参数则重积分。每帧重建，
  不跨完整步误用旧几何；两设备后端继续使用 Catalyst/ParaView 原有渲染。
- `runtime_products.py`：有物理坐标标注、显示放大说明、固定色标及终止元数据。
  小型终态粒子包含位置、接受弧长、方向、步数和状态，不是整条轨迹回读。

### 数值与生命周期证据

目录仍为 `tests/gpu_validation/out/tb2_wall_trace_20261103/`，各批次不覆盖旧证据。

| 证据 | 范围及结果 |
| --- | --- |
| `components_final.xml` | 44 passed：解析、CPU/CUDA 数学、两类设备驱动、七拓扑、原流体/物理坐标回归、内存及资源、种子控制/广播/恢复 |
| `real_final.xml` | 16 passed：64×48×16、原 dt=2.5e-4、两步 GRCBC；七种拓扑 standard-device 及 NP2 x direct-device；每配置 3 张瞬时、2 张平均 JPEG，合计 40 张 |
| 真实 CPU 对照 | 原生 CPU ASTR 独立生成正式二维壁面 HDF5；根 CMake 构建的 FP64 C++ 壁面积分参考处理该数据，不使用 Python 流动推进器。三种子/双向的终点、接受长度差为零，方向、步数和终止状态相同 |
| `real_range_offline/summary.json` | 在上述不变的二维证据上，追加 CPU 原生沿线剪切幅值参考，八配置最大范围差 4.336808689942018e-19；不重跑 CFD，也不下载 GPU 线几何 |
| `lifecycle.xml` | 3 passed：只换视角复用、换种子/弧长重积分；NP2 x 连续 12 与 5+7 步续算的 state/statistics 逐值一致、图片逐字节一致；输出关闭不改权威状态；实际 NP4 x/z 新路径 memcheck 四报告均 0 errors |
| `gap.xml` | 2 passed：晚加入、停用再加；有其他消费者时历史持续覆盖，无消费者时保留真实缺口，不补造历史 |
| `cost.xml` | 1 passed：无整场/检查点/诊断 oracle 的开关实测，监测不限额，详见下段 |
| `wall_regression.xml` | 2 passed：NP2 z 上原标量壁面图和剪切箭头，两设备后端均 active、共 30 张图；无场/几何大块回读 |
| `example.xml` | 随库配置单独复核通过；该 test ID 亦包含在 44 项组件检查内，不另计独立测试 |

以上分批覆盖当前 68 个不同测试 ID，不把重复执行或离线补验再算为新的流动工况。
真实轨迹在本夹具中近乎沿 x 的直线，不能仅凭端点一致推断剪切幅值正确；
因此另有沿线幅值复核。曲线及有效中间点由旋转/非均匀轴设备组件检查覆盖。

新加入图像若有其他消费者持续采样，完整覆盖为 0.00275；若无消费者，则
本次受控加入/停用的覆盖为 0.001375，起点为 3dt 而非伪造的窗口起点。
两者单位均为原算例时间单位。测试使用原有原子目录发布回放，只改产品
清单，不改变 CFD、边界、步长、物理模型或统计权重。

### 已定位问题及修复边界

1. `shared_wall_diagnostic` 中非均匀轴反向旋转的自适应采样弧长相差约 1e-9。
   以最近 CPU 顶点直接作同状态对照并不正确。修正的是测试相位：有效点
   按同一弧长的解析解、同一实际步长的 CPU Cash–Karp 重算检查；另外仍
   检查独立完整 CPU 轨迹的终点/总长度、状态、步数及跨 rank 无重漏步。
   阈值仍为 2e-10，不把自适应采样序列差改称坐标误差，也未改生产积分器。
2. 初次真实 NP4 x/z 在种子处 status=25。保留的设备诊断明确显示：局部
   `(-3,0,-3)` 角区几何为零，轴向面上实际为 `(3.984375,0,0.078125)`。
   这是新增二维填充访问了原算子无需读取的几何角区，不是主 CPU 程序
   必须交换所有角区的证据。仅在已证明的笛卡尔填充中按轴向面重建坐标/
   完整度量行，修后 NP4 真实产品、CPU 参考与内存检查通过。
3. `real_corner_diagnostic` 曾在诊断构建失败后误运行旧可执行文件，该目录
   只保留为失败记录，不计为新诊断。有效诊断为 `real_corner_debug_current`。
   设备 printf 已从源码移除；诊断本身没有进入生产默认。

### 时间、内存和传输

`cost/.../cost.json`：NP2 x、两完整步，关闭产品 0.066487775 s，开启瞬时
和平均两产品为 0.734391129 s；两步窗口含新统计、积分、渲染和 JPEG，
不含 step=0 图片。开启后的首/次步最大 rank 时间为 0.436935781/
0.297455348 s。这是小型短窗成本，不是稳定性能基准或生产开销比例。

20 ms 采样的进程树 RSS 峰值差 784056320 B，两个物理 GPU 的峰值差
61607936/57413632 B；非第三方分配硬上限或所有瞬时峰值证明。
原生和外部运行均不限人为预算，外部限额为 null、人工显存预留为零。

生产图片路径没有二维/三维场或最终线几何的大块 D2H。非空分区每次积分
输入控制为 160 B（8 个坐标标量及三种子的坐标/所有权包），另有每轮粒子
状态。瞬时派生入口原有 4 B 有效性标志也属于控制回读。私有 MPI 的 pinned
暂存量单独记录，不混称为图片流场下载。相机复用视图的积分/输入控制
回读记为零；最终粒子元数据是已有状态，不再触发场或几何下载。

示例：`scripts/output/input.products.sbli.wall-shear-lines.example.jsonl`。
仅示范本夹具的索引、弧长、色标与相机，未替其他网格自动缩放。
§16.1 首轮本地实施已收口；完整 goal 保持 active，下一项为 TB3 的瞬时
lambda2/lambda_ci 设备选区提取。TB1 的更宽工况和生产显示收口仍依证据
推进，不据本段声明长时湍流统计定常、生产分离识别或新增 AIR5/CURVE/FP32。

## TB3 候选接线与暂停时状态（2026-11-03）

用户随后要求暂停原 goal 检查平台作业；本段记录保存工作区时的实际进展，
不将前段的 active 当作当前调度状态。TB3 尚未关闭，TB4 尚未实施。

### 实现范围

- 新增 `lambda2`、`lambda_ci` 瞬时字段，使用完整三维物理速度梯度。
  梯度仍来自已批准的独立六阶后处理模板，不读取旧 RK 梯度。
- `src/insitu_vortex_math.h` 为 CPU/CUDA 共用特征值数学实现。
  lambda2 是 `S^2+Omega^2=sym(A^2)` 的中间特征值，lambda_ci 是
  速度梯度特征值虚部的非负幅值。局部补偿运算不改变流场存储精度，
  不按经验阈值把微小虚部截成零。
- 数值入口分别为 `src/insitu_vortex.cpp` 与 `src_gpu/insitu_vortex.cu`。
  GPU 选区内临时保存九分量梯度，判据计算后释放；没有梯度主机回读。
  同区域两种等值面按共享采样包组织，重启和成本检查仍待收口。
- 候选产品为 `lambda2_surface`、`lambda_ci_surface`，支持固定原生阈值，
  或以 `reference_length`/`reference_velocity` 明确换算固定归一化阈值。
  `volume_index` 在求导和提取前限定三方向全局索引区域，读取必要框外模板。
  首轮限定非反应 FP64 笛卡尔状态及两类严格设备管线，不声明新增 CURVE、
  AIR5、FP32 或主机兼容渲染准入。

### 已观测证据

目录：`tests/gpu_validation/out/tb3_vortex_math_20261103/`。

| 证据 | 已观测结果 |
| --- | --- |
| `math_expanded.xml` | 15 passed；解析、随机、近退化、尺度及非有限输入，CPU/CUDA 同矩阵检查 |
| `math_memcheck.xml`、`math_memcheck.log` | 14 passed、1 deselected；数学探针 memcheck 为 0 errors，不代替真实场/渲染内存验收 |
| `runtime_matrix.xml` | 7 passed；七种拓扑的小型真实平板 CPU/GPU 完整步梯度与判据检查 |
| `catalog.xml` | 9 passed；显式阈值、固定尺度换算、索引区域及非法配置 |
| `catalog_regression.xml` | 51 passed、14 skipped；跳过项不计为本次通过证据 |
| `surface_matrix.xml` | 10 passed；七种平板拓扑 standard-device、NP2 x direct-device，以及 TGV NP2 两后端；场和最终几何回读为零 |
| `roi.xml` | 2 passed；CPU/GPU 的 NP4 x/z 数值选区与全域计算后裁切逐值一致 |
| 根 CMake 构建 | `build_cpu_probe/bin/astr` 和 `build_m12_device_render/bin/astr` 均构建通过 |

`surface_matrix.xml` 对应共享采样优化前的出图版本。随后共用采样包及
临时显存日志改动已构建，但不得把旧矩阵冒称这些后续改动的完整回归。

### 同状态与推进轨迹的区分

`runtime_current` 曾因直接对比两条 CFD 轨迹的 lambda_ci 而失败。
例如第 2 步同一位置的微小梯度差会将近纯剪切场的 lambda_ci 从零变为
`1.06495471768814e-9`。对这两个实际 FP64 输入分别做 80 位特征值计算，
各自复现了其结果。这不是通过降低同状态门槛或裁零处理解决的问题。

`runtime_matrix` 改为将每个真实梯度包原样交给 CPU/CUDA 数学实现，按原
`2e-10` 缩放误差门槛检查；梯度轨迹差仍按原门槛检查，判据轨迹差单独
写入 `comparison.json`，不宣称所有近退化判据的跨轨迹差均小于该门槛。
初次配置失败 `runtime_before`、`runtime_first`、`runtime_configured`
和数值比较失败 `runtime_current` 均保留，不作为通过证据。

### 未通过项与下一步

`lifecycle.xml` 的首项精确续算在读取产品控制状态时失败：
`ASTR INSITU SAMPLE ERROR: native dynamic-product control tail`。
连续运行和五步前缀已生成，5+7 步续接未通过；同批次因 `pytest -x` 停止，
真实 NP4 内存检查和成本测量也没有执行，不能记为通过。

恢复原 goal 后，应先定位该续算错误，再完成共享采样后的回归、数值/图像
续接、输出开关无副作用、真实场 memcheck、显存/时间/传输记录及配置说明。
完成 TB3 后才进入 TB4。当前仅保存进展，不启动这些计算。

另行提出的按需时间统计服务见
[独立设计草案](ASTR_GPU_TIME_STATISTICS_SERVICE_PLAN.md)，仍是待讨论、
暂不执行事项；用户要求停止 grill，直至其主动重新发起。
