# ASTR GPU 性能优化推进计划

## AIR5 同状态复用与通信 Goal（2026-09-25，本地验收完成）

基线为 `7fd3823`，冻结可执行文件为
`tests/gpu_validation/out/air5_flux_opt_relink_gate_20260925/astr`，
SHA256 `e1eb957930048f817599a2413f7e1f5b62e2370c64f420024c8d4b8a0f4c57e5`。
不操作平台或 Git，不改变 FP64、化学算法、容差、同步及默认数值路径。

两个独立候选均默认关闭：

- A：`ASTR_AIR5_PRIMITIVE_REUSE=chemistry`。化学半步已恢复成功的
  rho/u/T/Y/Tv/p 只在该半步后立即复用，范围为化学更新域与本 rank
  严格内点的交集。物理边界、MPI 重复端点及 halo 一律重新恢复。
  不分配全域缓存，不保存跨调用有效标志。`off` 是强制全量重算对照。
  RK、滤波、restart 和一般 prepare 调用不接收复用凭据，仍走原路径。
  待恢复区域按三个不重叠的外层长方体集合紧凑映射，256 线程一维
  block，每点只访问一次；复用区域为空时退回原全量核。
- B：`ASTR_AIR5_CHEMISTRY_REDUCTIONS=packed`。化学 status 与四个
  整型计数共用 MAX 归约，失败则在任何后续边界/通信前终止。三个
  min 取负后与两个 max 共用浮点 MAX 归约，每半步由四次减为两次。
  不合并存在状态依赖的边界失败检查，不改 halo 或共享面协议。
  默认 `baseline`。启动时集体检查两项配置是否在各 rank 一致。

选择依据：上一轮 NP2 Nsys 稳态两步，两个 rank 的内域恢复分别约
0.163/0.090 s，暴露 MPI 等待约 0.159/0.125 s；等待不等同于网络传输。
首版只针对可证明重复的恢复及同依赖归约，不据此承诺整体通信收益。

验收先 A/B 独立、再组合，沿用逐阶段逐元素容差，覆盖周期/非周期、
零/痕量组分、双温能量、激波传感器、NP1/NP2 x/y/z、restart/滤波，
保正、守恒、边界收支与最终共享面逐位一致，补故障传播及 memcheck。
通过后基线/A/B/组合各 NP1/NP2 三轮交错计时，仍采用 Mach4 无入射
激波 64x512x8 检查点、一步预热三步测量、逐步最慢 rank，无 profiler
及中间大场输出。报告中位数、波动、显存与 Nsight 调用/等待变化。
收益不超过波动者仅保留可选候选，不升级默认，不以单核收益结题。
以下验收仅针对本地短回放与数值矩阵，不替代长时反应 SBLI 物理验证。

### 最终实现与数值证据

最终候选 SHA256：
`7e26451c85cfd1153c733d65d1805cd67d832e6504decafb51c78bf2ff705687`，
冻结于 `out/air5_compact_combined_np2_20260925/astr`。
下列 `out/` 均相对于 `tests/gpu_validation/`。

- 根 CMake 构建通过，45 个相关单测通过。只修改两个 GPU 源文件，
  不修改 CPU 算法。新增持久全域设备数组/缓存容量为 0 字节，直接
  复用原始量数组。未据此宣称进程总显存峰值严格相同。
- 首版 A/B/组合独立 Mach4 NP2 回放各 20 份同相位场与冻结 GPU
  基线差值为零，六份 rank memcheck 零错误，收支和共享面通过：
  `out/air5_{reuse,packed,combined}_gate_20260925/`。
- 最终紧凑版 A NP2、组合 NP1/NP2 共 50 份同相位场与相应冻结
  基线差值为零，五份 rank memcheck 零错误。45 份接受态的保正、
  双温能量范围、物性域和质量闭合通过。NP2 对流边界收支最大缩放
  残差 1.389390e-16，各 18 对共享面逐位一致：
  `out/air5_compact_reuse_gate_20260925/`、
  `out/air5_compact_combined_np{1,2}_20260925/`。
- 最终 off/A/B/组合的周期反应 TGV 在 NP1/NP2 x/y/z 共 16 个 GPU
  回放通过不可变 CPU 和 GPU 对照，GPU 场差值为零，504 对共享面
  一致。CPU 和输入未变，不重复计算 CPU：
  `out/air5_compact_periodic_20260925/summary.json`，参考矩阵为
  `out/air5_chem_perf_periodic_20260925/`。
- 组合 full/scalar 滤波、NP1/NP2 x/y/z 八组通过，140 份场与首版
  GPU 对照差值为零。严格零组分和 NO=1e-12 的滤波 restart 通过
  CPU 对照，单组 20 份场，最大绝对差 1.164154e-10，逐元素容差仍
  为 atol=1e-8、rtol=2e-11。滤波矩阵及 restart 共 324 对共享面
  一致，restart 最大相对守恒漂移 8.900738e-15：
  `out/air5_compact_filter_20260925/summary.json`。
- 测试专用 MPI 拦截库仅在 rank 1 注入 status=99，其他 rank 同步
  退出，无后续 post_chemistry/pre_rhs 快照。生产程序没有故障开关。
- 最终版的默认 full_state 回归通过：`out/air5_compact_default_20260925/`。
  首版冻结激波管回归通过，激波管有
  1536 个传感器标记界面，24 份 CPU/GPU 阶段场满足原容差：
  `out/air5_chem_perf_shock_20260925/`。紧凑版未修改激波通量实现，
  该冻结算例不进入化学半步，复用其传感器回归检查。

首次零组分补测误用了要求所有阶段组分严格为正的旧 TGV 检查器，
CPU 初始精确零被判失败。保留 `out/air5_chem_perf_filter_20260925/`
失败记录及八个已通过滤波条目。新补测使用精确非负性，任何负值仍
失败，其余 TGV 约束和阈值不变；旧检查器未修改。

### 完整步性能与负结果

两个 RTX 4000 Ada，Mach4 无入射激波 64x512x8，step1090、dt=2ns。
每配置三个独立轮次，每轮预热一步、测三步，取逐步最慢 rank 后的
三步均值，表中列三个轮次均值的中位数、范围和样本标准差，单位 s。
证据：`out/air5_compact_perf_ab_20260925/summary.json`。

| 配置 | NP1 中位数 [最小, 最大] | NP1 标准差 | NP2 中位数 [最小, 最大] | NP2 标准差 |
| --- | --- | ---: | --- | ---: |
| 冻结基线 | 1.127277 [1.125307, 1.127465] | 0.001196 | 0.736057 [0.735392, 0.737209] | 0.000919 |
| A 同状态复用 | 1.107205 [1.107037, 1.107209] | 0.000098 | 0.714084 [0.713230, 0.714186] | 0.000525 |
| B 归约合并 | 1.126187 [1.126141, 1.127746] | 0.000914 | 0.735619 [0.733836, 0.736337] | 0.001288 |
| A+B | 1.107789 [1.106675, 1.108023] | 0.000720 | 0.716591 [0.713063, 0.716959] | 0.002151 |

A 的完整步耗时降低 NP1 1.78%、NP2 2.99%，两种规模均超过轮间波动。
双卡效率 T1/(2*T2) 由 76.58% 到 77.53%。A+B 分别降低 1.73%、
2.64%，双卡效率 77.30%；并不优于单独 A。B 单独变化仅 0.097%/
0.059%，不超过波动，不能宣称有独立整步加速。

首版按原全域 block 发射、内点提前退出，数值通过但性能没有稳定
收益，原始四方计时保留在 `out/air5_chem_perf_ab_20260925/`，候选
SHA256 为 `7f3afcd825ecc7fd3e7449e652c11cc6b47f4526205b58309b2c076d07f39409`。
Nsight 显示其 shell 与原内域核单次耗时接近，才改为紧凑外层映射。

### Nsight 复查与下一步

`out/air5_compact_nsys_np{1,2}_20260925/` 保存原始报告、SQLite 和
`steady.json`，第三至第七次化学核之间的两个稳态周期，只用于诊断。
Nsight 基线复用上一轮 `out/air5_flux_opt_nsys_np{1,2}_20260925/`
的同源码报告（b2ff45e 二进制）；完整步计时则使用上述 e1eb957 冻结
重链接基线并重新交错测量，不把两种二进制的历史计时混写。

- NP2 每步 Allreduce 从 82 到 78，Sendrecv 保持 172 次。
  D2H/H2D 拷贝保持每 rank 每步 169/121 次，字节数与原报告相同。
  本轮没有改变 halo 传输、最终面所有权或通信后端。
- NP2 两周期内域恢复由 12 次全量改为 8 次全量加 4 次紧凑外层。
  rank 1 的外层四次合计由首版 53.682 ms 到 10.361 ms；rank 0
  由 29.832 ms 到 10.162 ms。NP1 的全部内域恢复两周期合计由
  164.742 ms 到 125.024 ms，寄存器仍为每线程 124。
- NP2 暴露 MPI 等待按每步计，rank 0 由 62.371 ms 到 47.143 ms，
  rank 1 由 79.492 ms 到 78.784 ms。不能把此变化全部归因于 B，
  A 也改变 rank 到达时序，且独立 B 无显著计时收益。

推荐在已验证 AIR5 路径显式选择 A；保留 B 为经过数值验证的可选
候选，默认仍 off/baseline。尚未解决的通信问题是 halo 主机中转及
分阶段到达不均，下一轮须单独量化打包/搬运和依赖等待后再选择方案。
本轮不启动 CUDA-aware、通信计算重叠或取消同步，也不操作超算任务。

## AIR5 通量重构优化 Goal（2026-09-25，完成）

范围限定为 `symmetric_species` 的 GPU base/face/apply 三个核。
base 只计算低阶六面，face 只计算当前方向所需面，apply 只计算当前
方向低阶两面，直接复用现有面通量函数。不改变公式、投影、六面预算、
双温约束、MPI 最终共享面、FP64、同步和默认 `full_state`，不新增
全域通量缓存。基线计时二进制 SHA256 为
`89028e956cb0b0a4af66abafda93ce0c789f3a25d4a08065664242060ad572e7`，
保留于 `out/air5_perf_timing_gate_gpu2_20260925/astr`。

先验证同相位场及保正/物性域，再覆盖周期接触、痕量/零组分、能量
压力、非周期边界、激波传感器激活和 x/y/z 分解，共享面保持原逐位
门槛，相关 memcheck 零错误。CPU 不变时复用不可变参考。
通过后进行基线/候选 GPU NP1/NP2 各三轮交错 A/B 完整步计时，保持
一步预热、三步测量，定点复查 Nsight，并报告内存变化。
只有完整步收益超出轮间波动且两种规模无可分辨退化才推荐替代。
无整步收益时保留可选候选或补丁与负结果，不升级默认，不虚称加速。
不操作超算或 Git 提交，不扩大至化学算法/精度/同步优化。

### 本轮实现与验收

仅修改 `src_gpu/chemistry_solver_gpu.cuf` 的三个核，新增 23 行、删除
13 行。CPU 数值实现、通量公式及低阶 RHS 按方向累计顺序不变。
根 CMake 构建通过。候选二进制 SHA256：
`b2ff45ee94a1c859daea085b439605d074d2cf42c0aa54250add2482ec39f3b1`。

- Mach4 同一检查点 GPU NP1/NP2 共 30 份阶段场满足原
  atol=1e-9、rtol=1e-10；额外逐位探测未完全通过，最大差
  8.077936e-28，仅在痕量 N/O/NO。密度、动量、总能量、振动能量及
  N2/O2 在该回放的 post_update 场一致。不宣称全字段逐位一致。
- NP2 十八份状态满足保正及物性域检查，边界三个 RK 阶段净最终通量
  与对流 RHS 收支通过，最大缩放残差 1.38939e-16；18 对共享面逐位一致。
- x/y/z 各四类周期压力算例（接触、低 N2、能量脉冲、组分脉冲），
  CPU NP1/NP2 与 GPU NP2 共 36 次运行通过。1080 对面逐位一致，
  最大守恒残差 2.920763e-14，24 份 rank memcheck 零错误。
  含严格零组分、痕量组分自身尺度及双温约束检查。
- 三个默认 full_state 接触对照通过，GPU 默认路径与旧归档场逐位一致。
  十二组候选压力场与旧归档逐阶段满足原容差，最大绝对差
  2.273737e-13、最大缩放差 2.166762e-16。
- 冻结激波管 NP2 五步通过 CPU/GPU 较严格原门槛
  atol=2e-10、rtol=2e-11。传感器快照累计标记 1536 个界面，
  CPU/GPU 各 36 对最终共享面逐位一致；不是零激波掩码空测。
- Mach4 NP1 及最终重链接 NP2 额外 memcheck 零错误。
  本轮合计 27 份干净 rank 日志。
  36 项相关计时、限制器、残差和激波格式检查器测试通过。

收口根构建重链接后 `build_gpu_probe/bin/astr` 哈希变为
`e1eb957930048f817599a2413f7e1f5b62e2370c64f420024c8d4b8a0f4c57e5`。
源码未再修改；补跑 NP2 的二十份阶段场与实测候选逐位一致，memcheck
两 rank 零错误，证据在 `air5_flux_opt_relink_gate_20260925/`。
下述性能记录严格归属于冻结 `b2ff45ee...`，不重标为重链接版本。

### 无 profiler 交错 A/B

使用 `run_air5_flux_ab.py`，每种卡数三轮、基线与候选顺序交错；
每次均从相同检查点预热一步、测三步，逐步取最慢 rank，每轮取均值，
最后报告三轮均值的中位数。检查初始备份哈希、检查点时钟及无中途场
输出。测量期间未并行运行其他本轮 GPU 测试或 profiler。

| 配置 | 基线中位数 s/步 | 候选中位数 s/步 | 耗时降低 | 相对基线加速 |
| --- | ---: | ---: | ---: | ---: |
| NP1 单卡 | 1.340148 | 1.125848 | 15.99% | 1.19035 |
| NP2 双卡 | 0.848744 | 0.736944 | 13.17% | 1.15171 |

单卡基线范围 1.337133--1.342142，候选 1.125672--1.128618 s/步；
双卡基线 0.847804--0.848959，候选 0.735810--0.737383 s/步。
对应轮间标准差依次为 0.002522、0.001652、0.000614、0.000812 s/步。
两组候选与基线范围均不重叠，满足完整步收益准入。

双卡强扩展效率由本轮基线的 78.9489% 降至候选的 76.3863%，因为
单卡缩短比例更大。本轮没有解决 MPI 扩展效率问题，不以更快的双卡
绝对时间冒充更高扩展效率。复用 CPU NP1 80.098913 s/步历史冻结参考，
候选单/双卡对应 71.15 / 108.69 倍；CPU 非同期重测，也不是整节点基线。

### 热点与内存变化

Nsys 沿用两个稳定化学周期的窗口，不使用其墙钟作为正式加速比：

| 单卡每周期核时间 s | 优化前 | 优化后 |
| --- | ---: | ---: |
| symmetric_base | 0.048071 | 0.011047 |
| symmetric_face | 0.460419 | 0.408797 |
| symmetric_apply | 0.136070 | 0.013669 |

三核合计减少约 0.211 s/步，与无 profiler 完整步约 0.214 s 的降低
接近。Nsys NP1/NP2 及 NCU face/apply 均实际采集成功。
NCU 首次匹配调用的 spilling requests：face 从 15886520 降至
5283540，apply 从 16123716 降至 571200。两核仍为 128 寄存器/线程，
理论 occupancy 33.33%；实际分别 25.90% 和 32.09%，因此不是通过
提高 occupancy 获得本轮收益。

内存口径：未新增/改变全域数组分配，逻辑持久工作区增量为 0。
显式局部通量数组 base 从 132 降至 66 个 FP64 数，face/apply 从
132 降至 22 个；编译后 NCU 报告的两核 stack size 均由 4480 降至
4384 字节。该数值不是整卡峰值显存，不能按网格点直接乘成显存节省。
本轮没有测量驱动级总峰值显存，也没有声称显著降低全域常驻显存。

双卡候选 Nsys 暴露 MPI 时间 rank0/1 约 62.4/79.5 ms 每周期，
原约 65.0/67.9 ms；不同阶段负载不均仍在，不能将这次计算裁剪作为
通信优化完成证据。后续另立目标评估同状态物性复用、全局检查和通信。

结论：保留三个核的局部优化作为可选 `symmetric_species` 路径实现，
不新增运行时数值模式、不改变默认 full_state。该有界目标完成，
真实 SBLI/长时物理与 A800 生产验收仍独立进行。

### 本轮证据与复现

目录前缀均为 `tests/gpu_validation/out/`：

- `air5_flux_opt_gate_np{1,2}_20260925/`：基线阶段场、状态、边界门槛。
- `air5_flux_opt_{primary,axis_y,axis_z,default}_20260925/summary.json`：压力矩阵。
- `air5_flux_opt_shock_20260925/`：激波管 CPU/GPU 阶段场与传感器。
- `air5_flux_opt_ab_20260925/summary.json`：12 轮 A/B 契约与耗时。
  同目录 `stress_baseline_compare.json` 保存 13 组旧归档对照。
- `air5_flux_opt_nsys_np{1,2}_20260925/`：时间线、SQLite、steady.json。
- `air5_flux_opt_ncu_{face,apply}_20260925/`：硬件计数器与 details.txt。

```bash
python tests/gpu_validation/run_air5_flux_ab.py \
  --baseline tests/gpu_validation/out/air5_layered_multistep_20260924/gpu_dt2/gpu \
  --reference tests/gpu_validation/out/air5_perf_timing_gate_gpu2_20260925/astr \
  --candidate tests/gpu_validation/out/air5_flux_opt_ab_20260925/candidate_np1_r1/astr \
  --output tests/gpu_validation/out/air5_flux_ab_new
```

## AIR5 本地性能诊断 Goal（2026-09-25）

状态：本地诊断已完成，优化实现尚未开始。根 CMake 构建通过，23 项
受影响的计时/回放/无场输出检查通过；未重新运行无关长时算例。

### 完整步计时结果

新增默认关闭的 `ASTR_COMPLETE_STEP_TIMING=1`，在 CPU/GPU 共用的
`steploop` 中包围 `crashcheck + time_integration_rk`，逐 rank 输出时间。
包含化学、输运、边界和步内统计，不新增同步；不包括步后控制文件读取、
按 checkpoint 频率触发的 CFL 输出和初始化。驱动避免窗口内 checkpoint
事件，保留启动阶段 `flowinit` 的初始场写出，该启动 I/O 不计入样本。

GPU NP2 计时开关后的 20 份同相位场与基线最大差为零；GPU NP1/NP2
六组阶段场满足原容差，最大缩放差 5.77316e-15，单卡九份状态检查通过。
驱动 `run_air5_performance_diagnosis.py` 使用预热一步、计时三步、三轮
独立运行，每步取最慢 rank，然后统计每轮每步均值的中位数和离散性。

| 本地配置 | 中位数 s/完整步 | 三轮范围 s/步 | 轮间标准差 s/步 |
| --- | ---: | ---: | ---: |
| CPU NP1 | 80.098913 | 79.800890--80.441939 | 0.320788 |
| NP1，单卡 | 1.336195 | 1.334227--1.336720 | 0.001314 |
| NP2，双卡 x-slab | 0.848752 | 0.848405--0.849070 | 0.000333 |

单卡/双卡相对 CPU NP1 加速比分别为 59.9455 / 94.3726。
双卡加速 1.57431，强扩展效率 78.7153%。CPU 为 Xeon Gold 6248R 的
单 MPI rank，并非整台双路 48 物理核节点。GPU 为两张 RTX 4000 Ada。
当前一般 halo 通信为原基线 pageable host staging，显式同步保留；
对称最终面交换自身使用持久分配的 pinned 缓冲及 MPI_Sendrecv。
这是小规模平板状态，不代表生产 SBLI，也不是最佳通信后端测试。
合并证据：`tests/gpu_validation/out/air5_perf_diagnosis_20260925/timing_summary.json`，
包含九轮输入契约、检查点和二进制哈希。原 CPU/GPU 证据分别位于
`out/air5_perf_cpu_timings_20260925/` 和
`out/air5_perf_gpu_timings_startup_checked_20260925/`。
首轮旧检查器把初始化重写误判为窗口内写场，失败记录保留于
`out/air5_perf_gpu_timings_20260925/`；已核实 `flowinit` 在 `steploop` 前
调用 `writeflfed`，改为核对原始备份、输出检查点时钟及唯一初始写出。
CPU/GPU NP1 的十份同相位场通过原逐元素 atol=1e-9、rtol=1e-10 门槛，
最大缩放差 1.35417e-14；CPU 九份状态均满足保正及物性域检查。
同二进制 CPU NP1 计时开/关的十份阶段场逐位一致，证据为
`out/air5_perf_timing_gate_cpu1_20260925/timing_invariance.json`。
未进行算法优化或 Git 提交。

### 稳定窗口的 Nsight Systems 归因

NP1/NP2 各采集四个完整步。`summarize_air5_nsys.py` 从每 rank 第三次
化学核启动起，到第七次化学核启动前，提取两个稳定的
“第一化学半步到下一步第一化学半步”周期。该窗口排除初始化和首步，
包含周期之间的主机工作，不冒充公共计时器的完全相同调用边界。
下表除占比外均为窗口累计除以二。正式加速比只取无 profiler 的计时。

| 每周期时间，s | NP1 rank0 | NP2 rank0 | NP2 rank1 |
| --- | ---: | ---: | ---: |
| 时间线窗口 | 1.337735 | 0.847389 | 0.847372 |
| GPU kernel 区间并集 | 1.318808 | 0.737176 | 0.736520 |
| H2D+D2H 设备执行 | 0.005476 | 0.016532 | 0.016591 |
| 不与设备工作重叠的 MPI | 0.000152 | 0.065041 | 0.067930 |
| 其余设备空档 | 0.013299 | 0.028640 | 0.026331 |

单卡 kernel 忙碌约 98.58%；双卡分别约 87.00% / 86.92%。双卡 MPI
暴露时间约占窗口 7.68% / 8.02%。每 rank 每步约 82 次 Allreduce 和
172 次 Sendrecv；Allreduce 约 53.0--55.5 ms，Sendrecv 约 12.0--12.4 ms。
Allreduce 时间包含等待另一 rank 的到达偏差，不等同网络传输成本。
每 rank 每步约 119--120 MB H2D+D2H，包含 halo、最终界面面数据和标量；
它不是整场每核往返拷贝。NP1 也有约 43.1 MB/步的传输，不能全部归入
跨卡通信。

双卡主要核仍有明显负载不均：界面限制核 rank0/1 为 0.282/0.229 s，
内点原始量恢复核为 0.0448/0.0812 s。化学核为 0.174/0.176 s。
观察说明各阶段工作量不均，尚不能单凭该记录确定是状态相关迭代、
边界负载还是硬件差异。原始量恢复调用双温反演，其中振动温度使用
有提前退出的二分迭代，是需要定点计数的候选，不是已证实的唯一根因。

显式 cudaDeviceSynchronize 在单卡约 210--211 次/周期，双卡每 rank
约 326 次。其 API 时间绝大部分覆盖正在执行的 kernel，不能与 kernel
时间相加，更不能把全部同步时长当作取消同步的潜在收益。稳定窗口内
未见 cudaMallocHost/cudaFreeHost 重复分配；现有缓冲复用不是首要缺陷。
“其余设备空档”仅指无设备活动、无 MPI 覆盖的区间，不等于全部可消除。

### 热点核与 Nsight Compute

| 核函数短名 | NP1 每周期 s | 占 kernel 时间 | NCU SM 吞吐 | DRAM 吞吐 | 实际 occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| air5_symmetric_face_kernel | 0.460419 | 34.91% | 82.03% | 11.30% | 26.93% |
| air5_chemistry_half_step_kernel | 0.333099 | 25.26% | 85.32% | 1.53% | 16.66% |
| air5_symmetric_apply_kernel | 0.136070 | 10.32% | 57.23% | 58.22% | 31.70% |

前三者合计约 70.49% 的单卡 kernel 时间。另有面/内点原始量恢复合计
0.173929 s，约 13.19%。NCU 对前三核分别采集首个匹配调用、full 指标集，
均成功生成报告，无权限阻塞；这是调用样本，不是所有轴/状态的平均。
NCU replay/时钟控制会改变核耗时，表中核时间来自 Nsys 而非 NCU。

界面核每线程 128 寄存器、理论 occupancy 33.33%，报告约 1589 万次
local spilling requests。化学核每线程 242 寄存器、理论 occupancy
16.67%，该次采样 spilling 为零。应用核每线程 128 寄存器，约 1612 万次
spilling requests。前两核最高利用管线均为 FP64，分支效率约
99.78% / 99.91%；当前样本不支持把 warp 拒绝步发散列为首要瓶颈。
这些比例取决于 Ada 的 FP64 吞吐，不能外推到 A800。

### 原诊断提出的优化候选

以下第 1 项已由本文件顶部的通量重构 goal 实现并验证；其余仍未实施。

1. **先减少重复通量重构，工作量中、数值风险中。**
   `chemistry_solver_gpu.cuf` 中 base、三个方向的 face 和 apply 都调用
   `air5_full_state_node_face_fluxes`。face/apply 实际只使用当前方向，
   apply 只需低阶通量。优先评估按轴/按高低阶拆分，减少大局部数组和
   无用 FP64 运算；不改变五组分对称投影、六面预算或最终共享面所有权。
   全通量缓存只作为额外显存换计算的候选，不能默认引入。
2. **原始量恢复与化学计算复用，工作量中高、风险中高。**
   先测恢复次数、反演迭代数和化学接受/拒绝分布。只复用同一状态版本
   已有物性，禁止跨阶段复用旧温度/组成；不先放宽 ROS-2 或反演容差。
3. **全局状态检查与通信编排，工作量中、风险中。**
   先区分 Allreduce 到达等待与通信本身，再合并确实同依赖的诊断。
   独立测试 pinned/device-aware 后端、最终面交换及重叠，不能因通用
   HaloTransport 支持 device-aware 就宣称所有 AIR5 专用通信已经直传。
   不省略关键失败检查，不在本轮取消显式同步。
4. **寄存器/块尺寸候选，工作量低中、收益不确定。**
   以完整步而非 occupancy 单指标验收。限制寄存器可能加剧溢出，
   盲目提高 occupancy 未必改善 FP64 饱和核；任何混合精度另设目标。

下一轮先选第 1 项建立独立目标，同相位场、保正、质量/双温能量相容性、
MPI 共享面和 memcheck 不退化，再比较三轮完整步计时。本轮诊断不提升
默认数值选项，不替代更长匹配窗口和真实入射激波物理验证。

### 可复现证据

下列目录均位于 `tests/gpu_validation/out/`，原始大文件不加入 Git：

- `air5_perf_timing_gate_{cpu1,gpu1,gpu2}_20260925/`：同相位场/状态门槛。
- `air5_perf_nsys_np{1,2}_20260925/`：`profile.nsys-rep`、SQLite 和 `steady.json`。
- `air5_perf_ncu_{face,chemistry,apply}_20260925/`：`kernel.ncu-rep`、`details.txt`。
- `air5_perf_diagnosis_20260925/`：合并计时与 CPU/GPU/编译器/MPI 元数据。

驱动：`run_air5_performance_diagnosis.py`；采集：
`run_air5_sbli_domain_replay.py --nsys` 或 `--np 1 --ncu-kernel REGEX`；
稳定窗口提取：`summarize_air5_nsys.py profile.sqlite --output steady.json`。

基线提交 `7bf99c5`，保持 `symmetric_species + layered`、FP64、显式同步、
现有化学及物性模型。复用 Mach4 无入射激波平板检查点
`tests/gpu_validation/out/air5_layered_multistep_20260924/gpu_dt2/gpu`，
step1090、t=2.18 us，网格区间 63x511x7，dt=2 ns。
本阶段做诊断，不修改算法、精度、同步策略，不宣称完整 SBLI 性能验收。

计时契约：CPU NP1、GPU NP1、GPU NP2 x-slab，各至少三次独立运行，
均从同一检查点开始。初始窗口为预热一步、计时三步，保留实际化学、
输运、边界和 MPI。阶段快照和中间大文件输出不得进入计时窗口，少量
统计可保留。报告每步中位数、离散性、CPU 基线加速比和双卡强扩展效率。
若窗口不能形成稳定计时，报告原因再调整设计，不挑选最有利样本。

初步源码审查：`benchmark_runtime` 的无场 I/O/CPU RK 模式仅允许周期
TGV/Shu-Osher，不能直接用于 AIR5。AIR5 的 GPU 分支在常规 phase timing
之前返回。`ASTR_GPU_COMPLETE_STEP_TIMING` 虽覆盖反应调用，但仅输出
I/O rank，缺少同位置 CPU 对照。必须建立相同调用边界的逐 rank 完整步
计时，不能把输运 RK 时间与包含化学的完整步时间混比。

实施与验收顺序：
1. 新增最小默认关闭的公共完整步计时与驱动，检查无中途大文件写入。
2. 验证计时开关不改变同相位场，运行 CPU/GPU 小窗口保正、物性域及
   原逐元素误差门槛；失败即停止下游性能结论。
3. 完成三组、各三轮无 profiler 计时，保存输入、二进制及环境来源。
4. Nsight Systems 采集 GPU NP1/NP2 CUDA/MPI 时间线，区分初始化和推进，
   报告热点核、同步、拷贝、通信和主机空档。
5. 对实测最主要的 1--3 个核用 Nsight Compute 采集指标；权限或兼容性
   失败须留证并明确缺失，不能推定或伪造硬件计数。
6. 同步本文件和当前状态，交付有证据的优化候选与风险，不在本 goal
   实施优化，不操作远程任务、不进行 Git 提交。

本地两张 RTX 4000 Ada 约 19 GiB 显存；本轮 Nsight Systems 2025.6.1、
Nsight Compute 2025.4.0 均已实际采集成功。

以下保留此前非反应流性能优化计划与阶段记录。

状态更新：2026-09-07。P1、P2 和 P3 本机阶段均已收口；P0 A800 基线仍待执行。

## 1. 目标

本计划面向当前 ASTR CUDA Fortran 实现，在不改变既定数值方法和物理语义的前提下，提高单 GPU 计算效率，并为后续多 GPU 通信优化和激波边界层算例性能优化建立统一流程。

性能优化必须同时满足以下条件：

- CPU/GPU 数值结果继续通过既有误差门槛；
- RK3、边界处理、MPI halo 和滤波时序不变；
- 主时间循环保持 GPU 变量常驻；
- 性能结论来自完整 RK 计时，不以单个 kernel 的 NCU replay 时间代替；
- 每项优化具有明确的 NCU 或 Nsight Systems 瓶颈证据；
- 失败候选保留测试记录，但不保留在正式计算路径中。

当前源码起点为提交：

```text
0d684a84bb689654bfc5a53a6928140c3a8e6fdb
perf(gpu): close TGV profiling and candidate validation
```

该提交是本轮性能优化的冻结起点，不是当前仓库 HEAD。P2 正式实现收口于
`566b5bc`，P3 HaloTransport 收口于 `547a398`。后续 OpenSBLI 功能提交不得
被重新标注为上述历史性能数据的执行版本。

## 2. 当前性能基线

当前可靠的单 GPU 性能结果来自 RTX 4000 Ada 工作站上的三维周期 TGV：

| 项目 | 固定设置 |
|---|---|
| 网格 | `256x256x256` |
| MPI | NP=1，`1x1x1` |
| 精度 | FP64 |
| 对流和黏性导数 | 六阶显式中心差分 `643e/643e` |
| 滤波 | 十阶显式中心滤波，开启 |
| 时间推进 | RK3 |
| 同步 | 每个 kernel 后显式同步 |
| 输出 | 计时区间内关闭 checkpoint 和全场输出 |
| 重复方式 | 一次进程预热，五个独立计时进程 |

| 指标 | 原始 GPU 基线 | 当前 GPU 实现 | 变化 |
|---|---:|---:|---:|
| 完整 RK 中位时间 | `0.762855769 s` | `0.580044424 s` | 降低 `23.964%` |
| GPU 版本内部加速比 | `1.0x` | `1.31517x` | 吞吐提高 `31.517%` |
| 重复测试离散度 | `2.105%` | `2.027%` | 通过 5% 门槛 |
| 峰值显存 | `9766 MiB` | `9738 MiB` | 降低 `0.287%` |

这里的 `1.31517x` 只表示优化前后两个 GPU 版本的差异，不表示相对 CPU NP=1 的整体加速比。后续论文和汇报必须分别报告：

1. GPU 优化增益：优化后 GPU 与冻结 GPU 基线比较；
2. 应用整体加速比：优化后 GPU 与同口径 CPU NP=1 比较；
3. 多 GPU 扩展效率：NP=N 与 NP=1 GPU 比较，并要求一个 MPI rank 对应一张物理 GPU。

## 3. 已保留的优化

当前正式路径已经保留以下优化：

1. 周期 x/y/z 对流、diffusion-RHS 和滤波 kernel 使用线性线程映射，减少无效线程，并改善 Fortran 存储顺序下的连续访问。
2. 守恒量转原始量时复用局部速度和压力，避免写入全局数组后立即重新读取。
3. 完全周期且开启滤波的 TGV 跳过冗余的第一阶段原始量刷新；物理边界算例继续执行原刷新路径。
4. x 周期 primitive halo kernel 让 halo 位置成为最快变化的线程索引，并复用局部原始量。
5. 六阶显式导数统一使用 `constdef::num1d60`，避免 `/60.d0` 在多个设备调用点生成重复 FP64 倒数指令。
6. 建立完整 RK 计时、12-kernel NCU 矩阵、Nsight Systems 常驻分析、Compute Sanitizer 和候选验收 runner。

`num1d60` 优化已经获得正常执行和 NCU 两类证据：

- diffusion-flux 正常执行时间降低约 `31.10%`；
- convection x/y/z 总时间降低约 `28.77%`；
- `gradcal` 时间降低约 `29.73%`；
- 对应六阶导数热点中的 `MUFU.RCP64H` 被消除；
- 该轮总 GPU kernel 时间降低约 `10.05%`。

## 4. 已拒绝或暂不采用的优化

### 4.1 diffusion-flux 线程块变更

将 diffusion-flux 从 `(8,8,8)` 改为 `(32,4,4)` 后，完整 RK 时间慢约 `0.093%`。该变化没有达到性能门槛，已经撤销。

### 4.2 大量取消显式同步

受控 selective-sync 实验将性能区间内的 `cudaDeviceSynchronize` 从 275 次降低到 15 次，但配对测试中位时间从 `0.682327627 s/RK` 变为 `0.683028141 s/RK`，慢约 `0.103%`。另一个 compute-only 候选慢约 `3.426%`，离散度达到 `6.677%`。

因此：

- `explicit` 继续作为正式默认模式；
- `selective` 只保留为 TGV 受限实验入口；
- 当前硬件上的主要矛盾是 kernel 执行，而不是主机同步调用开销；
- 后续候选不得把取消同步和计算 kernel 优化混在同一次测试中。

### 4.3 当前不进入的方向

以下方向不属于当前性能优化范围：

- `--use_fast_math`、`-Mfprelaxed`、`-fast`；
- FP32、TF32、混合精度和 Tensor Core 替换；
- 修改六阶显式差分或十阶显式滤波系数；
- compact 差分和 compact 滤波；
- 跨 RK、边界、滤波或 MPI 阶段的 kernel fusion；
- CUDA Graphs；
- GPU HDF5/checkpoint 输出；
- 仅凭 GPU 利用率或单个 NCU replay 时间宣布整体加速。

## 5. Phase P0：A800 基线冻结

状态：暂缓。A800 排队任务不作为本机 P1 探索的前置条件；本机候选仍以
`0d684a8`、RTX 4000 Ada 和 `0.580044424 s/RK` 冻结基线判定。任何本机
结论都不得外推为 A800 结论。

### 5.1 目的

RTX 4000 Ada 的测试结果不能直接外推到 A800。A800 的 FP64 吞吐、显存带宽、寄存器和 occupancy 平衡不同，热点排序可能发生变化。所有后续正式候选应优先以 A800 同机数据作为基线。

### 5.2 操作

1. 从提交 `0d684a8` 建立新的 A800 构建目录，不复用工作站 CMake cache。
2. 使用项目根目录 `CMakeLists.txt`、NVHPC RELEASE 和 `ASTR_WITH_CUDA=ON` 构建。
3. 记录 GPU 型号、UUID、驱动、NVHPC、MPI、Nsight、功率状态和编译命令。
4. 运行 `256^3` NP=1 五次完整 RK 基准。
5. 运行相同算例的 Nsight Systems 常驻分析。
6. 运行 12-kernel Nsight Compute 矩阵。
7. 运行 NP=1/NP=2 十步场和统计量比较以及 Compute Sanitizer。
8. 在显存允许时增加 `512^3` NP=1 测试，用于判断更大问题规模下的带宽和计算饱和状态。`512^3` 结果作为规模效应证据，不替代 `256^3` 固定验收算例。

### 5.3 CPU 基线补充

建立同口径 CPU NP=1 完整 RK benchmark：

- 使用相同网格、步数、数值格式、滤波和扩散设置；
- 排除初始化、checkpoint、HDF5 和结束输出；
- 至少运行三个独立进程，条件允许时运行五次；
- 报告中位时间和离散度；
- 只用该结果计算 CPU NP=1 到 GPU NP=1 的整体加速比。

### 5.4 验收

- 五次 GPU 测试离散度不超过 `5%`；
- NP=1/NP=2 CPU/GPU 误差不超过 `1e-10`；
- 常驻区间没有禁止的全场 H2D/D2H；
- memcheck 为 0 errors；
- A800 baseline timing TSV、NSYS、NCU 和环境记录完整保存。

## 6. Phase P1-C1：convection 和 diffusion-flux 读取复用

状态：已于 2026-09-06 按替代终止条件关闭，没有候选进入正式源码。

### 6.1 当前证据

NCU 显示 convection 和 diffusion-flux 同时具有较高计算吞吐、TEX throttle 和 short-scoreboard stall。它们重复读取速度、压力、Jacobian 和网格度量，是下一项优先候选。

### 6.2 候选原则

- 在单个线程内复用已经读取的 primitive 和 metric 值；
- 合并数学上完全相同的局部计算；
- 不新增全场临时数组；
- 不改变浮点运算顺序时优先；确需改变局部求值顺序时必须先说明误差预期；
- 不改变既定 x/y/z block shape；
- shared memory 只能在 NCU 证明跨线程重复读取足够高时作为第二候选。

### 6.3 重点指标

- executed instructions；
- global load sectors 和 bytes；
- L1/L2 命中率；
- TEX throttle；
- short/long scoreboard stall；
- registers/thread 和 local-memory spill；
- 完整 RK 中位时间。

### 6.4 验收

- 完整 RK 中位时间相对 A800 冻结基线至少降低 `3%`；
- 候选离散度不超过 `5%`；
- 峰值显存增长不超过 `5%`；
- NP=1/NP=2 十步 TGV 场和统计量通过 `1e-10`；
- 12-kernel 矩阵没有隐藏其他常用 kernel 的显著回退。

### 6.5 候选结果

固定基线为 `0.580044424 s/RK`。三个候选均使用 `256^3`、NP=1、FP64、
`643e/643e`、十阶显式滤波、扩散开启和逐 kernel 显式同步。每项候选在
测试后均已从计算路径撤销。

| 候选 | NCU 机制证据 | 五轮完整 RK 中位时间 | 相对基线 | 结论 |
|---|---|---:|---:|---|
| C1-1 `flux_at_global` 局部标量复用 | `conv_x` 的 registers、instructions 和 excessive L2 均不变；编译器已完成等价复用 | `0.589632722 s` | 慢 `1.653%` | 拒绝 |
| C1-2 扩散常量倒数/组合系数前移 | `diffusion_flux` 从 `27.430 ms` 降至 `25.268 ms`，instructions 从 `430.70 M` 降至 `411.05 M` | `0.581123491 s` | 慢 `0.186%` | 单核有效但整步无收益，拒绝 |
| C1-3 复用 `qwork_d` 存储方向通量 | 六个新核合计约 `53.02 ms/stage`，DRAM throughput 为 `87.27%` 至 `93.79%` | `0.638189768 s` | 慢 `10.024%` | 新增全场写读成为带宽瓶颈，拒绝 |

三项候选的 run-to-run spread 分别为 `1.832%`、`1.345%` 和 `1.653%`，
峰值显存均未增长。C1-1 证明手工局部变量不能优于 NVHPC 已有公共子表达式
消除。C1-2 证明单核优化不能替代完整 RK 验收。C1-3 证明当前显式同步约束下，
以全场工作数组换取通量复用不适合 RTX 4000 Ada。

证据目录：

- `tests/gpu_validation/out/p1_c1_baseline_conv_diff_0d684a8/`
- `tests/gpu_validation/out/p1_c1_candidate1_flux_local_reuse/`
- `tests/gpu_validation/out/p1_c1_candidate1_flux_local_reuse_ncu/`
- `tests/gpu_validation/out/p1_c1_candidate2_combined/`
- `tests/gpu_validation/out/p1_c1_candidate2_combined_ncu/`
- `tests/gpu_validation/out/p1_c1_candidate3_stored_conv/`
- `tests/gpu_validation/out/p1_c1_candidate3_stored_conv_ncu/`

下一阶段进入 P1-C2。C1-2 的逐点常量系数前移可在将来与更大范围的
diffusion 重构组合测试，但在完整 RK 未达到 `3%` 前不得单独保留。

## 7. Phase P1-C2：diffusion-RHS 寄存器生命周期

### 7.1 当前证据

三个 diffusion-RHS kernel 当前约为 128 registers/thread，achieved occupancy 约为 32%。优化目标不是单纯追求更高 occupancy，而是在不增加全局流量的条件下减少长生命周期临时量和 spill。

### 7.2 候选顺序

1. 缩小局部变量作用域，减少不同时使用的临时量同时存活。
2. 复用只读系数和已经加载的数据。
3. 检查内联设备函数是否导致过多临时量。
4. 只有前述方法无效时，测试受控的局部计算拆分。

不得为了降低寄存器数量而增加全场中间数组或额外 H2D/D2H。

### 7.3 验收

采用与 C1 相同的完整 gate。寄存器下降或 occupancy 上升只是机制证据，若完整 RK 改善不足 `3%`，该候选不进入正式路径。

### 7.4 筛选结果

P1-C2 已关闭，没有保留计算候选。所有测试均使用固定的 512 线程方向块、
FP64、`256^3` TGV 和逐 kernel 显式同步。

| 候选 | NCU 机制证据 | 完整 RK | 结论 |
|---|---|---:|---|
| 六组通量数组复用为一组正负缓冲 | 三方向仍为 128 registers/thread，三核合计仅降低约 `0.974%` | 未进入整步门 | 寄存器生命周期未改变，拒绝 |
| 全文件 `maxregcount=96` | occupancy 仍约 `32%`，出现约 `9.55 M` spill 指令 | 未进入整步门 | 未跨过双块驻留阈值，拒绝 |
| 全文件 `maxregcount=64` | occupancy 提升到 `61-63%`，但出现约 `146.46 M` spill 指令；三核合计慢 `35.45%` | 未进入整步门 | spill 抵消 occupancy，拒绝 |
| 周期 diffusion-RHS 拆为动量核和能量核 | 动量核仍为 128 registers/thread，能量核为 96-98 | `0.601406599 s/RK`，慢 `3.68%` | 增加 launch、同步和度量读取，拒绝 |

结论是当前固定 512 线程块下，寄存器数只有降至不高于 64 才能增加 block
residency，但编译器限寄存器会造成不可接受的 local-memory spill。后续若重新
打开该方向，必须通过更大范围的算法重组降低活跃状态，而不能继续调整
`maxregcount`。

## 8. Phase P1-C3：y/z 十阶滤波访存

### 8.1 当前证据

x 滤波 occupancy 约为 60%，y/z 滤波约为 32%，并具有明显 TEX 和 scoreboard stall。Fortran 第一维连续存储使 y/z stencil 天然存在较大访问跨度。

### 8.2 候选顺序

1. 调整线程到数据的映射，但保持既定 block shape。
2. 减少同一线程对 stencil 数据的重复读取。
3. 检查只读缓存和 L2 reuse。
4. 只有 NCU 证明复用收益足够时，测试共享内存 tile。
5. 保留当前全 `q` ping-pong 实现为默认正确性路径。

此前规划的“单个 3D work array”继续作为额外低显存方案，而不是替代当前 ping-pong。该方案主要解决可计算网格规模，不预设其速度更快。它必须单独测量额外 pass、复制和同步成本。

### 8.3 验收

- 首先通过滤波模块级 CPU/GPU 差值；
- 再通过 TGV、物理壁面和曲线网格滤波回归；
- 最后应用完整性能 gate；
- 只有完整 RK 改善达到门槛才作为性能默认实现。

### 8.4 筛选结果

P1-C3 已关闭，没有保留性能核。全周期专用 halo 核证明去除物理边界分支可将
y/z 寄存器从 80 降到 58/60，并将单核时间分别降低约 `11.45%` 和 `12.39%`，
但五轮完整 RK 中位时间为 `0.582806489 s`，相对冻结基线慢 `0.48%`。

逐分量 shared tile 将 excessive L2 从约 `25.5 MiB` 降到 `5-6 MiB`，但十次
block barrier 和附加寻址令 y/z 核变慢到 `10.270/11.379 ms`。一次装载五个
分量的 tile 将 barrier 降为一次，y/z 为 `8.931/9.433 ms`，两核合计只改善
约 `4.94%`，折算完整 RK 的理论收益约 `0.5%`，未进入五轮验收。共享内存
版本还会使 z 核 DRAM throughput 升至约 `83.4%`。

上述“拒绝”结论限定于当前 RTX 4000 Ada、当前 NVHPC/CUDA 工具链和冻结的
`0.580044424 s/RK` 基线。它表示候选不应成为当前工作站默认实现，不表示这些
机制在 A800 上必然无效。A800 的 FP64 吞吐、显存带宽、寄存器驻留阈值和缓存
行为不同，因此保留各候选的补丁机制、NCU 报告和输出目录，待 A800 环境稳定后
逐项重建。复测必须先建立同机五轮完整 RK 基线，再检查相同 kernel 的寄存器、
spill、吞吐和缓存指标；只有完整 RK 中位时间改善至少 `3%` 且全部正确性门通过，
候选才可恢复。不得直接使用工作站上的单 kernel 百分比推断 A800 收益。

十阶滤波继续使用原全 `q` ping-pong 默认实现。六个有效滤波系数集中到
`constdef.F90` 的编译期 `parameter`，GPU 通过常数乘法使用；NCU SASS 已确认
热点中为立即数 `DMUL/DFMA`，不存在由这些固定分数生成的运行时 FP64 除法。

## 9. Phase P2：激波和 SBLI 专项优化

TGV 不能代表 selective-Roe 和特征重构的成本。激波路径需要独立性能基线。

### 9.1 基准算例

- 保留 TGV 作为规则 stencil 性能基准；
- 使用三维周期挤出 Shu-Osher 作为可控激波占比和 kernel 定位基准；
- 使用已验证的三维高超声速边界层/SBLI 路径作为完整激波性能基准；
- 分离物理正确性、CPU/GPU 数值一致性和性能三类结论。

P2-0 不修改数值 kernel。Shu-Osher 默认网格为 `256x64x32`，SBLI 默认网格为
`256x192x32`。两者采用 NP=1、逐 kernel 显式同步、20 steps、一次进程预热和
五轮独立计时。mask 激活比例通过独立的一步 dump 离线计算，不在计时循环中加入
reduction、D2H 或文本输出。checkpoint 和场输出频率必须大于 `maxstep`。
SBLI 性能初场使用已验证的解析斜激波叠加场，使首个 RK 即覆盖 selective-Roe
路径；上边界仍使用相同斜激波目标的 `bctype=52` NSCBC。无激波初场继续保留为
边界注入正确性回归，不作为激波 kernel 性能基准。

P2-0 同时解除一个非数值 capability 耦合：已验证的 characteristic Shu-Osher
路径不再要求设置 `ASTR_SHOCK_SENSOR_DUMP` 才能运行。`lchardecomp=t` 已经要求
每个 RK 计算 sensor 和 mask，dump 仅是可选诊断输出，不能作为数值能力开关。
仅传感器验证使用的 S0-A4/A5 仍保持 dump 限制。

### 9.2 优化对象

- characteristic interface-flux kernel 的寄存器和 spill；
- 激波传感器、扩展 mask 和特征通量之间的重复读取；
- 激波区域比例变化引起的线程分支成本；
- adaptive 单 kernel 与 physical/shock 分离 kernel 的实际成本；
- 传感器 halo 交换和通量计算之间的等待时间。

不得在没有测量激波单元占比和 branch 指标前引入压缩界面列表。前缀扫描和不规则索引可能比线程发散更昂贵。

### 9.3 验收

- Sod/Shu-Osher 精度与超调门槛不退化；
- SBLI CPU/GPU 场、壁面压力、摩擦和热流诊断不退化；
- 单 kernel 改善必须转化为完整 RK 改善；
- 多 rank 下 shock-sensor halo 和 characteristic flux 保持一致。

### 9.4 P2-0 基线结论

P2-0 已在 RTX 4000 Ada 上完成，完整证据见
`documents/ASTR_PHASE_P2_BASELINE_REPORT.md`。Shu-Osher `256x64x32` 和 SBLI
`256x192x32` 的五轮完整 RK 中位时间分别为 `0.151583855 s` 和
`0.523264916 s`，离散度分别为 `1.933%` 和 `0.363%`。两者 shock-active
节点比例分别为 `9.3385%` 和 `3.1713%`。

Nsight Systems 显示三方向 characteristic interface-flux 占两算例总 kernel
时间约 `94.1%` 和 `87.5%`，sensor 与 mask expansion 合计不足 `1%`。y 向
特征通量使用 128 registers/thread，occupancy 约 `32%`，并产生大量 local
spill；branch efficiency 约 `99.94-99.95%`。因此 P2-1 首选 y 向“全域物理
通量 + active 界面特征覆盖”双 kernel 筛选，优化假设是隔离高资源特征路径，
而不是消除严重线程发散。该候选修改数值 kernel，实施前需单独审批。

### 9.5 P2-1A y 向双 kernel 筛选结果

P2-1A 已完成实现、正确性、memcheck、NSYS、NCU 和五轮完整 RK 验收，并按
失败规则撤销源码。Shu-Osher `256x64x32` 从 `0.151583855 s/RK` 增至
`0.184307279 s/RK`，慢 `21.588%`；SBLI `256x192x32` 从
`0.523264916 s/RK` 增至 `0.529303436 s/RK`，慢 `1.154%`。两项离散度分别为
`0.928%` 和 `0.176%`，负收益不是运行噪声。

NCU 显示物理基底 kernel 仍为 128 registers/thread、`32.54%` occupancy 和约
`63.96 M` local-memory spilling requests；特征覆盖 kernel 另有约 `3.90 M`
spilling requests。拆分未实现预期的低资源物理路径，反而增加一次完整界面网格
遍历和显式同步。因此不推进 x/z 同类拆分，生产路径继续使用原单 kernel 实现。
完整证据见 `documents/ASTR_PHASE_P2_BASELINE_REPORT.md`。

上述拒绝仅适用于当前 RTX 4000 Ada 与当前 NVHPC/CUDA 工具链。A800 的 FP64、
寄存器驻留和缓存行为不同，P2-1A 可在冻结 A800 本机基线后重建复测；但必须继续
满足完整 RK 改善至少 `3%`、正确性门全部通过的条件，不能外推工作站单 kernel
指标。

### 9.6 P2-1B 非激波物理通量五分量复用

P2-1B 保留单个 sensor-coupled interface kernel，不采用 P2-1A 的双 kernel
结构。非激波分支原先对五个守恒分量分别调用标量 Steger-Warming 分裂函数，
导致每个 stencil 点重复计算五次网格度量归一化、声速、特征值分裂和平方根。
新路径在每个 stencil 点一次生成五分量物理通量，并复用该线程原本已经存在的
`split_plus(5,7)` 与 `split_minus(5,7)`。未新增全场数组、kernel、同步或
H2D/D2H。

候选保留 `sqrt(tmp)/mach` 声速定义、FP64 运算、MP7/Roe 选择语义和固定线程块。
物理 x/y 方向的非激波和 Roe 分支均令 `npdc=3` 显式使用 `-hm:dim+hm`，与
原标量物理实现一致，避免未来出现内部 MPI rank 时错误钳制 halo stencil。

RTX 4000 Ada 五轮完整 RK 结果如下：

| 算例 | P2-0 基线 | P2-1B | 改善 | spread | 决策 |
|---|---:|---:|---:|---:|---|
| Shu-Osher `256x64x32` | `0.151583855 s` | `0.109288716 s` | `27.902%` | `0.244%` | 保留 |
| SBLI `256x192x32` | `0.523264916 s` | `0.268521579 s` | `48.684%` | `0.431%` | 保留 |

SBLI y 核 NCU replay 从 `103.30 ms` 降至 `33.53 ms`，local spill requests 从
`68,911,722` 降至 `15,059,748`。寄存器仍为 128/thread，说明收益来自消除
重复标量物理分裂工作并缩短局部临时量压力，而不是通过寄存器上限提高 occupancy。

十步 SBLI 完整场继续通过，primitive 与重构守恒量最大差分别约为 `1.02e-14`
和 `6.00e-15`。此前在线 `massflux` 的 `9.05e-9` 差异已定位为统计相位错误：CPU
在 `rkfirst` 前完成 `bctype=52` 上边界 x/z 横向滤波，GPU 统计曾在滤波前取样。
GPU 现复用 `qsave_d` 保存完整 RK 状态，构造同相位滤波状态完成统计，再恢复状态
执行正式 RK。十步 `massflux` 最大差降至 `5.03e-13`，且不改变场演化或增加全场
device 数组。完整证据与复现路径见
`documents/ASTR_PHASE_P2_BASELINE_REPORT.md`。

### 9.7 P2-1B 修正后热点

修正后 Nsight Systems 中，三方向 characteristic interface flux 在 Shu-Osher
和 SBLI 分别占 kernel 时间 `91.7%` 和 `73.1%`。raw sensor 与 expanded mask
合计仅 `1.2%` 和 `1.4%`，其完整 RK 理论上限不足 `3%`，不进入候选。

SBLI x/y/z 特征通量 NCU 均为 128 registers/thread，local spilling requests
分别为 `19,472,739`、`15,059,748` 和 `6,762,124`；Shu-Osher x 为
`8,485,620`。该证据支持再筛选一个减少 thread-local split storage 的候选，
但不支持 mask fusion 或压缩 active-interface list。

### 9.8 P2-1C 单 split 数组筛选结果

P2-1C 让正、负分裂通量顺序复用一个 `split_flux(5,7)`，并用 5 个标量保存正向
特征重构值。一步 Shu-Osher 和 SBLI 正确性通过。同为每轮 10 个 RK 样本的五轮
正式比较仅改善 `1.975%` 和 `1.208%`，低于 `3%`。SBLI y 核寄存器仍为
128/thread，spill requests 反而从
`15,059,748` 增至 `18,007,902`。候选源码已撤销，原始 timing、NCU 和源码副本
保留在 `tests/gpu_validation/out/p2_full_p21c_split_reuse_*`。

### 9.9 Shock-Sensor Halo 等待结论

NP=2 y-slab 的 `cuda,mpi` 时间线按每张 GPU 配对 raw sensor 与 expanded mask。
Shu-Osher 的 halo 关键路径为 `7.706589 ms`，占 profile 内完整 RK 的 `3.488%`；
SBLI 为 `8.765948 ms`，占 `1.421%`。Shu-Osher 中 P2 可改的 sensor pack/unpack
kernel 只占完整 RK 约 `0.074%`。其余是 pageable host staging 与 blocking MPI，
属于 P3 通信后端，不在 P2 内用 kernel 改动处理。

### 9.10 P2 关闭状态

最终生产路径通过顶层 CPU/GPU 构建、54 个 Python 测试、全部 shell 语法检查、
Sod/Shu-Osher/SBLI 十步对照、Shu-Osher NP=2 x/y/z 与 NP=8 `2x2x2`、SBLI
壁面统计及 Compute Sanitizer 0 errors。恢复后五轮复测为：

| 算例 | 最终中位时间 | spread | 峰值显存 | 相对 P2-0 |
|---|---:|---:|---:|---:|
| Shu-Osher `256x64x32` | `0.110301406 s/RK` | `0.958%` | 810 MiB | 改善 `27.234%` |
| SBLI `256x192x32` | `0.267333482 s/RK` | `0.770%` | 1,500 MiB | 改善 `48.910%` |

P2 已关闭。正式源码只保留 P2-1B。P2-1A/P2-1C 均拒绝，sensor/mask 无 3%
收益上限，halo 剩余机会转交 P3。

## 10. Phase P3：多 GPU HaloTransport 优化

2026-09-07 本机验收完成。默认仍为 pageable blocking；保留可选 `pinned`
及仅适用于全周期 stored diffusion 内部区域的 `pinned-overlap`。
独立 nonblocking 因收益不足拒绝，CUDA-aware 因当前软件栈准入失败暂缓。
以下早期筛选记录保留，最终验收结论见本文件末尾和 P3 基线报告。

当前 host-staged blocking 交换继续作为可移植正确性基线。solution/sponge 的
qswap-compatible 路径保留 `hm+1` 和接口面平均，filter/qwork、diffusion 和
shock-sensor 路径保留固定 `hm`。优化按以下顺序进行：

1. pageable host buffer 改为 pinned host buffer；
2. blocking host-staged MPI 改为 nonblocking host-staged MPI；
3. 分离内部区域和 halo 邻近区域，评估计算通信重叠；
4. 在 NVIDIA 平台评估 CUDA-aware MPI；
5. 为未来 AMD/HIP/DCU 保留 host-staged 后端和设备感知接口边界。

通信重叠会改变同步和阶段组织，不得与 P1 kernel 优化同时实施。它需要单独设计、审批和性能基线。

### 10.1 硬件和完成状态

P3 已完成本机最终验收，详细证据见 `ASTR_PHASE_P3_IMPLEMENTATION_PLAN.md`
和 `ASTR_PHASE_P3_BASELINE_REPORT.md`。最终结论为：

- pageable blocking 保留为默认、可移植的正确性基线；
- pinned blocking 作为可选后端保留，九个正式组合的完整 RK 时间减少
  `2.829%--23.728%`；
- pinned-overlap 仅对完全周期 stored-diffusion 内部区域启用，TGV x/y/z
  相对配对 pinned 的增量为 `2.616%/3.085%/2.944%`；
- 独立 paired nonblocking 最大可复现增量为 `2.425%`，未达到 `3%` 门槛，
  已从正式源码撤下；
- CUDA-aware MPI 在当前 HPC-X 软件栈的大消息或 sanitizer 门槛失败，暂不接入；
- 物理边界闭合、shock sensor 和 SBLI 不直接复用周期扩散的内部区 overlap。

- 性能测试必须一个 MPI rank 对应一张物理 GPU；
- 两张 GPU 上的 NP=4/8 oversubscription 只能作为正确性测试；
- NP=4/8 扩展效率必须在具有相应物理 GPU 数量的平台上测试。

### 10.2 验收

- 所有 x/y/z 和组合拓扑继续通过 halo 正确性门槛；
- 通信时间或完整 RK 时间有可重复下降；
- overlap 必须由时间线证明，而不是根据非阻塞 API 名称推断；
- CUDA-aware 路径失败时可以回退到 host-staged 正确性后端。

### 10.3 Phase P4：参考 OpenCFD-SCU 的任务图优化

OpenCFD-SCU 静态源码审计表明，其主要参考价值是将内部区、边界带、halo
传输组织为多 stream/event 任务图，以及针对 y/z 模板计算改善数据移动。其
通信本身仍采用 host-staged、逐场阻塞 `MPI_Sendrecv`，不作为 ASTR 的实现基线。

P4 保留 ASTR 当前一次分配并注册的 pinned host 缓冲、聚合消息、固定 tag 和
`MPI_Irecv/Isend/Testall` 状态机。执行顺序为：

1. 将 x-slab 单 context 路径推广到 y/z slab，覆盖 solution、filter 和融合
   diffusion halo；
2. 为 x/y/z 建立独立 context，随后支持双轴和三轴分解；
3. 按每个算子真实 stencil 半宽划分内部区和边界带，使用 event 释放边界计算；
4. 只有 NCU 证明 y/z 访存受限时，测试共享内存转置或 warp shuffle；
5. 只测试局部生产者与消费者融合，不建立全 RHS 巨型 kernel。

明确不采用 OpenCFD-SCU 的逐场阻塞通信、RHS `atomicAdd`、全局 rank 取模绑卡、
循环内临时分配和朴素线程内前缀计数。OpenCFD-SCU 的实现仅提供架构候选来源，
不构成 ASTR 性能收益证据。

2026-09-15 已完成执行项 1 和 2。`pinned-pipeline` 现覆盖周期 TGV 的 x/y/z
单轴以及双轴、三轴分解。每个活动轴拥有独立 stream、event、request、邻居、
计数和私有固定 tag。solution halo 会先发布所有活动轴的 MPI，再按 x/y/z 顺序
等待与 unpack。filter 保持有序的三方向 ping-pong；融合 diffusion 遍历所有轴，
但目前只在首个活动轴通信期间执行一次内部 RHS，尚未实现多个 diffusion 轴同时
推进。

NP=2/3 三轴生产 halo 合约和双 context 精确合约通过。NP=2 单轴及 NP=4
`2x2x1`、NP=8 `2x2x2` 的 1/10/100 步 CPU/GPU 门槛全部通过。多轴 100 步最大
守恒场差分别为 `7.1054e-13` 和 `7.6739e-13`。NP=4 完整求解器的四 rank
full leak-check 均为 `0 errors`、`0 bytes leaked`，racecheck 均为
`0 errors, 0 warnings`。

重启后的本地 `256^3`、NP=2、五轮完整 RK 配对中，y-slab 从
`0.510051923` 降至 `0.479824547 s/RK`，降低 `5.926%`；z-slab 从
`0.508967702` 降至 `0.478513546 s/RK`，降低 `5.984%`。四组相对极差均不超过
`1.057%`，候选显存增量为 `32--33 MiB`。每轴 context 重构后，同一时段 x-slab
五轮中位时间从 pinned explicit 的 `0.529076911 s/RK` 降至 pipeline explicit 的
`0.506394356 s/RK`，降低 `4.287%`；pipeline dependency 为
`0.504894878 s/RK`，只比 pipeline explicit 再低 `0.296%`。默认同步继续使用
`explicit`，`dependency` 保持 opt-in。

下一执行项是在四卡 A800 上按一秩一卡完成 NP=4 plane/cube 的 Nsight Systems
时间线和五轮完整 RK 配对。本地 NP=4/8 共享双卡只证明正确性。只有 A800 证明
多轴 MPI 尾部仍限制完整 RK，才继续并行推进多个 diffusion 轴或测试持久 MPI
request；不根据本地 oversubscription 排序改变默认后端。

## 11. 候选执行流程

每个候选使用：

```bash
CANDIDATE_ID=<candidate-id> \
BASELINE_REF=<frozen-git-ref> \
BASELINE_TIMINGS=/absolute/path/to/baseline_timings.tsv \
TARGET_KERNELS='<kernel names>' \
ALLOWED_PATHS='<comma-separated source paths>' \
HYPOTHESIS='<measured bottleneck and expected mechanism>' \
GATE_SET=full \
OUT_DIR=/tmp/astr_candidate_<candidate-id> \
  tests/gpu_validation/run_gpu_optimization_candidate_gate.sh
```

执行顺序：

1. 检查候选元数据和改动范围；
2. 从顶层 CMake 构建 CPU/GPU；
3. NP=1 和 NP=2 十步场/统计量比较；
4. 五次完整 RK benchmark；
5. 与冻结基线比较时间、离散度和显存；
6. Nsight Systems 常驻检查；
7. 12-kernel NCU 因果检查；
8. Compute Sanitizer；
9. 补跑候选影响模块对应的物理边界、曲线网格或激波回归矩阵。

## 12. 停止条件

出现以下任一情况时停止当前候选，不继续堆叠优化：

- CPU/GPU 误差超过对应门槛；
- 数值格式、边界或 MPI 语义发生未批准变化；
- 新增禁止的全场数据传输；
- Compute Sanitizer 报错；
- 五次测试离散度超过 `5%`；
- 完整 RK 改善不足 `3%`；
- NCU 指标改善但完整 RK 没有改善；
- 候选只在 oversubscription rank 配置中表现更快；
- 需要通过放宽误差、改变精度或启用 relaxed math 才能获得加速。

失败后保留报告和日志，源码候选单独回退。不得删除失败证据，也不得将多个失败候选混合后重新测试。

## 13. 推荐执行顺序

近期执行顺序固定为：

```text
P0  A800 基线与 CPU NP=1 同口径基线（暂缓，不阻塞本机探索）
 -> P1-C1 convection/diffusion-flux 读取复用（已关闭，无保留候选）
 -> P1-C2 diffusion-RHS 寄存器生命周期（已关闭，无保留性能候选）
 -> P1-C3 y/z 滤波访存（已关闭，仅保留 constdef 常量集中化）
 -> P2 激波和 SBLI 专项优化（已关闭，仅保留 P2-1B）
 -> P3 多 GPU HaloTransport 优化（已完成本机收口）
```

下一性能阶段为 P0 A800 NP=1/2/4 基线和强扩展测试。609x255x9 OpenSBLI
薄层用于物理验证，不单独承担四卡三维扩展结论。在 P0 完成前，不对 A800
做性能结论。取消显式同步、kernel fusion、CUDA Graphs 和混合精度均不属于
既有验收结论，需要单独立项审批。

## 14. 交付物

本节先保留 P3 执行期间的过程记录，最终状态以末尾“P3 最终本机验收”为准。

P3 CUDA-aware 准入更新：当前 HPC-X 2.25.1 软件栈的小消息设备缓冲区测试通过，
但默认 UCX 大消息出现接收数据未更新。关闭 IPC 后逐元素比较通过，
Compute Sanitizer 仍未达到零错误门槛，提前绑定设备也未解决。
本轮停止该候选接入，不给出其性能结论，也不据此断言硬件不支持。
独立探针、配置、失败日志和冻结哈希见 `ASTR_PHASE_P3_BASELINE_REPORT.md`。
默认 pageable 回退不变，pinned 与 pinned-overlap 的最终完整回归仍待收口。
当前收口检查点已通过三种后端、三个算例、五种拓扑的 45 组十步逐场及统计量比较，
以及九组小网格 memcheck，18 份 rank 日志均为零错误。
SBLI 使用完整 RK 同相位比较。该内存检查只限定于 host-only MPI 配置。
随后已补齐 18 处物理端点 `MPI_PROC_NULL` 接收上传保护，重新通过 45 组主回归、
九组开启滤波和扩散的物理壁面专项，以及 18 组 memcheck 的 36 份零错误 rank 日志。
生产 pack/unpack 冲突接口值专项随后也已通过：三种后端分别运行 NP=2/3，
直接检查生产模块的三轴平均、所有权和分量边界，另有九份 memcheck rank 日志零错误。
最终源码的 TGV 256^3 x-slab 新测量中，pinned 完整 RK 时间比 pageable 下降 13.839%。
overlap 组 spread 为 8.020%，整组不确定，不能剔除单次结果后声称通过。
仍需完整配对重测和其余方向、算例的最终性能复核，尚不宣布 P3 完成。
Shu-Osher/SBLI 三方向的最终源码性能组现已完成，Shu z 的高波动组也完成了
完整反向重测。有效组中 pinned 的 RK 时间降幅分别为 2.829%--7.611% 和
7.823%--23.728%，最大单卡显存增幅 3.436%。这两类算例的 overlap 分支未启用。
剩余 TGV y/z 和 TGV x 配对重测的性能收口。

请求完成状态证据已补齐：仅用于 profiling 的 PMPI/NVTX 包装库记录真实
MPI_Testall 返回值，不增加轮询，也不改变求解器。受控双 rank 探针确认
未完成/已完成标记有效，空请求不被计入。最终源码 TGV 256^3 y-slab
捕获到 4237 个未全部完成标记和 24 个全部完成标记位于同进程扩散 kernel
期间。这证明该次插桩执行存在请求与计算重叠，不代表网络带宽或正式整步收益。
所有正式性能测试禁用该包装库。证据、重现方法和分析器测试见 P3 报告及验证 README。

### P3 最终本机验收

- 九个正式组合采用最终冻结源码、NP=2 双物理 GPU、五次完整进程重复。
  pinned 相对 pageable 的完整 RK 时间减少 2.829%--23.728%。
- overlap 相对配对 pinned 的 TGV x/y/z 增量为 2.616%/3.085%/2.944%。
  y 达到 3% 单点门槛，其他有效组没有超过 1% 的稳定退化。
- TGV x 两组噪声结果全部保留，最后一次完整配对重测通过稳定性门槛。
  共审计 34 组、204 个含预热的进程日志，未剔除单次结果。
- 最大单卡显存增长 3.436%。主机锁页内存另记，不与显存混淆。
- 最终源码 45 组 CPU/GPU 对比、九组物理壁面检查、生产 halo 精确合同及
  host-only MPI 的零错误 sanitizer 矩阵通过。68 个 Python 测试、96 个
  shell 语法检查和顶层 CPU/GPU 构建通过。
- 新增冻结 L0、小网格 overlap、NSYS 插桩及重链接版本的直接逐场对照，
  六个原始场和五个重构守恒场最大差值均为零。

逐 kernel 显式同步、FP64、RK3、数值格式及边界/halo 语义保持不变。
此结论仅覆盖本机 host-staged 路径，不代表 A800、多节点或 CUDA-aware
准入，不将短步 SBLI 回归作为真实物理验证。正式性能始终归属于冻结二进制
`aee22855...`；重新链接后的文件哈希及对照结果单独记录。详细表格、失败组、
输入哈希和证据路径见 `ASTR_PHASE_P3_BASELINE_REPORT.md` 的最终验收章节。
现有 SQLite 的请求句柄不能单独证明通信在核执行期间尚未完成。

每个通过的阶段必须提交：

- 冻结基线 Git reference；
- 环境与编译选项记录；
- 原始 timing TSV 和汇总报告；
- NP=1/NP=2 场与统计量误差报告；
- Nsight Systems 报告和常驻审计；
- Nsight Compute 原始报告、source CSV 和热点矩阵；
- Compute Sanitizer 日志；
- 候选保留或拒绝的明确结论；
- 更新后的性能报告和可复现命令。

性能报告必须明确区分 GPU 内部优化增益、CPU 到 GPU 的整体加速比和多 GPU 扩展效率。
