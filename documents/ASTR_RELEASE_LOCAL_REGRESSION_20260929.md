# 发布候选本地数值与重启回归

## 范围

基准提交 6952dc0，加已批准的 CMake 安装路径修复。使用 R4 独立构建、
安装后的两个非反应流二进制，不使用旧 build_cpu_probe/build_gpu_probe。
FP64、explicit 同步、pinned 主机中转，AIR5 未编入。没有执行远程测试。

CPU SHA256：`b744cfa6ca057066eb1604566621eaa86399a6d2303a2fcf9dcae1980dc6e80d`。
GPU SHA256：`6d444c69000e6b3e0c052cf54199d546d5a46bfcab981ad7484f9b11f50ae806`。
路径分别为 `/tmp/astr-release-install-20260929a/test_release_install_prefixes_0/override-install/bin/astr`
和同目录编号 1 的 GPU 文件。

证据根目录：`tests/gpu_validation/out/release_6952dc0_installfix_20260929/`。
保留每项输入、日志、HDF5 和逐分量比较报告。以下数值是五个重构守恒量
的最大绝对差，不是统一物理单位下的误差，也不是统计置信区间。

## 短窗数值检查

| 目录 | 配置 | q 最大绝对差 | 状态 |
|---|---|---:|---|
| channel_np2 | 32³，2×1×1，固定驱动力，黏性与 scalar 滤波 | 3.20e-14 | 场与统计通过 |
| tgv_np1 | 128³，NP1，黏性与 scalar 滤波 | 2.84e-13 | 场与统计通过 |
| tgv_np2 | 128³，1×1×2，黏性与 scalar 滤波 | 2.84e-13 | 场与统计通过 |
| curve_np1_original_xy | 64×64×16，NP1，曲线平板，黏性与 scalar 滤波 | 1.55e-15 | 场、统计、几何及壁面检查通过 |
| curve_np2_y | 64×64×16，1×2×1，同上 | 1.55e-15 | 场、统计、网格和壁面检查通过 |
| open_shock_np2 | 64×64×16，2×1×1，bc12/52，选择性 Roe，无黏无滤波 | 4.00e-15 | 同相位场、统计和传感器通过 |

网格列沿用脚本 im/jm/km 输入约定，不统一等同于含端点存储节点数。
各短窗 maxstep=2，从零启动。沿用原脚本容差：槽道场 atol=1e-8、rtol=1e-10，
槽道统计 atol=1e-9、rtol=1e-10；其余场与统计 atol=rtol=1e-10。
壁面不变量 atol=1e-12。曲线 CPU 几何度量独立检查限 NP1，不能记成 NP2
也执行了同一几何 dump 检查。

开放激波传感器最大绝对差 6.44e-16，掩码差异为零，两端均标记 14603 个
节点。实际触发了激波分支，不是空掩码通过。传感器原门槛 atol=rtol=1e-12。

使用的既有脚本：run_channel_phased_compare.sh、run_tgv_mpirank2_field_compare.sh、
run_curvilinear_hbl_c8_filter_compare.sh、run_s2_hbl_selective_roe_s2c4_compare.sh。
开放激波显式设置 SAME_PHASE_FIELD=t、COMPARE_SENSOR=t。槽道、曲线及
开放边界场比较使用 CPU 完整 RK 快照，不用常规 CPU HDF5 代替。

## 重启与滤波

`opensbli_restart_v2`：CPU NP1、GPU NP2 2×1×1，33×33×9 节点，dt=1e-6。
连续到 step4，与 step2 checkpoint 续算到 step4 比较，原始变量最大差
1.33e-15，重构 q 最大差 4.44e-16，符合 atol=1e-10、rtol=0。
两端均拒绝错误的 checkpoint 步号，统计记录检查为 1,2,3,4。
这组不启用滤波，不能单独覆盖 scalar 滤波重启。

`tgv_filter_restart_100_v2`：32³、NP2 2×1×1，dt=1e-3，maxstep=100，
黏性及滤波开启；CPU 与 GPU full/default 对照，另从 step50 checkpoint 续算。

| 比较 | q 最大绝对差 | 门槛 |
|---|---:|---|
| CPU 完整 RK / GPU default | 4.83e-13 | atol=rtol=1e-10 |
| GPU full / GPU default | 0 | atol=rtol=0 |
| GPU 连续 / GPU default 重启 | 6.73e-11 | atol=1e-10，rtol=0 |

default 不设置 ASTR_GPU_FILTER_WORKSPACE，验证当前 scalar 默认分支。
CPU/GPU 动能、拟涡能、耗散率统计比较通过。GPU 连续/重启的统计记录均为
1–100 步，严格逐步递增，原始统计输出比较符合 atol=rtol=1e-10。
该窗口是有界回归，不是 t=20 的 TGV 物理验证，也不是任意壁面或 CURVE
滤波长时间/重启组合的验收。

驱动：`tests/gpu_validation/run_release_filter_restart.py`。
重启统计另用 `compare_flowstate.py` 比较，报告为 restart_statistics.txt；
检查两侧 nstep 序列相同且相邻差为 1。

## 失败记录与处理

1. `curve_np1` 曾将原测试的 x/y 分辨率从 64 减至 32。逆度量内部误差
   4.39e-4 超过 1e-4；停止后续矩阵。恢复 64 后为 1.39e-5，原门槛通过。
   粗网格失败保留，不调整几何容差，不据此修改求解器。
2. `opensbli_restart` 在输入生成阶段失败，未启动求解器。公共 write_input
   新增参数后旧调用遗漏 turbinf/z_bctype。补传原静态 prof、周期 1，输入测试
   及既有参考生成测试共 3 项通过。重启脚本新增 CPU_EXE/GPU_EXE 覆盖入口。
3. `tgv_filter_restart_100` 首版驱动错误使用 CPU 常规 HDF5，q 最大差
   2.47e-7，门槛失败。按现有方法改用完整 RK 快照后，v2 同相位比较通过。
   保留原失败目录，不放宽容差，不修改 CPU/GPU 求解器。

## 壁面与曲线 100 步补测

使用上述同一对已安装二进制，执行前 SHA256 再核对一致。两张本地
RTX 4000 Ada 空闲时启动；FP64、explicit、pinned，未改数值源码。

| 目录 | 配置 | CPU/GPU q 最大绝对差 | 结果 |
|---|---|---:|---|
| channel_scalar_100 | 32³，NP2 2,1,1，dt=5e-4，固定驱动力 1e-4，黏性/滤波 | 8.53e-14 | 同相位场及统计通过 |
| curve_scalar_100 | 64×64×16，NP2 1,2,1，dt=1e-5，warp=0.4/0.2，黏性/滤波 | 8.55e-15 | 同相位场、统计、网格及下壁不变量通过 |

两项 maxstep=100，沿用短窗脚本原容差；从 step0 启动的更新次数仍遵循
原主循环语义，不解释为统计收敛。NP2 曲线检查不包含 NP1 独立度量 dump。
channel_full_100 与 channel_scalar_100 的 GPU 场在零容差下完全一致。

### 槽道重启失败，停止后续矩阵

驱动 run_release_boundary_filter_restart.py 复用已通过的输入，在新目录
执行 full 对照及 scalar 第 50 步恢复到第 100 步。输出目录
`channel_filter_restart_100` 的 full 场零差且统计通过，但 restart 场失败：
q 最大差 4.683903870983386e-6，压力最大差 1.873826157350322e-6，
原绝对门槛 1e-10，未放宽。result.json 记为 failed；没有继续重启统计
验收或 CURVE full/restart 矩阵。

定向对照 `channel_checkpoint50_control` 只改变连续运行的 checkpoint
间隔，由 100 改成 50。最终场与原连续 scalar 场完全一致。该对照备份的
step50 与分段运行 checkpoint_step50，以及重启目录 bakup 中的同一步
ro/u1/u2/u3/p/t 逐字段差均为零，步号为 50、time 为 0.025。
因此未发现 checkpoint 间隔改变连续结果或分段恢复点选择错误的证据。

待验证假设：src/mainloop.F90 的 GPU checkpoint 分支在主机副本上调用
boucon/qswap 以匹配 CPU 输出相位，但不改变设备常驻状态。恢复后的状态
重建或边界初始化可能与连续推进不一致。当前未证明这就是根因，不能据此
直接更改 CPU/GPU 输出相位。需继续定位恢复后的首步差异；涉及输出语义
或既有 CPU 数值行为的修复应先人工决策。

全部已启动计算结束。此失败阻止关闭壁面滤波重启门槛，不影响前面已通过
的独立场比较记录，也不能用 full/scalar 一致性掩盖重启失败。

### 逐阶段诊断：写出状态与推进状态不同

进一步使用同一 GPU 二进制的 ASTR_VALIDATION_RHS_STEP=49 和
ASTR_VALIDATION_RHS_STEP_SECONDARY=50，捕获连续运行和恢复运行的 pre_rhs、
post_update 快照。驱动为 diagnose_release_channel_restart.py，输出为
`channel_restart_diagnosis`。未修改求解器源码。开启诊断后的连续 step50
checkpoint 与原 step50 的六个原始变量字段完全一致。

第 49 步 RK3 末的原始设备 q 与 step50 文件重建 q 的差异分布：

- 两个 rank 各自远离壁面和 MPI/周期接口的内点最大差 1.07e-14。
- 壁面 q 最大差 1.6942638316770342e-3。
- 非壁面的 x 共享节点差最高约 4.75e-6。
- 文件壁面值与 CPU noslip 的压力外推、固定壁温和零壁速处理所得状态
  吻合至 7.11e-15；x=16 文件值与两 rank 原始共享节点 q 的算术平均
  吻合至 7.11e-15。

原因链条位于 src/mainloop.F90:418：D2H 后在主机副本执行 boucon/qswap，
但不回写设备；文件与 GPU 下一步实际推进的 q 因此不同。
src_gpu/mainloop_gpu.cuf:54 在下一步先滤波，再施加物理边界。重启通过
readcheckpoint/updateq 恢复的是已处理的文件状态，不能撤销写出前的壁面
投影和共享节点平均。两条路径的滤波输入不同，不是 HDF5 小数保存精度问题。

因果对照仅在新的诊断目录 `restart_raw_wall_diagnostic` 修改恢复数据副本：
将两面壁面的原始变量替换为连续 step49 末 q 重建值，其余字段不动。
这不是受支持的恢复方式或正式修复，尤其没有恢复各 rank 独立共享节点值。
step50 RK1 pre_rhs 在距 x/z 接口至少 6 个节点的区域中，原恢复差为
2.18e-5 / 2.13e-5；替换壁面后降至 1.78e-14 / 2.13e-14。
全场仍有约 9.70e-7 差异，不能据此宣称重启门槛通过；共享节点状态仍未恢复。

结论：壁面写出投影已通过局部反事实对照确认是主要误差来源，另外存在共享
节点平均造成的状态差异。尚未测试 CPU 连续/重启是否有同类误差，不将此
结果直接判定为 CPU 数值 bug。修改方案须区分用于展示/CPU 同相位比较的场
与精确恢复状态，并处理重复共享节点；仅删除 boucon 或提高 HDF5 精度不足以
证明解决全部问题。另一方案是在每一步都统一规范状态，但它会改变连续推进
的离散结果，不能作为未审批的部署修复。当前保留代码与容差，等待方案决定。

## 已批准修复与复测

用户批准保留原 HDF5 展示场，同时独立保存实际 GPU 推进状态，不改变连续
算法顺序。新增 src_gpu/checkpoint_gpu.cuf，在主机 boucon/qswap 投影前
保存每个 rank 的完整 q，包括 halo 和重复共享节点。重启先读取原 HDF5，
再恢复 q，首次设备准备从 q 重建物性。AIR5 补偿恢复路径不变。

文件格式版本、网格和拓扑尺寸、步号、生成代次、部分物性/数值参数、负载长度
及有限性检查均在读取时执行。生成代次采用日历时间：最小独立检查确认 NVHPC
首次 system_clock 返回 0，不适合作为每进程首次 checkpoint 的代次标识。
文件重命名错误码先归一化再 MPI 汇总，防止负错误码被其他 rank 的零掩盖。
以上为恢复协议修复，不是 CPU 连续推进算法或滤波闭合修改。

复测使用根 CMake 新建 CPU/CUDA 非反应流构建，BUILD_TESTING=OFF，两者编译通过。
GPU：build_release_restart/bin/astr，SHA256
`9e7d11f458203c9f33d00abf3522fe9479e6f723fbfdfa6227a26da978e0c166`。
CPU：build_release_restart_cpu/bin/astr，SHA256
`dc763a5bd13afcb124c084426d46388eabc8de559ad44e57872060a38b663dcd`。
它们包含未提交的恢复修复，不与开头的安装二进制混同。

| 证据目录 | 检查 | 结果 |
|---|---|---|
| channel_exact_restart_final | 32³，NP2 2×1×1，50→100，scalar，固定驱动力 | 六个原始字段、重构 q、步号对齐统计差均为零 |
| curve_exact_restart_final | 64×64×16，NP2 1×2×1，50→100，scalar | 六个原始字段、重构 q、统计差均为零 |
| tgv_exact_restart_final | 32³，NP2 2×1×1，50→100，默认 scalar | 重启场差为零，full/default 差为零，CPU/GPU 最大场差 4.831690603168681e-13 |
| opensbli_exact_restart_final | CPU NP1、GPU NP2，连续4步及2→4步 | 两条路径重启、统计步号续接、错误 auxiliary 步号拒绝通过 |
| exact_restart_rejection_final | 10 项文件生命周期检查 | 缺失、.tmp、截断、NaN、错误代次/拓扑、超长负载、未知版本、备份失败均拒绝；无标识旧格式警告后可读取 |
| exact_restart_sequence_final | 槽道 NP2，序列输出，连续4步及2→4步 | 序列辅助文件选定、按步号/rank 状态恢复及最终场零差通过 |

槽道和 CURVE 的新版 full 连续场还与修复前 scalar 连续场逐字段零差，表明
此次受测路径没有改变连续计算结果。原场绝对门槛 1e-10 未放宽。
槽道 flowstate 在重启时重建为 50..100 步，CURVE 则续接 1..100 步。检查
分别严格核对这两个步号序列，再与参考的对应行比较，没有修改统计写出逻辑。
中间测试曾因比较全长日志与重启窗口而报告行数错误；对应失败目录保留，
最终结论只引用表中 final 目录。

## 精简源码安装版复核

在独立的 242 文件源码副本中重建非反应流 CPU/CUDA，移除 tests/ 和 chem/，
保留明确列出的新增 checkpoint_gpu.cuf。配置检查 7 passed，独立构建及双前缀
安装检查 2 passed，安装不改变源码。日志、源码指纹及 installed_receipt.json
位于 /tmp/astr-release-final-install-20260929/。目标程序位于其中
test_release_install_prefixes_0/override-install/bin/astr（CPU）和
test_release_install_prefixes_1/override-install/bin/astr（GPU）。

- CPU SHA256：73f1dcc131b0ff9cbad512e306af2e41a2bfae678f9f53df68e9b778e4215e4c。
- GPU SHA256：d0462e62e7a99a5e972bdac360a676307d05f9f4b7cad2ea8197c2fb1a3b51ec。
- installed_exact_tgv：NP2 32³，100 步，CPU/GPU 最大场差 4.831690603168681e-13，
  full/scalar 及 50→100 重启场差为零，统计通过。
- installed_exact_channel：NP2 32³，100 步，与修复前参考的 full 连续场零差，
  scalar 50→100 重启场及对应统计零差。

安装版已检查动态依赖并实际运行，不将构建目录程序的测试直接冒充安装版测试。
本轮没有扩展已批准容差、物理模型或目标平台范围。

## 当前未关闭事项

- 本地短回归只覆盖表中组合，不等于全部边界方向和参数组合。
- R7 已覆盖上述 HDF5 checkpoint 及文本统计续接，未覆盖全部输出类型。
- R8 的上述周期、槽道和 CURVE 滤波重启问题已修复并通过本地指定矩阵；
  不推广为全部边界、不同拓扑迁移或长时物理验收。
- CUDA-aware MPI、混合精度、AIR5、目标计算节点依赖不由这组 pinned/FP64
  检查放行。未新增 Compute Sanitizer 或性能测试。
- 安装和重启修复及本轮测试脚本修改尚未提交，最终发布版本仍需重新固定。
