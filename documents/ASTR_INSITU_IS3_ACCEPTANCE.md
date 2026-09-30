# IS3 TGV 原位后处理验收

日期：2026-09-30。范围：`feature/gpu_dev` 当前工作区，本地三维
`32^3` 周期 TGV，NP=1 与 NP=2、拓扑 `2,1,1`，FP64、643e/RK3、
黏性与显式滤波开启，显式 GPU 同步。未提交 Git。

结论：IS3 TGV 批处理闭环及本目标所需 IS0-IS2 前置门槛通过。
这是有界功能与数值验收，不是 IS4 全拓扑、壁面/AIR5、CURVE 或生产规模认证。
CPU 验证统计与精确续接；正式 EGL 渲染使用 GPU 求解器绑定设备。

后续失败边界：256^3 每步渲染仅完成 48/100 帧，因节点主机增量
16.052 GiB 超过批准的 16 GiB 而中止，设备内存也随帧数增长。
本报告的 32^3 短窗事实保留，但不能外推长序列资源稳定性。
优先整改见主计划第 5.1 节，修复前不放行更大规模长序列展示。

## 逐项证据

以下运行目录均位于 `tests/gpu_validation/out/`。运行报告需与所列验证器
一同阅读，单个报告的历史状态字符串不代表整个阶段结论。

| 原目标要求 | 实现与验证器 | 权威证据与结果 |
|---|---|---|
| 默认关闭、CPU/GPU 构建解耦 | 根 CMake、`insitu_run_config.F90`、`test_cmake_production_decoupling.py` | CPU-only 与 CUDA 正式构建完成；7 项无 tests 源码副本配置检查通过。`insitu_cpu_extension_statistics_20260930` 的未配置/关闭路径无原位统计输出，权威场不变 |
| 依赖与设备身份、失败不静默替换后端 | `insitu_runtime.F90`、`insitu_device_map.cpp`、`insitu_mesh_adapter.cpp` | 当前正式二进制 `test_insitu_admission.py` 14 项通过，含 NP=1/2 生命周期、缺依赖、配置错配与关闭构建拒绝；UUID 映射及字段定义单元测试 7 项通过；真实帧检查实际 EGL UUID |
| 完整 RK 相位、只读字段与物理导数 | `insitu_fields.F90`、`run_insitu_sample_validation.py` | `insitu_device_spatial_checked_20260930` 同相位 CPU/GPU 字段最大差 `1.990e-13`，派生量 `1.358e-14`，均低于 `2e-10`；含制造梯度与全域模板对照。当前正式程序通过 `insitu_cpu_extension_statistics_20260930` 12 组开关/参考场逐位比较 |
| 独立时间加权统计、窗口裁剪、Reynolds/Favre 均值/协方差/应力与两类 RMS | `insitu_time_integral.F90`、`insitu_velocity_statistics.F90`、`insitu_spatial_statistics.F90`、设备累计模块 | 当前积分、速度统计、空间统计探针通过：不外推、裁剪、状态续接、局部 RMS=1 而区域信号 RMS=0；真实 TGV 独立离散参考与 CPU/GPU 统计对照通过。RMS 仍按已批准的平方值门槛比较，并报告原值差 |
| steps/物理时间调度，真实时刻、端点去重 | `insitu_schedule.F90` 及正式调度 | 当前 schedule 探针通过跨多个触发点、非法时钟、状态续接和端点去重；`insitu_device_owned_time_20260930` 正式时间调度输出 0/2/4，与步数调度一致，无重复末帧 |
| CPU 同拓扑精确配对续接 | `insitu_session.F90`、`run_insitu_cpu_paired_restart.py` | `insitu_cpu_pair_final_20260930` 和 `insitu_cpu_pair_cuda_build_20260930`：纯 CPU 构建及 CUDA 构建 CPU 模式的 NP=1/2 q/统计状态逐字节一致，HDF 逐位一致；缺失、损坏、拓扑、窗口、时间步错配均拒绝 |
| GPU 同拓扑精确配对续接及旧格式兼容 | `run_insitu_paired_restart.py` | 当前程序 `insitu_is3_final_render_20260930`：NP=1/2 精确 q、统计、HDF、图像与几何续接通过，渲染开关/调度错配拒绝；`insitu_device_owned_legacy_pair_20260930` 的 ASTRPS01 兼容性通过，原批次校验和不变 |
| 不完整批次拒绝、不覆盖旧批次 | `insitu_checkpoint_batch.F90` | 当前 NP=2 batch 探针 `insitu_is3_final_batch_hpcx_20260930` 通过 CRC64、半组、未就绪、配置与损坏拒绝，复制/发布不覆盖；真实重启驱动亦检查源批次未修改 |
| 预设切片、Q 等值面、瞬时与平均流线 | `scripts/insitu/tgv_pipeline.py`、`tgv_streamlines.py` | 当前 `insitu_is3_final_render_20260930` 输出预设 JPEG/EPS，固定相机与色标，Reynolds/Favre 平均流线在有效覆盖后产生；与已验收基准逐像素/几何一致，人工查看平均流线非空 |
| 跨 MPI 流线、字段/时间元数据及离线读取 | `check_insitu_extracts.py`、恒定速度诊断副本 | 当前最终渲染目录 `offline.json` 独立回读 44 份几何，检查必需字段、有效统计时长、步/时间、切片位置和 Q 值；16 个种子跨 x=pi 连续性及端点误差最大 `1.7764e-15`，低于 `2e-10` |
| 关闭常规整场写出仍有多帧图像与提取数据 | `run_insitu_native_render.py` | `insitu_device_owned_render_20260930` 与 `insitu_device_owned_time_20260930`：至少 0/2/4 三帧，无常规 HDF/restart 或测试场快照，JPEG/EPS 和 VTK 几何与对照一致 |
| 每物理 GPU/节点资源准入、超额停止 | `insitu_resource_budget.F90`、`insitu_resource_observer.cpp` | 当前预算探针通过溢出/原子拒绝。`insitu_native_observer_limits_20260930` 实际拒绝主机、单卡、共享卡合计增量超额和空闲余量不足；当前最终渲染逐阶段 CSV 在已批准预算内。20 ms 匹配开/关观测见 `insitu_device_owned_time_20260930` |
| 内存检查 | Compute Sanitizer | `insitu_restore_memcheck_20260930` NP=2 统计保存/释放/恢复及累计路径，两份进程日志均零错误；不以此替代数值或资源验证 |

## 明确边界

- 设备统计常驻并不等于零拷贝。仍下载非法值标记、区域归约、按帧流场/均值、最终统计和检查点累计状态。
- 资源门槛为本地每物理 GPU 附加 2 GiB、每节点附加 4 GiB、设备空闲至少 1 GiB。阶段观测加独立采样不能保证捕获所有瞬时分配峰值，不作为生产默认值。
- EPS 为已批准的栅格嵌入格式，不宣称矢量流场图。流线到外表面终止，不做周期回绕。
- 流线依赖本地 ParaView 6.1.1 的可复现 FP64 坐标补丁，见 `scripts/insitu/patches/paraview-6.1.1-streamline-fp64.patch`；无补丁版本不能继承精度验收。
- CPU 精确附属文件只服务显式原位配对批次，保留默认重启与旧 HDF 格式；不支持 CPU/GPU 批次互换、跨拓扑恢复或一般算例迁移。
- 一次空间统计探针直接启动因 HPC-X 安装前缀初始化失败；使用配套 `mpiexec --mca coll_hcoll_enable 0 -np 2` 后通过。不是数值失败，亦未更改算法绕过。

## 下一阶段

IS4 才扩展 NP=2 三方向 slab 和 NP=4 双轴、空提取分区及更广故障矩阵。
本次不自动开展 IS4，不触碰远程任务，也不因本结论执行 Git 提交。
因后续 256^3 展示失败，当前先处理生命周期、分段计时及写出契约，
完整 100 步展示通过后再恢复上述 IS4 顺序。
