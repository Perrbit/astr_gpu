# 固定索引截面数值存档扩展执行进度

日期：2026-10-29。分支：`feature/gpu_dev`。SL0–SL8：本地实现与有界验收完成。
用户已授权按[专项计划](ASTR_INDEX_SLICE_ARCHIVE_EXTENSION_PLAN.md)完整推进SL0–SL8及必要公共前置。
不使用子代理或worktree，不执行Git写操作，不操作平台作业。

## 当前阶段

| 阶段 | 状态 | 已有证据与剩余工作 |
|---|---|---|
| SL0 | 本地完成 | 结构化JSONL、27项登记、精确int64、五变量/AIR5及真实平均来源；最终公共宿主/工具74项通过 |
| SL1 | 本地完成 | 实际完整步热更新、调频、增删及显式重启用；步数、物理时间、已有自适应窗口及恢复通过 |
| SL2 | 本地完成 | 三轴整面/子区、独立HDF5/XDMF、唯一所有权/空rank；CPU/GPU FP32/FP64来源逐值及ParaView读回通过 |
| SL3 | 本地完成 | 曲线几何仿射场NP=1/2/4、七节点边界、解析收敛阶>5.8、双rank新核memcheck通过；无整场诊断副本 |
| SL4 | 本地完成 | 正式时间均值及已有AIR5采样均值、来源/窗口、等待与拒绝、输出只读性及均值精确续算通过；缺失均值不伪造 |
| SL5 | 本地完成 | ASTRIR07、12对5+7续算、自包含新段、目录/HDF错误隔离、CUDA分配失败后继续计算通过 |
| SL6 | 本地完成 | Python/ParaView原生读回、局部JPEG、按行剖面、移动与显式跨段索引，以及CPU/GPU OutputTools安装通过 |
| SL7 | 本地完成 | TGV、channel壁面、静态CURVE、AIR5及原随机吹吸M12；CPU/GPU、渲染共存、输出开关不改变权威场/统计通过；不是生产物理验收 |
| SL8 | 本地完成 | 96×48×32双卡M12五种成本配置、二维传输/私有通信字节、资源采样、文件读回及完整交付报告已保存 |

## 已实施

- `src/vendor/yyjson/`保留上游0.13.0的原始C/H源码与MIT许可，根CMake静态编译，无新增系统JSON运行库。
- `insitu_run_config`提供类型化数值产品与字段登记；`insitu_catalog`逐行解析，文件级错误拒绝整次更新，
  条目错误保留对应诊断但不吞后续条目。字段选择排序去重，索引使用全局范围。
- `insitu_product_schedule`持有新登记表，保留版本与已消费启用请求，按上次实际输出锚定频率。
  删除保留ID/版本历史；可识别错误停用原产品；资源停用不因调频或重复请求重试。
  新定义可更新其有效版本，但资源停用仍需新请求或已批准的配置纠正规则才能恢复。
- `insitu_catalog`按完整步轮询，仅rank 0 stat/读取正式文件。文件签名来自实际打开的描述符，
  包括inode与时间/模式元数据，不计算哈希；不变文件不重解析，不到检测时刻不访问文件。
  广播字段为显式字符串、整数和浮点元数据，不广播带编译器填充的结构体或设备指针。

动态数值截面现已接完整步入口，不需要Catalyst。旧`slices`不能再定义启用、位置或派生量，
仅空/disabled占位组可被读取；生产定义只来自JSONL。新检查点使用ASTRIR07外层格式，
保存动态控制，即使清单为空；旧检查点不再兼容。各阶段完成依据为下述独立测试，而非仅接入入口。

## 分阶段验证记录

以下1–19项保留实施过程中的阶段状态；最终矩阵和成本以末尾收口记录及验收报告为准。

1. 根CMake的`build_gpu_probe/bin/insitu_catalog_probe`构建通过，Catalyst关闭；该探针不调用GPU。
2. 新测试先观察到有效记录被报告为“runtime catalog parser is not implemented”的预期失败，再实现解析。
3. 首批42项宿主解析检查通过，包括CRLF、长行、坏条目隔离、重复ID、未知字段、非有限时间、
   全局索引边界、AIR5布局存在性，以及超过2^53但仍可表示的int64请求号。
4. CPU-only/Catalyst OFF构建的解析探针也执行过上述42项，全部通过。
5. 状态机测试先观察到“first accepted definition must have version 1”的预期失败，再实现状态转移。
6. 状态机版本完成后，42项解析＋8项状态转移，共50项通过。
7. 文件观察器先复现“initial file was not applied”的预期失败，再实现。最新矩阵增加NP=1/2/4
   的正常原子发布和坏格式头，共56项通过，零跳过；非root的文件读取/stat计数均为零。
   证据：`tests/gpu_validation/out/index_slice_catalog_20261029/host_r1.xml`。
8. 受影响旧配置/调度回归125项通过、39项因未提供对应探针/MPI环境跳过，不将跳过计为通过。
9. 小型控制往返测试先失败，再确认二进制长度/记录边界正确。新探针的自引用
   `definitions=[definitions,definitions(1)]`在当前NVHPC构建中损坏首项的可分配字段数组；
   改为独立分配与`move_alloc`后往返通过。生产恢复检查未放宽，并断言新增产品不能改变既有字段。
   尚未接入原生`insitu_control.bin`，这不是实际CFD检查点验收。
10. 公共宿主矩阵现为65项通过，发布/显式重启用工具8项通过，见`host_r2.xml`、`tools_r1.xml`。
11. 选字段IO测试CPU/GPU各8项通过，见`selected_cpu_r3.xml`、`selected_gpu_r1.xml`。
    选`velocity_x/pressure`的21节点子区精确读回，GPU实际下载336字节；含NP=1、2的x/y/z分解与NP=4共享双卡。
    独立坐标仅写一次，帧不重复坐标或velocity向量；旧完整场默认写法通过同组回归。
12. 新局部诊断链的首个MPI版本在调用方使用了邻域列表旧长度，导致远端条带下标越界。
    现改为一次分配元数据槽、显式传递有效长度。CPU/GPU各5项曲线几何仿射场检查通过，
    见`diagnostic_cpu_r3.xml`、`diagnostic_gpu_r1.xml`。不修改求解器q或halo。
13. 双rank新核memcheck见`memcheck_r4.xml`，零错误。前几轮HPC-X/UCX在MPI初始化和
    主机缓冲识别时触发CUDA上下文/API探测错误；本私有通道使用主机条带通信，按仓库既有
    ob1＋host-only MPI配置关闭这些外部CUDA探测后执行，无屏蔽本项目kernel错误的抑制项。
14. 真实TGV CPU NP=1、GPU NP=1/2的12对5+7续算3项通过，见`runtime_r2.xml`。
    恢复新段仅含8/12步，与连续参考逐值一致，且不采用磁盘冲突字段清单。
    已有均值/输出只读性、缺字段局部拒绝、单产品目录错误隔离及CPU/GPU同相位对照4项通过，
    见`runtime_means_r1.xml`。更新后的共享输出配置矩阵83项通过，见`output_config_r1.xml`。
15. channel壁面与静态CURVE开放边界2项通过，见`representatives_r1.xml`的已完成项。
    该轮M12误用默认空UDF，坐标为零，未完成推进；已停止该项并选择现有M12 UDF重新构建，
    不修改原M12工况或只读源目录。
16. AIR5两卡初始截面CPU/GPU逐值一致。4步后原始速度差约1.1e-12，经SI短尺度导数放大后
    Q差约3.2e-9；首轮误用观测值幅度缩放，不能代替计划要求的SI参考尺度。
    按既有AIR5热力学参考声速、密度、温度及固定Ly构造梯度/Q尺度后通过，
    见`air5_r2.xml`与测试目录`dimensional_diagnostic_comparison.json`。未放宽2e-10归一化门槛。
17. 正确的M12 UDF构建下，64×32×24、原dt=0.02、10个完整步、原743e/643e、原边界与随机吹吸，
    NP=2 x分解CPU/GPU截面及派生量对照通过，见`m12_r2.xml`。
    M12/AIR5原始日志、输入、截面及相应诊断证据复制保留在
    `tests/gpu_validation/out/index_slice_catalog_20261029/representative_evidence/`，不依赖pytest临时目录续存。
    CPU已有均值读取另改为只读均值/覆盖访问器，不为单个速度标量反复重建RMS/协方差；不改变统计累计。
18. 均值专用访问器改动后CPU/GPU只读性2项再通过，见`means_accessors_r1.xml`及同名前缀的持久测试目录。
    已补分项计时入口、各rank二维下载/私有条带字节日志，HDF中增加现有自由来流/参考值记录；这些新增元数据与计时的代表运行待下一轮联合检查。
19. 离线4项通过，见`offline_r2.xml`和`offline_cases_r2/`。真实M12坐标、字段与类型在ParaView XDMF3
    逐值一致；只读JPEG重画不默认生成EPS。首次ParaView验收因其HDF5与用户h5py ABI冲突失败，
    现按仓库已有NPZ参考方式隔离两个进程，不在pvpython内导入另一套h5py，不绕过数值对照。
    列字段只读HDF元数据，剖面只读对应行，单字段作图不顺带读取其余字段。

命令：

```bash
cmake -S . -B build_gpu_probe
cmake --build build_gpu_probe --target insitu_catalog_probe insitu_catalog_state_probe insitu_catalog_mpi_probe --parallel 8
ASTR_CATALOG_PROBE="$PWD/build_gpu_probe/bin/insitu_catalog_probe" \
ASTR_CATALOG_STATE_PROBE="$PWD/build_gpu_probe/bin/insitu_catalog_state_probe" \
ASTR_CATALOG_MPI_PROBE="$PWD/build_gpu_probe/bin/insitu_catalog_mpi_probe" \
ASTR_CATALOG_MPIEXEC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
pytest -q tests/gpu_validation/test_insitu_catalog.py \
  --junitxml=tests/gpu_validation/out/index_slice_catalog_20261029/host_r1.xml
```

上述早期证据分别支持对应的接口、数值、续算或内存范围，不单独作为SL7/SL8完成依据。
纯CPU探针构建在`build_cpu_probe`，已将该测试构建的`BUILD_TESTING`打开；未运行或改写生产算例。

## 最终收口

- `catalog_closed.xml`最终74项通过，无跳过，含修复后的自适应名称绑定与MPI根rank观察。
- `runtime_io_probes_r1.xml`、`production_ownership_r1.xml`分别验证真实文件热更新/HDF故障隔离，
  以及不对称网格、三轴/子区/空rank与NP=4多轴唯一所有权。
- `device_recovery_r2.xml`验证解析收敛及可返回的设备分配失败；`memcheck_final.xml`为最终双rank新核零错误。
  首次设备`allocate(stat=...)`仍致命退出，已改为显式配对的`cudaMalloc/cudaFree`，并检查失败后下一核仍可运行。
- `runtime_final.xml`18项、`native_widths_final.xml`4项及均值/自适应续算的独立证据覆盖TGV、壁面、CURVE、M12、AIR5，
  原生来源读回、输出无扰动及12与5+7精确续算。各矩阵有重叠，不汇总为独立测试总数。
- `adaptive_restart_final.xml`核对密集窗口输出[0,3,4,5,6,11]与恢复段[6,11]。
  NVHPC不同长度字符`FINDLOC`及逻辑替代的运行库问题，改由新桥接层小表直接比较修复，未改物理算法。
- `shared_buffer_final.xml`核对同完整步匹配需求复用最近一个二维结果，第二产品字段D2H为零，
  前一产品写盘失败不污染后续产品；不为每产品保留整场缓存。
- CPU-only FP32、CUDA FP32/FP64与Catalyst关闭构建通过；现有Catalyst开启构建的实际共存及
  `legacy_render_envelope_final.xml`两项续算通过。CPU/GPU OutputTools安装后离开源码目录运行通过。
- `offline_r2.xml`、`paraview32_final.xml`验证真实FP64/FP32截面与坐标读取，
  `cost_readback_final.xml`逐帧检查最终成本运行文件和传输记账。
- 中等M12成本原始输入、日志、资源及数据位于`m12_cost_final/`。
  关闭输出、整面基础、整面九字段、九字段子区、两产品的10步窗口分别约
  0.2924/0.2977/0.3084/0.2981/0.3122秒；单轮短窗不作为生产加速率结论。

完整证据路径、来源能力矩阵、分项计时与局限见
[本地验收报告](ASTR_INDEX_SLICE_ARCHIVE_VALIDATION_REPORT.md)，
使用接口见[截面存档说明](../scripts/output/INDEX_SLICE_ARCHIVES.md)。
后续完整X2图片家族热更新、移动网格、任意物理平面及平台生产规模另行推进。
M12源目录始终只读，采用其原UDF构建；未执行Git写操作或平台作业。
