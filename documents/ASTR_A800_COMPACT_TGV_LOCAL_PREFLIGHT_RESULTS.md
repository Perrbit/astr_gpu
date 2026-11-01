# A800 紧致TGV本地预测试结果

日期：2026-11-01。源码基点为`feature/gpu_dev`的`7ab0eca`加本轮未提交修改。
范围：本地缩网格正确性、CPU物理核绑定、通信参数传递、计时开关及无场I/O入口。
未连接超算，未运行A800测试，未执行Git暂存、提交或推送。

## 1. 实际修改

| 文件 | 修改 |
| --- | --- |
| `src/mainloop.F90` | CPU算法RK时间排除独立`rkfirst`统计／输出，新增逐rank计时；不重排数值调用 |
| `tests/gpu_validation/compact_benchmark.py` | 共享CPU／GPU拓扑清单、独立通信配置、逐步rank最大计时解析 |
| `tests/gpu_validation/run_compact_execution.py` | 公共halo及UCX通道可配置，与紧致接口选择解耦；保留旧默认 |
| `tests/gpu_validation/run_compact_performance.py` | 透传两层后端和UCX；禁止将不同传输设置的预检混入同一正式组；旧CP9口径不变 |
| `tests/gpu_validation/run_compact_a800_preflight.py` | 同rank缩网格预测试、真实CPU亲和性记录、同相位场／统计对照 |
| `tests/gpu_validation/run_compact_a800_campaign.py` | 正式38种配置、五轮、独立失败继续、同格式CPU1—32基线汇总；未运行正式规模 |
| `.gitignore` | 排除本轮原始快照，避免把约3.1GiB诊断目录加入Git |

当前GPU底层PCR、MPI通信算法和滤波公式未改写；这不是新增PCR通信重叠实现。
公共halo和紧致接口独立选择pinned／device-aware，两者之一需要设备通信时才选择UCX。
本地通道为`self,sm,cuda_copy`，平台可通过参数显式增加`cuda_ipc`，不能把本地结果称为A800 IPC验收。

## 2. 构建与运行环境

- 根CMake的CPU-only及CUDA `astr`目标均通过，`--parallel 8`。
- 本地NVFORTRAN26.1、Open MPI4.1.9a1、HPC-X2.25.1、HDF5 1.14.6。
- 两张RTX4000 Ada；CPU共48物理核，96逻辑处理器，CPU预测试最多使用32个不同物理核。
- CPU构建：`build_compact_cp01_cpu/bin/astr`，SHA256 `c0bd98c51498daaa0bec6295eb8e51c3f69f1684eb48939039ccd36b1f72df19`。
- GPU构建：`build_compact_cp01_gpu/bin/astr`，SHA256 `6da41d4c91f3722841988acbb7e85e0a2dc7d2cf6dc17174a56900f1badb13cb`。
- 原有缓存选择M12网格UDF，但TGV使用自身解析网格分派；没有修改`M12_C13`。
- 本地编译架构沿用自动识别，不将本地二进制直接部署A800；平台仍须按cc80重新构建。

## 3. 预测试清单

共同设置为FP64、Re1600、Ma0.1、周期TGV、RK3、dt=1e-4、开启对应十阶滤波、
关闭crashfix、无反应、无原位产品，两个格式为643e/643e与643c/643c。
外层explicit同步，紧致私有dependency，批次65535，full滤波存储／批处理。

| 项目 | 实际覆盖 |
| --- | --- |
| 缩小强扩展 | 32³；两格式GPU十种拓扑各有同网格CPU参考 |
| 缩小弱扩展 | 每rank16³；全局16³、16×16×32、16×32×32 |
| GPU rank | NP1；NP2的211／121／112；NP4的411／141／114／221／212／122 |
| CPU额外rank | NP8=222、NP16=422、NP32=442；连同NP1／2／4构成完整物理核数覆盖 |
| 主运行数 | 58个，每个10完整步；26组同拓扑CPU/GPU配对 |
| 共享GPU | NP4的14个运行共享两卡，只作正确性证明，不作四卡性能测量 |
| 计时开／关 | 两格式各补一个CPU NP2、10步运行，末步全局物理q逐值一致 |
| CLI后端组合 | pinned/pinned、device-aware/pinned、pinned/device-aware、device-aware/device-aware均通过4步短测 |
| 正式无快照入口 | 两格式分别CPU NP32跑7步、GPU NP2跑22步；保留5／20个计时样本且没有诊断快照／HDF5 |

共68次实际短运行：58主运行＋2计时关闭对照＋4通信CLI检查＋4正式入口缩网格检查。
正式512³、A800硬件、GPU独占四卡、平台HDF5／MPI依赖以及平台IPC仍未验收。

## 4. 数值结果

同相位场检查末步全部三个RK stage和各rank物理节点，分量缩放分母为max(1,对应CPU分量最大模)。
统计量比较相同样本身份与原统计输出。两项门槛均为2e-10，未放宽。

| 格式 | CPU/GPU最大绝对场差 | 最大缩放场差 | 统计量最大绝对差 |
| --- | ---: | ---: | ---: |
| 显式 | 3.1263880373e-13 | 1.9984014434e-15 | 4.9071857688e-14 |
| 紧致 | 7.2191141953e-12 | 4.0215906794e-14 | 4.8960835386e-14 |

CPU各强扩展分解相对同格式CPU NP1，在周期唯一节点上重组后最大缩放场差为：

| CPU rank数 | 两格式／已测同核数拓扑中的最大值 |
| ---: | ---: |
| 1 | 0 |
| 2 | 1.8366319638e-14 |
| 4 | 1.9474632030e-14 |
| 8 | 1.7574667930e-14 |
| 16 | 1.8524649980e-14 |
| 32 | 2.0424614081e-14 |

所有主运行的守恒状态有限，rho及恢复内能为正；最大报告CFL为0.0163108。
每个CPU rank的亲和性文件确认只占一个物理核，同一运行rank间物理核互不重复。
两格式CPU计时开／关末步物理q逐值一致。以上不构成长时物理验证或A800性能结论。

## 5. 工具检查与已知旧断言

- 针对性单测27项通过：21项共享清单／两层后端／新汇总／既有性能工具检查，以及6项CPU计时schema／CPU历史汇总／扩展汇总检查。
- Python语法与`git diff --check`通过。构建存在既有CMake策略及C编译警告，没有本轮编译错误。
- 扩大调用旧静态合同测试时发现3项已有断言与当前源码不符：
  `test_explicit_boundary_correction_uses_a_shared_scoped_predicate`仍匹配旧边界字符串；
  `test_benchmark_policy_controls_startup_and_new_output_products`仍假设只有旧的三个调用位置；
  `test_benchmark_admission_requires_independent_periodic_boundary_codes`仍要求旧负向表达式。
- 本轮未修改上述边界、输出和准入源码来迁就旧字符串，也未把这3项写成通过。未运行全库测试或新增长时Sanitizer矩阵。

## 6. 证据与复现

原始目录：`testout/a800_compact_local_preflight_20261101/`。
`summary.json`保存58个主运行、26组配对、CPU拓扑误差及计时不变性；
`cli_*/summary.json`保存通信组合；`formal_entry_short/summary.json`保存无场I/O入口检查。
各运行的`launch.json`和`affinity.rank*.json`保存环境、MPI命令和实际绑定。

```bash
python tests/gpu_validation/run_compact_a800_preflight.py \
  --cpu build_compact_cp01_cpu/bin/astr \
  --gpu build_compact_cp01_gpu/bin/astr \
  --out testout/a800_compact_local_preflight_NEW \
  --halo-transport device-aware --compact-transport device-aware \
  --ucx-tls self,sm,cuda_copy

python tests/gpu_validation/run_compact_a800_campaign.py \
  --mode manifest --out testout/not_started_a800
```

第二条只打印正式清单，不创建算例、不启动MPI、不提交任务。
正式清单为26个GPU配置、12个CPU配置，各五轮，总190次正式启动。
正式CPU采用1／2／4／8／16／32个物理核；GPU保持一rank一卡，拒绝把共享GPU记录纳入正式扩展结果。
