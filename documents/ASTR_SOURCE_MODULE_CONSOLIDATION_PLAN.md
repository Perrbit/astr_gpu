# ASTR Source Module Consolidation Plan

状态：已完成
日期：2026-09-19

## 1. 目标

在保持 `src/` 与 `src_gpu/` 严格隔离的前提下，减少 chemistry 源文件的平铺碎片，
整理 conservative boundary 的编译单元边界。此次重构只改变文件组织，不改变
Fortran module 名称、公开过程签名、数值公式、边界语义或运行时默认值。

## 2. 文件布局

CPU chemistry 从 16 个文件整合为 8 个文件：

| 新文件 | 保留的 module |
|---|---|
| `src/chemistry_air5_data.F90` | `chemistry_air5_data`，由版本化 JSON 自动生成 |
| `src/chemistry_core.F90` | `chemistry_model`, `chemistry_state_layout` |
| `src/chemistry_properties.F90` | `chemistry_thermo`, `chemistry_flow_state`, `chemistry_transport`, `chemistry_relaxation` |
| `src/chemistry_kinetics.F90` | `chemistry_source`, `chemistry_linear6`, `chemistry_ros2` |
| `src/chemistry_boundary_state.F90` | `chemistry_hbl_profile`, `chemistry_hbl_boundary_state` |
| `src/chemistry_boundary.F90` | `chemistry_postshock_boundary`, `chemistry_hbl_boundary` |
| `src/chemistry_runtime.F90` | `chemistry_flow_runtime` |
| `src/chemistry_solver.F90` | `chemistry_flow_solver` |

GPU chemistry 从 12 个文件整合为 8 个文件。transport/solver 保留独立编译单元，
因为它们单独使用 `maxregcount=128`；relaxation/coupling 不得继承该限制。
`flow_state_gpu` 也保持独立，否则 NVHPC 26.1 将其设备过程编译为 132 registers，
无法被 128-register 的 solver kernel 调用：

| 新文件 | 保留的 module |
|---|---|
| `src_gpu/chemistry_core_gpu.cuf` | `chemistry_model_gpu`, `chemistry_thermo_gpu` |
| `src_gpu/chemistry_flow_state_gpu.cuf` | `chemistry_flow_state_gpu` |
| `src_gpu/chemistry_transport_gpu.cuf` | `chemistry_transport_gpu` |
| `src_gpu/chemistry_relaxation_gpu.cuf` | `chemistry_relaxation_gpu` |
| `src_gpu/chemistry_kinetics_gpu.cuf` | `chemistry_linear6_gpu`, `chemistry_source_gpu`, `chemistry_ros2_gpu` |
| `src_gpu/chemistry_boundary_gpu.cuf` | `chemistry_postshock_boundary_gpu`, `chemistry_hbl_boundary_gpu` |
| `src_gpu/chemistry_solver_gpu.cuf` | `chemistry_flow_solver_gpu`，使用 `maxregcount=128` |
| `src_gpu/chemistry_coupling_gpu.cuf` | `chemistry_coupling_gpu` |

Conservative boundary 保留三个独立职责：配置解析、纯面算子和阶段调度。它们被
最小 CPU/GPU probe 独立编译，合并成一个对象会引入主数组、MPI 和运行时依赖，
因此不进行单体化。`conservative_boundary_faces_body.inc` 保留为 CPU/GPU 面算子的
唯一共享数值实现，避免两套公式漂移；CPU/GPU wrapper 分别只负责过程属性与 kernel
启动。该组文件已经使用一致的 `conservative_boundary_*` 命名，不做无收益改名。

## 3. 依赖顺序

CPU chemistry 编译顺序固定为 air5 data、core、properties、kinetics、boundary state、
boundary、runtime、solver。
GPU chemistry 编译顺序固定为 core、flow state、transport、relaxation、kinetics、
boundary、solver、coupling。GPU 文件继续使用 CPU core/runtime module，不复制
机理常量和运行时配置。

## 4. 验收

1. 更新 `src/CMakeLists.txt` 和所有直接编译源文件的测试路径。
2. 静态检查旧 chemistry 文件名不再出现在构建与测试合同中。
3. 运行 conservative boundary 的 CPU/GPU probe 和 Python 合同测试。
4. 运行 chemistry CPU/GPU 的 thermo、flow、source、ROS-2、HBL 和 CMake 合同测试。
5. 通过顶层 CMake 完成 CPU、CUDA、CPU+AIR5 和 CUDA+AIR5 构建。
6. 重构前后 module 名称集合必须一致；不以构建成功替代数值测试。

## 5. 非目标

- 不修改化学机理、ROS-2 参数、输运模型或 6x6 线性求解。
- 不合并 CPU 与 GPU 实现。
- 不修改 conservative boundary 的物理条件。
- 不移动其他主求解、通信、统计或 I/O 文件。
- 不进行 Git commit 或 push，除非用户另行要求。

## 6. 完成记录

- chemistry 源文件由 28 个减少为 16 个，28 个 module 的名称集合保持不变；
- CPU+AIR5 与 CUDA+AIR5 均通过顶层 CMake 干净构建；
- 三个 GPU chemistry probe 构建通过，补齐其缺失的 `MPI::MPI_Fortran` 依赖；
- CPU、结构、边界和维护合同为 `171 passed, 3 skipped, 35 subtests passed`；
- 使用重建 probe 的 GPU 数值合同为 `9 passed`；
- 生成源码清单更新为 93 个条目并通过自检。
