# ASTR 核心仓库结构与职责

本页给出维护边界和源码职责分组。逐文件声明、`use`、`call`、include 与 CMake 成员关系以[生成源码清单](generated/source-inventory.md)为准。

## 核心目录结构

```mermaid
flowchart TB
    accTitle: ASTR core repository structure
    accDescr: The tracked core solver is divided into CPU orchestration and numerics, a CUDA Fortran backend, validation evidence, and maintenance documentation. Peripheral applications are outside detailed scope.

    root["ASTR repository"]
    root_cmake["CMakeLists.txt：依赖与构建选项"]
    src["src：CPU 主程序与共享编排"]
    src_cmake["src/CMakeLists.txt：astr target 成员"]
    src_gpu["src_gpu：CUDA Fortran 后端"]
    validation["tests/gpu_validation：验证证据"]
    maintenance["documents/maintenance：维护知识"]
    audit["scripts/maintenance：只读源码审计"]
    peripheral["外围：pastr、miniapps、chemMech、非关键 examples"]

    root --> root_cmake
    root --> src
    src --> src_cmake
    src_cmake --> src_gpu
    root --> validation
    root --> maintenance
    root --> audit
    root -.-> peripheral
    audit --> maintenance
    validation --> maintenance

    classDef build fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef source fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef evidence fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef excluded fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#374151

    class root_cmake,src_cmake build
    class root,src,src_gpu source
    class validation,maintenance,audit evidence
    class peripheral excluded
```

根 `CMakeLists.txt` 负责 compiler、MPI、HDF5、chemistry 和 `ASTR_WITH_CUDA` 构建能力。`src/CMakeLists.txt` 将 CPU 文件纳入 `ASTR_SOURCES`，并在 `ASTR_WITH_CUDA=ON` 时将 CUDA Fortran 文件纳入 `ASTR_GPU_SOURCES`。程序是否进入 GPU 路径仍由输入中的 `use_gpu` 在运行时决定。

## CPU 源码职责分组

下表覆盖生成清单中的 48 个 `src/` production source。分组表达维护职责，不替代模块依赖审计。

| 职责组 | 文件 | 维护边界 |
|---|---|---|
| 入口与时间编排 | `astr.F90`, `mainloop.F90`, `test.F90` | 命令分派、初始化顺序、时间循环和内置测试入口 |
| 共享状态与通用支撑 | `cmdefne.F90`, `commarray.F90`, `commcal.F90`, `commfunc.F90`, `commtype.F90`, `commvar.F90`, `constdef.F90`, `singleton.F90`, `strings.F90`, `utility.F90` | 输入状态、主数组、常数、类型和通用函数 |
| MPI 与并行 I/O | `parallel.F90`, `mpiio.F90` | 拓扑、collective、halo 交换和 MPI 文件访问 |
| 网格与几何 | `geom.F90`, `gridgeneration.F90`, `ibmethod.F90`, `rectilinear_metric_halo.F90` | 网格生成、度量、几何 halo 与现有浸入边界 CPU 路径 |
| 初始化与模型选择 | `initialisation.F90`, `models.F90` | 算例初场、入口 profile 和模型入口 |
| 数值计算 | `comsolver.F90`, `derivative.F90`, `fdnn.F90`, `filter.F90`, `fludyna.F90`, `flux.F90`, `interp.F90`, `riemann.F90`, `solver.F90`, `thermchem.F90`, `perfect_gas_transport.F90` | 导数、滤波、通量、RHS、热化学和输运 |
| 物理边界与源区 | `bc.F90`, `conservative_boundary_config.F90`, `conservative_boundary_faces.F90`, `conservative_boundary_faces_body.inc`, `conservative_boundary_runtime.F90`, `perfect_gas_boundary.F90`, `perfect_gas_boundary_body.inc`, `sponge_layer.F90` | legacy 与 conservative boundary、perfect-gas 边界、sponge |
| 统计、输入输出与后处理 | `hdf5io.F90`, `pp.F90`, `readwrite.F90`, `statistic.F90`, `stlaio.F90`, `tecio.F90`, `validation_io.F90`, `vtkio.F90` | 输入、HDF5/checkpoint、统计、格式输出和验证快照 |

主入口证据为 `src/astr.F90::astr`。运行输入由 `src/readwrite.F90::readinput` 读取，时间推进由 `src/mainloop.F90::steploop` 进入。

## CUDA Fortran 源码职责分组

下表覆盖生成清单中的 19 个 `src_gpu/` production source。

| 职责组 | 文件 | 维护边界 |
|---|---|---|
| facade、设备与能力 | `gpu_runtime.cuf`, `device_runtime_gpu.cuf`, `gpu_check.cuf`, `case_capability_gpu.cuf` | CPU 可见 facade、设备绑定、同步/错误检查和能力准入 |
| device state | `commarray_gpu.cuf`, `commvar_gpu.cuf` | 常驻数组、host/device 显式复制和 device 常量 |
| GPU 数值与编排 | `gradcal_gpu.cuf`, `mainloop_gpu.cuf`, `solver_gpu.cuf`, `shock_sensor_gpu.cuf`, `statistic_gpu.cuf`, `sponge_gpu.cuf` | RK 编排、梯度、通量/RHS、sensor、统计和 sponge kernel |
| GPU 边界 | `boundary_gpu.cuf`, `conservative_boundary_faces_gpu.cuf`, `conservative_boundary_stage_gpu.cuf`, `perfect_gas_boundary_gpu.cuf` | 物理边界、conservative stage 和 perfect-gas device helper |
| GPU halo | `halo_exchange_gpu.cuf`, `halo_transport_gpu.cuf`, `qswap_gpu.cuf` | pack/unpack、host-staged MPI transport 和本地周期交换 |

CPU 对 GPU 的稳定入口是 `src_gpu/gpu_runtime.cuf::gpu_runtime`，而不是直接调用各数值 kernel。当前目录名 `src_gpu/` 表示已实现的 CUDA Fortran 后端，不代表 HIP/DCU 后端已经存在。

## 维护规则

1. 新 production source 必须进入对应 CMake source list，随后重新生成 inventory。
2. `src/` 对 backend-specific module 的依赖应集中在 facade 调用和 `_CUDA` 条件边界。
3. generated inventory 出现未归属文件时停止文档推进，先判断是 CMake 漏项、include owner 缺失还是范围定义错误。
4. CPU 与 GPU 同名物理过程不表示实现机械一致，必须在运行顺序、halo 深度和字段所有权层面核对契约。

