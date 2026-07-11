# ASTR GPU Porting Status Since `47737cf`

本文档整理当前 `feature/gpu_dev` 最新版本相对于基线提交
`47737cf9b6a25b00bc5f1139cdaec36d55e5024e` 的主要功能变化、当前 GPU
已验证支持的算例、已验证支持的 `bctype`，以及 `128^3` channel flow 的整体加速比。

基线提交：

```text
47737cf fix compilation errors  问题 7: `present(timerept) .and. timerept` -- NVHPC 求值顺序导致段错误
```

当前最新提交：

```text
451fe0e Extend GPU boundary validation to channel wall multi-rank
```

## 相对基线的功能变化

从 `47737cf` 到当前版本，GPU 移植从早期编译修复推进到可运行、可验证的
CUDA Fortran 多算例、多 MPI rank、显式格式 GPU 路径。

主要新增提交包括：

```text
df1961b Implement first 2-rank x-slab GPU halo exchange for TGV CUDA Fortran path
5597e69 Extend CUDA Fortran TGV GPU path to multi-direction MPI halo validation
7a5fdce add GPU porting plans and validation records
fd250ab Port explicit 2dvort GPU path and multi-rank validation
b5ed8c3 Extend explicit GPU validation to HIT and x-zeroextrap boundary
451fe0e Extend GPU boundary validation to channel wall multi-rank
```

主要功能变化如下。

1. GPU 主循环从 TGV 单 rank 扩展到多 rank。
   - 支持 TGV GPU resident time loop。
   - 支持 x/y/z slab 以及组合拓扑 halo。
   - 已验证 NP=2、NP=4、NP=8 核心矩阵。
   - 高 rank oversubscription smoke 覆盖到 NP=16、NP=32。

2. GPU 变量常驻能力加强。
   - 扩展 `src_gpu/commarray_gpu.cuf`、`mainloop_gpu.cuf`、`solver_gpu.cuf`、
     `qswap_gpu.cuf`、`halo_exchange_gpu.cuf` 等模块。
   - 主计算路径中 `q`、primitive、梯度、通量、统计量主要在 GPU 上推进。
   - 文件输出、checkpoint、HDF5 写场仍保持 CPU-owned boundary，暂不作为 GPU 化范围。

3. 四个主要数值模块 GPU 化范围扩大。
   - `convection`
   - `diffusion`
   - `filter`
   - `gradcal`

4. 多 MPI rank GPU halo 通信。
   - 实现 host-staged、CPU `qswap/dataswap` 兼容的 halo 交换。
   - 支持 x/y/z 三方向内部 rank 接口。
   - 物理边界通过 `MPI_PROC_NULL` gating，只在真实物理面施加边界 kernel。

5. 新增非 TGV 显式算例。
   - `2dvort`：3D extruded 显式非反应算例，支持单 rank、多 rank。
   - `HIT`：生成确定性 `velocity.h5`，支持 NP=1、NP=2 x-slab、NP=8
     `2x2x2` stats-only 验证。

6. 新增边界条件 GPU 支持。
   - `bctype=50` zero extrapolation：x/y/z 单 rank 和多 rank matrix。
   - `bctype=60` symmetry：x/y/z Cartesian slice 和多 rank matrix。
   - `bctype=41` isothermal no-slip wall：
     - 人工 TGV wall41 x/y/z slice。
     - 真实 channel y-wall 路径。
   - `bctype=42` adiabatic no-slip wall：
     - 人工 TGV wall42 x/y slice。
     - z 方向不支持，因为 CPU `noslip_adibatic` 当前没有 `ndir=5/6` 分支。
   - `bctype=411` slip-nonslip isothermal wall：
     - 人工 TGV wall411 y slice。
     - x/z 方向不支持，因为 CPU `slipisotwall` 当前只有 `ndir=3/4` 分支。
   - `bctype=421` slip-nonslip adiabatic wall：
     - 人工 TGV wall421 y slice。
     - x/z 方向不支持，因为 CPU `slipadibwall` 当前只有 `ndir=3/4` 分支。
     - 当前 CPU 有效公式未使用 `xslip` 分段，GPU 按 CPU 有效公式实现。

7. Channel flow GPU 路径。
   - 支持 `examples/Channel` 的 `x/z periodic + y bctype=41`。
   - 增加 deterministic channel 初始化。
   - 增加 GPU channel source 和 `massflux/fbcx/forcex/wrms` 统计。
   - 已验证 NP=1、NP=2、NP=4、NP=8、NP=27 的不同验证层级。
   - `128^3, deltat=7.5d-4, maxstep=100` 的 fixed-force stats-only 长步验证已通过。

8. 输入文件 CRLF 问题处理。
   - 大量 `examples/*/datin/input.*` 和 `controller` 统一换行为 LF。
   - `src/readwrite.F90` 对 `flowtype` 等字符串加入 `char(13)` 清理，避免 NVHPC
     将 `\r` 作为字符串内容导致 `select case` 匹配失败。

9. 验证体系扩展。
   - 新增 `tests/gpu_validation` 下多套 compare/run 脚本。
   - 增加 `flowstate.dat`、`flowfield.h5`、GPU 统计量、Nsight profile 驱动。
   - `documents/GPU_VALIDATION_MATRIX.md` 记录 TGV、2dvort、HIT、zeroextrap、
     symmetry、wall41、channel 的验证证据。

10. 文档和架构规划。
    - 新增 CUDA Fortran porting roadmap。
    - 新增 full GPU architecture plan。
    - 新增 multi-rank porting plan。
    - 新增 Phase C symmetry plan。
    - 新增 ADR，记录显式格式、线程块、同步、runtime `use_gpu`、CPU/GPU 分离、
      GPU resident、file output 边界等设计决策。

## 当前 GPU 已支持的算例

当前“支持”指已有 CPU/GPU 对比或验证矩阵证据，不表示 ASTR 所有物理模型均已 GPU 化。

| 算例 | `flowtype` | 当前 GPU 支持范围 |
|---|---|---|
| Taylor-Green Vortex | `tgv` | 周期边界，显式 `643e`，支持 NP=1/2/4/8 核心矩阵，高 rank smoke 到 NP=16/32 |
| 3D extruded Vortex Transport | `2dvort` | 从 `examples/Vortex_Transport` 改成 3D 显式周期算例，支持 NP=1/2/4，NP=8 stats smoke |
| HIT | `hit` | 生成确定性 `velocity.h5`，显式周期算例，支持 NP=1、NP=2 x-slab、NP=8 stats smoke |
| Channel flow | `channel` | `x/z periodic + y bctype=41` isothermal no-slip wall，支持 NP=1/2/4/8/27 验证；`128^3` 100 steps stats-only 已通过 |

另有用于边界路径验证的人工 TGV boundary slices：

| 验证 slice | 目的 |
|---|---|
| TGV + `bctype=50` | zero extrapolation x/y/z 边界验证 |
| TGV + `bctype=60` | symmetry x/y/z 边界验证 |
| TGV + `bctype=41` | artificial wall41 x/y/z kernel 和 MPI physical-face gating 验证 |
| TGV + `bctype=42` | artificial wall42 x/y kernel 和 MPI physical-face gating 验证 |
| TGV + `bctype=411` | artificial wall411 y kernel、`xslip` 切换和 MPI physical-face gating 验证 |
| TGV + `bctype=421` | artificial wall421 y kernel 和 MPI physical-face gating 验证；当前 CPU 有效公式未使用 `xslip` 分段 |

共同限制：

- 只支持显式格式：`643e,643e`。
- 不支持 compact 差分/滤波、三对角/五对角求解。
- 当前支持范围限定在 `numq=5`。
- 当前支持范围限定在 `num_species=0`。
- 无化学反应。
- 无湍流模型。
- 文件输出、checkpoint、HDF5 写场仍是 CPU 边界。

## 当前 GPU 已支持的 `bctype`

| `bctype` | 含义 | 当前 GPU 支持范围 |
|---:|---|---|
| `1` | periodic | x/y/z 周期方向，TGV、2dvort、HIT、channel x/z 均已使用 |
| `50` | zero extrapolation | x/y/z 单一物理方向，另一两方向 periodic；支持 NP=1/2/4 matrix |
| `60` | symmetry | x/y/z Cartesian symmetry slice；支持 NP=1/2/4 matrix |
| `41` | isothermal no-slip wall | GPU kernel 支持 x/y/z Cartesian wall slice；真实算例已验证 channel 的 y-wall |
| `42` | adiabatic no-slip wall | GPU kernel 支持 x/y Cartesian wall slice；z 方向不支持，因为 CPU `noslip_adibatic` 未实现 `ndir=5/6` |
| `411` | slip-nonslip isothermal wall | GPU kernel 支持 y Cartesian wall slice；x/z 方向不支持，因为 CPU `slipisotwall` 未实现 `ndir=1/2/5/6` |
| `421` | slip-nonslip adiabatic wall | GPU kernel 支持 y Cartesian wall slice；x/z 方向不支持，因为 CPU `slipadibwall` 未实现 `ndir=1/2/5/6` |

当前不应扩大解释的范围：

- `bctype=41` 当前不是完整通用壁面模型，只是显式、Cartesian、无 species、
  无 turbulence、无 wall blowing/suction 的 no-slip isothermal wall 路径。
- channel 的真实物理路径是 `x/z periodic + y bctype=41`。
- 人工 TGV wall41 只用于验证 x/y/z wall kernel 和 MPI physical-face gating，
  不代表物理 TGV-wall 算例。
- `42` 当前只支持 x/y Cartesian no-slip adiabatic slice，不支持 z、wall blowing/suction、
  turbulence wall model 或通用曲线壁面。
- `411` 当前只支持 y Cartesian slip-nonslip isothermal slice。下壁按 `x <= xslip`
  切换 slip/no-slip，上壁按 CPU 分支为 isothermal no-slip；不支持 x/z、wall blowing/suction、
  turbulence wall model、species 或通用曲线壁面。
- `421` 当前只支持 y Cartesian slip-nonslip adiabatic slice。当前 CPU 有效公式未使用
  `xslip` 分段；GPU 按 CPU 有效公式实现，不支持 x/z、wall blowing/suction、
  turbulence wall model、species 或通用曲线壁面。
- inflow/outflow/farfield/NSCBC，例如 `11`、`12`、`21`、`22`、`23`、`51`、`52`，
  当前不属于已支持 GPU 范围。

## `128^3` Channel Flow 整体加速比

本节使用 NP=1 CPU 作为统一基线，重新计算 `128^3` channel flow 的整体加速比。

测试参数：

```text
GRID=128,128,128
DELTAT=7.5d-4
MAXSTEP=100
LFILTER=f
DIFFTERM=t
CHANNEL_FORCE_MODE=fixed
CHANNEL_FORCE_FIXED=1.d-4
RUN_FIELD=f
```

统一基线：

```text
NP=1 CPU wall time ~= 517 s
```

加速比：

| GPU 配置 | GPU wall time | 相对 NP=1 CPU 加速比 |
|---|---:|---:|
| NP=1 `1x1x1` | 14 s | 36.9x |
| NP=2 `2x1x1` | 13 s | 39.8x |
| NP=2 `1x2x1` | 12 s | 43.1x |
| NP=2 `1x1x2` | 13 s | 39.8x |
| NP=4 `2x2x1` | 15 s | 34.5x |
| NP=4 `2x1x2` | 15 s | 34.5x |
| NP=4 `1x2x2` | 14 s | 36.9x |
| NP=8 `2x2x2` | 18 s | 28.7x |

结论：

- 当前两张 GPU 环境下，按单 CPU rank 作为统一基线，channel `128^3`、100 steps
  的整体加速比大约是 `28.7x` 到 `43.1x`。
- NP=8 因为 8 个 MPI rank 共享 2 张 GPU，属于 oversubscription correctness
  验证，不能作为生产多卡 scaling 结论。
- 若未来有一 rank 一 GPU 的多卡环境，需要重新测量 NP=4、NP=8 等真实多卡性能。
