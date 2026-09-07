# ASTR 数值格式维护契约

本页不重复推导系数。完整 CPU 公式、selector 和边界公式见 [ASTR CPU numerical schemes](../ASTR_CPU_NUMERICAL_SCHEMES.md)。这里记录 GPU 维护时必须保持的入口、数据依赖、执行顺序和适用范围。

## 能力解释原则

1. CPU 文件中存在某个格式，不表示 GPU 已支持该格式。
2. GPU 文件中存在 device function，不表示 capability predicate 已允许对应算例。
3. CPU/GPU field 等价不等于物理验证，短步稳定不等于长时间稳定。
4. 当前 GPU 主线不移植 compact derivative、compact filter 或 tridiagonal/pentadiagonal solve。

## 当前数值契约

| 数值族 | CPU reference | GPU entry | 数据与顺序不变量 | 当前 GPU 边界 |
|---|---|---|---|---|
| Explicit sixth-order central derivative | `src/derivative.F90::diff6ec`, `src/comsolver.F90::gradcal` | `src_gpu/gradcal_gpu.cuf::gradcal_gpu`, central kernels in `solver_gpu` | 读取 `hm` halo、`dxi/jacob` 和同一 RK stage primitive state；物理边界使用已验证显式 closure | 3D、五守恒变量、已准入 periodic/physical/CURVE 组合 |
| Explicit central convection | `src/solver.F90::convrsdcal6` | `src_gpu/solver_gpu.cuf::convective_rhs_kernel` and direction kernels | 三方向贡献限制在同一 active box；curvilinear flux 使用 resident metrics | `conschm='643e'` 的已准入 central cases |
| Explicit diffusion | `src/solver.F90::diffrsdcal6` | diffusion flux/RHS kernels in `src_gpu/solver_gpu.cuf::solver_gpu` | gradient/primitive 先闭合，`sigma_d/qflux_d` 用固定 `hm` raw halo，不做 interface averaging | perfect-gas、Sutherland/已准入 transport 与 physical boundary 组合 |
| Explicit tenth-order central filter | `src/comsolver.F90::filterq_explicit10` | `filter_x_global_kernel`, `filter_y_pong_to_q_halo_global_kernel`, `filter_z_halo_global_kernel` in `solver_gpu` | full-`q` ping-pong；x 后交换 y work halo，y 后交换 z input halo；物理面使用已批准 closure | `lfilter=t` 的已准入 explicit cases；不含 compact filter |
| Explicit upwind reconstruction | `src/solver.F90::convrsduwd`, reconstruction functions in `src/flux.F90::flux` | WENO7/MP7 and boundary degradation helpers in `solver_gpu` | 先做 Steger-Warming split，再按 interface 重构；方向 kernel 读取该方向 halo | 已验证 first-order split、WENO7、MP7 路径；CPU 其他 reconstruction 不自动支持 |
| Ducros-style sensor | `src/commcal.F90::ducrossensor` | `src_gpu/shock_sensor_gpu.cuf::compute_shock_sensor_gpu` | raw sensor kernel 后显式同步，跨 rank 交换 `ssf` 的 `hm` halo，再扩展 mask | shock validation 与 selective-Roe admitted cases |
| Selective Roe characteristic path | `src/solver.F90::convrsduwd` | characteristic interface/RHS kernels in `solver_gpu` | smooth interface 走 physical-space reconstruction；mask 激活 interface 用 Roe characteristic override | 已验证的 explicit MP7 shock/SBLI combinations；不是全格式通用开关 |
| RK integration | `src/mainloop.F90::time_integration_rk` | `src_gpu/mainloop_gpu.cuf::time_integration_rk_gpu` | 第一 stage 保存 base state；每 stage 在 filter/boundary/halo/RHS 后更新；RHS 符号匹配 CPU | GPU 为 RK3；CPU RK4 存在但不属于当前 GPU contract |
| Case sources | `src/solver.F90::src_chan`, `src/solver.F90::src_rti` | `src_gpu/solver_gpu.cuf::apply_case_sources_gpu` | convection 取负和 diffusion 累加完成后、RK update 前加入 RHS | channel fixed forcing、RTI gravity 等已准入单组分路径 |
| Sponge | `src/sponge_layer.F90::spongefilter` | `src_gpu/sponge_gpu.cuf::apply_xmax_sponge_gpu` | RK update 后作用，随后 primitive refresh；需要正确 solution halo | 当前为受限 x-max sponge cases，不表示六面通用 sponge |
| Physical boundary closure | `src/bc.F90::boucon` and conservative boundary modules | `src_gpu/boundary_gpu.cuf::apply_boundary_conditions_gpu` and conservative stage | boundary plane ownership、filter closure、RHS closure 与 halo refresh 顺序必须成组验证 | 仅已列入 capability predicates 的 wall/symmetry/inflow/outflow/farfield/NSCBC 组合 |

## RHS 组装契约

CPU `src/solver.F90::rhscal` 的公共语义为：convective contribution 先累积到 `qrhs`，随后整体取负；diffusion 和 flow-specific source 再以各自符号加入。GPU 可以融合或拆分 kernel，但完整 stage 的 `qrhs_d` 必须保持同义顺序。

```text
qrhs = - divergence(convective flux)
      + divergence(viscous and heat flux)
      + case or user source
```

对 curvilinear grid，三项都必须使用同一时刻的 `jacob/dxi` 和 active ranges。只比较单方向 kernel 不能替代 complete-RK same-phase field comparison。

## Filter 与工作数组

当前 GPU explicit tenth-order filter 使用 resident `q_d` 与完整 `qwork_d` ping-pong，按 x、y、z 顺序推进。该实现增加一个完整守恒变量工作场，但避免 in-place stencil 覆盖和逐变量 host round trip。规划中允许保留当前实现，同时评估“单个 3D work array 按变量处理”的额外模式；该模式尚未实现，不能列入当前 capability。

物理边界不能直接套用 centered tenth-order stencil。当前已批准的显式 closure 保留物理 face，并按距边界位置使用 one-sided sixth、centered sixth、centered eighth，再进入 centered tenth order。修改 closure 必须同时验证 CPU oracle、GPU field、wall invariant 和 MPI interface，不得只以无崩溃验收。

## Shock sensor 与 selective Roe

Ducros-style sensor 通过速度散度与旋度区分压缩主导区域。GPU 将 sensor 与 flux 分开，形成 raw sensor、halo exchange、expanded mask 和 flux reconstruction 阶段。该结构增加网格遍历，但避免在一个大 kernel 中同时承担跨 rank mask、physical boundary 和两套 reconstruction 的复杂控制流。

selective Roe 仍包含 interface-level 分支：mask 未激活时使用 physical-space reconstruction，激活时使用 Roe characteristic projection/reconstruction/back projection。是否值得 compact active-interface list 属于性能问题，必须由 shock activity fraction、memory traffic 和 end-to-end profile 决定，不能仅凭线程发散直觉改写。

## 不支持或 deferred 的数值路径

| 路径 | CPU 状态 | GPU 状态 | 维护动作 |
|---|---|---|---|
| Compact central derivative/filter | 已有 line solve | 不移植 | capability 明确拒绝，不为适配算例而静默换义 |
| Compact upwind | CPU 已有 | 不移植 | 将输入算例改为 explicit 前需记录物理与数值基线变化 |
| CPU RK4 | 已有 | 未支持 | 保持 GPU RK3 gate |
| Species transport | 已有相关数组/路径 | deferred | `num_species>0` 保持拒绝 |
| Chemistry/combustion | Cantera/CPU coupling | deferred | 不把 reaction-rate kernel 当作完整 thermochemistry replacement |
| RANS/LES | CPU model path | deferred | `turbmode!='none'` 保持拒绝 |
| Moving grid | 部分 CPU framework | 未验证 | static CURVE 证据不能外推 |
| Immersed boundary | CPU `ibmethod` | deferred | `limmbou=t` 不进入现有 GPU contract |

## 变更验收最小集

数值格式改动至少需要：局部 operator/contract test、单步 same-phase field diff、多步统计量、相关 physical boundary closure、多 rank same-topology 对比、Compute Sanitizer，以及独立 residency/performance profile。误差阈值必须由量级、累计步数、topology 和 CPU oracle 状态分别定义，不能全项目共用一个 tolerance。

