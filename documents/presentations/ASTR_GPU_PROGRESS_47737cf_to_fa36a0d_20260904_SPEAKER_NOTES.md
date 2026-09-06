# ASTR GPU 移植阶段进展讲稿

## Slide 1

开场只保留三个信息：问题是 CPU 计算周期长；方法是把计算和数据生命周期迁到 CUDA Fortran/MPI；结果是代表性 Channel 获得 37.84 倍单卡加速，最新 256 立方曲线 C10 双卡相对单卡获得 1.58 倍加速。后续页面分别回答为什么可信、能算什么和还有什么没做。

[Sources]
- git 47737cf9b6a25b00bc5f1139cdaec36d55e5024e
- git fa36a0d657d0c22ae6d456ec442013d2f723364f
- tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903
- tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark_after_reduction_20260904

## Slide 2

这页只解释收益。37.84 倍来自同一个 128 立方 Channel 工况，CPU NP=1 用时 526.677 秒，GPU NP=1 三次数据的中位数为 13.917 秒。它是端到端壁钟时间，不是单个 kernel 的理论峰值。

[Sources]
- tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903/benchmark_times.tsv
- tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903/gpu_repeats.tsv

## Slide 3

常驻不是完全没有传输。MPI halo 必须交换，输出时仍要回到 CPU。关键变化是 q、原始变量、网格度量、梯度和工作数组在 RK 循环内持续驻留，不再为每个 kernel 搬整场数据。

[Sources]
- src_gpu/mainloop_gpu.cuf
- src_gpu/commarray_gpu.cuf
- documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md

## Slide 4

算例名称用于说明能力层次。TGV、2dvort 和 HIT 是基础周期流动；Channel 与 LDC 引入壁面；RTI 和 Sod 系列引入源项与间断；Mach 5 边界层与 CURVE-C15 组合曲线网格、黏性扩散、激波传感器和选择性 Roe。

[Sources]
- CONTEXT.md
- documents/GPU_VALIDATION_MATRIX.md
- tests/gpu_validation/README.md

## Slide 5

图中每根柱子取该验证门中最不利的已记录差值。CURVE-C12 的统计差异约 5e-11，仍低于 1e-10。该图说明 CPU/GPU 数值等价，不代表物理模型已经被实验数据验证。

[Sources]
- documents/GPU_VALIDATION_MATRIX.md
- documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md

## Slide 6

纵轴是细网格变化量与粗网格变化量之比。小于 1 表明继续加密后结果改变量在缩小。C14 同时检查壁面摩擦、冷壁热流和速度/温度剖面。热壁热流接近零点，因此只作为信息项，不用不稳定的相对误差作验收。

[Sources]
- tests/gpu_validation/out/curvilinear_hbl_physical_dual_after_reduction_20260904
- tests/gpu_validation/analyze_curvilinear_hbl_physics.py
- tests/gpu_validation/summarize_curvilinear_hbl_refinement.py

## Slide 7

左图和右图回答两个不同问题。左图比较 CPU 与 GPU 的整体加速，右图只比较一张和两张 GPU。右图最新三次中位数为 57.517 秒和 36.487 秒，对应 1.5764 倍加速和 78.82% 并行效率。

[Sources]
- tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903
- tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark_after_reduction_20260904/benchmark_summary.md

## Slide 8

C15 是本次相对旧稿的关键新增。原来的第一层归约产生 65536 乘 6 个 double，合计 3,145,728 字节。新实现使用第二个 GPU kernel 归约到 6 个标量。最新 trace 在 221 个 kernel 的 RK 区间内只有两次 176 字节 H2D 和两次 48 字节 D2H。

[Sources]
- src_gpu/statistic_gpu.cuf
- tests/gpu_validation/out/curvilinear_hbl_c10_256_residency_after_reduction_20260904/residency_report.txt
- tests/gpu_validation/out/curvilinear_hbl_c10_256_residency_after_reduction_20260904/nsys_summary.txt

## Slide 9

这页直接回答有什么用。当前程序可用于高频回归、两卡大网格计算和复杂流动能力开发。它还不是任意工程外形求解器，也没有完成生产级 SBLI 物理对标。

[Sources]
- documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md
- documents/GPU_VALIDATION_MATRIX.md

## Slide 10

下一阶段建议围绕两条主线：真实复杂工况的物理可信化，以及通信后端性能化。多组分、化学、湍流模型、IBM、动网格和 GPU HDF5 当前都不应混入主线，否则会同时扩大数值、物理和工程风险。

[Sources]
- documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md
- CONTEXT.md

## Slide 11

结束时只重复三句话：以前只有 CPU；现在 GPU 主线、多卡和曲线激波能力已经建立；下一步要把工程正确性推进到真实复杂工况的物理可信度。

[Sources]
- Summary of slides 1-10

## Slide 12

附录不在主线中逐页讲解，只在提问涉及具体提交、格式、边界、MPI 或性能协议时使用。

[Sources]
- Local project evidence

## Slide 13

行数只反映工作量，不等同于功能价值。最新 fa36a0d 单次提交包含 30 个文件、2082 行新增和 85 行删除，重点是 C10-C15。

[Sources]
- git log 47737cf..fa36a0d
- git diff --shortstat 47737cf..fa36a0d

## Slide 14

CPU 代码仍然是主要 oracle，并负责尚未 GPU 化的文件边界。GPU facade 避免 src/ 直接依赖大量 CUDA 模块。能力门控确保未验证的数值/边界组合明确拒绝，而不是静默进入错误路径。

[Sources]
- src/astr.F90
- src/mainloop.F90
- src_gpu/gpu_runtime.cuf
- src_gpu/case_capability_gpu.cuf

## Slide 15

显式格式更适合当前 GPU 路径，因为不需要沿整条网格线求解线性系统。选择性 Roe 只在 shock mask 触发处进入特征空间，平滑区保留物理空间重构。

[Sources]
- documents/ASTR_CPU_NUMERICAL_SCHEMES.md
- documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md
- src_gpu/solver_gpu.cuf

## Slide 16

边界方向限制来自 CPU 现有实现和已完成验证。例如 42 当前只支持 x/y，411 和 421 当前只支持 y。曲线几何优先使用局部法向投影，不能把 Cartesian 分量钳制直接推广到曲线面。

[Sources]
- documents/GPU_VALIDATION_MATRIX.md
- src_gpu/case_capability_gpu.cuf
- src_gpu/boundary_gpu.cuf

## Slide 17

NP 表示 MPI rank 数量，不自动等于 GPU 数量。NP=8 和 NP=27 在本机只用于检查三方向邻居、内部 rank 和边界 ownership。当前正式性能证据只有 NP=1 单卡和 NP=2 双卡。

[Sources]
- src_gpu/halo_exchange_gpu.cuf
- documents/ASTR_GPU_MULTI_RANK_PORTING_PLAN.md
- documents/GPU_VALIDATION_MATRIX.md

## Slide 18

每一级回答不同问题。CPU/GPU 一致只说明移植一致；物理诊断说明离散结果是否呈合理趋势；Sanitizer 检查越界和非法访问；Nsight 与 nvitop 证明计算和数据驻留。

[Sources]
- tests/gpu_validation/README.md
- documents/GPU_VALIDATION_MATRIX.md

## Slide 19

Channel 的 CPU 基线只测一次，因此 37.84 倍是工程证据，不是发表级统计。C10 每个 GPU 配置有预热和三次有效计时，且报告 min/median/max。两组都包含 CPU-owned 文件边界开销。

[Sources]
- tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903
- tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark_after_reduction_20260904/timings.tsv

## Slide 20

旧 CPU 程序在物理边界滤波、三维壁面积、NSCBC halo 时序、Ducros 索引和 symmetry 法向上暴露过确定性问题。所有明确修复都先经过人工确认。对物理语义不唯一的 NSCBC 分支不做擅自泛化。

[Sources]
- documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md
- documents/GPU_VALIDATION_MATRIX.md
- tests/gpu_validation/out/curvilinear_hbl_c10_256_residency_after_reduction_20260904/nsys_summary.txt
