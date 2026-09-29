# ASTR GPU

ASTR GPU 是基于 CUDA Fortran 和 MPI 的结构化高阶可压缩流动求解器。
主要时间推进状态常驻 GPU，MPI 交换子域边界数据。
基础交付范围为受支持配置下的非反应流；固定 AIR5 双温反应流为可选实验功能。

## 内容

| 路径 | 用途 |
|---|---|
| src/、src_gpu/ | CPU、共享物理接口及 CUDA Fortran 源码 |
| user_define_module/ | 用户模块及示例 |
| CMakeLists.txt | 主构建入口 |
| examples/ | 算例输入、安装规则和 GPU_Quickstart 启动工具 |
| USER_GUIDE.md | 部署、输入、边界条件、可选功能、统计与重启 |
| documents/ | 能力边界、输入格式、部署验收及验证摘要 |
| LICENSE | 根许可证；各源码已有版权及许可声明保留 |

本交付目录不含测试脚本、开发历史、编译产物或运行结果。
示例存在不表示该示例的全部物理配置均可在 GPU 上运行。

## 构建

先配置兼容的 NVHPC、MPI 和 HDF5 Fortran/HL 环境，在源码根目录执行：

```bash
cmake -S . -B build_prod \
  -DCMAKE_Fortran_COMPILER=nvfortran \
  -DCMAKE_BUILD_TYPE=Release \
  -DASTR_WITH_CUDA=ON \
  -DASTR_WITH_AIR5_CHEMISTRY=OFF \
  -DCHEMISTRY=OFF \
  -DBUILD_TESTING=OFF
cmake --build build_prod --target astr -j 2
```

可执行文件为 build_prod/bin/astr。纯 CPU 构建使用 ASTR_WITH_CUDA=OFF。
usegpu 是运行时输入开关；编入 CUDA 不会自动选择 GPU。
本目录不含 tests/，必须设置 BUILD_TESTING=OFF。遗留 Cantera 构建不在基础交付范围。

## 使用

- [使用说明](USER_GUIDE.md)
- [算例生成与启动](examples/GPU_Quickstart/README.md)
- [能力边界](documents/ASTR_RELEASE_PROD_CAPABILITY_SCOPE.md)
- [输入数据字典](documents/ASTR_DEPLOYMENT_INPUT_CONTRACT.md)
- [部署验收清单](documents/ASTR_RELEASE_READINESS_CHECKLIST.md)
- [验证结果摘要](documents/ASTR_RELEASE_LOCAL_REGRESSION_20260929.md)
- [源码许可说明](documents/ASTR_RELEASE_SOURCE_PROVENANCE_REVIEW.md)

基础配置采用 FP64、scalar 滤波存储、explicit 同步及主机中转通信，输出采用 iomode=h。
在实际目标节点完成部署验收后再启动长时计算。CUDA-aware MPI、混合精度及 AIR5
按各自条件单独验证，不把短跑成功视为物理统计收敛。
