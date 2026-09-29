# ASTR GPU 发布执行清单

更新日期：2026-09-29。状态：R1 方案 A 已执行；R2 已获准提交发布准备变更。

## 执行约束

在 feature/gpu_dev 当前目录逐步完成发布准备。不创建 worktree，不自动
创建 release/prod，不执行 Git 暂存、提交、推送或目录裁剪。不干预远程任务。
物理模型、数值方法、验收标准或交付范围需要人工决定时停止。
历史验证只有在源码、输入和环境适用时才复用；不以历史通过替代最终发布包验收。

## 基线盘点

- 当前 HEAD：`bd5207b521787c3c2074e538b9e00d40b6210a23`。
- HEAD 说明：`feat(air5): support compensated filtering with characteristic top boundaries`。
- 当前工作区不干净，不能把已有构建和运行结果全部归属于这个 HEAD。
- 跟踪文件变化包括根及 src CMake、src/test.F90、scalar 滤波默认值、README
  和验证文档/测试；USER_GUIDE、GPU_Quickstart、发布说明及诊断脚本还未跟踪。
- 未跟踪的手册、演示材料及 USER_GUIDE.md.save 不自动列入交付，不删除。
- 原有许可证文件存在；尚未完成全部第三方来源和交付许可审查。

## 顺序与状态

| 顺序 | 项目 | 当前状态 | 完成条件 |
|---|---|---|---|
| R1 | 冻结发布能力和默认配置 | 方案 A 已批准并落实 | 非反应流基础发布；AIR5 显式启用且标为实验功能 |
| R2 | 源码基线与变更归属 | 已核对差异并获提交授权 | 以本清单本次提交对应的版本为候选基线，后续仍需发布验收 |
| R3 | 发布文件清单 | 已有裁剪建议，未实际裁剪 | 确认源码、构建、examples、许可证和最小文档清单 |
| R4 | 最终包独立构建 | 既有临时副本构建记录可参考 | 对 R3 实际交付内容独立构建，不依赖 tests 或本地遗漏文件 |
| R5 | 目标平台依赖 | 未执行本轮检查 | NVHPC、MPI、HDF5、驱动和计算节点运行库匹配；远程操作先停下确认 |
| R6 | 最小数值回归 | 未执行发布版本回归 | 冻结二进制上覆盖周期、壁面、开放边界、CURVE、CPU/GPU 和 MPI |
| R7 | 输出与重启 | 历史证据待映射 | 发布配置下输出相位、连续/重启一致性及统计续接可追溯 |
| R8 | 默认 scalar 滤波 | 已有短测试记录，待检查长窗证据 | 核对适用的 full/scalar、长窗和重启记录，只补必要缺口 |
| R9 | CUDA-aware MPI | 作为条件能力，未完成本轮平台准入 | 目标软件栈和拟交付边界组合通过，提供明确主机中转配置 |
| R10 | 使用说明 | 已有草稿和五套启动示例 | 默认值、选项、允许组合、失败信息与实际交付一致 |
| R11 | 发布清单及已知问题 | 待前述门槛完成 | 版本、依赖、证据、能力限制、许可和部署恢复方法齐全 |

上述状态不等于全部能力从未验证，而是区分研发历史与本轮最终交付验收。
本轮完成源码/文档核对及 30 项针对性测试，没有重新执行构建、GPU 算例或平台测试。

## R1 已批准方案 A

修改前核查发现：

- 根 CMake 的 `ASTR_WITH_AIR5_CHEMISTRY` 默认 OFF。
- `examples/GPU_Quickstart/build.sh` 的 `AIR5` 默认 ON，为五套示例共同构建。
- 发布能力说明建议基础部署默认关闭 AIR5，按需启用受限验证版。

这不是求解器错误，但会让“默认发布配置”产生歧义。用户已选择方案 A：

**A，推荐：非反应流稳定版为默认。** AIR5 源码保留，启动包构建默认改为
OFF。两个 AIR5 示例仍保留，但明确要求先 `AIR5=ON` 构建，并标注实验范围。
同步脚本、指南和相关测试，不更改物理模型或删除 AIR5 源码。

共同基础配置：FP64、scalar 滤波、explicit 同步；主机中转作为
无需 CUDA-aware MPI 的基础通信选择，device-aware 按平台和组合单独准入。
混合精度、流水线通信和 fused 通量不自动成为默认。

启动包默认已改为 OFF，README 和 USER_GUIDE 同步显式 AIR5 构建及 EXE
选择方法。新增构建脚本参数测试，覆盖未设置、空值、OFF 和 ON。测试通过：

```bash
python -m pytest -q tests/gpu_validation/test_gpu_quickstart.py \
  tests/gpu_validation/test_scalar_filter_workspace_contract.py
# 30 passed
```

参数测试使用模拟 CMake 记录实际 shell 参数，不冒充真实编译检查。
未改变根 CMake 默认、求解器物理模型或 AIR5 源码。

## R2 提交范围已获授权

用户已授权 git add、git commit，未授权 push 或创建分支。本次提交将下列
发布准备内容固定为候选基线；实际提交号以 Git 记录为准，不在提交内部自引用。

- CMakeLists.txt、src/CMakeLists.txt、src/test.F90：测试构建解耦。
- src_gpu/commarray_gpu.cuf：此前批准的 scalar 滤波默认值。
- examples/GPU_Quickstart/：完整启动包，不包含 runs 和缓存。
- README.md、USER_GUIDE.md、发布能力说明、本清单和当前状态文档。
- tests/gpu_validation/README.md 及三个对应测试：
  test_cmake_production_decoupling.py、test_gpu_quickstart.py、
  test_scalar_filter_workspace_contract.py。
- 当前状态文档所链接的 ASTR_AIR5_PRECURSOR_ADMISSION_20260929.md，
  及其 check_air5_precursor_admission.py、test_air5_precursor_admission.py，
  保持已记录诊断可追溯；不纳入稳定版物理验收结论。

不使用 git add 全目录，不纳入手册、演示文件、USER_GUIDE.md.save、
运行结果或其他未审查脚本。提交前再核查暂存差异。
本次只提交上述范围，不创建 release/prod，不删除文件，不推送。
下一步是 R3 文件清单与 R4 最终包独立构建。版本被提交不等于发布验收完成。

## 依据

- `documents/ASTR_RELEASE_PROD_CAPABILITY_SCOPE.md`
- `documents/ASTR_AIR5_PRECURSOR_ADMISSION_20260929.md`
- `CMakeLists.txt`
- `examples/GPU_Quickstart/build.sh`
- `tests/gpu_validation/test_cmake_production_decoupling.py`
