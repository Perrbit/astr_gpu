# 发布源码来源初查

日期：2026-09-29。状态：按用户决定暂缓核查，不是完整许可审查结论。

## 检查范围

静态检查根 LICENSE、src/、src_gpu/、user_define_module/ 和
GPU_Quickstart 中显式版权、许可和复制来源注释，核对 src/CMakeLists.txt。
没有修改第三方代码、许可证或构建列表，也未执行 Git 操作。
关键词搜索没有命中不代表文件不存在第三方来源。

## 已发现的来源记录

| 位置 | 源码现有声明 | 待核实事项 |
|---|---|---|
| LICENSE | Apache License 2.0，Copyright 2021 Jian Fang | 保留原文，不据此覆盖各段代码已有声明 |
| src/stlaio.F90:383 等多处 | GNU LGPL | 确认各例程来源、具体许可版本和对应声明文件 |
| src/initialisation.F90:2960，r8_random | GNU LGPL；Wichman/Hill 原算法、John Burkardt Fortran90 版本 | 确认引入版本及许可文本 |
| src/geom.F90:3779，point_in_polygon | GNU LGPL；John Burkardt | 确认引入版本及许可文本 |
| src/utility.F90:645，progress_bar | 来自 macie/fortran-libs，Maciej Zok，2010 MIT License | 核实上游版本及完整版权/许可声明 |
| src/singleton.F90:9 | 改编自 fftn.c，提及 Mark Olesen 和 John Beale | 本次查看的文件头未给出明确许可，需追溯原始来源 |
| src/initialisation.F90:1576、2004 | copied from xcompact | 确认对应版本、复制范围和适用声明 |

上述五个 src 文件均在主程序 CMake 源列表中。即使某条路径运行时不启用，
文件仍属于当前源码交付范围。不以未使用 GPU 浸入边界等理由擅自删去文件。

Git 跟踪的许可相关文件还包括
documents/reference_data/opensbli_katzer/LICENSE.upstream 和
tests/gpu_validation/data/HTR_LICENSE.txt；它们分别随对应资料使用，不能
代替上述生产源码的来源记录。精简包不自动复制这些参考数据。

## 用户决定及保留事项

用户决定跳过本轮来源追溯，维持根目录 Apache-2.0 许可证。保留原有
第三方声明及实现，不修改许可证，不为第三方代码另行标注 Apache-2.0。
上表留作未核实来源记录，不据此新增许可文本或推断具体许可版本。

该项不再阻塞当前发布准备，不判定项目已违规，也不宣称已通过许可审查。
R11 的来源/许可项记为暂缓核查。构建和数值验证结果不受这一文档检查影响，
但不能替代发布来源确认。
