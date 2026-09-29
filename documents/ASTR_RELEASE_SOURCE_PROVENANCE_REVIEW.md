# 源码版权与许可说明

文档版次：2026-09-29。本文件汇总源码已有声明，不重新授予或替换第三方许可。

## 根许可证与第三方声明

根 LICENSE 保持 Apache License 2.0。各源码文件中的原作者、版权、许可、
公开来源和参考文献声明应同时保留，不能仅根据根许可证覆盖已有第三方条款。

| 位置 | 源码现有声明 | 待核实事项 |
|---|---|---|
| LICENSE | Apache License 2.0，Copyright 2021 Jian Fang | 保留原文，不据此覆盖各段代码已有声明 |
| src/stlaio.F90:383 等多处 | GNU LGPL | 确认各例程来源、具体许可版本和对应声明文件 |
| src/initialisation.F90:2960，r8_random | GNU LGPL；Wichman/Hill 原算法、John Burkardt Fortran90 版本 | 确认引入版本及许可文本 |
| src/geom.F90:3779，point_in_polygon | GNU LGPL；John Burkardt | 确认引入版本及许可文本 |
| src/utility.F90:645，progress_bar | 来自 macie/fortran-libs，Maciej Zok，2010 MIT License | 核实上游版本及完整版权/许可声明 |
| src/singleton.F90:9 | 改编自 fftn.c，提及 Mark Olesen 和 John Beale | 文件头未给出明确许可，需追溯原始来源 |
| src/initialisation.F90:1576、2004 | copied from xcompact | 确认对应版本、复制范围和适用声明 |


## 核查状态

上述条目为已有声明的来源索引，不是完整许可审查结论。
其中部分引入版本、完整许可文本及复制范围仍待核实；这些事项保留为已知限制。
本交付不修改这些实现及原声明，不将尚未核实的片段统一重新标注为 Apache-2.0。

未随本目录交付的参考数据、论文、图片或第三方运行输入，不能从源码许可证
推定其使用权限。对外提供额外资料时，应另行核对资料本身的许可和共享条件。
构建、数值验证或性能测试通过不能替代来源与许可核查。
