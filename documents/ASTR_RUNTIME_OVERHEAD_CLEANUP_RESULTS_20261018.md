# 运行时重复工作清理实施记录

2026-10-18，feature/gpu_dev；基线2e3d76e。完成已批准审查项1至5，
没有推进其余审查项，没有操作Git或平台任务。

## 修订与工作量变化

| 项目 | 实现 | 实测或检查结果 |
| --- | --- | --- |
| TGV/HIT统计 | 同相位计算一次动能、拟涡能、耗散率，供原四个文件共享；独立写出入口仍可自行计算 | 三步TGV的独立Nsys记录中，三种统计部分归约核各由4次减为2次 |
| 三维场和切片 | 发布采用名称清单，不再回读场文件或共享几何计算CRC | 同一四步归档窗口，strace记录的相关data.h5只读打开由48次减为0 |
| JPEG/EPS | 删除普通图片的逐文件fsync，保留flush、close和发布 | JPEG-only与JPEG/EPS两种单测均无fsync调用，内容及故障清理通过 |
| M12诊断 | 静态参数检查前移；动态有效性合并到一次MIN归约；探针CSV在会话内保持打开，逐步flush，结束时close | NP2、52步探针文件写打开由52次减为1；逐步72探针和第50步壁面文件逐字节不变 |
| 检查点分区 | 本地检查整张表的重叠、覆盖、计数和溢出，然后一次全局放行 | 写入和精确读取的MPI_Allreduce总调用数均不再随NP=1/2/4增长，见下表 |

分区探针用测试专属PMPI包装计数，未向生产astr链接计数器。
计数覆盖一次探针操作，不仅是分区检查本身。

| 操作 | 旧NP1 | 旧NP2 | 旧NP4 | 新NP1 | 新NP2 | 新NP4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 写入 | 171 | 199 | 261 | 133 | 133 | 133 |
| 精确读取 | 206 | 234 | 296 | 168 | 168 | 168 |

本地两两重叠比较仍为O(P²)；消除的是循环内的全局归约，不宣称本地
算法已经线性化。Nsys用于核调用计数，不用其受插桩影响的时间计算收益。
本轮没有运行正式整步计时矩阵，不给出加速百分比，也不外推A800收益。

## 数值与生命周期检查

| 范围 | 结果 | 证据 |
| --- | --- | --- |
| 根CMake构建 | 纯CPU、CUDA/AIR5、CUDA/Catalyst/M12三个现有配置构建通过 | build_cpu_probe、build_gpu_probe、build_m12_device_render |
| TGV旧新程序 | 16³、NP1/2、四步；CPU与GPU各自旧新state/statistics逐值一致，GPU四个诊断文本逐字节一致 | native_runtime.xml中四项数值检查；native_pure_cpu.xml中两项纯CPU检查 |
| TGV精确续算 | 同后端同拓扑连续四步与2+2步续算，state/statistics逐值一致 | 同上 |
| 分区与状态角色 | 46项通过；5/11/34分量、缓存/补偿/统计/历史角色、精确续算/重分区与损坏元数据拒绝 | native_state.xml |
| 归档数据与回读 | HDF5数据集逐值不变，三维场/切片序列可读，新名称清单可离线重建 | native_io_profile.xml，两项通过 |
| 归档发布失败 | EIO注入创建/发布阶段，失败帧及未来帧不发布，已完成初始帧保留 | native_archive_recovery_r2.xml中两项故障检查 |
| 归档同根续接 | CPU/GPU各NP2，连续12步与5+7步续算，最终场/统计逐值一致；原生与独立索引相符，ParaView读取通过 | native_parent_r3.xml，两项通过 |
| 原生父归档 | 31项通过；旧CRC与新名称清单、五变量/AIR5字段布局、三维场/切片/派生量、损坏元数据拒绝 | native_lineage_final.xml |
| 显式离线工具 | 98项通过；旧CRC校验保留，新清单支持补齐 | host_offline.xml |
| 图片与源码约束 | 30项通过，含四个子测试；保留受控图片I/O故障、未知错误及内容检查 | host_final.xml |
| M12真实短窗 | 64×32×24区间、NP2 x-slab、dt=0.02、52步；原随机吹吸、FP64和MP通量保正不变，无图片到期事件 | native_m12_r2.xml，两项通过 |
| M12诊断与续接 | 原新CSV逐字节一致；当前连续52步与50+2步state逐值一致，恢复段CSV与连续尾部逐字节一致 | 同上及file_opens.json |

这些检查有重叠，不累加为独立验收总数。没有把新建短窗称为HIT、
反应SBLI、长时物理定常或生产规模验收。M12在第5/10步的只读诊断
隔离检查确认q及RNG不变；该窗口没有测试新的渲染产品。

所有证据位于
`tests/gpu_validation/out/runtime_overhead_cleanup_20261018/`。
调用计数文件为：

- `native_io_profile/test_tgv_statistics_kernel_cal0/kernel_counts.json`
- `native_io_profile/test_archive_payloads_and_no_c0/archive_reads.json`
- `native_m12_r2/test_m12_probe_and_wall_output0/file_opens.json`
- `red_counts/test_partition_gate_has_consta{0,1}/counts.json`
- `native_state/test_partition_gate_has_consta{0,1}/counts.json`

## 程序来源

旧TGV程序冻结自本轮开始时的纯CPU及CUDA构建。M12参考来自原有
2026-10-13的Catalyst/M12构建，不将它称为2e3d76e的全量重构建。
本轮分别验证其诊断输出等价，以及当前M12程序自身的精确续接。

| 程序 | SHA256 |
| --- | --- |
| frozen/astr_gpu | 9e9c56dc05b43e7aabe03df7c5cafd19909aef56f0b4a50629c57a23ccc43124 |
| frozen/astr_cpu | b0071b9cd66c685f5a4a900593a180c08a7f6a28f0feb41b754ef86197675382 |
| frozen/astr_m12_render | 482934288bb0a8e203a8c6f322b265d46e9f36132d3d55db6daa6f7c717e481e |
| 最终build_gpu_probe/bin/astr | 382fc9a6ce774478dce37d39d34cf3f29678b2d02212152356be63276dfc63c5 |
| 最终build_cpu_probe/bin/astr | e6e7be75feefbf5a696f94b8b054ba6d220524c88da479d72d0f9c797e1021f0 |
| 最终build_m12_device_render/bin/astr | 7ca377f144a45ccb72aeecaf77bbaf10bc52157abe937e832584ec054f973aa0 |

最后一处清单识别修改前，实际参与TGV数值/Nsys检查的CUDA程序为
`9c1056d3f0daf7edb2df93110b990ae4ae47b2d05165c49db0dd538f36133777`，
纯CPU程序为
`5938f5cb36c4533b1d1857974f88bd9df611864e320f5a80f8923095724a1136`，
M12诊断程序为
`162f3aa9197a15c5ebd782f3874434f8e1450053539bae3ff22a9b95dcaab30e`。

旧新TGV数值、Nsys与M12诊断检查发生在最后一处原生清单识别修复
之前。该修复只增加对名称清单的识别，不改变数值或图片路径；相关
原生父归档与CPU/GPU真实同根续接已用最终GPU构建重验。三个配置
均重新构建，不为这处识别修改重跑昂贵数值或计时矩阵。
收口时证据目录共700356247字节，未向旧AIR5目录追加；无残留
ASTR、MPI、strace或Nsys任务。

## 暴露并解决的问题

取消归档CRC后，显式Python离线工具和原生父归档资源登记仍只识别
旧清单。分别补齐v2名称清单识别，同时保留旧CRC格式处理及必需
成员、数量、末尾额外行和HDF5元数据约束。原生同根续接首次真实
失败保留在`native_archive_recovery_r2.xml`，修复后记录见r3。

其他首轮失败也保留：测试模块导入路径、NVHPC以O_RDWR而不是
O_WRONLY打开探针文件，以及故障测试错误地拒绝已完成初始帧。
对应测试已修正并通过；没有通过放宽场差门槛解决失败。

## 保留与不再承诺的行为

- 不改FP64、RK/化学控制、空间格式、保正、halo、同步策略和生产默认。
- M12仍每步采样72点、每50步输出壁面量，保留全场正有限检查；
  本地CUDA或I/O错误直接MPI_Abort，不再先进行成功路径全局归约。
- 普通图片不再逐张保证掉电持久性。写入、flush、close及发布错误仍处理。
- 新归档没有payload CRC，不能发现任意静默改写；正常文件I/O、
  必需成员、结构和时间检查仍保留，旧带CRC文件仍可显式离线校验。
- 不增加CRC回扫开关、不清理旧结果、不修改平台作业，不自动提交代码。
