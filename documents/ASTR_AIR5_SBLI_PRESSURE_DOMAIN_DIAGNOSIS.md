# AIR5 SBLI Pressure-Domain Diagnosis

更新：2026-09-24。范围：当前 `air5sbli` 双卡工程算例，不是物理通过报告。

## 结论

长时运行的直接停止原因已经定位：壁面二阶压力外推产生超过 `1 MPa` 的
有效域外状态，随后输运物性检查返回 `chemistry_status_out_of_domain=4`。
CPU 和 GPU 在同一 checkpoint、步号、RK 阶段重现此条件。
从同一 checkpoint 将时间步减半、减至四分之一，仍在接近的物理时刻越界。
这不是机器舍入量级的超限，也没有证据表明是 GPU 独有的错误。

当前工况还存在需要先决策的压力覆盖问题：冻结气体规则反射估算为
`1.926424 MPa`，已超过已验证有效域。不能假设细化时间步或网格就能让整个
压缩过程保持在 `1 MPa` 以下。该估算不是黏性、反应、热壁 SBLI 的峰值预测，
目前也未完成空间网格收敛，不能据此宣布现有粗网格压力峰值物理正确。

## 壁面赋值链路

CPU `src/chemistry_boundary_state.F90:build_air5_hbl_wall_state` 与 GPU
`src_gpu/chemistry_boundary_gpu.cuf:air5_hbl_wall_boundary_kernel` 均采用：

\[
p_w=\frac{4p_1-p_2}{3},\qquad
Y_{s,w}=Y_{s,1},\qquad
\rho_w=\frac{p_w}{R(Y_w)T_w},\qquad
\boldsymbol{u}_w=0,\quad T_w=T_{v,w}=2925\ {\rm K}.
\]

这是均匀法向网格上二阶单边零法向压力导数闭合。它是外推，不是凸组合，
当 `p1 > p2` 时允许 `pw > p1`。当前检查确认两端实现了同一公式，
但没有将此闭合在未解析激波/壁面相互作用区的适用性认定为已验证。

step 741/RK2、全局节点 `(20,0,3)` 的 GPU `pre_rhs` 快照：

| 法向索引 | 压力 Pa | 温度 K | 密度 kg/m3 |
|---:|---:|---:|---:|
| 0，壁面 | 1020294.032274 | 2925.000000 | 1.209939189 |
| 1 | 871181.818413 | 3919.256642 | 0.771026260 |
| 2 | 423845.176830 | 1871.932262 | 0.785384300 |

壁面值比最近内点高 `17.116%`，并精确满足外推关系。此时全域内点最高压力
只有 `0.871182 MPa`，所以本次最先触发上限的是边界重构。CPU 对应峰值为
`1020294.032653 Pa`，相对 GPU 差 `3.72e-10`。这只是此位置拒绝条件的一致性，
不替代整个晚期场的 CPU/GPU 误差验收。

物性入口 `src/chemistry_properties.F90:air5_transport_properties` 在计算输运
系数前调用压力域检查。`chemMech/air5_kimjo12.json:validity` 的范围是
`1000--1000000 Pa`，`air5_pressure_is_in_domain` 仅附加舍入容差，约
`1.42e-8 Pa`。原步长超限 `20294 Pa`，不是浮点容差问题。`1 MPa` 是当前
固定模型的已声明/已测试边界，不应写成空气物性在该压力必然失效的理论极限。

## 同 Checkpoint 时间步对照

起点均为原始 checkpoint 700、`t0=0.35 us`，输入状态不插值，FP64、显式
同步、NP=2 x-slab，化学、边界、网格、滤波和物性范围均不变。

| dt / s | 失败步号 | 阶段 | 失败步起始时间 / us | 峰值 / MPa |
|---:|---:|---:|---:|---:|
| 5e-10 | 741 | RK2 | 0.370500 | 1.020294032 |
| 2.5e-10 | 785 | RK2 | 0.371250 | 1.004656220 |
| 1.25e-10 | 870 | RK2 | 0.371250 | 1.001483115 |

时间按 `t0 + (nstep-700)*dt` 计算，不是 `nstep*dt`，也不是 RK2 的精确
子阶段时间。所有运行遇到有效域外状态立即结束。较小步长的超限幅度减小，
不意味着可继续积分：其首次采样越界更接近阈值。此对照只排除了“局部减小
步长即可避免越界”的解释，不是从初始时刻开展的时间收敛研究。

新增脚本只驱动 ASTR，不积分流场，也不放宽验收。复现示例（输出目录必须新建）：

```bash
python3 tests/gpu_validation/run_air5_sbli_domain_replay.py \
  --baseline tests/gpu_validation/out/air5_sbli_long_20260924/gpu \
  --output tests/gpu_validation/out/sbli_domain_replay_new \
  --dt 2.5e-10 --updates 120
```

`result.json` 的 `domain-failure-not-pass` 表示重现拒绝条件，不是通过。
该诊断只用于已知 NP=2 x-slab 算例。它保留原 checkpoint，复制可执行文件，
记录输入 checkpoint 与可执行文件摘要，最长运行 300 s，无自动重启。

## 冻结规则反射的压力覆盖估算

从 `incident_shock_metadata.json` 读取入射波下游状态：
`M2=4.589585464`，`gamma=1.4`，`p2=626769.375958 Pa`，
速度相对壁面方向为 `-10.983012810 deg`。将流动转回平行壁面方向所需的
转角为 `theta=10.983012810 deg`，不是入射波相对原来流的 `13.362535364 deg`。

冻结组成和振动能，以规则反射弱支估算：

\[
\tan\theta=2\cot\beta\,
\frac{M_2^2\sin^2\beta-1}{M_2^2(\gamma+\cos2\beta)+2},\qquad
\frac{p_3}{p_2}=\frac{2\gamma M_2^2\sin^2\beta-(\gamma-1)}{\gamma+1}.
\]

取第一个附着激波根，得到 `beta=21.291509538 deg`、`p3/p2=3.073577191`、
`p3=1926424.058081 Pa`。公式依据
[NASA Oblique Shock Waves](https://www.grc.nasa.gov/WWW/BGH/oblique.html)。
这是独立的压力范围预检，不调用越域的机理物性，不改变模型有效域。
不包含壁面黏性、分离、有限速率反应或突加入射波的瞬态效应。

## 证据路径与下一步

以下路径均相对 `tests/gpu_validation/out/`：

- `air5_sbli_long_20260924/`：原任务、checkpoint 700、边界状态 JSON。
- `air5_sbli_failure_replay_gpu_fields_20260924/`：GPU step741/RK2 `pre_rhs`。
- `air5_sbli_failure_replay_cpu_20260924/`：匹配 CPU 快照。
- `air5_sbli_domain_halfdt_20260924/`：减半步长的 `contract.json/result.json`。
- `air5_sbli_domain_quarterdt_20260924/`：四分之一步长的对应记录。

读取快照用 `check_air5_c5_hbl._assemble_global`，恢复状态用
`check_air5_c5_normal_shock.primitive_metrics`。壁面数据来自相同 RK 阶段，
不是 checkpoint 与中间阶段混比。

下一步先人工确定工况与有效域的关系，再投入长时/网格计算：

1. 若保留原工况，先审查动力学、弛豫和输运模型的压力适用依据；只有依据允许，
   才另行批准扩展范围，并补新压力区间的 CPU/GPU 物性、源项、积分验证。
2. 若保留当前 `1 MPa` 门槛，另建一致的低压工程工况，重生成入口、初场和
   顶部激波状态。降压会改变 Reynolds 数、扩散及化学/弛豫时间尺度，不能仅
   缩放压力后称为原工况的等价验证。
3. 范围决策之后，再做空间细化及启动瞬态对照，判断壁面峰值、反射结构和
   边界闭合的敏感性；不把压力截断、盲目改为一阶外推作为通过手段。

本次没有修改求解器、CPU/GPU 壁面格式、压力范围或原算例参数。长时目标未完成。
