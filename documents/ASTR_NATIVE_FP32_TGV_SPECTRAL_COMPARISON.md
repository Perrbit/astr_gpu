# TGV 原生 FP32／FP64、显式／紧致的空间相关与能谱对照

日期：2026-10-28。只读分析已保存的完整步检查点，未重跑 CFD、改动求解器或操作平台任务。

## 1. 主要结果

在相同格式下以 FP64 为参考，名义时间 $t^*=10$ 的显式 FP32 速度场相对 L2 差为 **0.1590%**，紧致 FP32 为 **16.4019%**。
紧致 FP32 的普通速度能量仅增加 **0.3409%**，密度加权动能却增加 **22.6919%**。
最终保存场的平均密度为 **1.22305024**，因此此前约 22% 的密度漂移不只是原生统计归约误差。

| 相同格式的 FP32−FP64 | 名义时间 | 速度相对 L2 差 | 普通速度能量变化 | 密度加权动能变化 | 归一化相关函数最大绝对差 |
|---|---:|---:|---:|---:|---:|
| 显式 | 8.75 | 0.05833% | +0.00947% | +0.11247% | 9.8572e-5 |
| 显式 | 10.00 | 0.15903% | +0.01228% | +0.12960% | 2.1661e-4 |
| 紧致 | 8.75 | 7.17945% | +0.22194% | +19.48453% | 0.0111534 |
| 紧致 | 10.00 | 16.40193% | +0.34094% | +22.69188% | 0.0222150 |

相关函数最大差覆盖三速度分量、三分离方向和整个周期滞后序列。速度相对 L2 差比较各自减去瞬时体积均值后的场。
没有新设跨精度误差阈值，也不将“正常结束”判作精度通过。

## 2. 数据、相位与比较对象

原始目录为 `tests/gpu_validation/out/native_fp32_tgv_checkpoints_20261022/`，四组分别为：

- `643e_fp64`、`643e_fp32`：对流和黏性采用六阶显式差分，主滤波开启。
- `643c_fp64`、`643c_fp32`：对流和黏性采用六阶紧致差分，主滤波开启。

均为 $256^3$ 区间、$[0,2\pi]^3$ 周期盒、Re=1600、Ma=0.1、RK3、NP=2 x 分解，输入步长 $2.5\times10^{-4}$。
检查点为第 35,000、40,000 完整步，名义时间分别为 8.75、10。

存储节点为 $257^3$。每个方向剔除最后一个周期重复端点，只使用 $256^3$ 唯一节点。
HDF5 数据按 z/y/x 保存，读取时明确转换为 x/y/z。使用 q1–q5 权威守恒量，离线以 FP64 重建速度，不依赖保存的物性缓存。

**这是两套完整差分／滤波组合的比较，不是只改变导数公式的单因素对照。**
GPU 主循环对紧致模式调用 `compact_filter_state_gpu`，显式模式调用 `apply_explicit_filter_gpu`。
不能把所有谱差都归结为导数格式本身。

FP32 存储时钟在两帧分别为 8.748099327、9.997412682。FP64 时钟接近名义值。
但 FP32 实际存储步长为 0.000250000011874，其乘以 40,000 为 10.000000475，而不是 9.997412682。
后者主要反映时钟逐次累加舍入，不能解释为自动无强迫 TGV 少推进了 0.002587 个时间单位。
本次按相同步号比较，不用只有两帧的场进行伪共同时间插值。

## 3. 普通速度谱与密度加权动能谱

### 3.1 定义

先定义瞬时体积平均及速度偏差：

$$
\overline{u_i}^{\,V}=N^{-3}\sum_{\boldsymbol{x}}u_i(\boldsymbol{x}),
\qquad u_i'=u_i-\overline{u_i}^{\,V}.
$$

采用前向归一化的离散 Fourier 系数：

$$
\widehat{u_i'}(\boldsymbol{n})=N^{-3}\sum_{\boldsymbol{j}}
u_i'(\boldsymbol{j})\exp(-2\pi\mathrm{i}\boldsymbol{n}\cdot\boldsymbol{j}/N).
$$

整数中心球壳 $S_k$ 满足 $k-1/2\leq |\boldsymbol{n}|<k+1/2$。普通速度谱和真实动能对应的谱分别定义为

$$
E_u(k)=\frac12\sum_{\boldsymbol{n}\in S_k}\sum_i|\widehat{u_i'}|^2,
\qquad
E_\rho(k)=\frac12\sum_{\boldsymbol{n}\in S_k}\sum_i|\widehat{\sqrt{\rho}u_i}|^2.
$$

第二个谱不减去均值，因此

$$
\sum_k E_u(k)=\frac12\langle |\boldsymbol{u}'|^2\rangle_V=K_u,
\qquad
\sum_k E_\rho(k)=\frac12\langle\rho |\boldsymbol{u}|^2\rangle_V=K_\rho.
$$

这不是将密度加权谱再除以平均密度，也不是 Favre 脉动能谱。压缩流文献中可用速度与 $\sqrt{\rho}\boldsymbol{u}$ 分别研究速度分布和动能分布，参见 [San 与 Maulik，2018，§3.2–3.3](https://doi.org/10.5194/npg-25-457-2018)。本次归一化由上面的离散恒等式定义。

### 3.2 实现与图像

实现：`scripts/gpu_validation/analyze_native_precision_tgv_spectra.py` 中的 `read_checkpoint`、`spectral_grid`、`forward_velocity`、`shell_sum`、`density_weighted_spectrum`。
实 FFT 仅保存一个方向的半谱，非零非 Nyquist 模态权重为 2，零模态与偶数网格 Nyquist 模态权重为 1。
球壳使用模态能量求和，不除以模态数，也不再次乘 $4\pi k^2$。FFT 轴与归一化约定参考 [SciPy 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfftn.html)。

所有球壳包括角部模态都保存到 CSV。能谱主图显示壳中心 $2\leq k\leq128$，避免接近对称零值的第一壳拉伸纵轴。
最后一壳跨过 Nyquist 内切球边缘，不能将其解释为完整各向同性采样球壳；$k>128$ 角部模态不用于谱斜率解释。
这里是瞬时球壳归约，不预设流动各向同性，不进行惯性区斜率拟合。

![普通速度能谱，名义时间10](../tests/gpu_validation/out/native_fp32_tgv_spectra_20261028/figures/velocity_energy_step40000.jpeg)

![密度加权动能谱，名义时间10](../tests/gpu_validation/out/native_fp32_tgv_spectra_20261028/figures/density_weighted_energy_step40000.jpeg)

绘图入口为 `scripts/gpu_validation/plot_native_precision_tgv_spectra.py::plot_all`。两帧的 JPEG/EPS 均已输出。

### 3.3 绝对量与高波数尾部

| 第40,000步 | 保存场平均密度 | $K_u$ | $K_\rho$ | $64\leq k\leq128$ 占普通速度能量比例 |
|---|---:|---:|---:|---:|
| 显式 FP64 | 1.00000000 | 0.07368255 | 0.07363661 | 0.001793% |
| 显式 FP32 | 1.00118464 | 0.07369160 | 0.07373204 | 0.001797% |
| 紧致 FP64 | 1.00000000 | 0.07437736 | 0.07433083 | 0.085284% |
| 紧致 FP32 | 1.22305024 | 0.07463094 | 0.09119790 | 0.123042% |

显式精度配对的谱相近。紧致 FP32 的总速度能量接近 FP64，却在高波数段保留了更多能量。
密度加权谱的变化更大，与其保存场平均密度增加相一致。不能仅展示归一化谱形或普通速度谱，而遗漏质量和真实动能的漂移。

在 FP64 两种格式之间，紧致的 $64\leq k\leq128$ 能量约为显式的 **48.01 倍**，但该波段占紧致总速度能量仅约 **0.0853%**。
这个大倍数主要描述小能量尾部，不能说成总能量增加 48 倍，也不能据此单独认定紧致更准确。

第一壳在末帧的能量为：显式 FP64 2.61e-30、显式 FP32 7.27e-13、紧致 FP64 5.31e-27、紧致 FP32 3.86e-9。
这解释了主图为何单独限制显示窗口；该壳数据没有被删除，也没有被作为物理反向级联的证据。

## 4. 两点空间速度相关函数

### 4.1 定义与实现

分别保留速度分量 $i\in\{u,v,w\}$ 和分离方向 $a\in\{x,y,z\}$：

$$
R_{ii}^{(a)}(r)=\langle u_i'(\boldsymbol{x})u_i'(\boldsymbol{x}+r\boldsymbol{e}_a)\rangle_V,
\qquad C_{ii}^{(a)}(r)=R_{ii}^{(a)}(r)/R_{ii}^{(a)}(0).
$$

位移按周期回绕。该量是单帧的空间平均，不是时间平均或系综平均。
不先把三个方向合并为一条“各向同性相关曲线”。若分量方差为零，归一化相关函数记录为 NaN，而不是伪造为 0 或 1。

实现为 `velocity_diagnostics`，由模态功率沿横向波数归约，再做一维逆变换。
独立小网格检查逐项与直接周期移位乘积的空间平均比较。
CSV 保存未再归一化的 $R$（沿原算例的无量纲速度约定）和归一化 $C$；图横轴为 $r_a/L_{\mathrm{box}}$，$L_{\mathrm{box}}=2\pi L_{\mathrm{ref}}$。

![x分离方向三速度分量相关，名义时间10](../tests/gpu_validation/out/native_fp32_tgv_spectra_20261028/figures/correlation_axis_x_step40000.jpeg)

y、z 分离方向另有独立图片，所有九个组合均保存在每帧的 `*_correlations.csv`。

### 4.2 结果解释

第40,000步，显式 FP32／FP64 的归一化相关函数最大绝对差为 **2.1661e-4**，紧致为 **0.0222150**。
这表明归一化后的主要空间相关尺度仍比点态速度差更接近。
但归一化会消去整体幅值变化，不能用相关曲线接近来否定密度加权动能漂移。

FP64 显式／紧致的速度场相对 L2 差为 **35.53%**，而普通速度总能量仅相差 **0.9430%**，相关函数最大绝对差为 **0.03528**。
总能量和空间相关接近，不代表瞬时涡结构逐点一致。没有同时间高分辨率参考场，本次不把任一 FP64 场视为另一格式的真解。

## 5. 差值能谱与归一化实交叉谱

定义差值能谱为

$$
E_{\Delta u}(k)=\frac12\sum_{S_k}\sum_i|\widehat{u'_{i,b}}-\widehat{u'_{i,a}}|^2.
$$

定义谱对齐度为

$$
\gamma_u(k)=\frac{\operatorname{Re}\sum_{S_k,i}\widehat{u'_{i,a}}\widehat{u'_{i,b}}^{\,*}}
{\sqrt{\sum_{S_k,i}|\widehat{u'_{i,a}}|^2\sum_{S_k,i}|\widehat{u'_{i,b}}|^2}}.
$$

这是单帧、壳内归约的归一化实交叉谱，范围为 $[-1,1]$，不是多样本估计的平方相干函数。
纯正幅值缩放可以保持 $\gamma=1$，同时具有非零差值能谱，因此两个量联合报告。
实现为 `compare_transforms`；差谱积分通过 Parseval 给出相对速度 L2 差。

![两种格式的跨精度差值谱与谱对齐度](../tests/gpu_validation/out/native_fp32_tgv_spectra_20261028/figures/precision_error_and_alignment_step40000.jpeg)

下表按波段先归约交叉能量和各自能量，再计算谱对齐度，不是简单平均每个壳的 $\gamma$。

| 第40,000步 FP32／同格式FP64 | $k=1..7$ 能量比 | $k=8..31$ 能量比 | $k=32..63$ 能量比 | $k=64..128$ 能量比 | $k=64..128$ 谱对齐度 |
|---|---:|---:|---:|---:|---:|
| 显式 | 0.999916 | 1.000483 | 1.001789 | 1.002253 | 0.999957 |
| 紧致 | 0.987082 | 1.020677 | 1.287034 | 1.447652 | 0.486289 |

紧致 FP32 在高波数段既有能量幅值变化，也有明显的模态对齐损失，不只是统一乘上一个幅值系数。
两帧之间，其速度相对 L2 差从 7.18% 增长到 16.40%；本次只有两个保存时刻，不以此建立误差增长定律，也不据此判定唯一误差来源。

## 6. 分析正确性与后续定位

5 项小网格检查全部通过：单 Fourier 模态、直接移位相关、Nyquist 不重复加权、密度加权动能积分、幅值缩放下的交叉谱关系。
8 份真实检查点的两类 Parseval 积分最大绝对差为 **2.78e-17**。这验证了本次谱归一化，不是新的物理精度门槛。

目前证据支持优先审查紧致 FP32 的质量和能量漂移，而不是只靠归一化统计曲线判断可用性。
下一步若定位原因，应分开冻结状态上的紧致导数、三方向紧致滤波常数模态保持和实际 RK 更新，逐项量化全域守恒增量。
本轮没有修改这些算法，没有调小步长、增强耗散、替换滤波，也没有重新提交长算例。

## 7. 文件入口

- 分析：`scripts/gpu_validation/analyze_native_precision_tgv_spectra.py`。
- 作图：`scripts/gpu_validation/plot_native_precision_tgv_spectra.py`。
- 小网格验证：`tests/gpu_validation/test_native_precision_tgv_spectra.py`。
- 结果：`tests/gpu_validation/out/native_fp32_tgv_spectra_20261028/report.json`。
- 图片：同目录 `figures/`，12 组 JPEG/EPS。
- 数据：每个时刻、每个格式／精度组合都有 `*_spectra.csv`、`*_correlations.csv` 与小型 NPZ；交叉谱另有配对 NPZ。

运行：

```bash
python -m pytest -q tests/gpu_validation/test_native_precision_tgv_spectra.py
python scripts/gpu_validation/analyze_native_precision_tgv_spectra.py
python scripts/gpu_validation/plot_native_precision_tgv_spectra.py
```

源和输出目录在两个脚本顶部手动指定。所有离线计算使用 FP64，不反馈到求解器。
