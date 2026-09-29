# ASTR 部署输入数据字典

日期：2026-09-29。适用于当前工作区读取实现；不是任意历史版本的兼容承诺。
本文件记录现有格式，不新增求解器格式版本。数据生产方应随运行包保存
生成器版本、单位、网格索引、参考量和文件校验值。

## 网格

`lreadgrid=t` 时读取主输入 gridfile 指定的 HDF5 文件，坐标数据集为
`x`、`y`、`z`，三个数组尺寸相同。Quickstart 平板生成的全局节点采用
Fortran 逻辑顺序 `(i,j,k)`，范围为 `0:ia,0:ja,0:ka`；h5py 中写出顺序为
`(ka+1,ja+1,ia+1)`。生成器先构造 ijk 数组再 transpose(2,1,0)。
这是该三维全局网格文件的约定，不应套用到监测文件或二维切片。

坐标使用当前算例单位：无量纲平板为参考长度归一化坐标，AIR5 有量纲示例
为米。应检查坐标对应、正 Jacobian、网格度量及边界法向，而不是只核对尺寸。
依据：src/readwrite.F90 的 readgrid；examples/GPU_Quickstart/prepare.py。

## 外部初场与重启的区别

以下为 lrestart=f 时 ninit=1/2/3 的现有读取行为，不是任意模型的 GPU
准入保证。lrestart=t 时读取 checkpoint，优先于这些初场选择。

| ninit | 固定文件名 | 实际读取字段 | 初始化处理 |
|---|---|---|---|
| 1 | datin/flowini1d.h5 | ro、u1、t；num_species>0 时还读 sp001 等 | 沿 x 的一维状态复制到所有 j/k，u2/u3 置零，重建压力 |
| 2 | datin/flowini2d.h5 | ro、u1、u2、t | x-y 二维状态沿 z 复制，u3 置零，重建压力 |
| 3 | datin/flowini3d.h5 | ro、u1、u2、u3、t | 三维状态，按当前状态方程重建压力 |

数据集名称 t 是温度，不是时间。二维数组在 h5py 中为 `(ja+1,ia+1)`，
三维为 `(ka+1,ja+1,ia+1)`，一维为 `(ia+1,)`。提供原始变量而非守恒动量。
文件中的 p 和 time 不用于上述初场恢复；初始化步号和时间重新置零。
这些读取器不读取 Tv，二维/三维读取器也不从该文件读取组分场，不能用其
代替 AIR5 完整初场或精确重启。AIR5 应使用已准入的专用初始化或完整 checkpoint。

数组按现有全局索引读取和分块，不执行网格到网格的任意插值。单位与当前
算例一致，不因 HDF5 文件扩展名而自动无量纲化。依据：src/readwrite.F90 的
readflowini1d/2d/3d、src/initialisation.F90 的 flowinit。

## 单组分静态剖面

文件固定为 `datin/inlet.prof`。当前 j 向读取约定如下：

| 行 | 内容 |
|---|---|
| 1 | 说明文字，可包含精确标记 density=provided、pressure=provided |
| 2 | 说明行 |
| 3 | nomi_thick、disp_thick、mome_thick、fric_velocity 四个实数 |
| 4 | 列说明 |
| 5 起 | ja+1 行，依次对应全局 j=0 到 ja，无坐标列 |

常规数据列为 `rho u v T`；出现 pressure=provided 时改为 `rho u v T p`。
不要在首列添加 y。读取后第三速度分量设为零。各量与算例单位及参考量一致。
第 3 行依次表示名义厚度、位移厚度、动量厚度和摩擦速度；启动模板数值
只是输入示范，不是从已收敛流场测得的结果。

对当前无量纲路径，标记含义如下：

| density=provided | pressure=provided | 读取后的处理 |
|---|---|---|
| 无 | 无 | 用自由来流压力与给定 T 重建 rho，文件 rho 不作为最终密度 |
| 有 | 无 | 保留给定 rho/T，检查其状态方程压力与自由来流压力一致 |
| 无 | 有 | 用给定 p/T 重建 rho |
| 有 | 有 | 保留 rho/p/T 并核验状态方程一致性 |

不要将这些无量纲分支推广为任意有量纲或反应流剖面合同。
依据：src/initialisation.F90 的 inletprofile；src/parallel.F90 的 preadprofile。

## 单组分动态入口

当前受支持的无量纲 `11,intp` 路径使用 `inflow/isliceNNNNN.h5`，五位编号。
首次从 00000 至 00003 装入四个时间平面。恢复时由 auxiliary.txt 的
ninflowslice 决定四片窗口，不能任意重新编号或重置时间。

| 数据集 | 当前无量纲路径的含义 |
|---|---|
| time | 切片物理时间，使用与求解器相同的时间单位 |
| ro | 相对于 rho_prof 的密度增量 |
| u1 | 相对于第一速度剖面的增量 |
| u2 | 相对于第二速度剖面的增量 |
| u3 | 第三速度增量，当前静态剖面的第三速度为零 |
| t | 相对于 tmp_prof 的温度增量 |

time 为标量，五个场在 h5py 中的全局形状为 `(ka+1,ja+1)`，对应 Fortran
`(j,k)` 面。增量可以有负值；插值并叠加后的密度和温度必须为正。
不读取压力场，也不读取 AIR5 组分或 Tv。时间戳须严格递增且均匀间隔，
文件需覆盖启动、各 RK 阶段、换片及重启所需窗口，不自动周期复用缺失切片。

现有行为：四点三次时间插值后加到静态剖面；若叠加后的流向速度小于零，
该速度恢复为静态剖面值。这是已有实现，不是无条件适用的入口回流模型。
complete_state 模式由 rho/T 计算压力；mach_pressure 模式还会在亚声速流入
时用内部外推压力重建密度。不得将完整瞬时场直接当增量，否则会重复叠加基态。

依据：src/bc.F90 的 inflowintp；src_gpu/inflow_timeseries_gpu.cuf；
src_gpu/boundary_gpu.cuf 的 s1_bl_dynamic_inflow_x_kernel。
开发分支切片生成器仅用于测试，不作为已发展的湍流入口数据源。

## AIR5 平板输入（实验范围）

`datin/air5_hbl_domain.dat` 的 Quickstart 格式：第一行
`air5_hbl_domain_v1`，第二行域长 Lx、Ly、Lz，单位米。

`datin/air5_hbl_profile.dat` 至少包含两行数据，允许空行及 # 注释。
必须提供 `# x_origin=正数`，单位米。数据按 y 严格递增，13 列依次为：

```text
y rho u v w p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO
```

y 为米，rho 为 kg/m^3，速度为 m/s，p 为 Pa，T/Tv 为 K，Y 为质量分数。
这里是完整状态，不是上节的动态增量格式。组分须非负、和为 1，rho/p/T
须满足当前 AIR5 混合气体状态方程。读取器同时检查状态转换可行性。
不可调整字段顺序，也不可将第 13 列误当作闭合 N2；文件中包含完整五组分。

依据：src/chemistry_boundary_state.F90 的 read_air5_hbl_profile 和
validate_air5_hbl_profile_row；Quickstart 的 air5_flatplate_data。
本节仅覆盖现有平板启动包，不宣称入射激波状态、任意反应初场及动态 AIR5
入口均可使用同一格式。

### AIR5 可选二维完整初场

AIR5 平板专用初始化可额外读取 `datin/air5_hbl_initial_field.dat`，并非
将 ninit 改成 2。首个非空、非 # 注释行是 nx、ny 两个节点数，均至少为 2。
随后 nx*ny 行为 `x y q(1:11)`，每行 13 个实数。外层依次遍历 x，内层
遍历 y；所有 x 截面须使用同一套严格递增 y 坐标，x 也须严格递增。
坐标以米计，q 布局及单位见下节入射激波表，不能填入质量分数或 primitive 速度。

初始化按计算节点的 x/y 取样，再转换为原始变量，z 不作为输入坐标。
该文件不是三维湍流入口或补偿重启文件。即使它存在，入口剖面和边界配置
仍必须配套；没有该文件时，专用平板初始化沿流向和展向复制入口剖面。
依据：src/chemistry_boundary_state.F90 的 read_air5_hbl_initial_field，
src/initialisation.F90 的 air5hblini。

## AIR5 激波状态文件（实验范围）

以下文件由固定 AIR5 模型读取，均为有量纲状态。与单组分 OpenSBLI 的
namelist、静态入口 13 列剖面及动态增量切片不同，不能互相替换。

### 正激波

`datin/air5_normal_shock_states.dat` 必须恰有两行数值，依次为左、右状态，
每行 12 列：

```text
rho u p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO Ev q5
```

允许空行及 # 注释。可选元数据 `# outlet_pressure_pa=数值` 指定出口目标
压力，若提供则需为模型允许范围内的压力，不得重复。未提供时内部目标值为
零，边界不进入该可选压力出口分支。其含义不是物理出口压力为零。

rho、u、p、T/Tv 分别使用 kg/m^3、m/s、Pa、K。横向速度在此格式中固定为零。
Ev 为单位体积振动能，q5 为单位体积总能量，均为 J/m^3。q5 包括动能、
平动/转动内能、振动能和当前机制的生成能；不是单组分 p/(gamma-1) 加动能。
读取器重建守恒状态并检查 Ev、q5、压力的一致性。初始化以
`x <= 0.5*ref_len` 取第一行，其余取第二行，文件本身没有可自由填写的激波位置列。

依据：src/initialisation.F90 的 read_air5normalshock_states 和
air5normalshockini；src/chemistry_boundary.F90 的压力出口分支。

### 入射激波

`datin/air5_incident_shock.dat` 固定四行：

```text
air5_incident_shock_v1
x_top y_top nx ny nz
q_upstream(1:11)
q_downstream(1:11)
```

最后两行各为 11 个实数，上述括号文字只是布局说明，不能写入文件。
x_top/y_top 为米；法向量无量纲，必须单位化，当前要求 nx>0、ny>=0、
abs(nz)<=2e-12。顶面交点须在计算域内，y_top 与域高一致。
状态布局为：

| 分量 | 含义与单位 |
|---|---|
| 1 | rho，kg/m^3 |
| 2:4 | rho*u、rho*v、rho*w，kg/(m^2 s) |
| 5 | 上述总能量 q5，J/m^3 |
| 6:10 | rho*Y_N2、rho*Y_O2、rho*Y_N、rho*Y_O、rho*Y_NO，kg/m^3 |
| 11 | Ev，J/m^3 |

文件保存完整五组分密度，不是只写四个独立质量分数；这不改变推进器的
N-1 独立组分约束。读取器检查状态可接受性、正法向速度、下游密度/压力
增大以及两侧法向守恒通量一致。上游守恒状态还须与入口剖面外缘匹配。
这些是输入一致性检查，不等于长时反应 SBLI 已完成物理验证。

依据：src/chemistry_boundary_state.F90 的 read_air5_incident_shock；
src/chemistry_boundary.F90 的 configure_air5_hbl_boundary；
src/chemistry_core.F90 的 chemistry_state_layout；
src/chemistry_properties.F90 的 air5_primitive_to_conservative。

## 单组分 OpenSBLI 固定工况

环境变量 `ASTR_CONSERVATIVE_BOUNDARY_FILE` 指向 Fortran namelist，组名为
`conservative_boundary`，字段为 schema=1、split_x、q_left(5)、q_right(5)。
两组 q 为无量纲的 rho、三个动量及总能量，不是 primitive 速度/温度。
文件尾仅允许空行或 ! 注释，不允许追加第二个配置组。解析通过之后仍有
固定工况的 Mach、Re、边界、格式和目标状态检查，不能当作通用激波边界接口。
精确参数应取匹配版本的已有 OpenSBLI 运行包，不以 AIR5 状态或任意斜激波值替代。
依据：src/conservative_boundary_config.F90、src/conservative_boundary_runtime.F90。

## 已核验启动组合

下表限定已有 Quickstart 短启动记录，不是全部支持条件的笛卡尔积。
共同设置：FP64、RK3、显式同步、pinned 主机中转、无湍流模型。
CPU NP=1、GPU NP=1 和 GPU NP=2 的 2,1,1 分解已在该记录中检查。

| 模板 | 对流/扩散 | 边界 x-/x+/y-/y+/z-/z+ | 入口与滤波 |
|---|---|---|---|
| tgv | 643e/643e | 1/1/1/1/1/1 | 内置初场，十阶滤波 |
| channel | 643e/643e | 1/1/41/41/1/1 | 固定驱动力，十阶滤波 |
| flatplate | 543e/643e，recon=3，lchardecomp=f | 11/21/41/51/1/1 | prof、complete_state，十阶滤波 |
| air5_tgv | 以随包输入为准 | 1/1/1/1/1/1 | 固定 AIR5 双温，滤波关闭 |
| air5_flatplate | 以随包输入为准 | 11/50/41/51/1/1 | AIR5 专用完整剖面，prescribed 顶面、补偿、十阶滤波 |

参数及证据见 examples/GPU_Quickstart/README.md。四套开启滤波模板的
full/scalar 短窗结果一致不等于长期物理验收。AIR5 始终为显式启用的实验范围。
动态 intp、CURVE、其他壁面编号、device-aware 和混合精度不由此表放行，
需查各自配置检查及验证记录。源码拒绝某组合时，不通过删去检查来扩大范围。

## 数据交付记录

每套运行数据记录源码版本、生成器版本、字段约定、量纲与参考量、全局网格
及拓扑、文件清单、时间覆盖范围和校验值。已有格式没有统一版本字段时，
将这些信息放在旁附清单，不擅自给顺序读取的文件增加头行。
checkpoint 和统计恢复规则见 USER_GUIDE 10.4；主输入和 controller 见
USER_GUIDE 对应章节。其他外部初场及未列出的专用数据仍需逐项补齐合同。
