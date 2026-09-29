# ASTR GPU 使用说明

文档版次：2026-09-29。适用于随本说明交付的 CUDA Fortran/MPI 源码。
本说明覆盖部署、输入配置、算例启动、边界条件、可选功能、输出与恢复。
基础配置面向非反应流；AIR5 和受限性能选项的使用条件在相应章节中单独列明。
完成部署和短时数值对照不等于目标物理工况已经达到统计收敛。

## 目录

1. 能力与适用范围
2. 部署与构建
3. 算例目录与启动
4. 主输入文件
5. 控制文件与时间步
6. 边界条件调用
7. 网格与入口数据
8. MPI 与 GPU 绑定
9. 数值与性能可选项
10. 输出、统计与重启
11. 算例选择与专用处理
12. AIR5 使用条件与配置
13. 故障处理
14. 部署验收及复现记录
15. 源码与配套文档索引

## 1. 能力与适用范围

ASTR GPU 面向 NVIDIA GPU 上的结构化可压缩流动显式求解。
主状态在设备常驻，MPI 交换子域 halo；文件输出和 checkpoint 仍由主机参与。

| 项目 | 当前范围 |
|---|---|
| 时间推进 | GPU RK3 |
| 空间离散 | 六阶显式中心差分；受限一阶 Steger-Warming、WENO7、MP7 |
| 激波处理 | Ducros 及已验证的选择性 Roe 特征重构组合 |
| 滤波 | 十阶显式滤波及物理边界闭合 |
| 网格 | Cartesian；静态单块三维结构化曲线网格 |
| 物性 | 单组分理想气体及 Sutherland 黏度；可选固定 AIR5 双温物性与化学模型 |
| 并行 | MPI 多子域；生产建议一 rank 一 GPU |
| 精度 | FP64 默认；指定配置可将部分中间量改用 FP32，条件见第 9 节 |

GPU 不承诺紧致格式、RK4、RANS/LES、多块非共形、动网格、通用浸入边界、
任意 Cantera 机制、任意方向曲面 NSCBC 或 HIP/DCU 后端。
CPU 源码中存在某功能不表示 GPU 已支持。
AIR5 长时真实反应 SBLI、真实湍流入口统计和外部数据库验证尚不能统一宣告完成。

“支持”需同时限定物理模型、网格、边界方向、格式、精度、通信与验证窗口。
完整声明见 [部署能力边界](documents/ASTR_RELEASE_PROD_CAPABILITY_SCOPE.md)。

## 2. 部署与构建

### 2.1 环境要求

- Linux、CMake、Fortran 编译器。CUDA 构建使用 NVIDIA HPC SDK 的 nvfortran。
- MPI Fortran 开发环境，与实际运行 MPI 一致。
- HDF5 Fortran 和 HL 库；建议使用与编译器及 MPI 匹配的并行 HDF5。
- GPU 节点有兼容驱动和足够显存。普通 gfortran 不能编译 CUDA Fortran。

本地构建隔离检查使用 NVHPC 26.1、配套 HPC-X 和 HDF5 1.14.6。
这是已用环境，不表示其他版本全部支持或全部不支持。
Fortran 的 .mod 文件存在编译器兼容性要求，不能只看 HDF5 版本号。
关闭场输出也不会消除 HDF5 动态库依赖。

环境示意，路径应对应实际安装：

```bash
export NVHPC_ROOT=/path/to/nvhpc
export HDF5_ROOT=/path/to/parallel-hdf5
export PATH="$NVHPC_ROOT/compilers/bin:$NVHPC_ROOT/comm_libs/hpcx/bin:$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:$NVHPC_ROOT/compilers/lib:${LD_LIBRARY_PATH:-}"
command -v nvfortran
command -v mpifort
nvfortran --version
mpifort --showme
h5pfc -showconfig
```

MPI 初始化应通过匹配的启动器进行。HPC-X 额外环境采用平台模块或厂商初始化脚本，
不要混入另一套 MPI 库。计算节点与登录节点的动态库可见性需分别检查。

### 2.2 统一构建入口

只通过根 CMakeLists.txt 构建。旧 Makefile 和 build_nvhpc*.sh 不是当前 GPU 发布权威入口。

```bash
export ROOT=/path/to/astr
cmake -S "$ROOT" -B "$ROOT/build_prod" \
  -DCMAKE_Fortran_COMPILER=nvfortran \
  -DCMAKE_BUILD_TYPE=Release \
  -DHDF5_ROOT="$HDF5_ROOT" \
  -DASTR_WITH_CUDA=ON \
  -DASTR_WITH_AIR5_CHEMISTRY=OFF \
  -DCHEMISTRY=OFF \
  -DBUILD_TESTING=OFF
cmake --build "$ROOT/build_prod" --target astr -j 2
ldd "$ROOT/build_prod/bin/astr"
```

输出为 build_prod/bin/astr；ldd 不应存在 not found。
更换编译器或工具链使用独立构建目录，不混用 CMakeCache 和模块文件。
登录节点可以编译；无 GPU 编译时，应通过目标平台已核验配置指定正确的
NVHPC GPU 架构，并检查最终编译命令，不直接复用本地 GPU 目标。

| CMake 选项 | 默认 | 含义 |
|---|---|---|
| ASTR_WITH_CUDA | OFF | 编译 CUDA 支持，不是运行时 GPU 开关 |
| ASTR_WITH_AIR5_CHEMISTRY | OFF | 固定 AIR5，与遗留 CHEMISTRY 互斥 |
| CHEMISTRY | OFF | 遗留 Cantera CPU 接口，不等于任意机制 GPU 支持 |
| BUILD_TESTING | ON | 新增验证命令和 probe；生产包设 OFF |
| ASTR_BUILD_MPI_COMPLETION_TRACE | OFF | MPI 诊断工具，要求 BUILD_TESTING=ON |
| ASTR_S0A_MAXREGCOUNT | 128 | 特定通量核寄存器上限，不建议任意调整 |
| ASTR_AIR5_MAXREGCOUNT | 128 | 特定 AIR5 核寄存器上限，不是精度开关 |

纯 CPU 使用独立构建目录并设置 ASTR_WITH_CUDA=OFF。
AIR5 使用 ASTR_WITH_AIR5_CHEMISTRY=ON、CHEMISTRY=OFF。
发布基础配置及 `examples/GPU_Quickstart/build.sh` 均默认关闭 AIR5。
需要实验性 AIR5 功能时，显式运行
`AIR5=ON BUILD_DIR="$PWD/build_air5" bash examples/GPU_Quickstart/build.sh`，
启动 AIR5 示例时设置 `EXE="$PWD/build_air5/bin/astr"`。
编译启用只表示包含该实现，不代表反应流生产物理验收完成。

BUILD_TESTING=OFF 已支持无 tests/ 的源码包，但仍需 examples/ 和
user_define_module/userdefine.F90。关闭测试不关闭 CUDA-aware MPI 检测。
未指定安装前缀时，默认安装到构建目录下的 opt。可在配置时传入
`-DCMAKE_INSTALL_PREFIX=/absolute/install/path`，或在构建完成后运行
`cmake --install "$ROOT/build_prod" --prefix /absolute/install/path`。
可执行文件安装到 bin，原 CPU 示例输入安装到 examples，二维和三维 TGV
使用不同目录。GPU_Quickstart 启动包仍从源码包的 examples/GPU_Quickstart
使用，通过 EXE 指向安装后的 bin/astr；不会由旧 examples 安装规则自动安装。
安装不是运行库打包，目标环境仍需匹配 MPI、HDF5 和 NVHPC 运行库。

## 3. 算例目录与启动

```text
case_name/
  datin/
    input.dat          主输入，可换名称
    controller         固定名称
    grid.h5            读网格时需要
    monitor.dat        可选监测点
    slice.dat          可选切片设置
    ...                算例专用入口、初场和域参数
  outdat/              checkpoint 和其他输出
  bakup/               备份
  monitor/             监测结果
  islice/ jslice/ kslice/
  run.log              启动命令重定向日志
```

程序建立部分输出目录。相对路径通常相对于进程工作目录，不是二进制目录。
必须进入算例目录再启动，两个任务不得共用输出目录。

```bash
export ROOT=/path/to/astr
export CASE=/absolute/path/to/case_name
cd "$CASE"
export OMP_NUM_THREADS=1
export ASTR_FORCE_MPI_TOPOLOGY=1,1,1
export ASTR_GPU_PRECISION_MODE=fp64
export ASTR_GPU_FILTER_WORKSPACE=scalar
export ASTR_GPU_SYNC_MODE=explicit
export ASTR_GPU_HALO_TRANSPORT=pinned
mpirun -np 1 "$ROOT/build_prod/bin/astr" run datin/input.dat >run.log 2>&1
```

需预先提供有效输入，并设置 usegpu=t。纯 CPU 用 usegpu=f，清除不适用 GPU 环境变量，
尤其 device-aware 会在 MPI_Init 前触发设备预绑定。
主命令为 run、test、pp。生产版保留原 CPU 内置测试，但新增
bcad/bciv/bcst/bcrh/bcgc 要求 BUILD_TESTING=ON，否则明确报错。
pp 是遗留后处理入口，不表示已实现 ParaView 原位可视化。

## 4. 主输入文件

### 4.1 固定行序

readinput 使用固定跳行和 list-directed 读取，**不是键值配置**。
注释标题不驱动解析，不能任意插入空行或重排字段。
沿用模板，采用 LF、无 BOM，逻辑值 t/f，双精度实数如 1.d-3。
部分字符串处理了 CRLF，但不能保证所有路径都能容忍回车符。

### 4.2 字段读取顺序

| 顺序 | 字段 | 说明 |
|---:|---|---|
| 1 | flowtype | 区分大小写，控制初场、部分源项和 GPU 配置检查，不能随意改名 |
| 2 | ia,ja,ka | 全局索引上限/分段数；物理节点常为 0:ia 等，不直接等于 HDF5 数组尺寸 |
| 3 | lihomo,ljhomo,lkhomo | 齐次/周期方向，与边界和平均方向一致 |
| 4 | 九个逻辑开关 | 见下表，旧示例可能缺少 usegpu |
| 5 | lrestart | 从 outdat 恢复 |
| 6 | alfa_filter,kcutoff | 遗留滤波/频谱参数；不是当前显式滤波任意强度开关 |
| 7 | 参考量 | 无量纲：ref_tem,Re,Mach；有量纲：ref_tem,ref_vel,ref_len,ref_den |
| 8 | conschm,difschm,rkscheme | 对流、扩散、时间格式；遗留 Cantera 构建还读取 odetype |
| 9 | recon_schem,lchardecomp,bfacmpld,shkcrt | 重构方法、特征分解及相关参数，必须与所选对流格式及算例配置一致 |
| 10 | num_species[,Schmidt...] | 基础单组分填 0；AIR5 按其布局填写 |
| 11 | turbmode,iomode | GPU 基础通常 none,h；h 为 HDF5 |
| 12 | 六行 bctype | 六面顺序固定 |
| 13 | ninit | 0 内置初场；1/2/3 读取相应维数初场 |
| 14 | 六个 sponge 宽度[,spg_def] | 六面顺序，0 不设置；非零值会启用边界缓冲区，并触发相应 GPU 配置检查 |
| 15 | gridfile | 即使 lreadgrid=f，模板中这一行仍保留 |

开关顺序：

```text
nondimen,diffterm,lfilter,lreadgrid,lfftk,limmbou,ltimrpt,lcomb,usegpu
```

| 开关 | 含义 |
|---|---|
| nondimen | 无量纲设置，影响参考量与物理解释 |
| diffterm | 黏性/扩散 |
| lfilter | 滤波 |
| lreadgrid | 外部网格，否则按算例生成 |
| lfftk | 遗留 Fourier 相关开关，GPU 基础保持 false |
| limmbou | 浸入边界，当前 GPU 不支持 |
| ltimrpt | 时间开销报告 |
| lcomb | 化学布局/路径，还需编译选项与 flowtype 配合 |
| usegpu | 运行时 GPU 开关 |

示例标题可能写 lfftz，源码实际变量为 lfftk。旧八字段输入最后一项是 lcomb，
不是 usegpu；此时 GPU 开关保持 false。输入文件表面包含 GPU 参数并不足以启用 GPU，
要核对日志实际路径。

### 4.3 显式周期 TGV 模板

以下为显式周期 TGV 输入模板。复制时保留头部、字段顺序和空行。
完整生成与启动方法见 [GPU Quickstart](examples/GPU_Quickstart/README.md)。

```text
########################################################################
# ASTR explicit GPU TGV
########################################################################

# flowtype
tgv

# ia,ja,ka
32,32,32

# lihomo,ljhomo,lkhomo
t,t,t

# nondimen,diffterm,lfilter,lreadgrid,lfftk,limmbou,ltimrpt,lcomb,usegpu
t,t,t,f,f,f,t,f,t

# lrestart
f

# alfa_filter,kcutoff
0.49d0,48

# ref_tem,reynolds,mach
273.15d0,1600.d0,0.1d0

# conschm,difschm,rkscheme
643e,643e,rk3

# recon_schem,lchardecomp,bfacmpld,shkcrt
3,f,0.3d0,0.05d0

# num_species
0

# turbmode,iomode
none,h

# bctype: x-min,x-max,y-min,y-max,z-min,z-max
1
1
1
1
1
1

# ninit
0

# sponge widths
0,0,0,0,0,0

# gridfile
datin/grid.h5
```

examples/ 是原 CPU 算例集合，不是 GPU 白名单。将 c 后缀改为 e 后，
必须重新确认时间步、边界和物理精度，不能认为方法完全等价。

## 5. 控制文件与时间步

datin/controller 的文档模板：

```text
############################################################
# ASTR controller
############################################################

# lwsequ,lwslic,lavg,lcracon
f,f,f,f

# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg
9,10,100,50,1,50

# deltat
1.d-3
```

| 字段 | 含义 |
|---|---|
| lwsequ | 完整场序列输出 |
| lwslic | 切片输出，需 slice.dat |
| lavg | 统计累计，不等于统计已收敛 |
| lcracon | 遗留崩溃修复/续算；可信化与 AIR5 补偿保持 false |
| maxstep | 绝对步号上限，不是追加步数或终止物理时间 |
| feqchkpt | checkpoint 与控制文件重读相关频率 |
| feqwsequ/feqslice | 场序列/切片频率 |
| feqlist | 日志/诊断频率 |
| feqavg | 统计采样频率 |
| deltat | 时间步，单位随模型设置 |

频率使用正整数，**不要填 0 关闭输出**，路径存在 mod(nstep,frequency)。
lwsequ=f、lwslic=f、lavg=f 不等于关闭 checkpoint。

主循环为 nstep<=maxstep。从 0 开始且无其他停止条件，maxstep=9 对应十次循环更新。
重启时核对 checkpoint 实际步号和相位，以日志 state_time、HDF5 time 为准。
controller 在指定步号重读，不是逐步实时监控。主输入和多数环境变量不能动态更改。
需要改 controller 时原子替换，避免读入半写文件；不要强杀进程代替完整 checkpoint 停止计划。

固定 deltat 不会自动变为 CFL 自适应步长。检查局部网格、黏性限制和化学状态。
上面的 1.d-3 仅为 TGV 示例，不能用于有量纲 AIR5 高焓边界层。

## 6. 边界条件调用

### 6.1 编号、语法和范围

六面固定顺序：x-min、x-max、y-min、y-max、z-min、z-max，每面一行。
曲线网格中是逻辑 i/j/k 面，不代表任意笛卡尔法向平面。

| 编号 | 输入形式 | 当前 GPU 范围 |
|---:|---|---|
| 0 | 0 | CPU udf_bc；无通用 GPU 用户接口，只有特定算例处理 |
| 1 | 1 | 周期，需与齐次标志一致 |
| 11 | 11,prof 或 11,intp | 受限 x-min 剖面或时序入口 |
| 12 | 12 | 受限 x-min 特征入口，不追加 turbinf |
| 21 | 21 或 21,pout | 受限出口，方向和几何检查仍生效 |
| 22 | 22 或 22,pout | 受限 NSCBC 出口 |
| 23 | 23 | CPU gcnscbc，未完成对应 GPU 移植 |
| 31 | 31 | 受限 y 向固定边界，典型 RTI |
| 41 | 41,Twall | 等温无滑移；曲线零吹吸六面及受限 y-min 吹吸 |
| 42 | 42 | x/y 绝热无滑移，z 不支持 |
| 411 | 411,xslip,Twall | y 向滑移/无滑移组合 |
| 421 | 421,xslip | y 向对应绝热组合 |
| 50 | 50 | 六面零外推 |
| 51 | 51 | 受限远场；swbli 有特殊附加数据 |
| 52 | 52 | 受限 upper-y NSCBC/非反射，不是任意六面 |
| 60 | 60 | 曲面局部几何投影对称边界 |

输入格式正确不代表对应方向、网格和物理模型均已实现。Twall、xslip、pout 与当前单位/归一化一致。
省略 pout 时读取器默认 1.d0，不意味着任意有量纲工况都可省略出口压力。
turbinf、xslip、pout 等部分量是共享配置，不能假定同一输入为不同面设置不同值
就获得独立面参数。

flowtype=swbli 的 51 行额外读取 xrhjump,angshk，不能推广到 bl 或 AIR5。
AIR5 入射激波使用独立状态文件格式。CPU intx 和 udef 入口也不是当前通用 GPU 能力。

### 6.2 典型组合

周期 TGV：六面 1，齐次 t,t,t。
槽道 x/z 周期、y 双壁：1/1/41/41/1/1，并分别填写壁温。
方向、驱动力和速度分量不会仅因改变壁面编号而自动旋转。

平板的受控边界组合示意：

```text
11,prof
21
41,1.0d0
51
1
1
```

1.0d0 只是无量纲壁温语法示意，不是推荐物理温度。仍需提供相匹配的剖面、网格和参考量，并通过 GPU 配置检查。

### 6.3 特征远场与专用边界

`ASTR_NSCBC_FARFIELD_MODE` 改变 bctype=52 的计算方法，不是性能开关。

| 取值 | 实际作用 |
|---|---|
| `compatibility`（默认） | 保留原有远场特征量修改方法；不采用下面的五个目标状态环境变量。不能将此值理解为“自动选择最合适边界”。 |
| `incoming_only` | 按局部外法向速度与声速识别传入的特征波，只将这些波向指定远场状态松弛，传出波保留内部计算值。 |
| `sbli_shock` | 使用与 incoming_only 相同的传入波处理，但目标状态在指定流向位置前后分别取激波上游和由斜激波关系计算的下游状态。 |
| `nonreflecting` | 不向给定远场状态松弛；按传入/传出特征分解，并平衡已实现的横向及黏性源项，以减少边界反射。不表示离散反射严格为零。 |

后三种模式只用于上侧逻辑 j 面（y-max）的 52 边界，且为非反应五方程系统。
曲线网格另有几何、格式和拓扑检查，不能直接用于 AIR5。

incoming_only 和 sbli_shock 读取 `ASTR_NSCBC_FARFIELD_RHO`、
`ASTR_NSCBC_FARFIELD_U`、`ASTR_NSCBC_FARFIELD_V`、
`ASTR_NSCBC_FARFIELD_W`、`ASTR_NSCBC_FARFIELD_T`。
分别指定目标密度、三个速度分量和温度；未设置项沿用算例自由来流值，
单位与算例一致，密度和温度必须为正。压力由当前理想气体状态方程计算。
sbli_shock 还读取 `ASTR_NSCBC_FARFIELD_SHOCK_X`（流向分界位置）和
`ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG`（激波角，度，严格介于 0 和 90 之间）。
角度不是壁面转角，位置和角度应在运行配置中明确记录。

`ASTR_CONSERVATIVE_BOUNDARY_FILE` 的值是配置文件路径；设置后启用
OpenSBLI 固定工况处理，并检查 Mach 数、Re、格式和边界组合。
它不是通用 SBLI 参数文件，具体固定条件见第 11 节。

## 7. 网格与入口数据

lreadgrid=t 从 gridfile 读取 x/y/z HDF5 数据集。维数、索引、共享端点及
Fortran/HDF5 排布应沿用已验证工具，不能凭 NumPy 数组外观猜测。
lreadgrid=f 根据 flowtype 生成网格，不代表域长可由输入中任意新字段指定。

曲线网格检查 Jacobian、度量、面法向和边界兼容性，先做自由流保持与短窗对照。
读入成功不等于任意曲面边界受支持。

| 初场/入口 | 用法 | 数据要求 |
|---|---|---|
| 内置 | ninit=0 | flowtype 决定初始化及可能的源项 |
| 外部初场 | ninit=1/2/3 | 对应读取例程、文件名、维数与状态定义 |
| 单组分静态剖面 | 11,prof | 剖面与网格、物性成套提供 |
| 单组分时序切片 | 11,intp | 四切片、均匀时间间隔三次插值，覆盖运行及重启时间 |
| CPU intx/udef | 遗留模式 | 不承诺对应 GPU 路径 |

当前受支持的无量纲动态入口将五变量切片作为静态剖面的增量，经过时间插值后
叠加，不能直接填入完整瞬时状态。压力按边界模式由状态方程或内部外推确定，
不是该切片的输入字段。五变量切片不直接适用 AIR5 十一变量。
网格、静态剖面、切片和 AIR5 数据的具体字段见
[部署输入数据字典](documents/ASTR_DEPLOYMENT_INPUT_CONTRACT.md)。

`ASTR_PROFILE_INFLOW_MODE` 用于已实现的 x-min、bctype=11 入口：

| 取值 | 实际作用 |
|---|---|
| `complete_state`（默认） | 从入口数据指定密度、速度、压力及组分，并核对温度与状态方程一致性。 |
| `mach_pressure` | 根据入口数据的局部法向马赫数分段处理：达到或超过 1 时指定完整状态；小于 1 时指定速度、温度及组分，压力由内部节点外推，再由状态方程计算密度。 |

mach_pressure 不表示固定 Mach 数或固定压力；当前实现要求法向速度指向域内或为零，
不支持用此选项处理入口回流。切换该值会改变边界物理条件，不应作为纯性能调整。

## 8. MPI 与 GPU 绑定

```bash
export ASTR_FORCE_MPI_TOPOLOGY=1,1,2
mpirun -np 2 /absolute/path/to/astr run datin/input.dat >run.log 2>&1
```

拓扑乘积等于 MPI rank 数，对应 i/j/k 分解。未设置时使用自动分解。
子域需满足模板和 halo 最小尺寸，不能无限增加 rank。
datin/parallel.info 记录分解信息，强制分解通过上述环境变量设置。

设备选择根据节点内 local rank 和进程可见 GPU 集合。核对绑定日志和实际进程。
生产一 rank 一卡，多 rank 共享少量 GPU 的正确性测试不能当成扩展性能。
Slurm 的典型资源意图为 ntasks=4、gpus=4、cpus-per-task=4，但具体启动方式
服从平台 MPI 集成，避免双层启动。调度器按进程屏蔽设备时，不再强行覆盖
CUDA_VISIBLE_DEVICES。多分配 CPU 核不会自动启用多线程。

ASTR_GPU_HALO_TRANSPORT：

| 值 | 含义 |
|---|---|
| pageable（默认） | GPU 将待交换数据复制到普通主机内存，由 MPI 发送；收到的数据再复制回 GPU。 |
| pinned | 仍经主机中转，但预先分配并注册固定内存缓冲区，运行中重复使用，避免反复注册主机内存。 |
| pinned-overlap | 使用固定注册的主机缓冲，在交换数据期间计算不依赖新 halo 的黏性项内部区域；不是所有计算与通信都重叠。 |
| pinned-pipeline | 将打包、GPU/主机复制、MPI 发送接收及内部计算分阶段安排，通过 CUDA stream/event 管理先后关系。仅允许至少一个方向有 MPI 分解的三维周期 TGV。 |
| device-aware | 将 GPU 缓冲区地址直接交给 MPI，求解器不再为这些交换显式安排主机中转；实际传输方式由 MPI/UCX 和硬件决定。 |

halo 是相邻子域提供的网格层，用于本子域边缘的差分计算。
pinned-overlap 的黏性计算重叠要求 diffterm=t、周期边界、超过一个 MPI rank，
且局部 im/jm/km 都不小于 6；以日志 `periodic_diffusion_active=T` 确认实际启用。
不满足条件时，选择该名称不等于发生了重叠。
pinned-pipeline 不能搭配同步模式 selective，可搭配 explicit 或 dependency。
主机缓冲注册失败时程序可能退回 pageable，应检查启动日志而非只看环境变量。

CUDA-aware MPI 不是 NCCL，也不自动证明跨节点 GPUDirect RDMA 生效。
需要构建查询、运行时支持及实际传输验证。device-aware 的 MPI_Init 前绑定
需要启动器提供可识别 local-rank 环境，未知平台不能直接照搬别处配置。

任务工作目录、可用资源和调度器启动命令由部署平台规定。
运行脚本应使用该平台允许的目录，不混用其他环境的模块或启动器配置。

## 9. 数值与性能可选项

### 9.1 数值设置

- 643e 为六阶显式路径的主要代码。
- 543e 为已有迎风路径使用的代码，还需对应 recon_schem。
- recon_schem=-1/1/3 分别用于已验证的一阶 Steger-Warming/WENO7/MP7 配置。
- lchardecomp 和 shkcrt 必须与对应重构方法匹配；AIR5 激波捕获实现不同于单组分。
- 十阶显式内点滤波采用固定系数，alfa_filter=0.49 不表示按紧致 alpha 公式调强度。
- 物理边界闭合与内点离散不同，不仅比较格式名称，还需核对边界和滤波行为。

### 9.2 运行时选项

以下“默认”指未设置环境变量时的程序行为，不等于所有算例的推荐选择。
各 rank 应使用相同配置；按表中小写值填写，不假定所有读取器都忽略大小写或空值。
可先用 `unset 变量名` 清除上次运行的设置。多数选项只在初始化时读取。

**精度：`ASTR_GPU_PRECISION_MODE`**

- `fp64`（默认）：使用双精度数值实现。
- `mixed_workspace`：主守恒状态和 RK 更新仍为 FP64，只将指定中间数组及对应计算改用 FP32。不是将整个求解器改为单精度。

`ASTR_GPU_MIXED_CANDIDATE` 指定哪一类中间量使用 FP32，每次只能选一项：

| 取值 | 改动对象与使用条件 |
|---|---|
| `flux`（默认选择） | 物理空间迎风通量工作数组。要求 conschm=543e、recon_schem=1 或 3、lchardecomp=f、三个方向周期。 |
| `derivative` | 黏性计算使用的导数工作数组。要求 diffterm=t、difschm=643e、五方程、无组分和附加模式方程、RK3；不允许需要这些导数的激波传感器或特征重构。 |
| `viscous_flux` | 黏性通量工作数组。要求 diffterm=t、difschm=643e、五方程、无组分和附加模式方程、RK3。 |
| `characteristic_flux` | 特征空间迎风通量工作数组。要求 543e、MP7（3）、lchardecomp=t、五方程 RK3、lfilter=f；仅开放代码已检查的周期无黏、x 向物理边界或特定黏性平板组合。不是任意特征边界均可用。 |

这些条件还需同时满足算例本身的 GPU 支持检查。未启用 mixed_workspace 时，
默认 flux 不改变 FP64 计算；显式指定其他三项会报错。
AIR5 不在这里的混合精度支持范围。滤波保持 FP64，不提供 FP32 滤波选项。
混合精度改变舍入误差，应先与同工况 FP64 比较，而非直接用于未检查的长时计算。

**滤波存储：`ASTR_GPU_FILTER_WORKSPACE`**

| 取值 | 实际作用 |
|---|---|
| `full` | 为全部守恒分量保留滤波工作数组，原状态和工作数组交替保存各方向结果。显式设置后恢复原存储方式。 |
| `scalar`（默认） | 逐分量滤波，重复使用一个三维工作数组；减少滤波工作区显存，不删除主状态的任何分量，也不降低精度。 |

这项选择不改变滤波系数，也不取消 AIR5 保正和能量检查。
省下的是滤波工作区，不能将整个程序的显存需求按分量数等比例缩小。
本版本默认 scalar；旧二进制的默认行为须核对启动日志。
运行脚本显式设置 full 时仍使用 full。lfilter=f 时，单组分仍可能
保留完整兼容工作数组，AIR5 则不分配滤波工作区，不能仅凭默认选项推算显存。

**同步：`ASTR_GPU_SYNC_MODE`**

| 取值 | 实际作用与限制 |
|---|---|
| `explicit`（默认） | 在通用核调用检查点执行设备同步，并检查启动错误。 |
| `selective` | 通用检查点保留启动错误检查，但不逐次等待设备完成；需要主机读取或跨执行流依赖的位置仍同步。仅允许三维周期 TGV，不能搭配 pinned-pipeline。 |
| `dependency` | 同样取消通用检查点的逐次等待，配合流水通信中的事件和执行流依赖保证先后顺序。必须使用 pinned-pipeline，因此要求多 rank 三维周期 TGV。 |

后二者不是关闭所有同步，强制同步点仍保留；不能用于当前 AIR5 或壁面算例。

**迎风核组织：`ASTR_GPU_FLUX_PAIR_MODE`**

- `split`（默认）：分开计算正、负方向的分裂通量。
- `fused`：合并正、负通量计算，减少重复读取和核调用。当前实际启用条件是三维无量纲周期 TGV、五方程、turbmode=none、543e/643e、recon_schem=1 或 3，且 lchardecomp、lfilter、diffterm 均为 false。不满足时不启用合并计算。该选项不更换空间离散格式，也不合并整个 RK 步。

**物性参数**

- `ASTR_PERFECT_GAS_PRANDTL`：指定单组分理想气体的正 Prandtl 数，改变热传导与黏性输运的比例。
- `ASTR_SUTHERLAND_TEMPERATURE_K`：指定 Sutherland 定律中的正温度常数，单位 K；不是自由来流温度或壁温。

未设置时保留原算例物性设置。改变这两项会改变物理模型，不应归入不改变结果的性能选项。

**槽道驱动：`ASTR_CHANNEL_FORCE_MODE`**

| 取值 | 实际作用 |
|---|---|
| `feedback`（默认） | 根据当前质量流率与目标值的偏差及壁面摩擦，更新驱动力。 |
| `fixed` | 保持给定驱动力；优先读取 ASTR_CHANNEL_FORCE_FIXED，未提供时使用现有 force 值，不自动生成推荐数值。 |
| `frozen`（也接受 freeze） | 首次调用时计算一次反馈驱动力，此后保持该值。不同于由用户直接指定固定数值。 |

`ASTR_CHANNEL_FORCE_FIXED` 是实数。一旦提供，即使 MODE=feedback 或 frozen，
也优先使用这个固定力值。它不是指定速度；动量方向和单位按槽道源项定义。
当前读取器对无法识别的 MODE 会落到反馈分支，因此尤其要检查拼写和实际驱动力。
外部脚本的 CHANNEL_FORCE_MODE 等别名不能直接当成求解器环境变量。

### 9.3 无场输出与计时

受限周期三维 TGV/Shu-Osher 可设置：

```bash
export ASTR_GPU_BENCHMARK_NO_FIELD_IO=on
export ASTR_GPU_RK_TIMING=on
```

这不是任意壁面/AIR5 的通用零输出开关，不适合需要 checkpoint 的长时生产。
纯 CPU 对照使用 ASTR_CPU_RK_TIMING=on，同样要求无场输出 benchmark。
ASTR_COMPLETE_STEP_TIMING=on 记录完整时间推进调用的逐 rank 耗时；
ASTR_GPU_RANK_RK_TIMING=on 用于 GPU 分 rank RK 诊断。不同标签不混算。

性能比较固定网格、数值、输出、同步和时间步，排除启动与预热。
并行一步应考虑最慢 rank，不能平均 rank 时间。
反应流完整步包括化学和输运，不等同仅 RK 输运计时。

## 10. 输出、统计与重启

### 10.1 文件含义

本版本支持的标准输出方式为 iomode=h。底层还保留 s 的一维串接 HDF5 接口，但主程序
write_io_tree 直接采用结构化写入，并未完整按 h/s/n 分派。因此 s 不列为
本版本已验证的主程序输出方式，n 也不能视为全局关闭文件 I/O 的保证。
这不是新增运行时限制；选择这些遗留选项前须另行核对、测试对应路径。

| 文件 | 含义 |
|---|---|
| run.log | 启动重定向日志、CFL、状态和异常 |
| flowstate.dat | 以表头为准；maxq1...maxq5 不是湍动能或时间平均 |
| outdat/flowfield.h5 + auxiliary.txt | 场和辅助元数据；GPU 精确恢复还需下述配套文件 |
| outdat/restart_q*.bin | 非反应流 GPU 实际推进状态，按 rank 保存 |
| bakup/ | 备份，恢复前核对完整性和时间 |
| monitor/ | 监测点及专用剖面、壁面量 |
| 切片/场序列 | 后处理或入口数据，不默认是完整 restart |

CPU/GPU 比较必须同相位。同名文件、同一步号可能处于不同边界处理阶段。
非反应流 GPU 的 restart_q 保留下一步滤波前的实际推进状态；HDF5 场可能经过
主机边界投影。精确场比较采用同相位快照或已核对相位的 checkpoint 重建。

### 10.2 统计与监测

lavg 和 feqavg 控制累计。五方程 bl/swbli 的受限 z 周期壁面组合有 GPU
紧凑统计路径，不应套用于所有算例和 AIR5。
平均前确定齐次方向、瞬态丢弃窗口、样本数及重启累计策略。

datin/monitor.dat 第一行为头行，后续为三元索引或坐标。读取器依数值格式区分
整数索引与物理坐标，不能混写。slice.dat 为头行后的 i/j/k 索引。
具体输入沿用读取器和已验证格式，不按文件后缀猜测。

ASTR_AIR5_FLOW_MONITOR_STRIDE 为正整数，用于已实现 air5hbl/air5sbli 监测。
它输出固定展向截面的三个剖面和壁面三点导数诊断，不是展向平均，
不自动等于 Favre/Reynolds 应力，也不是求解器离散壁面通量。
剖面不再变化不能证明三维湍流统计收敛。

### 10.3 重启步骤

1. 等待完整 checkpoint，核对 HDF5 与 auxiliary 步号、时间一致。
2. 建立新运行目录，使用匹配输入、网格、物性、环境及入口数据。
3. 同一次输出的文件成套放入新目录 outdat，设置 lrestart=t。
4. 使用新的绝对 maxstep，核对追加更新数和实际物理时间。
5. 检查恢复日志、CFL、状态和统计量后再长跑。

AIR5 补偿还需主状态、carry（浮点低位余量）和版本元数据；特征顶面也有必须匹配的配置字段。
紧凑统计可能另有累计状态文件。只保留 primitive 场不等于精确重启。
改变滤波、拓扑、物性或边界是新的续算配置，应记录并验证。

监测文件可能使用 status='new' 拒绝覆盖，因此优先新目录续算。
不得复制仍在写入的 HDF5 作为唯一恢复来源。
不要把场输出过程造成的 CPU/GPU 相位差直接当作数值误差。

### 10.4 重启文件配套与兼容边界

| 输出方式 | 新运行目录需要的文件 | 检查重点 |
|---|---|---|
| 非序列 HDF5 场 | outdat/flowfield.h5、outdat/auxiliary.txt | HDF5 nstep 与辅助文件 nstep 相同 |
| 新版非反应流 GPU 续算 | 上述场文件及全部 outdat/restart_q.rankNNNNNNNN.bin | 保存实际推进的 FP64 q，含每个 rank 的重复共享节点和 halo；不能只复制 HDF5 |
| 序列 HDF5 场 | 选定的 flowfieldNNNN.h5，及同代 auxiliaryNNNN.txt 的副本命名为 auxiliary.txt | 保持辅助文件 filenumb 指向该场，不混入其他步号的默认场 |
| 新版非反应流 GPU 序列续算 | 上述序列文件及该步全部 restart_q.stepSSSSSSSSSS.rankNNNNNNNN.bin | S 为十位时间步号，不是四位 filenumb；保持 lwsequ 设置不变 |
| 普通累计平均 | 上述流场及对应 meanflow.h5 | lavg=t 且 nsamples>0 时读取，步号及样本数须匹配 |
| GPU 紧凑统计 | 流场及全部 compact_stats.rankNNNNNNNN.bin | 原 MPI 拓扑、rank 编号、局部尺寸、步号、时间和样本数须匹配 |
| AIR5 补偿/专用边界 | 完整原始 checkpoint 及该模式要求的元数据 | 不从 primitive 字段重新拼装或删除 carry/版本字段 |

普通重启读取器先读 auxiliary.txt，根据 filenumb 查找序列场；找不到时会
尝试 flowfield.h5。因此“文件存在”不等于选中了期望的代次。应在独立目录中
只放确定的一套恢复数据，保存原件，不通过修改 nstep 绕过一致性检查。
输入文件、网格和入口数据也必须与该套状态配套。

非反应流 GPU 的 HDF5 场保留原有展示及 CPU 场比较语义。独立 restart_q
文件保存主机边界投影和共享节点平均之前的实际推进状态，避免重启改变下一步
滤波输入。辅助文件记录版本和生成代次，读取时核验 rank、拓扑、尺寸、步号、
时间及部分数值配置，并检查负载长度和有限性。该检查不是输入文件或网格的
完整校验和；用户仍须保持全部物理配置一致。每次 checkpoint 额外写出一套
含 halo 的五分量 FP64 q，非序列旧代随场文件保留于 bakup。

带版本标识的 checkpoint 若缺文件、存在未完成的 .tmp、代次不符或配置不匹配，
程序直接停止，不回退到展示场。不要删除标识绕过检查。精确 GPU 续算目前不支持
更换 MPI 拓扑或切换 CPU 路径。无版本标识的历史文件仍可按旧方式读取，但会
提示不能保证精确续算，尤其不能保证壁面滤波的连续/重启一致性。
此格式仅适用于非反应流 numq=5；AIR5 的补偿及专用边界恢复协议不变。

GPU 紧凑统计按 rank 保存二进制累计状态，当前读取器明确核验原拓扑和局部
分块。它不是跨拓扑统计迁移格式；不得因全局 HDF5 场可重新分块，就认为
这些统计文件也可直接迁移。完整统计恢复不能只用 flowstate.dat 替代。

默认兼容范围是同一已验证二进制、相同模型、网格、边界及统计配置的续算。
更换版本、CPU/GPU 路径、拓扑、精度或滤波方案时，先做短窗连续/重启对照，
不承诺任意版本间或 CPU/GPU 间的逐位一致。本版本验证覆盖的具体
组合见验证结果摘要，不以一个通过案例代替所有组合。

### 10.5 正常停止与失败恢复

生产运行宜在启动前设置明确的绝对 maxstep 和正的 checkpoint 间隔，确保
计划停止前有完整恢复点。controller 并非每步读取：主循环在 checkpoint
间隔处重新读取它。修改 controller 不等于即时停止，也不保证额外生成
用户指定时刻的 checkpoint；不要在并发读取期间逐行改写该文件。

进程退出码为零仍需检查正常结束标记、错误信息、场是否有限和恢复文件是否
配套。文件写出失败、MPI 中止或强制终止后，不使用最新但未完成的文件组，
应选择上一套已确认完整的恢复点，在新目录短跑核验后再续算。
程序存在备份文件并不等于全部输出构成一个原子事务，不混用不同代次备份。

若升级后的二进制失败，保留失败日志与新目录，用原二进制、原输入、原环境
及升级前的完整 checkpoint 恢复。不要要求旧版本读取新版本新增的状态字段。
任何数值异常都先定位，不通过关闭报错检查或放宽保正容差恢复生产。

## 11. 算例选择与专用处理

完整可启动包见 [GPU_Quickstart](examples/GPU_Quickstart/README.md)：
提供单组分 TGV、槽道、平板，以及 AIR5 TGV、平板的主输入、controller、
启动脚本和附加数据生成器。可选 CPU/GPU、NP=1/2、full/scalar，
不依赖 tests/。默认仅三次更新，用于安装和启动检查。
当前没有可直接启用的 AIR5 槽道实现，不能只改 lcomb 来构造反应槽道。

| 类别/名称 | 用途 | 限制 |
|---|---|---|
| tgv | 首次启动、周期验证、性能基线 | 小网格短跑后扩大，统计定义需匹配 |
| 2dvort、hit | 输运及周期流场 | 按已验证三维配置与可复现初场 |
| channel | 壁面、驱动、多 rank | 明确力、壁温、方向和统计窗口 |
| LDC、rti | 多面壁面/重力 | 使用对应现有 flowtype，不凭标题改名 |
| sod、shuosher | 激波捕获 | 程序检查格式、边界、黏性和滤波组合，不支持的组合会停止运行 |
| openshock | 受控开放边界 | 不是任意入射激波通用模板 |
| bl、swbli | 平板及激波边界层 | 剖面、参考量、入口与远场共同定义工况 |
| AIR5 系列 | 固定五组分研发验证 | 不列为全面完成的生产物理能力 |

OpenSBLI 专用模式固定 Mach 2、Re=950、壁温 1.676194、顶面分界 x=40 等，
不是自由参数接口。其他 GPU 支持条件检查也可能依赖 flowtype。
合理的专用物理处理需要保留，配置检查不应通过删代码绕过。

## 12. AIR5 使用条件与配置

固定 N2/O2/N/O/NO 双温模型，保留总密度与 N-1 独立质量分数物理架构，
按其专用十一变量布局使用；十一分量不是十一种组分。
ROS-2 局部 6x6 求解属于化学积分，与不移植紧致差分矩阵不矛盾。

已列出反应 flowtype 进入化学半步、输运步、化学半步。
air5reactor、air5postshock、air5normalshock、air5tgv、air5hbl、air5sbli
与冻结输运类型需区分；lcomb=t 不保证任意 flowtype 启用同样的反应路径。

| 环境变量 | 可选值 | 默认/限制 |
|---|---|---|
| ASTR_AIR5_SOURCE_MODE | coupled / chemical / vt / frozen | 默认 coupled；各值含义见下文，仍受 flowtype 控制 |
| ASTR_AIR5_CONVECTION_LIMITER | full_state / species_budget / consistent_species / symmetric_species | 默认 full_state；各值采用不同的界面通量限制方法，见下文 |
| ASTR_AIR5_DIFFUSION_LIMITER | full_state / layered | 默认 full_state；全通量共用限制系数，或分开限制组分扩散及能量 |
| ASTR_AIR5_COMPENSATION | off / on | 默认 off；on 保存并传播浮点更新中损失的低位余量，不是添加物理补偿源项 |
| ASTR_AIR5_COMPENSATION_RESTART | initialize | 仅旧 checkpoint 首次启用补偿，初始化零 carry |
| ASTR_AIR5_TOP_MODE | prescribed / characteristic | 默认 prescribed，直接指定顶面状态；characteristic 按特征波传播方向推进顶面状态 |
| ASTR_AIR5_TOP_TAU | 有限正实数，秒 | characteristic 必须给定的松弛时间尺度；减小值会增大基础松弛率 1/tau，不是自动优化参数 |
| ASTR_AIR5_PRIMITIVE_REUSE | off / chemistry | 默认 off，重新恢复原始变量；chemistry 复用化学积分已恢复的同一状态原始变量，避免重复计算 |
| ASTR_AIR5_CHEMISTRY_REDUCTIONS | baseline / packed | 默认 baseline，分开汇总状态码、极值和计数；packed 合并为两个 MPI_Allreduce，保持所汇总诊断量的含义 |

**反应与能量交换：ASTR_AIR5_SOURCE_MODE**

- `coupled`：启用化学反应和振动-平动能量交换。
- `chemical`：仅保留化学反应源项，不保留独立的振动-平动松弛源项；不意味着反应与能量无关。
- `vt`：仅保留振动-平动能量交换，关闭改变组分的化学反应源项。
- `frozen`：关闭上述两类局部源项。空间输运是否启用仍由算例和 diffterm 等设置决定，不是冻结整个流场。

**对流通量限制：ASTR_AIR5_CONVECTION_LIMITER**

限制器在高阶更新可能产生负组分或不可接受的双温状态时，减少高阶通量修正。
它改变离散结果，不是可以任意互换的性能选项。

| 取值 | 算法区别 |
|---|---|
| `full_state` | 对同一界面的全部 11 个守恒分量使用共同限制系数；组分非负及热力学约束共同决定这个系数，痕量组分可能因此影响动量通量。 |
| `species_budget` | 仍共用全部 11 分量的界面系数，但组分非负由各单元允许流出的组分量约束，去掉辅助面状态中重复的组分检查；密度和热力学检查保留。 |
| `consistent_species` | 先限制流体通量，再在固定流体通量下限制总和为零的组分修正；N2 通量仍由总质量通量减去其他组分通量得到。保留用于对照，不因已实现就认为压力平衡和长时物理误差已达标。 |
| `symmetric_species` | 对五个组分对称构造候选通量，联合约束组分非负、总质量通量一致性及气体常数加权通量，并同步修正组分携带的能量；不再让一个痕量组分直接决定全部流体分量的共用系数。 |

**扩散通量限制：ASTR_AIR5_DIFFUSION_LIMITER**

- `full_state`：组分、动量和能量扩散采用共同的界面限制系数。
- `layered`：先限制组分扩散并一致修正其携带的能量，再根据热力学可接受性检查能量相关修正。不是取消能量约束，也不是让各分量完全独立更新。

AIR5 受控验证采用 `symmetric_species + layered`，但程序默认仍是
`full_state + full_state`。要复用前者必须显式设置两个变量，不能依赖默认值。
交付范围和未关闭的物理门槛见
[部署能力边界](documents/ASTR_RELEASE_PROD_CAPABILITY_SCOPE.md)。具体生产配置
另附对应版本的验证记录，不将开发分支的全部历史验证作为随包能力承诺。

**补偿及顶面设置的条件**

COMPENSATION=on 要求 AIR5、11 分量、三维 air5hbl/air5sbli、RK3，
不使用浸入边界或海绵层。它额外保存低位余量并参与重启，不能理解为放宽保正容差。
TOP_MODE=characteristic 使用局部法向传播速度区分传入和传出波，并对需要指定的
传入状态采用松弛；TOP_TAU 只是基础时间尺度，实际强度还由边界处理决定。
这些 AIR5 顶面设置与第 6 节的单组分 ASTR_NSCBC_FARFIELD_MODE 不是同一接口。

ASTR_AIR5_FILTER_VALIDATION=on、ASTR_AIR5_TOP_GPU_VALIDATION=on、
ASTR_AIR5_TOP_RESTART_VALIDATION=on 分别允许程序进入 AIR5 滤波组合、
GPU 特征顶面和特征顶面重启。需要这些功能但未明确设置时，相应检查会停止运行。
它们不会执行验证，也不会自动证明算例正确，不应为消除报错而全部打开。
AIR5 full/scalar 滤波已有实现；使用补偿、滤波和特征顶面的组合时，
仍须核对对应工况的短窗、重启及多 rank 对照记录。

HBL 使用 datin/air5_hbl_profile.dat 等专用输入。
域长文件 datin/air5_hbl_domain.dat 格式：

```text
air5_hbl_domain_v1
Lx Ly Lz
```

实际文件第二行填写数值，不是字母。当前 AIR5 要求 nondimen=f、
num_species=5、turbmode=none；使用有量纲状态，不照搬无量纲 TGV 参考量。

剖面 datin/air5_hbl_profile.dat 每个数据行固定 13 列：

```text
y rho u v w p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO
```

此处为字段说明，不是可直接写入数据区的标题。空行和 # 注释可用，
# x_origin=... 可记录正的流向起点。采用 SI 单位：位置 m、密度 kg/m³、
速度 m/s、压力 Pa、温度 K，质量分数无量纲。
y 必须严格递增，组分非负且和为 1，rho/p/T 必须满足当前混合气体状态方程。
读取后按 y 线性插值，密度由插值后的 p、T 和组分重建；
超出剖面范围使用端点状态，因此必须检查剖面是否覆盖实际网格。

入射文件 datin/air5_incident_shock.dat 的结构为：

```text
air5_incident_shock_v1
x_top y_top normal_x normal_y normal_z
q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11
q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11
```

后三行为数值占位说明。两行 q 分别为上游、下游守恒状态，
存储顺序是 rho、三个动量密度、总能量密度、五个组分密度
（N2/O2/N/O/NO）、振动能量密度。五个组分槽位的存储不改变 N-1 独立组分约束。
总能量定义必须使用当前热化学模型，不能自行用常 gamma 公式替代。
当前读入器要求正的 x_top/y_top、单位法向且 normal_z 近零，
并检查压缩激波状态与法向守恒通量一致性。此文件不能用单组分 51 行代替。

已有补偿 checkpoint 自动恢复 acq01...acq11、acc01...acc11 和版本号。
initialize 仅允许不含补偿字段的旧 checkpoint 以零 carry 开始新配置，
不能修复缺少一部分字段的损坏文件，也不能当作精确恢复已有低位余量。

单位状态、短窗、拓扑、重启和声学通过不代替真实反应 SBLI 长时验收。
仍需检查模型适用性、分辨率、入口、热流/摩阻、元素和能量收支、滤波与统计敏感性。
非有限量、状态码或保正失败应定位原因，不靠截断组分或放宽容差强行跑完。

## 13. 故障处理

| 现象 | 首先检查 |
|---|---|
| 找不到测试源文件 | BUILD_TESTING=OFF 是否生效，是否复用了旧缓存 |
| 找不到 examples | 精简包仍需保留原示例目录 |
| HDF5 模块/链接错误 | Fortran ABI、HL/Fortran 库、MPI 是否匹配 |
| 计算节点缺库 | 计算节点环境、模块和挂载路径；登录节点 ldd 不足以保证 |
| MPI_Init/PMIx 错误 | 启动器、MPI 库、临时目录和权限；不一律归因 GPU profiling |
| GPU 不可见/误绑定 | 调度资源、CUDA_VISIBLE_DEVICES、local rank |
| device-aware 拒绝 | 编译查询、运行栈和 MPI 前预绑定 |
| 零场、CFL=0/Inf | flowtype、九字段顺序、CRLF、剖面和网格 |
| Unsupported configuration | 逐项核查格式、边界、方向、拓扑，不能删除配置检查来强行运行 |
| monitor 文件存在 | 新目录续算与 status=new；保留旧证据 |
| 重启字段缺失 | checkpoint 与辅助/补偿/顶面元数据是否成套 |
| 内存不足 | halo、工作区、host 缓冲和统计内存，不只估五变量物理场 |
| 统计突变 | 输出相位、累计重置、重启和实际物理开关 |

保留完整日志、输入、环境、版本和最后完整 checkpoint，记录首次失败步号和 rank。
启动依赖故障与科学数值失败分开处理，不自动修改物理条件来消除错误。

## 14. 部署验收及复现记录

部署不绑定某一超算平台。已有验证记录仅适用于记录中的环境和
二进制，不能替代新部署环境的验收；CUDA Fortran 版本仍要求受支持的 NVIDIA
GPU 和 NVHPC 工具链，不代表已经支持 AMD/DCU。

最小顺序：独立构建与依赖检查 -> 单 rank 短跑与设备绑定 ->
CPU/GPU 同相位 -> 多 rank 与目标通信 -> 输出重启 -> 目标物理长窗。

依赖检查不能只查 `ldd` 是否出现 `not found`。还要核对 MPI 启动器、
Fortran 编译包装器、HDF5 构建记录和二进制实际加载的 MPI 库是否属于
同一套兼容环境。编译器运行库及 MPI 动态加载的通信组件也需在实际计算节点
验证。登录节点检查通过不证明驱动、GPU 绑定或 CUDA-aware 传输可用。
平台模块名、路径、GPU 架构及调度器启动方式应由部署方填写，不能照搬
示例中的本地路径。先通过主机中转通信，再对 device-aware 单独验收。

更换编译器、MPI/UCX、驱动、GPU 或数值配置，按影响面重新验证。
文档更新无需重跑昂贵生产任务。生产包不带测试，可由开发分支的外部验收工具
检查同一二进制。

每个运行包记录：

- 源码版本、自定义修改记录、构建选项、二进制校验值。
- 编译器、MPI、HDF5、驱动、GPU、CPU/GPU 资源及拓扑。
- 输入、网格/入口版本、生效环境变量。
- 时间步、恢复时间、实际物理终点、输出和平均策略。
- 误差定义、比较相位、阈值，以及通过/失败/未测试结论。

校验值可写离线清单，不要求程序启动打印 SHA256。
对外共享运行记录前，移除账户、凭据、内部主机名及个人目录信息。

## 15. 源码与配套文档索引

| 主题 | 实现 |
|---|---|
| 构建 | CMakeLists.txt、src/CMakeLists.txt |
| 命令和输入/I/O | src/astr.F90、src/cmdefne.F90、src/readwrite.F90 |
| 初场/网格/分解 | src/initialisation.F90、src/gridgeneration.F90、src/parallel.F90 |
| 边界和支持条件检查 | src/bc.F90、src_gpu/boundary_gpu.cuf、src_gpu/case_capability_gpu.cuf |
| OpenSBLI 特例 | src/conservative_boundary_runtime.F90 |
| 设备和通信 | src_gpu/device_runtime_gpu.cuf、src_gpu/halo_transport_gpu.cuf |
| 推进和计时 | src/mainloop.F90、src_gpu/mainloop_gpu.cuf、src/benchmark_runtime.F90 |
| AIR5 | src/chemistry_runtime.F90、src/chemistry_boundary_state.F90、src/chemistry_boundary.F90 |
| 统计 | src/statistic.F90、src_gpu/production_statistics_gpu.cuf、src/chemistry_monitor.F90 |

配套文档：

- [算例生成与启动](examples/GPU_Quickstart/README.md)：非反应流和实验性 AIR5 示例。
- [输入数据字典](documents/ASTR_DEPLOYMENT_INPUT_CONTRACT.md)：网格、入口、初场及专用状态字段。
- [能力边界](documents/ASTR_RELEASE_PROD_CAPABILITY_SCOPE.md)：物理模型与可选组合限制。
- [部署验收清单](documents/ASTR_RELEASE_READINESS_CHECKLIST.md)：目标计算环境的检查顺序。
- [验证结果摘要](documents/ASTR_RELEASE_LOCAL_REGRESSION_20260929.md)：已执行检查及其适用范围。
- [源码许可说明](documents/ASTR_RELEASE_SOURCE_PROVENANCE_REVIEW.md)：已有版权声明与待核实事项。

新工况应另外准备单位与参考量说明、网格及入口数据、时间步依据、输出计划和
预先确定的误差判据。不要将示例成功运行直接作为目标工况物理正确性的证明。
