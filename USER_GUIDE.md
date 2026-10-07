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
| 5 | lrestart | 必须为 f；旧 checkpoint 恢复已关闭，新恢复点由 input.output 指定 |
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
| lwsequ | 遗留字段，忽略并提示；三维场改由 input.output 配置 |
| lwslic | 遗留字段，忽略并提示；切片改由 input.output 的全局索引配置 |
| lavg | 统计累计，不等于统计已收敛 |
| lcracon | 遗留崩溃修复/续算；可信化与 AIR5 补偿保持 false |
| maxstep | 绝对步号上限，不是追加步数或终止物理时间 |
| feqchkpt | 控制文件重读与CFL检查频率，不是新checkpoint保存间隔 |
| feqwsequ/feqslice | 保留的旧场/切片频率，不控制新产品 |
| feqlist | 日志/诊断频率 |
| feqavg | 统计采样频率 |
| deltat | 时间步，单位随模型设置 |

频率使用正整数，**不要填 0 关闭输出**，路径存在 mod(nstep,frequency)。
三类文件输出分别由 `input.output` 的 `enabled` 控制，不由这些遗留字段控制。

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

已准入的周期三维 TGV benchmark 可设置：

```bash
export ASTR_GPU_BENCHMARK_NO_FIELD_IO=on
export ASTR_GPU_RK_TIMING=on
```

这不是任意壁面/AIR5 的通用零输出开关，不适合需要 checkpoint 的长时生产。
benchmark 检查器保留 Shu-Osher 的分类，但当前默认新输出入口尚未准入
该算例，不能据此认为它可以使用本段的完整启动路径。
在已准入的 benchmark 中，该开关优先于新输出配置，关闭 checkpoint、
三维场和切片，即使这些产品已启用且到期也不写出。原位图像和正式统计
不由此关闭，须分别配置；该开关不意味着完全没有文件输出。
纯 CPU 对照使用 ASTR_CPU_RK_TIMING=on，同样要求无场输出 benchmark。
ASTR_COMPLETE_STEP_TIMING=on 记录完整时间推进调用的逐 rank 耗时；
ASTR_GPU_RANK_RK_TIMING=on 用于 GPU 分 rank RK 诊断。不同标签不混算。

性能比较固定网格、数值、输出、同步和时间步，排除启动与预热。
并行一步应考虑最慢 rank，不能平均 rank 时间。
反应流完整步包括化学和输运，不等同仅 RK 输运计时。

## 10. 输出、统计与重启

默认且唯一的正常场文件/恢复入口为完整步新接口。第10.6节给出配置和
准入范围，旧checkpoint不再接受，不存在自动转换或旧格式回退。

### 10.1 文件含义

新场和checkpoint采用并行HDF5/XDMF接口，`iomode` 不选择或关闭该接口。
旧输入仍保留该字段；关闭文件产品应使用各自的 `enabled=f`，而非 `iomode=n`。
日志、监测量及统计属于独立用途，关闭三类流场产品不等于没有任何文件写入。

| 文件 | 含义 |
|---|---|
| run.log | 启动重定向日志、CFL、状态和异常 |
| flowstate.dat | 以表头为准；maxq1...maxq5 不是湍动能或时间平均 |
| output根/checkpoints/step############/ | 完整步精确恢复批次，包含状态、控制、清单及按需统计/入口记录 |
| output根/resources/ | 共享几何、输入和冻结外部资源，恢复时必须保留 |
| monitor/ | 监测点及专用剖面、壁面量 |
| output根/fields、slices | 独立后处理产品及时间索引，不能代替checkpoint |

新产品均观察完整步状态。派生量从同一状态的私有副本计算，不改变推进场。
不能把历史旧场或RK内部诊断快照仅凭同一步号视为同相位数据。

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

1. 选择带有效 `MANIFEST/COMPLETE` 的完整新 checkpoint 目录。
2. 保留所引用的运行级 `resources/` 和历史产品段，使用匹配的程序与输入。
3. 主输入保持 `lrestart=f`，在 `datin/input.output` 的 `&output` 中填写
   `restore_directory`；由新入口验证并恢复，不复制旧场文件到默认文件名。
4. 设置绝对终止步号 `maxstep`，核对恢复步号、时间及下一步步长。
5. 先短跑检查 CFL、状态、统计续接和输出，再决定是否长跑。

AIR5 补偿余量、已登记边界历史和累计统计由同一批次负责恢复。
单独的可视化场不能代替 checkpoint，也不能用原始变量重新拼装低位余量。
不得将仍在写入的批次作为唯一恢复来源。配置及跨拓扑限制见10.6节。

### 10.4 重启文件配套与兼容边界

当前恢复入口只接受新格式完整批次。`flowfield.h5/auxiliary.txt`、旧序列场、
`restart_q*.bin`、旧原位配对恢复批次及旧AIR5 checkpoint均不再兼容。
没有自动转换或失败后的旧格式回退；旧结果若需续算，应保留匹配的旧程序。

每个新 checkpoint 保存 `state.h5` 和版本化控制/清单文件，按需保存统计、
动态入口及AIR5记录。共享几何、输入和入口源在运行级 `resources/`，
其余历史段由相对引用连接。整体搬移必须保留这些依赖，不能只拿一个HDF5。

同后端同拓扑精确续算要求匹配程序、模型、网格、边界和统计契约。
不同二进制、CPU/GPU互换和未登记拓扑不在精确续算承诺内。已登记的
有界重分区能力见10.6节；不通过删标识、改步号或改文件名绕过验证。

### 10.5 正常停止与失败恢复

生产运行宜在启动前设置明确的绝对 maxstep 和 `input.output` 中正的checkpoint间隔，确保
计划停止前有完整恢复点。controller 并非每步读取：主循环按旧控制频率
`feqchkpt` 对应的步号重新读取它，与新保存频率相互独立。
修改 controller 不等于即时停止，也不保证额外生成
用户指定时刻的 checkpoint；不要在并发读取期间逐行改写该文件。

进程退出码为零仍需检查正常结束标记、错误信息、场是否有限和恢复文件是否
配套。文件写出失败、MPI 中止或强制终止后，不使用最新但未完成的文件组，
应选择上一套已确认完整的恢复点，在新目录短跑核验后再续算。
程序存在备份文件并不等于全部输出构成一个原子事务，不混用不同代次备份。

若升级后的二进制失败，保留失败日志与新目录，用原二进制、原输入、原环境
及升级前的完整 checkpoint 恢复。不要要求旧版本读取新版本新增的状态字段。
任何数值异常都先定位，不通过关闭报错检查或放宽保正容差恢复生产。

### 10.6 默认完整步输出与精确重启

启动时自动读取必需的 `datin/input.output`，可分别配置 checkpoint、三维场
和切片。不需要设置环境变量；可选的 `ASTR_OUTPUT_CONFIG` 仅覆盖配置路径，
未设置或为空时均使用默认文件。缺失、损坏、各rank路径不一致时明确终止，
不回退旧输出，也不自动把旧controller频率转换成新产品调度。
保存相位位于完整时间步之后，不在 RK stage 之间。checkpoint 保存
权威守恒量、必要缓存、已登记的统计和调度状态；三维场和切片供后处理，
不能代替 checkpoint。普通文件输出不依赖 Catalyst 或 ParaView。

这是已完成首期有界本地验收的可选接口，不是所有算例和拓扑的通用保证。
首期交付范围按用户确认的方案A冻结，见输出重设计计划第10.77–10.78节；
旧controller的 `lwsequ/lwslic` 被忽略并提示；`feqchkpt` 仍控制既有的
controller重读与CFL检查，不控制新checkpoint频率。`lavg/feqavg` 的
统计定义和采样相位不变。关闭三类文件输出须在配置中分别设 `enabled=f`，
仍需提供合法配置和当前接口要求的预算，不能靠删除文件关闭输出。
基础字段已覆盖周期 TGV、bc41 槽道、已登记静态/动态 CURVE 平板、固定
AIR5 HBL/SBLI 的代表性短测。文件梯度、涡量、Q_rs 和散度目前准入
16³、五变量、无量纲、FP64、黏性开启、643e 求导、NP=1/2 的以下路径：
内部生成的周期 TGV；y 方向完整的 bc41 槽道；y 方向完整的已登记静态或
动态入口 CURVE 平板。前两者使用 643e 对流和显式滤波，平板使用 543e
对流。槽道和平板的双 rank 派生输出验证覆盖 x/z 分区，动态入口代表为 z。
内部用六阶中心差分，物理边界附近沿用求解器的二阶单边/二阶中心/四阶
中心闭合，再通过网格度量得到物理坐标梯度。

固定 AIR5 派生量另开放 16³、SI 单位、十一守恒量、五组分双温、
643e/643e、黏性及显式滤波开启、bc=[11,50,41,51,1,1] 的内部生成网格：
HBL 的 NP=1/2 z，SBLI 的 NP=1/2 x，y 方向均须完整。
九分量梯度、三分量涡量和散度单位为 s^-1，Q_rs 为 s^-2。
全部选择时，输出十二个基础标量及十四个派生标量，另有速度向量。
同相位独立离散参考及 CPU/GPU 差按计划第 10.61 节固定物理尺度检查，
不是用无量纲容差直接比较 SI 原值。其他尺寸或未登记组合仍拒绝；
已有基础字段支持不等于派生量支持。

启动顺序：

1. 按正常算例准备输入和 controller，主输入 `lrestart=f`。
2. 建立新的 `outdat/new`，其产品子目录不能预先存在。
3. 从 `scripts/output/input.output.tgv.example` 选择基础字段，或从
   `input.output.tgv.derived.example` 选择派生量，作为 `datin/input.output`。
4. 按正常命令 `astr run datin/input.tgv` 启动，不再要求设置输出环境变量。
   主输入中的 `usegpu` 仍决定 CPU/GPU，示例频率和预算不是生产推荐值。

AIR5 登记短测可用 `scripts/output/input.output.air5.derived.example`，
但该文件只配置输出，不代替 AIR5 物理输入、补偿及特征顶面的匹配设置。

`&output` 的 `device_budget_bytes` 限制受控设备工作区，
`device_reserve_bytes` 是另一个可选设置，指定 GPU 字段打包分配后必须
剩余的设备空闲字节数。后者默认 `0`，基础字段不新增空闲量查询；本地
16³ TGV 的 NP=1/2，以及已登记槽道、静态/动态 CURVE、AIR5 HBL/SBLI
的 NP=2 代表短测显式使用 `device_reserve_bytes=1073741824`。
启用后，在打包分配前和释放后查询 `cudaMemGetInfo`，记录各 rank 的
空闲量、计划分配量和保留量。查询失败或不足时集体终止，不发布失败帧，
不降低分辨率或改用 CPU。失败目录可能保留 `.tmp` 和空的索引表头，
这些不是可用帧。派生量分配继续至少保留 1 GiB；显式设置更大值时采用
更大的保留量。该选项不保护 checkpoint 的全部工作区，也不替代
`&insitu_run` 中原位处理的独立资源配置。CPU 路径不执行 CUDA 查询。
检查的是整个设备在这两个时点的空闲量，不能保证本作业独占显存或
捕获阶段内部所有瞬时峰值。此资源配置不改变文件格式、数值路径或
输出调度，可以按当前启动环境设置，不需要 `restart_output='override'`。

`&checkpoint`、`&volume`、`&slices` 各自支持 `mode='steps'` 或 `mode='time'`。
前者只设 `interval_steps`，后者只设 `interval_time`。文件记录实际完成步
时间，跨越调度阈值不会插值为阈值时刻。初末帧可独立选择；启用
checkpoint 时正常结束必须保存末步。`keep=1|2` 只轮换 checkpoint，
已发布三维场和切片全部保留。当前恢复源受保护，因此目录数可超过 keep。

重启时保持主输入的 `lrestart=f`，由 `&output` 的
`restore_directory='/path/to/run/checkpoints/step000000000005'` 选择新格式
批次，而非读取旧 HDF5 恢复文件。保留源根的 `resources`、所选完整批次
和其引用的历史段，不只复制 `state.h5`。新目标根须按配置新建；已验证的
同根续算会创建新段，不覆盖已有帧。`maxstep` 是恢复后的绝对终止步号。
`restart_output='saved'` 继承已保存调度。改变输出间隔、字段选择等须显式
选择 `restart_output='override'`，不能用它绕过物理/数值配置不匹配。

每段的 `series.xdmf` 只索引本段。布局兼容时，另自动生成
`lineage.xdmf`，按显式父链索引祖先保留帧和当前帧；祖先在各次恢复点
之后的帧不会混入新的分支。用 ParaView 的 XDMF3 Reader 打开该文件。
索引仍引用历史 HDF5 和共享坐标，不复制流场；搬移须保留完整相对引用树。
合法覆盖改变字段集合或切片布局时，日志报告统一索引不可用，本段及
历史段的 `series.xdmf` 继续可用，旧帧不删改，不自动补字段或取交集。
来源缺失、指纹错配或损坏会终止，不按布局变化忽略。当前原生父链最多
128 段、每个祖先段最多 8192 帧，路径上限 1200 字符；超过会明确拒绝。
这两个索引文件分别替换，不是双文件事务；失败残留须检查，不作为恢复状态。

16³ 动态入口 CURVE z、固定 AIR5 HBL z/SBLI x 的 CPU/GPU NP=2 短测已
覆盖整体搬移后续算、keep=1 保护恢复源及轮换、已停止段索引修复和父链
读回；入口历史、化学补偿及各自累计统计与连续运行一致。搬移须包含完整
相对引用树。该检查不放行改变 MPI 分解或在求解器运行时修复索引，详见
`scripts/output/README.md`。

同后端同拓扑的登记组合已通过精确续算检查。重新分区开放已验证的
周期五变量 TGV NP=1↔2 x/y/z 和 NP=2 分解方向互换。
另开放严格限定的槽道短测配置：内部生成 16³ 网格、ninit=3、无量纲、
bc=[1,1,41,41,1,1]、643e/643e、滤波和黏性开启、
ASTR_CHANNEL_FORCE_MODE=fixed、ASTR_CHANNEL_FORCE_FIXED=1.d-4，
支持同后端 NP=1↔2 x/z 及 NP=2 x↔z。源/目标 y 方向都不能分区，
不支持反馈/冻结驱动、紧致或原位累计统计；CPU 已登记 mean44 可迁移。
静态 CURVE 剖面平板另有已验证的有界配置：16³、无量纲、ninit=0、
turbinf=prof、bc=[11,21,41,51,1,1]、543e/643e、MP7 物理空间重构、
无湍流模型、滤波和黏性开启，不含渲染迁移。支持同后端 NP=1↔2 x/z
及 NP=2 x↔z；源/目标 y 方向必须完整。实际验收网格为 warp_x=0.08、warp_y=0.04
的三维挤出平板，dt=1e-5，不代表任意曲线网格或其他工况。
该配置允许 lavg=t 的 CPU mean44/GPU 紧凑累计统计迁移，采样身份不重置。
GPU 统计 HDF5 根数据保存当前分区的新增累计，`inherited/` 保存全域继承
历史。读取总累计时，继承历史只加一次，再加根数据的 z 分区和；几何
长度不能重复相加。迁移后不能改用旧 binary sidecar 保存统计历史。
同拓扑仍逐值精确恢复。该全域二维主机基线仅通过 16³ 有界验收，
不代表任意生产网格的容量或长期统计认证。详见输出工具说明。
动态入口 CURVE 另通过相同 16³ 挤出网格、dt=6e-6 的限定配置，
冻结十二份非多项式时序源，实际四帧缓存及换帧窗口随检查点保存。
支持同后端 NP=1↔2 x/z、NP=2 x↔z 和已登记累计统计的联合迁移。
恢复按新分区读取实际缓存，不重插值过去的帧，不清空或重新采样过去的统计。
入口窗口、时间戳、槽位、采样身份及调度保持一致；同拓扑仍逐值一致。
冻结源资源随检查点相对引用树搬移后，不再依赖原始入口目录。
本门槛不开放其他动态入口配置、y 分区、跨后端或渲染迁移。
固定 AIR5 HBL 另有有界周期 z 重分区门槛：16³、SI 单位、ninit=0、
bc=[11,50,41,51,1,1]、643e/643e、recon_schem=5、lchardecomp=f、
滤波和黏性开启，可携带已登记 mean44 累计统计及 GPU 全局守恒历史，
不含原位累计统计或渲染迁移；使用 coupled 化学源项、
compensation=on、symmetric_species 对流限制、layered 扩散限制及
characteristic 顶面，仍须填写正的松弛时间。支持同后端 NP=1↔2 z，
源/目标 x/y 都必须完整。实际验收域长为 0.08/0.01/0.002 m、
初始 rho=0.05 kg/m³、T=3000 K、N2/O2=0.767/0.233、dt=1e-10 s，
CPU scalar/GPU full；不是任意 AIR5 工况的跨拓扑认证。
mean44 按物理节点重分区，保留采样次数与首末样本身份，不重新采样过去。
GPU 守恒基线已是全域量，按原值恢复，不按目标 rank 数再求和或重建基线。
同拓扑仍逐值一致；跨拓扑累计矩按固定物理尺度乘样本数比较，守恒积分
按固定守恒尺度乘域体积比较，不能直接对有量纲字段套用统一绝对差。
这项能力只通过十二步的有界续算验收，不替代反应流长期物理验证。
固定 AIR5 SBLI 另支持同后端 NP=1↔2 x，源/目标 y/z完整。
保持上述16³、SI、边界/滤波/黏性/化学补偿配置，改用 recon_schem=3，
入口速度4*a_ref、入射角25°、顶面交点x=0.035 m及已冻结初场/Tv剖面。
mean44 和 GPU 全域守恒历史直接迁移，纯恢复逐值一致；跨拓扑场和累计量
按对应固定物理尺度检查，不是任意激波工况的迁移认证。详见输出工具说明。
其他壁面、CURVE y 分区、未登记动态入口、AIR5 HBL x/y 或 SBLI y/z 重分区、
CPU/GPU 互迁和渲染重新分区仍不开放。
滤波后缓存时序已修正，原滞后缓存轨迹的程序批次不能用于新程序精确续算。

已登记16³ TGV的CPU/GPU NP=1/2还通过同时开启检查点、三维场、三方向
索引切片及正式原位统计的联合短测，GPU可同时使用既定EGL渲染预设。
各产品独立调度，同拓扑5+7步续算与连续12步一致，关闭场/切片/渲染
不改变推进和统计。原位输出目录须与新文件输出根目录分开。
首期渲染主线为GPU TGV同拓扑，含下述有界静态CURVE及bc41壁面候选。
CPU渲染、一般AIR5三维渲染及任何渲染重新分区属于后续目标，当前仍明确拒绝，
也不表示生产长序列容量已经验证。

已验收的有界 IS8 `processing_backend='device'` 配置要求另行显式选择
`postprocess_transport='device-aware'` 或 `'pinned'`，不继承求解器 halo
通信设置。前者交给 MPI 设备缓冲，实际是否直传须检查 MPI/硬件；后者只将
共享节点面和 halo 面经固定页锁定主机缓冲中转，不允许下载整场作归属处理。
本地壁面追踪已观察到device-aware的MPI内部小面中转；应用暂存计数为零
不等于整个MPI路径没有主机复制，应与整场/几何回读分开核对。
该入口要求根构建同时开启 `ASTR_WITH_CUDA`、`ASTR_WITH_CATALYST` 和
默认关闭的 `ASTR_WITH_INSITU_DEVICE`，另提供 CUDA/FP64/MPI Viskores依赖。
几何常驻渲染另需默认关闭的 `ASTR_WITH_INSITU_DEVICE_RENDERING` 及相匹配、
已应用设备数组/绘制补丁的ParaView/VTK。不能直接使用未修补的普通安装。
TGV准入为内部生成的32³周期笛卡尔TGV、FP64、643e/643e、NP=1或NP=2 x/y/z，
所选产品为固定z=pi/4索引速度切片、Q_rs=0.25等值面及瞬时、Reynolds/Favre
速度流线。必须启用渲染、选择`derivative_backend='gpu'`并显式填写正的资源预算。
缺失/非法/各rank不一致的传输选择及未支持范围会报错，不退回CPU过滤器。
采样、节点归属、halo、梯度/Q、插值和提取在GPU执行。渲染入口如下：

| `rendering_pipeline` | 设备处理开启渲染时的行为 |
|---|---|
| `standard-device`（此时缺省选择） | 经Conduit/VTK设备数组进入标准绘制路径；几何、连接、颜色和已接受流线留在设备，不下载供主机整理。使用`device_render_pipeline.py`。 |
| `direct-device` | 同一设备几何进入专用VTK Mapper，仍由Catalyst/ParaView出图。使用`device_render_pipeline.py`，准入范围与标准入口相同。 |
| `compatible` | 显式选择原紧凑几何回读路径，使用`tgv_pipeline.py`；保留VTK几何导出，不具备几何常驻保证。 |

严格入口禁止几何writer和完整三维统计导出供渲染。JPEG/EPS编码、颜色/深度
图像回读及主机合成、有界元数据和每rank每轮至多2 KiB粒子续接状态仍存在；
pinned模式另有面暂存，不能称总D2H为零。正式设备统计累计及显式检查点
状态传输独立存在。未编译所选入口时直接报错，缺省选择也不会自动降级。
同后端同拓扑精确续接要求传输选择一致；只改传输须显式选择
`restart_output='override'`，保留累计统计和输出时钟，重建临时缓冲。
默认仍是`processing_backend='host'`，兼容路径不是全设备后处理。
X4另外完成了16³内部笛卡尔bc41槽道的有界设备壁面图片验证：
`products='channel_walls'`、`derivative_backend='gpu'`、
`processing_backend='device'`，显式选择两种面通信之一，采用
`standard-device`或`direct-device`及`device_render_pipeline.py`。
范围为FP64、643e/643e、x/z周期且y上下bc41、NP=1/2 x/y/z，
批准的物理盒为x∈[0,2π]、y∈[0,2]、z∈[0,π]。壁面压力、x向剪切
及入气体热流使用IS5原定义，在GPU从壁面及两层内点状态计算；
字段、三角形和显示颜色不回读供主机整理，仍保留图片及显式检查点I/O。
该设备壁面入口可选择`statistics=t`，直接由GPU壁面字段累计压力、
剪切及热流的时间均值、方差和RMS。同时选择`wall_mean_render=t`
可在GPU读取累计均值并渲染平均图片，不下载壁面场供主机整理。
仍要求`wall_separation=f`，该bc41入口不支持设备分离诊断。
统计窗口及完整步采样权重不变。显式统计导出/检查点会下载累计状态，
与采样和图片路径的传输分开计账。`processing_backend='host'`的
既有壁面统计/均值渲染不受此限制。设备壁面字段、双入口图像及同拓扑4步/3+1续算已有
有界证据，不能据此宣称长期槽道统计或全部X4已完成。

另准入32³静态y-wavy CURVE bc41 TGV短测的同三类瞬时壁面图片，
仍要求FP64、643e/643e、NP=1/2 x/y/z及`wall_separation=f`，
可选择`statistics=t`的GPU壁面累计和`wall_mean_render=t`的设备平均图。
该几何夹具不代表湍流槽道物理验证。其字段和两入口4步/3+1续算
已分别验证，固定短测色标为压力[80,120]、剪切[-0.2,0.2]、热流[0,15000]。
上述32³静态周期CURVE TGV现还可选择`products='q_surface'`，
FP64、643e/643e、NP=1/2 x/y/z、统计关闭、共用固定时钟，支持
两入口及两面后端。Q阈值0.25、u色标[-1,1]保持原预设。物理坐标
与字段共用边插值比例，最终几何在设备绘制；不是六阶几何重构。
非有限或零面积几何直接报错，不删面或合并不同边。该Q产品准入不含
独立PF/AP时钟、AIR5、256³或渲染重分区。

32³ y-wavy bc41曲壁夹具现也可使用上述`q_surface`产品。仅此新增
设备Q供给在物理壁面附近使用七节点六阶单边/偏置导数，内点和MPI
接口仍为六阶中心差分，然后以完整网格度量投影到物理坐标。不改变
求解器离散、原有输出导数或壁面剪切/热流公式。当前真实场验收覆盖
NP=2 x/y/z的standard-device/pinned及direct-device/device-aware组合，
并验证同拓扑精确续算；该预设仍使用Q=0.25。

`products='curve_demo'`另提供Q=0及瞬时流线，按速度模长着色，固定
1280×960及[0,1]色标。仅准入上述静态periodic/y-wavy TGV、FP64、
643e/643e，关闭统计，使用共用时钟及严格设备入口；不开放PF/AP独立
时钟。32³、64³、128³的NP=2 x分解已有短测，新增尺度使用dt=2e-5并逐步检查
CFL<=0.5。两入口/两面后端交叉、独立参考、开关隔离、同拓扑精确
续算和内存/传输检查通过。CURVE流线仍使用h=2π/32定义物理积分
步长，不随网格加密缩小；其余RK45参数和16个双向种子不变。
64³/128³预算为附加显存6 GiB/卡、主机16 GiB/节点、设备空闲至少2 GiB，
不是生产默认。64³/128³的半步长诊断已完成，不改变正式积分设置；
端点差包含不同末段接受弧长及折线插值误差，不作为积分误差上界。
256³候选仅开放NP=2 x分解，已构建，但首轮诊断矩阵中断，独立参考及
精确续算验收尚未完成，未准入100步长窗。

32³静态周期及上述y-wavy CURVE TGV另可选择`products='streamlines'`，
仍为FP64、643e/643e、NP=1/2 x/y/z、共用固定时钟、两严格入口及
两面后端。GPU用真实物理六面体反求与三线性插值，不按均匀索引网格
追踪。`mean_streamline_render`默认`f`；同时设置`statistics=t`及
`mean_streamline_render=t`时，额外供给原Reynolds/Favre平均速度流线。
不得用瞬时场替代零覆盖的均值；图片元数据保留实际累计起止与时长。
此选择不计算未请求的Q，不同时开放`all`中的其它CURVE产品。
物理坐标和最终轨迹不下载供主机整理，单元归属查询另允许每rank
每轮最多256 B，与原2 KiB续接状态分别计账。固定相机、种子和原RK45
门槛不变，场、几何、图片和精确续接已有有界证据。仍不准入任意
CURVE工况、独立PF/AP时钟、AIR5流线、256³或渲染重分区。
逐轨迹单元缓存优化后，本地32³、NP=2 x分解、direct-device/device-aware
五轮四步窗口中位数为周期CURVE约6.000 s、y-wavy约7.302 s，分别较
同输入冻结基线快2.21和2.41倍。窗口含统计及四类流线两帧，不代表
纯求解器加速或生产规模保证。扩大规模仍需逐级容量与正确性验收。

16³内部笛卡尔非催化AIR5平板底壁现可选择严格设备图片：
`products='air5_walls', processing_backend='device', derivative_backend='gpu'`，
采用`device_render_pipeline.py`及`standard-device`或`direct-device`，
显式选择pinned/device-aware面通信。要求SI双温五组分、643e/643e、
边界[11,50,41,51,1,1]、NP=1/2 x/y/z。
`statistics=t`可启用18项壁面标量的GPU时间累计；同时选择
`wall_mean_render=t`可输出十类平均壁面图片。严格设备入口暂不准入
`air5_volume_statistics=t`；原AIR5三维统计仍可在主机兼容入口显式选择。
本夹具物理盒为x∈[0,0.08] m、y/z∈[0,0.002] m，验收dt=1e-10 s、
四个完整步；根构建另需`ASTR_WITH_AIR5_CHEMISTRY=ON`。
GPU从底壁和两层内点的守恒状态重建物性，输出压力、T/Tv、五组分、
x剪切和入气体总热流十类JPEG/EPS；不下载壁面字段或三角形供主机
整理，不写VTK几何。保留原SI单位、固定相机及真实40:1平板比例。
状态/化学补偿及3+1精确续算已有有界证据。此热气体夹具入口速度为零，
不验证湍流、分离或生产SBLI。配合`statistics=t`可选择
`wall_separation=t`：GPU沿真实z长度积分有符号x剪切，只回读随x
变化的一维积分和有限性状态，沿用原CSV/零区间规则。该夹具仍
记录`not_applicable_no_positive_inflow`，不能解释为物理上没有分离。
这一选择不支持bc41/CURVE或任意AIR5工况。既有主机诊断及统计
入口保留，不能将其主机三层采样称为设备常驻。

上述三类设备壁面平均图使用与瞬时图相同的相机、色标及单位，文件名
增加`mean_`前缀。平均值采用原时间窗口裁剪后的累计权重，输出元数据
保留窗口和实际覆盖时长；覆盖时长为零时不输出平均图片，也不以瞬时
值替代。法向符号来自静态几何，不参与平均。均值端点的通信仍可使用
已选择的面后端，pinned及本机MPI内部面暂存、图片回读和显式统计/
检查点I/O分别计账。设备平均图的字段、隔离、4步/3+1精确续接和
双入口图片已在上述小网格验证，不代表长期生产统计或任意CURVE工况。

32³周期Cartesian或上述静态periodic CURVE TGV还可显式选择物理平面。
在已配置路径、渲染、预算和共用固定调度的`&insitu_run`中加入：

```fortran
 products='velocity_slice', statistics=f,
 processing_backend='device', derivative_backend='gpu',
 rendering_pipeline='standard-device', postprocess_transport='pinned',
 slice_definition='plane',
 slice_origin=3.d0,0.d0,0.d0,
 slice_normal=1.d0,0.25d0,-0.125d0,
```

也可显式选择`direct-device`或`device-aware`。origin是物理点，不能单位化；
normal由rank0单位化并广播，最大绝对分量统一为正，平局按x/y/z。
所有rank原始输入须一致；零/非有限法向及不可表示的规范化或几何均报错。
该路径使用一致四面体分解和FP64线性插值，不是六阶切片重构。
切片按全局顶点/边键去重并固定绕序，不下载几何供主机整理，不计算Q。
NP=1/2 x/y/z、两入口/面后端、4步与3+1精确续接及空产品已有短测证据。
此候选要求统计关闭，不支持独立产品时钟、壁面/AIR5平面、256³或重分区。
未填写`slice_definition`仍为既有`index`模式；plane模式不要同时填写
旧`slice_axis/slice_index`。续接须保持六个规范化平面值一致，改变平面
需显式输出override。不同边交点若舍入为同一坐标而形成退化面片，
程序停止，不自动吸附、合并、删面或改变平面。

唯一更大网格例外为
另行验收的256³、NP=2 x分解`products='tgv256_demo'`，只输出Q=0与瞬时
速度流线，不开累计原位统计、检查点、整场、切片或VTK几何写出。
其本地预算为附加6 GiB/GPU、16 GiB/节点、设备空闲至少2 GiB，不是
生产规模默认。依赖、实测与范围见`documents/ASTR_INSITU_IS8_RESIDENT_ACCEPTANCE.md`。

#### 各原位产品独立的固定输出周期

在已有的 `&insitu_run` 配置中可填写以下数组。示例省略了已有的依赖路径、
资源预算和渲染设置，不能作为完整启动输入使用：

```fortran
 products='all', statistics=f,
 rendering_pipeline='compatible',
 step_interval=0, time_interval=0.d0,
 initial_frame=f, final_frame=f,
 product_ids='q_surface.image','q_surface.geometry','instantaneous_streamlines.image',
 product_modes='steps','steps','time',
 product_steps=6,12,0,
 product_times=0.d0,0.d0,0.008d0,
```

此例每六个完整步输出 Q 图片对，每十二步输出 Q 的 VTK 几何，每隔
0.008 个模拟时间单位输出瞬时流线图片对。物理时间到期时取首个达到
目标时间的完整步状态；一个步跨过多个目标时只输出当前状态，不补造过去帧。
未列入 `product_ids` 的产品不输出。`products` 仍限定可选场景范围。

产品标识由下表场景加 `.image` 或 `.geometry` 构成。`.image` 始终把
JPEG/EPS 当作一对，不为两种格式分别设时钟。`.geometry` 写出 VTK
几何，仅允许 `compatible` 入口；严格设备入口仍禁止几何回读和写出。

| 场景标识 | 对应内容 |
|---|---|
| `q_surface` | 已有 Q 等值面预设 |
| `velocity_slice` | 已有速度切片预设 |
| `instantaneous_streamlines` | 同完整步瞬时速度流线 |
| `crossing_streamlines` | 恒定速度跨分区诊断流线，不是另一种实际流场 |
| `mean_reynolds_streamlines` | 已累计的 Reynolds 平均速度流线，要求 `products='all'`、`statistics=t` |
| `mean_favre_streamlines` | 已累计的 Favre 平均速度流线，要求 `products='all'`、`statistics=t` |

数组从第一项连续填写，长度一致，每项选择 `steps` 和正的 `product_steps`，
或 `time` 和有限正的 `product_times`；另一种间隔必须为零。不允许同时
设置共用的正 `step_interval/time_interval`、重复或未支持的标识。
同一组条目的排列顺序不影响配置身份。`initial_frame/final_frame` 共用于
所有已列产品。没有这些数组时，原共用时钟行为保持不变。

到期产品由原生调度层决定，脚本不另设周期；Q 导数和各类流线只在请求时
计算，均值只供给当步请求的种类。平均覆盖尚未建立时记录未覆盖，不用
瞬时流线代替。仅已批准的 JPEG/EPS 文件发布错误可记录整组缺帧并继续，
PF 固定时钟仍按原刻度推进；几何、数值、MPI 和编码错误仍终止。

产品时钟、最近尝试/成功/缺帧身份、缺帧原因和下一目标随现有检查点保存，
不增加独立续算文件。同配置同拓扑精确续接；只改变后处理通信方式，仍须
显式 `restart_output='override'`，但保留各产品时钟。其他获准配置变更需
显式覆盖，并以检查点完整步/时间重新建立产品时钟，不追补旧输出。
覆盖不绕过能力准入；渲染入口互换续算和渲染重分区仍拒绝。

该独立调度首轮仅用于非反应、周期笛卡尔 TGV，不准入壁面、CURVE 或
AIR5。验证为 16³ CPU/GPU 推进与统计隔离，加既有 32³ GPU 实际产品，
NP=1/2 x 分解。几何常驻入口的原范围不变；256³展示预设仍只允许原已
批准的两个图片产品，不因配置可解析而扩大性能或重启认证。
检查点、原生三维场/切片和正式统计沿用各自调度。可变频率见下一节。
详见 `documents/ASTR_INSITU_PF_ACCEPTANCE.md`。

#### 事件和重要时窗驱动的可变输出周期

该功能默认关闭。首版只准入内部生成的 16³/32³、周期笛卡尔、非反应
TGV。CPU 可使用原生场/切片和监测，实际图片仍要求既有 GPU/Catalyst
渲染入口。壁面、CURVE、AIR5、256³展示和渲染重分区未准入。
可分别加密已开启的三维场、切片、图片对及兼容入口的 VTK 几何。
正式统计采样、检查点周期和求解器时间步不变，不自动开启关闭的产品。

共享配置是 `datin/input.output` 最后的可选 `&adaptive_output` 组，
不需要 Catalyst 也可解析。首期指标为 `tgv_kinetic_energy`，定义为
`0.5*mean(rho*(u*u+v*v+w*w))/(roinf*uinf*uinf)`。在完整 RK 步结束后、
渲染和文件产品调度前，从权威守恒状态计算；周期节点只计一次，halo
不参与平均。GPU 上两级归约后，每 rank 每次监测仅回读两个 FP64 标量
共 16 字节，再做标量 MPI 归约，不为判断事件下载三维场。

| 共享字段 | 可选值或约束 |
|---|---|
| `enabled` | 缺省 `f`；设 `t` 才启用。 |
| `indicator` | 当前仅 `tgv_kinetic_energy`。 |
| `monitor_mode` | `steps` 配正 `monitor_steps`，或 `time` 配有限正 `monitor_time`，另一间隔为零。独立于任何写出周期。只有窗口、没有事件时，两间隔为零且不计算动能。 |
| `event_ids` | 最多8个连续填写的独立事件标识，使用1至63个字母、数字、下划线或短横线；不重复。当前均消费同一次动能监测，可用不同参考尺度和阈值。 |
| `s_ref`,`t_ref` | 与事件逐项对应的有限正参考尺度，与动能指标和模拟时间同量纲。无量纲输入采用同一无量纲约定，不使用当前指标幅值作分母。 |
| `r_on`,`r_off`,`hold_time` | 逐项对应。`0<=r_off<r_on`，保持模拟时长有限且非负，不是墙钟时间。 |
| `window_ids`,`window_modes` | 最多8个连续填写的独立窗口标识；每项可单独选 `steps` 或 `time`。 |
| `window_steps(:,i)`,`window_times(:,i)` | 该窗口的 `[start,end)`，非负且递增；未选的另一种范围为零。 |

初始化只建立首个动能历史。随后使用实际有效监测时间差计算
`r=(t_ref/s_ref)*abs((s_new-s_old)/(t_new-t_old))`。
`r>=r_on` 进入加密。最短保持时长达到后，只在新监测得到
`r<=r_off` 时退出，不在中间步用旧变化率退出。多个关联事件和窗口
取并集，不采用所有事件同时满足的条件。相邻完整步跨过整个窗口时，
记录已跨过，不补造窗内状态。无效指标、非正时间差或中间运算溢出报错。

产品仍在各自配置中给出常规间隔，并增加以下字段：

| 产品配置 | 新字段 |
|---|---|
| 原生 `&volume` / `&slices` | `adaptive=t`，`dense_interval_steps` 或 `dense_interval_time`，`event_ids` / `window_ids`。 |
| 原位 `&insitu_run` | 对应 `product_ids(i)` 的 `product_adaptive(i)=t`，`product_dense_steps(i)` 或 `product_dense_times(i)`，`product_events(:,i)` / `product_windows(:,i)`。 |

常规和加密间隔必须同一种口径，加密间隔严格较小。至少关联一个已登记
事件或窗口；未知标识、MPI 配置不一致、未启用却填写加密参数均拒绝。
配置顺序不改变身份。`&checkpoint` 不接受自适应参数。

进入加密时立即输出当前完整步状态，同步命中多个条件也只写一次。
已经加密时新增条件不额外插帧。每次成功输出后，从其实际步/时间起算
下一目标；退出后改用常规间隔，不强制退出帧。仅已批准的 JPEG/EPS
发布错误允许缺帧继续：最近成功身份不变，用最近失败尝试等待当前档位
的一个间隔，避免每步重复尝试。数值、MPI、编码和几何错误仍终止。

监测历史、事件保持、窗口位置及产品成功/失败时钟保存在已有检查点
控制文件内，不新增附属文件。同配置恢复逐值续接；变更仍要求显式
`restart_output='override'`。修改事件只重设依赖它的产品，修改窗口亦然；
改产品间隔或关联只重设该产品。正式累计统计和不受影响的固定时钟
保留。仅变后处理通信保留监测和全部时钟。重设日志含标识及原因。

原生短测示例：`scripts/output/input.output.tgv.adaptive.example`。
原位模板：`scripts/insitu/presets/tgv32/adaptive.nml.in`，需填入相匹配的
Catalyst 实现和脚本路径。示例阈值仅检验 Re=1、dt=1e-3 的短程调度，
不是通用湍流事件判据或生产默认。32³图片短测使用独立冻结的阈值配置。
可选 `ASTR_INSITU_TIMING=1` 分别记录监测、事件时钟/更新、产品时钟及
原生产品的包含性耗时；不得与其内部子阶段相加。
范围、续接和实测证据见 `documents/ASTR_INSITU_AP_ACCEPTANCE.md`。

原位首版 IS7 的有界能力矩阵、依赖条件和完整阶段耗时见
`documents/ASTR_INSITU_IS7_ACCEPTANCE.md`。独立短测入口为
`scripts/insitu/start_tgv_acceptance.py`，要求显式指定程序、MPI 和 Catalyst
实现路径；`--prepare-only`生成待检查输入，`--off`仅关闭新原位功能，
不会同时关闭检查点。模板32³、四个完整步、NP=1/2只用于本地验收，
不应直接作为生产规模默认值。已有算例目录拒绝覆盖。
设备入口可追加`--processing-backend device --postprocess-transport pinned`，
或显式选`device-aware`；缺省采用标准设备入口，可追加
`--rendering-pipeline direct-device`或`--rendering-pipeline compatible`。
不要把这两个后处理通信选项代入求解器halo选项。
可选`ASTR_INSITU_TIMING=1`记录完整窗口及后处理阶段耗时，不改变场或
统计续接、不增加逐阶段MPI屏障。包含子阶段的计时不能相加；记录首次
建立管线/视图的成本，不把小型首帧耗时当成长期生产平均。

可选 `products='channel_walls'` 支持内部笛卡尔 bc41 槽道的两张壁面，
输出壁压、沿 +x 的有符号黏性应力、壁面传向气体的导热热流。
切向为 +x，内法向下壁 +y、上壁 -y；保留求解器无量纲尺度，
不自动换算 Cf/St，不将短窗零剪切解释为分离。
当前准入尺寸不超过32、NP=1/2、643e显式差分，实际短测为16³。
壁面诊断要求 `derivative_backend='cpu'`；GPU只下载每壁三层五变量，
在主机计算壁面量，不是全GPU后处理。渲染仍需GPU求解与EGL。
可开启完整步速度及三项壁面标量统计，CPU/GPU均支持；GPU在设备累计，仅在导出
统计或保存检查点时下载累计数组。两壁节点各自保留，拉伸y网格按
真实坐标求积，不当作均匀方向。已验证四步与3+1步同拓扑精确续接，
不外推为长时间统计定常或更大网格性能认证。

主机诊断另有有界 `products='air5_walls'`，仅支持内部生成的笛卡尔非催化
AIR5平板底壁，边界 `[11,50,41,51,1,1]`、SI物性、643e，尺寸≤32、
NP=1/2；实际验收为16³、四个完整耦合步。要求
`processing_backend='host', derivative_backend='cpu'`；渲染要求GPU/EGL，CPU统计须 `render=f`。
GPU下载底壁三层11变量，在主机复用AIR5状态重建与输运物性，供给
ρ、三速度、T/Tv、压力、五组分、剪切和各热流贡献等18字段。
输出压力、T/Tv、五组分、剪切及总热流的JPEG/EPS/VTK预设，保留SI
单位和完整步身份；空壁面rank正常参加。总热流正方向为壁面向气体，
本静止非催化壁面组分焓通量为零。可选 `statistics=t` 和递增的
`statistics_window`，累计底壁18字段的时间均值、方差和RMS；默认渲染
瞬时字段。GPU主机壁面诊断之后上传紧凑壁面字段用于设备累计，不逐步
下载完整三维场。累计状态保存在现有 `statistics.h5` 内，同后端同拓扑
精确续接，统计重分区仍拒绝；最终标量结果见
`sample.wall_statistics.step*.rank*.bin`，布局见 `scripts/insitu/README.md`。
可显式增加 `air5_volume_statistics=t`，累计三维 T/Tv/五组分及速度
Reynolds/Favre 均值、协方差、密度加权应力。默认关闭，不给壁面
配置自动增加三维数组。三维节点包含物理 x/y 上端，周期 z 端点只
计一次，按真实坐标求积。可另设 `air5_volume_reduction=t` 输出全域
几何平均速度信号的 RMS 与局部方差体积汇总，两类诊断独立命名；
默认不作空间归约，也不据此假设全域统计均匀。该选项要求三维统计。
三维原始累计仍保存在 `statistics.h5`，精确续接时必须保持选择一致。
逐点最终结果为 `sample.air5_statistics.step*.rank*.bin`，区域结果为
`sample.air5_volume_rms.step*.csv`。GPU 直接读取设备物性缓存并累计，
每端点下载64字节归约量，累计数组只在保存或导出时下载。
此能力限上述小型 Cartesian 验收范围，不能作为任意AIR5边界、
生产物理验证或全GPU后处理的声明。

可选 `wall_mean_render=t` 为上述两种壁面产品增加时间平均图，要求同时
开启统计和渲染，默认关闭。采用与瞬时图相同的相机和固定色标；覆盖
时长为正才生成平均图，JPEG/EPS/VTK 文件名增加 `mean_` 前缀。
VTK 保留各壁面字段的均值、方差、RMS、统计窗口和覆盖时长；
图像只输出预设字段的均值，不自动生成三维平均场图。共享接口和周期
接缝补值来自私有统计副本，不改变流场。NP=1/2 的 x/y/z 分解、空壁面
rank、同后端同拓扑精确续接和内存安全已有短测证据。

可选 `wall_separation=t` 仅用于上述 AIR5 底壁统计，默认关闭。
沿 +x 定义切向，以真实 z 长度加权平均有符号壁面剪切，逐完整步输出
`sample.wall_separation.step*.csv`。相邻非零节点正变负识别为分离，
负变正识别为再附，位置用线性插值；精确零值保留为区间，未成对零点
不报告完整分离泡。当前热气体短测入口速度为零，文件明确标记
`not_applicable_no_positive_inflow`，不能把它解释为物理上没有分离。
零点算法由独立合成序列验收，未新增或认证生产 SBLI 工况。

原位配置可选 `derivative_backend='gpu'`，默认仍为 `'cpu'`。GPU派生量
候选准入32³周期TGV、GPU求解、NP=1/2；笛卡尔路径另允许NP=4的2×2×1分解，
以及643e显式差分，
计算完整步速度梯度、Q、散度及旋度，使用私有halo，不改变推进状态。
该有界候选通过数值、内存安全、实际出图及同后端精确续接检查；重启时
切换派生量后端需显式输出配置覆盖。默认 `products='all'` 仍传基本场及
全部派生量。可显式选择 `products='q_surface'`、`'streamlines'` 或
`'q_streamlines'`，仅向既定TGV预设供给速度及必要的Q。流线预设仍含
明确命名的恒定速度跨分区诊断图，该图不是物理流场。
选定产品共用原渲染调度，不改变统计累计或检查点时刻；紧凑产品不导出
平均流线，开启的统计仍保留。产品选择改变也需显式重启配置覆盖。
当前仍下载密度和三分量动量，在主机按完整步所有权重建速度；Q产品
还需上传私有速度及下载Q，因此只是减少字段供给，不是全GPU可视化。
更大网格和其他边界不在准入范围。自定义流水线需支持新增的产品参数，
详见[原位预设说明](scripts/insitu/README.md)。

静态CURVE原位统计候选限32³ TGV、NP=1或NP=2的x/y/z分解，显式643e和FP64。
仅图片的设备`curve_demo`另有64³ NP=2 x短测，不能据此扩大统计准入。
周期基准使用 `generate_curvilinear_tgv_grid.py --mapping periodic --amplitude 0.15`；
曲壁基准使用 `--mapping y-wavy --amplitude 0.15`，即
`x=xi, y=eta+0.15*sin(xi)*sin(zeta), z=zeta`，上下y边界为bc41，x/z周期。
网格通过输入文件的 `lreadgrid=t` 和 `gridfile` 读取，不由原位配置生成。
曲壁基准是几何诊断检查，不是已充分发展的槽道或湍流物理验收。
周期配置支持既有TGV产品和体积速度统计；曲壁须选择
`products='channel_walls', derivative_backend='cpu'`，可累计速度及壁面统计，
并可选择 `wall_mean_render=t`。渲染要求GPU/EGL，CPU可关闭渲染后累计统计。

CURVE速度梯度使用完整九项逆度量映射到物理坐标，Q/散度不使用计算坐标导数代替。
曲壁内法向由几何切向叉积并指向域内，剪切沿全局+x在壁面切平面的单位投影，
热流为沿内法向的导热通量；`wall_normal_y` 是该单位法向的y分量，通常不等于±1。
保留三层、二阶单边壁面闭合，周期切向使用六阶差分，不修改推进器的边界格式。
空间统计将每个三线性八节点单元的2×2×2 Gauss体积等分到八节点，
周期末端贡献合并至唯一节点。壁面四节点面使用双线性2×2 Gauss面积等分到四节点；
`ASTR_INSITU_CURVE_WALL` 记录每张壁面的面积及三个瞬时面积加权均值，
不表示该壁面在空间上统计均匀。所有积分点Jacobian必须严格为正。

静态网格冻结至新输出的 `resources/grid.h5`，续算不再依赖原输入网格路径，
必须保留完整资源目录和COMPLETE批次。32³、四步与3+1步同拓扑续接、
物理字段和独立测度已有短测检查；曲壁面积的解析截断误差另做16/32/64几何收敛。
恒定物理速度(1,0,0)流线只用于跨x=pi及直线/端点验证，不修改求解器；
实际TGV流线用于字段、接缝和续接检查，不作为解析粒子轨迹。
不准入CURVE NP=4、任意曲壁/AIR5曲线产品、移动网格或生产规模认证。

原生TGV预设仅在集体截图和编码完成后的JPEG/EPS文件暂存或发布阶段，
允许权限拒绝、磁盘满、I/O错误记录缺帧后继续。各rank必须写入一致的
`missing.<product>.step<step>.rank<rank>.json`，几何与统计保留，图片不补写
替代品。若清理或缺帧记录失败则停止。渲染、编码、几何、数值、统计及
未知错误不降级；正常检查点也不表示所有图片均已写出。

保存过程被中断时，没有COMPLETE标记的临时批次不能用于恢复。应显式
选择上一个完整checkpoint目录，不拼接不同完整步的流场、统计或渲染
控制文件；即使重新生成校验文件，跨步时钟不一致也会拒绝恢复。
32³ TGV的NP=2/4有界检查已覆盖这一流程，不表示生产任务MPI故障容错。

同一32³周期TGV、GPU、NP=1/2或NP=4的2×2×1范围还可选择轻量速度切片：

```fortran
 products='velocity_slice', derivative_backend='gpu',
 slice_axis='z', slice_index=4,
```

这些条目加入现有 `&insitu_run`，继续配置资源预算和步数或物理时间调度。
方向可选 `x/y/z`，索引为全局节点的0到31，默认 `z,4` 对应z=π/4。
不插值任意平面，不重复选择周期上端点32。GPU只打包该平面的密度和
三分量动量，主机完成端点一致性及速度重建，不下载三维场。切片不经过
的rank仍参与集合调用，分区面切片仅由一侧提供，避免重复单元。
输出JPEG/EPS及VTK几何；切片方向或索引改变需显式重启配置覆盖。
这不是全GPU渲染，也未开放壁面量、曲线网格、AIR5或平均场切片。

ParaView 使用 XDMF3 读取单帧 `data.xdmf` 或段内 `series.xdmf`。
移动目录须连同共享坐标和父段一起移动。仅在求解器及其他写入者均停止
后，才可用 `repair_series.py` 重建段内索引、`combine_series.py` 创建
独立父链索引。它们不修改流场或修复数值状态，也不保证断电耐久性。
完整配置、目录结构和离线导出说明见
[输出工具说明](scripts/output/README.md)，验收范围见
[输出与重启计划](documents/ASTR_OUTPUT_RESTART_REDESIGN_PLAN.md)。
根 CMake 安装会携带这三项工具、说明和两份配置示例；单独安装使用
`cmake --install <build> --prefix <prefix> --component OutputTools`。

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
| ASTR_AIR5_COMPENSATION_RESTART | initialize | 旧恢复接口参数；不用于新批次，旧checkpoint恢复已关闭 |
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

新格式AIR5批次自动恢复权威q、carry及已登记缓存/历史，不依赖旧字段文件。
旧补偿checkpoint和以 `initialize` 把旧文件转为零carry的恢复方式已关闭。
不得通过清零补偿或删除版本字段修复不完整批次。

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
