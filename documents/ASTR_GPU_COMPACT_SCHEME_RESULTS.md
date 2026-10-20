# ASTR 紧致格式实施记录

更新：2026-10-20。CP0—CP10在本文限定范围内已通过；CP6最终结论见§20，CP7续算验收见§21，CP8执行变体与归因见§22。
§16至§19保留此前停点和修复过程。
旧局部导数/滤波保留作几何初始化及组件参考；已支持的TGV主程序默认使用全局算子。
CP6第1至第7项具体修订均已获批并验证；CP9统一修复开关的正式结果见§24，CP10受影响回归及输出夹具修复见§25。§23保留历史失败及恢复决策。
不操作Git写入或平台任务。

CP3独立线组件通过§12检查，§11的CPU缓存时序修订获批并通过§13集成验证。

## 1. 当前状态

| 项目 | 状态 |
| --- | --- |
| CP0 源码、矩阵与环境基线 | 已完成；根CMake的CPU-only/CUDA完整主程序及组件构建均通过 |
| CP1 独立CPU参考与CUDA Fortran分块PCR | 已完成；独立组件通过V0，不接生产分派 |
| V0数值与Compute Sanitizer | CPU与两张GPU分别270组；四种Sanitizer检查通过，见§7 |
| CP2 局部643c导数 | 已完成；解析/同状态检查、周期TGV完整步及设备检查通过，见§10 |
| CP3 局部紧致滤波 | 已通过；CPU/GPU刷新修复、scalar/full、TGV完整步及设备检查见§13 |
| CP4 全局两层线求解组件 | 已通过，见§14；由CP5接入主程序；本机UCX IPC资源清理问题单独保留 |
| CP5 全周期TGV主程序 | 已通过，见§15；受支持643c/643c输入默认采用同一CPU/GPU全局算子 |
| CP6 平板与NSCBC | 已通过V4；54组CPU/GPU十步对照含工具重跑、两组边界各12布局无滤波、代表scalar/full滤波与原随机吹吸、设备检查及显式回归；最终汇总见§20 |
| CP7 检查点与续算 | 已通过周期TGV的V5，18项运行测试；算子身份、同拓扑精确续算及缺身份/不匹配非精确迁移见§21 |
| CP8 执行变体与归因 | V6通过；私有同步、复制/缓冲及分批，28项运行和39项工具检查；本地短窗与Nsight证据见§22 |
| CP9 | V7完成；TGV与统一crashfix平板各24次正式、4次阶段、8组预检，初始化/接口另测，见§24 |
| CP10 | V8完成；根CPU/CUDA构建、55项工具和受影响运行检查、文档与能力边界同步，见§25 |

## 2. 源码及环境基线

基线提交为`1476441d2ead89adc930e1a8f28ff87a139212e8`，分支`feature/gpu_dev`。
开始时无已跟踪源码修改；已有无关未跟踪手册、演示和其他文件保留不动。
构建必须由根CMake进入；新probe使用独立模块目录，不覆盖已有构建目录。

| 项目 | 观察值 |
| --- | --- |
| 编译器 | NVFORTRAN 26.1-0 |
| CUDA检查工具 | Compute Sanitizer 2025.4.0.0 |
| 已有MPI/HDF5配置 | HPC-X MPI及HDF5 1.14.6；新构建仍需实际验证发现与链接 |
| GPU | 两张RTX 4000 Ada，每卡总显存19195 MiB；首次检查空闲18734/18622 MiB，利用率均0% |
| 主机 | 62 GiB内存，首次可用约55 GiB |
| 工作盘 | 首次可用约693 GiB；本组件不生成三维场输出 |

这些是启动前快照，不保证后续可用量不变；不为组件设置任意显存/目录硬限或自动超时。

## 3. 643c导数的实际算子

来源：`src/derivative.F90::fd_scheme_initiate/compact_fd_rhs/df_compact`，
`src/commfunc.F90::tridiagonal_thomas_proprocess/tridiagonal_thomas_solver`。
这里的导数是单位计算坐标间距下的离散导数；物理网格度量在调用层投影，不重复引入物理间距。

内点线系统为

$$
\tfrac13 d_{i-1}+d_i+\tfrac13 d_{i+1}
=\tfrac79(f_{i+1}-f_{i-1})+\tfrac1{36}(f_{i+2}-f_{i-2}).
$$

| 位置 | 实际行 |
| --- | --- |
| 左物理端点 | `d0+2*d1=-2.5*f0+2*f1+0.5*f2`，三阶偏置闭合；LHS源码旁的二阶注释不能代替实际公式 |
| 右物理端点 | 对称反向的三阶闭合 |
| 物理端点内侧一行 | `0.25*d(i-1)+d(i)+0.25*d(i+1)=0.75*(f(i+1)-f(i-1))`，四阶 |
| 旧MPI局部求解端点 | 单位行，RHS采用`0.75*Δ1-0.15*Δ2+(1/60)*Δ3`显式六阶关系 |
| 新全局周期目标 | 首尾耦合循环矩阵；MPI接口使用内点关系，不沿用旧局部单位行 |

旧导数的求解区间依npdc为`0:dim+1`、`-1:dim`、`-1:dim+1`或`0:dim`。
因此组件的矩阵行数不能直接当作物理区间数；测试重建真实三对角系数而不计入无用的端部外侧系数。

## 4. 紧致滤波的实际算子

来源：`src/filter.F90::compact_filter_initiate/compact_filter_rhs/filter_coefficient_cal/compact_filter`。
内点为`alpha*g(i-1)+g(i)+alpha*g(i+1)=R(i)`，测试使用已批准输入中的alpha=0.49。
令源码系数为c0..c5，`R=2*c0*f(i)+sum(cj*(f(i-j)+f(i+j)),j=1..5)`，其中

$$
(c_0,c_1,c_2,c_3,c_4,c_5)=
\left(\frac{193+126\alpha}{512},\frac{105+302\alpha}{512},
\frac{-15+30\alpha}{128},\frac{45-90\alpha}{1024},
\frac{-5+10\alpha}{512},\frac{1-2\alpha}{1024}\right).
$$

| 位置 | LHS与RHS处理 |
| --- | --- |
| 旧局部非物理端部 | 一侧LHS系数beta=1.11，前三行使用coefh与11个已知场节点 |
| 真实物理端部 | 一侧LHS系数beta=0.98，前三行使用coefb与7个节点 |
| 距物理端部3/4个节点 | 分别使用coef6i/coef8i的RHS，随后才进入十阶内点 |
| 新全局周期目标 | 全部唯一节点使用内点关系，首尾角元保留alpha，不引入局部分区beta行 |

物理左端前三行的RHS为`sum(coefb(k,j)*f(i0+j),j=0..6)`，k=0..2；
旧接口前三行为`sum(coefh(k,j)*f(i0-2+j),j=0..10)`。右端按同表反向取值，
具体coefb/coefh公式由上述原函数生成；距物理端部3/4个节点使用coef6i/coef8i，而非未被调用的其他表行。

旧滤波求解区间为`0:dim+3`、`-3:dim`、`-3:dim+3`或`0:dim`。
实际CPU还在三对角求解完成后，将真实端点的返回值复制为原始f值。
因此原始线系统残差应检查求解器输出，而不能把随后施加端点保持的返回场误称为原矩阵的未修改解。
端点保持及其调用相位属于后续CP3/CP6的算子集成范围，本段不擅自删除或重设计。

## 5. CP1验证设计

组件只接收三对角/循环矩阵及RHS，不连接权威流场。
覆盖导数/滤波的四种物理端点归属，以及各自真正循环系统，共10类矩阵。
每类使用127/128/129、255/256/257、511/512/513行及1/5/11个独立RHS。
RHS由独立制造解经原始矩阵相乘形成；CPU完整矩阵带主元直接消元作参考。
11个RHS只检查通用线求解接口，不代表AIR5物理适配。

GPU分块保留每块首尾两个未知量，内部使用PCR，缩减端点系统继续使用PCR并作精确周期角元修正。
块内响应、缩减系数在初始化中预处理，不截断跨块耦合，不在GPU对整线执行Thomas。
固定算子系数可在主机预处理后一次上传，求解接口本身不回读或上传完整RHS。
输入RHS不可被求解器修改，组件显式同步，工作区预分配复用。

数值门槛：原矩阵归一化残差≤1e-12，CPU/GPU逐RHS归一化最大差≤2e-10，并保留绝对差。
CPU参考还与制造解比较，避免把同源参考误差误判为GPU通过。

## 6. 组件实现

| 文件 | 实际作用 |
| --- | --- |
| `src/compact_line_solver.F90` | 后端中性矩阵描述、完整矩阵带主元LU参考，以及固定PCR/块响应预处理 |
| `src_gpu/compact_line_solver_gpu.cuf` | 设备块内PCR、缩减端点PCR和解重建，预分配设备工作区 |
| `tests/gpu_validation/compact_line_solver_probe.F90` | 独立制造解、实际矩阵族、原矩阵残差和CPU/GPU对照，输出逐组CSV |
| `src/CMakeLists.txt` | 只新增EXCLUDE_FROM_ALL的CPU/GPU probe目标；未将新模块接入astr生产源码列表 |

组件使用至多128节点的平衡块。例如129节点分为65/64，513节点分为103/103/103/102/102，
不是把余下一个节点复制或丢弃。每块只消去内部未知量，保留两个端点。
对块内部矩阵D、RHS r，以及两端耦合向量a、c，预处理及解重建为

$$
y=D^{-1}r,\quad v=D^{-1}a,\quad w=D^{-1}c,\qquad x_I=y-vx_L-wx_R.
$$

左端缩减行的对角、块内上对角和RHS分别为
`bL-cL*v(1)`、`-cL*w(1)`、`rL-cL*y(1)`；右端为
`-aR*v(m)`、`bR-aR*w(m)`、`rR-aR*y(m)`。
原块间aL/cR保持不变，不按大小截断。缩减系统仍在GPU上用PCR求解。
循环系统对真实首尾角元作rank-2 Woodbury修正，不把填充长度当作周期长度，
也覆盖单块时缩减系统只有两个未知量、角元与非对角位置重合的情况。

CPU预处理只消费固定矩阵系数；场RHS没有进入主机求解路径。
`solve_compact_gpu`内无allocate、主机场数组赋值或显式Memcpy，复用已分配工作区，
每个kernel后均经现有gpu_check强制同步。测试驱动上传RHS一次，随后依次复用同一plan求解1/5/11列，
并核对输入RHS未被修改。CUDA类型仅存在设备模块中，CPU公共模块不依赖cudafor。
设备成员采用allocatable形式，接口依据[NVIDIA CUDA Fortran指南](https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/)，
并已在本地26.1编译器上实际编译验证。

CSV中ntype=1..4对应旧CPU四种端点归属，5只是本probe的循环测试标签，不是新增ASTR边界枚举。
这里的块内/端点两层均属于单设备组件，不能称为已实现CP4的跨MPI两层PCR。

## 7. 已完成的证据

### 7.1 构建与数值

根CMake分别配置`ASTR_WITH_CUDA=OFF/ON`、`BUILD_TESTING=ON`，使用NVFORTRAN 26.1、
现有HPC-X及HDF5 1.14.6；两者的astr完整目标均构建通过，相关probe也构建通过。
没有指定新的GPU架构值。CPU-only主程序与probe的ldd检查未出现libcuda、cudart或cudafor依赖。
既有CMake兼容性和C文件警告未被隐藏，也未为本段改动无关代码。

| 运行 | 组数 | CPU/GPU最大绝对差 | 最大归一化差 | 原矩阵最大归一化残差 |
| --- | --- | --- | --- | --- |
| CPU参考 | 270 | 不适用 | 不适用 | 4.0065237402204433e-16 |
| GPU 0 | 270 | 1.1546319456101628e-14 | 1.0640423132570715e-14 | 6.9309898184017677e-16 |
| GPU 1 | 270 | 1.1546319456101628e-14 | 1.0640423132570715e-14 | 6.9309898184017677e-16 |

CPU参考相对制造解的最大绝对差为1.4432899320127035e-14。
每份CSV均有270个唯一的矩阵/长度/RHS组合，指标均有限且通过既定门槛。
这些是组件结果，不是导数截断误差、流动CPU/GPU场差或性能收益。

### 7.2 设备检查与资源

| 工具 | 实际覆盖 | 结果 |
| --- | --- | --- |
| memcheck，含leak-check full | 全部270组 | 0 errors，0 bytes leaked |
| racecheck | 全部270组 | 0 errors，0 warnings，0 hazards |
| initcheck | 全部270组 | 0 errors |
| synccheck | 全部270组 | 0 errors |

检查均设置`--error-exitcode 99`并正常返回0，各自CSV完整通过。
所用CUDA probe为带lineinfo的优化构建，不把调试慢速运行作为性能结果。
单次最大受控设备数组逻辑字节数为261432，约0.249 MiB，包含plan、内部工作区及测试输入/输出数组。
这不是CUDA上下文、驱动或第三方运行时的总显存峰值。
最大CPU参考LU矩阵为513×513个FP64值，逻辑大小2105352字节。
包括最终重链接后二次复核在内，整组原始CSV和检查日志约560 KiB，没有生成完整流场或检查点。

## 8. 复核入口

以下均从仓库根目录执行。工具链配置使用§2的现有安装，不依赖子目录独立构建或另一份ASTR代码。

```bash
cmake --build build_compact_cp01_cpu --target astr compact_line_cpu_probe --parallel 8
cmake --build build_compact_cp01_gpu --target astr compact_line_gpu_probe --parallel 8
./build_compact_cp01_cpu/bin/compact_line_cpu_probe testout/compact_cp01_20261019/v0_cpu.csv
./build_compact_cp01_gpu/bin/compact_line_gpu_probe testout/compact_cp01_20261019/v0_gpu0.csv 0
./build_compact_cp01_gpu/bin/compact_line_gpu_probe testout/compact_cp01_20261019/v0_gpu1.csv 1
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 --log-file testout/compact_cp01_20261019/memcheck.log ./build_compact_cp01_gpu/bin/compact_line_gpu_probe testout/compact_cp01_20261019/v0_memcheck.csv 0
```

racecheck/initcheck/synccheck使用相同可执行文件与完整矩阵，分别保存同名log和`v0_<tool>.csv`。
原始证据目录为`testout/compact_cp01_20261019/`；最终重链接后的GPU 0/1及四种Sanitizer复核在其`final/`子目录，
各270组且指标与首轮一致。目录由本段专用.gitignore规则保留在本地，不进入Git。
今后复核使用新的运行目录，保留本次原始证据；上述命令记录的是本次调用方式。

## 9. CP0—CP1结束时的阶段边界

CP0与CP1通过时，主求解器的CPU/GPU导数、滤波、能力检查及默认格式均未修改。
该组件结论不代表GPU紧致TGV或平板已准入。随后独立授权的CP2接线记录于下节。
跨MPI接口PCR、全局主程序、NSCBC联测、精确续算及完整步性能均仍待后续阶段。

## 10. CP2局部导数与主循环

### 10.1 实现与准入

`src_gpu/compact_derivative_gpu.cuf`构造旧643c的批量RHS，复用CP1求解器。
它保存固定系数、输入打包、RHS和解数组，不在每次调用中分配或将流场送往CPU求解。
`src_gpu/compact_operator_gpu.cuf`负责三个方向的字段适配：

1. 原始量梯度：逐线求`u/v/w/T`的计算坐标导数，按CPU次序用`dxi`投影并累加。
2. 中央对流：构造`jacob`加权的投影Euler通量，紧致求导后以负号累加RHS。
3. 黏性项：用紧致物理梯度构造应力和热/机械能通量，沿用已有halo交换，再对投影黏性通量紧致求导并正号累加。

方向顺序为x、y、z，权威`q_d`布局不变。所有新增kernel及PCR阶段都显式同步。
当前开发路径不使用原显式扩散的内点/边界通信重叠回调，避免将六阶显式RHS混入紧致算子。
CPU数值源码没有修订；`src/`仅修改根构建下的源文件及测试target接线。

该路径必须同时满足`BUILD_TESTING=ON`及`ASTR_GPU_COMPACT_DEVELOPMENT=local`，
且仅准入FP64、显式同步、非反应五变量、三维周期TGV、`643c/643c`、`lfilter=f`、无crashfix。
该标记是阶段性开发入口，不是长期局部/全局生产选项。
缺少标记的643c运行已检查为明确报错；普通643e分派不受影响。
CP4/CP5之前不能把该路径称为全局两层PCR，MPI数值仍是旧CPU的局部闭合。

设置`ASTR_GPU_COMPACT_COMPARE=1`仅在测试构建中启用同状态对照。
每个方向/用途第一次调用时读取该次已打包的设备输入，调用真实CPU
`fds%central/compact_fd_rhs`核对，再检查原局部矩阵残差。
这些小网格诊断回读不是推进必需路径，未设置该诊断标记时不发生。

### 10.2 解析导数及组件检查

`tests/gpu_validation/compact_derivative_probe.F90`覆盖16/32/64/128区间、四种旧局部端点类型，
每组同时输入常数、一次多项式、三次多项式和`exp(2x)`。
在GPU上构造RHS/求解，与独立CPU全矩阵LU及解析导数分别比较。
GPU 0/1各16组，四种Sanitizer各16组通过。

| 指标 | 两卡最大值 |
| --- | --- |
| GPU与独立CPU LU的导数解绝对差（单位计算间距） | 1.1657341758564144e-15 |
| 原矩阵归一化残差 | 1.9984014443252813e-16 |
| 常数导数绝对误差 | 0 |
| 一次/三次多项式物理导数绝对误差 | 1.6653345369377348e-15 / 5.3290705182007514e-15 |

`exp(2x)`的64→128区间全线最大误差观测阶：纯接口闭合6.14，
左物理端点3.02、右物理端点2.98、双物理端点2.98，与旧算子的端点闭合相符。
边界、内部中段和全线误差分别写入CSV，不将物理边界的三阶误差称为全域六阶。

### 10.3 TGV完整步对照

输入为Re=1600、Ma=0.1、参考温度273.15 K、dt=1e-3、关闭滤波、开启黏性。
每组CPU/GPU运行10个完整RK步。主循环以`nstep=0`起始并包含`maxstep`，
因此测试设`maxstep=9`，最终场对应`t=0.01`，不是把配置中的9误当成9个完整步。
新输出的checkpoint/volume/slices均关闭，保留已有统计量、小网格诊断快照和初始化网格文件。

| 区间网格 | NP与拓扑 | 最终q最大绝对差 | 结论 |
| --- | --- | --- | --- |
| 32³ | 1，1×1×1 | 1.4210854715202004e-13 | 通过 |
| 32³ | 2，2×1×1 | 1.9895196601282805e-13 | 通过 |
| 32³ | 2，1×2×1 | 1.7053025658242404e-13 | 通过 |
| 32³ | 2，1×1×2 | 1.4210854715202004e-13 | 通过 |
| 33×35×37 | 1，1×1×1 | 1.9895196601282805e-13 | 通过 |
| 33×35×37 | 2，2×1×1，不等长局部段 | 1.9895196601282805e-13 | 通过 |

NP=2日志确认rank 0/1分别绑定物理GPU 0/1；使用pinned host-staged halo通信。
这里比较的是每个拓扑内CPU与GPU，未要求旧局部闭合在不同拓扑间产生同一全局算子。

| 守恒分量 | 最大绝对差 | 按该分量CPU最大模及1归一化的最大差 |
| --- | --- | --- |
| rho | 1.5543122344752192e-15 | 1.5543114215489500e-15 |
| rho*u | 2.8588242884097781e-15 | 2.8588242884097781e-15 |
| rho*v | 2.8865798640254070e-15 | 2.8865798640254070e-15 |
| rho*w | 3.1480894280289107e-15 | 3.1480894280289107e-15 |
| rho*E | 1.9895196601282805e-13 | 1.1083125981541590e-15 |

初次同状态全线检查的最大绝对差为1.7763568394002505e-15，归一化差1.1840793126273111e-15，
原矩阵残差4.3318180324880200e-16。三类用途、三个方向及所有rank均有记录。
最终物理节点均有限、rho及内能为正；CFL最高0.1780797。

已有`flowstate.dat`在相同step/物理时刻的动能、拟涡能、耗散率最大绝对差分别为
4.8502868388311526e-14、4.7684078907650473e-14、5.1174342541315810e-17。
CPU文本格式只有较少有效数字，不能要求解析出的时间浮点数逐值相等；
脚本对step严格匹配、对时间按文本精度比较，未放宽场或统计量的2e-10门槛。
首次因时间文本舍入差1.734723475976807e-18失败后，仅重做文件分析，没有重跑相同流动。

最小显式回归为32³、NP=2 x分解、643e/643e、同样10完整步，CPU/GPU场与统计量通过，
最大场绝对差1.7053025658242404e-13。CPU-only及CUDA主程序均构建通过。

### 10.4 内存检查与证据

批量导数probe的memcheck（含完整leak检查）、racecheck、initcheck、synccheck均无错误，racecheck无hazard。
32³ TGV的10完整步memcheck也无错误，并再次通过场及统计量比较。
首次完整步检查使用默认UCX，工具在MPI初始化的CUDA上下文/主机指针探测中报告错误，退出99；
失败原件保留在`tgv32_np1_memcheck/`。随后仅对该host-staged检查显式使用
`pml=ob1`、`btl=self,vader,tcp`、`osc=pt2pt`、`coll=^hcoll,ucc`、`opal_cuda_support=0`，
工具正常返回0。没有关闭CUDA API错误报告，也没有据此声明CUDA-aware通信已通过检查。

证据根目录为`testout/compact_cp2_20261019/`，收口时约1.3 GiB。
包括`derivative_gpu0/1.csv`、四种`derivative_<tool>.csv/.log`、上述各TGV目录的输入、日志、
诊断快照及`summary.json`，完整步干净检查在`tgv32_np1_memcheck_ob1/`。
这些局部诊断文件不进入Git；复核应另选新目录。

```bash
cmake --build build_compact_cp01_gpu --target astr compact_derivative_gpu_probe --parallel 8
./build_compact_cp01_gpu/bin/compact_derivative_gpu_probe NEW_DIRECTORY/derivative_gpu0.csv 0
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/tgv32_np2_x --topology 2,1,1
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/tgv32_memcheck --gpu-memcheck
```

`--check-only`复用已完成的同一输入/拓扑证据，仅重新分析，不推进求解器。
CP2已通过。下一段为CP3紧致滤波RHS/闭合及scalar/full联测，生产默认仍为原显式路径。

## 11. CP3前置检查：滤波后内部原始量刷新遗漏

状态：2026-10-19源码证据已确认，用户随后批准最小CPU/GPU同步修复并继续goal；修复已通过§13验证。

### 11.1 源码证据

1. `src/mainloop.F90::time_integration_rk`的`refresh_filtered_primitives`要求
   `conschm(4:4)=='e'.and.difschm(4:4)=='e'`，643c/643c不满足。
2. 同一循环每个RK stage调用`filterq`更新守恒量`q`，随后仅在上述条件为真时调用`updatefvar`。
3. `src/comsolver.F90::filterq`的紧致分支仅改`q`，不更新`rho/vel/prs/tmp`。
4. 周期TGV的`bc.F90::boucon`不执行物理边界处理。后续`parallel.F90::qswap`以及
   `refresh_reconciled_qswap_halos`只重建接口面和halo，不遍历严格内部物理节点。
5. 随后`gradcal`读取缓存的`vel/tmp`，`convrsdcal6`同时读取新`q`和旧`vel/prs`。

因此，对于滤波实际改变的内部节点，同一stage会混用滤波前后的状态。
这与共享节点平均后的halo刷新是不同问题，后者不能替代内部原始量重建。
CP2关闭滤波，其已通过结果不受本问题影响。
GPU两个准备/推进入口目前也只为显式格式设置相同刷新条件；紧致滤波尚未接入，不能照搬遗漏。

### 11.2 已批准的最小修订

- CPU：将现有非反应五变量滤波后刷新条件扩展至`643c/643c`，每次紧致滤波完成后、
  边界/halo及导数通量计算之前调用既有`updatefvar`。
- GPU：在本段局部紧致开发入口执行对应设备原始量重建；不借此开放AIR5、紧致迎风、壁面或其他未验证组合。
- 不修改滤波系数、端点投影、x/y/z顺序、EOS或容差。CPU/GPU对照采用修正后的同阶段语义，
  不要求GPU复现旧CPU的状态混用结果。
- 验证先检查内部原始量与当前q的EOS一致性，再做已批准的组件、scalar/full及TGV完整步矩阵。

根据D3和§30的CPU缺陷人工决策规则，首次发现时先提交人工决定，获批后才执行该修复及依赖它的流动验收。

## 12. CP3独立滤波线组件

### 12.1 实现及边界

在等待§11决定期间，仅推进不依赖时间推进缓存的线组件。
新增`src_gpu/compact_filter_gpu.cuf`，由CPU初始化的标准`compact_scheme`及已生成的
`coef6i/coef8i/coef10i/coefb/coefh`复制固定系数，设备构造RHS，复用CP1 PCR求解器。
物理边界和旧MPI局部接口采用各自实际闭合；物理端点求解后恢复原输入值。
raw solution与端点恢复后的filtered数组分开保留，线系统残差只针对前者。
输入、RHS、解及系数均预分配，每个新增kernel和PCR阶段显式同步。

当前仅在`BUILD_TESTING`下的`compact_filter_gpu_probe` target中编译新组件，
未加入ASTR主程序源清单，未修改CPU数值源码、现有滤波默认值或CP2准入条件。
此组件不含x/y/z体数组编排、MPI halo时序或滤波后EOS重建。
其逐分量/批量测试不能代替最终`ASTR_GPU_FILTER_WORKSPACE=scalar/full`主程序验收。

### 12.2 组件对照

`tests/gpu_validation/compact_filter_probe.F90`使用真实CPU
`compact_filter_initiate/compact_filter_rhs/compact_filter`及独立全矩阵LU。
对可选note显式传空串，选择普通滤波而非`boundary_no_filter`。
覆盖区间数16、32、64、127、128、129、255、256、512，四种局部端点类型及1/5/11个RHS批次，
每卡108组；11个RHS仅验证代数接口，不表示AIR5准入。
输入包括常数、一次/三次多项式、光滑指数函数和Fourier模态。

| 指标 | 两卡及设备检查的最大值 |
| --- | --- |
| GPU RHS与实际CPU RHS绝对差 | 2.6645352591003757e-15 |
| raw solution与独立CPU LU绝对差 | 4.8405723873656825e-14 |
| raw solution归一化差 | 1.7398121743692514e-14 |
| 原矩阵归一化残差 | 9.6553702630077630e-16 |
| 端点恢复后与实际CPU滤波结果绝对差 | 2.9753977059954195e-14 |
| 逐个RHS与整批RHS输出差 | 0，逐值一致 |
| 常数保持误差 | 3.3306690738754696e-15 |
| 一次/三次多项式保持误差 | 1.5987211554602254e-14 |
| 物理端点保持 | 与输入逐值相等 |

Fourier测试使用计算坐标相位间隔theta=pi/2；512区间线的中间三分之一与内点传递函数

$$
H(\theta)=\frac{2c_0+2\sum_{k=1}^{5}c_k\cos(k\theta)}{1+2\alpha\cos\theta}
$$

对照，最大绝对差5.4400928206632670e-15。
这里检查远离局部端点的内区响应，不把有限线端点闭合称为精确循环滤波。
未将整个有限线的所有点都强行套用平移不变传递函数。

memcheck（含完整leak检查）、racecheck、initcheck、synccheck各覆盖108组并正常返回0，
均无错误，memcheck为0 bytes leaked，racecheck为0 hazards/0 warnings。
最终源码复核在`testout/compact_cp3_20261019/final/`，包含两卡CSV及四种工具的CSV/log。
六份CSV均完整包含108个唯一组合，所有记录有限。首轮原件保留在上级目录，整组约312 KiB。
逐分量及整批plan同时存在时，最大受控设备数组逻辑量500672字节；不包含CUDA上下文开销。

### 12.3 复核与未完成项

```bash
cmake --build build_compact_cp01_gpu --target compact_filter_gpu_probe --parallel 8
mkdir -p NEW_DIRECTORY
./build_compact_cp01_gpu/bin/compact_filter_gpu_probe NEW_DIRECTORY/filter_gpu0.csv 0
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 --log-file NEW_DIRECTORY/filter_memcheck.log ./build_compact_cp01_gpu/bin/compact_filter_gpu_probe NEW_DIRECTORY/filter_memcheck.csv 0
```

本节线组件收口时，§11具体修订尚未获批，未对有缺陷的CPU时序开展完整步验收。
随后用户批准修复；x→y→z编排、方向halo、scalar/full与CPU/GPU原始量刷新及其集成验收另记§13。

## 13. CP3主循环集成与收口

### 13.1 滤波后缓存修复

用户批准§11的具体修订后，将CPU `time_integration_rk`及GPU两个准备/推进入口的
`refresh_filtered_primitives`条件从原显式组合扩展到非反应五变量`643c/643c`。
每次滤波完成后、边界/halo及导数通量计算之前，CPU调用既有`updatefvar`，GPU调用既有设备原始量核。
本路径的黏度和导热系数在通量计算时从新温度重新求值，不继续使用滤波前的温度。
未修改滤波系数、端点投影、EOS、RK系数或误差门槛。

修复前后均用实际ASTR CPU入口运行16³、dt=1e-3的单完整步，在相同`pre_rhs`相位读取q和缓存原始量。
独立使用理想气体EOS重建rho/u/v/w/p/T：

| 相位一致性指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 速度分量最大归一化差 | 1.1887287689111029e-8 | 0 |
| 温度最大归一化差 | 3.4560193870851254e-8 | 2.2088495456932160e-16 |
| 压力最大归一化差 | 3.4560194215431610e-8 | 1.9791292085413102e-16 |

原件在`cache_before/`与`cache_after/`，保留三stage的q/原始量快照及日志。
`run_compact_derivative_tgv.py::check_primitive_eos`在旧快照上实际失败，在修正后三stage通过，
并检查当前物理节点rho/p/T严格为正。源级刷新条件回归也经历先失败、后通过。
正式短测对CPU/GPU、首末诊断步及三个stage均检查同一EOS一致性，最大归一化差为2.2088500345601643e-16。
其压力最大绝对差为1.4210854715202004e-14，来自独立EOS计算的FP64运算顺序差异。

### 13.2 设备滤波编排与存储

`compact_operator_gpu::compact_filter_state_gpu`在原滤波调用位置接线，顺序保持x→y→z。
每个方向先调用`exchange_field_halo_gpu(...,direction=axis)`，只更新hm层halo，不平均共享节点，
与CPU紧致滤波前的`dataswap(q,direction=axis)`一致。新direction为可选参数，
既有调用不传时仍依次执行原来的三个方向交换。
本方向全部分量完成后才进入下一方向的halo刷新；滤波后的原始量重建和原有qswap仍在主推进层完成。

`ASTR_GPU_FILTER_WORKSPACE=scalar/full`继续使用原选项，默认scalar：

- scalar：每方向按单守恒分量打包全部独立线，完成该分量后复用同一plan处理下一分量。
- full：每方向打包全部五个守恒分量，批量求解。

打包输入和RHS已脱离q，完成本批次求解后可直接覆盖相应q分量，不需要额外3D滤波场或完整qwork。
这不是无缓冲求解：仍保存输入、RHS、PCR消元工作区、raw solution和端点处理后的解。
固定系数与三个方向的plan均预分配，在各stage循环复用；跨方向共享工作区等后续性能候选尚未实施。
既有显式滤波的scalar/full实现不变。

| 32³、单rank滤波私有设备数组 | 逻辑字节数 |
| --- | --- |
| scalar | 5220744 |
| full | 25815912 |
| 额外场数组qwork/filter_work | 两种紧致模式均不申请 |

以上是滤波plan受控数组，不是求解器总显存；不包含导数plan、场、halo缓冲和CUDA上下文。
本轮没有进行速度收益或峰值总显存的性能验收。

### 13.3 开启滤波的TGV验证

Re=1600、Ma=0.1、参考温度273.15 K、alpha=0.49、dt=1e-3、开启黏性，均运行10个完整步至t=0.01。
沿用§10的`maxstep=9`完整步计数及关闭checkpoint/volume/slices配置。
均使用FP64、显式同步和pinned host-staged通信，NP=2的两个rank绑定不同物理GPU。

| 区间网格 | NP与拓扑 | 模式 | 最终q最大绝对差 | 逐分量归一化最大差 |
| --- | --- | --- | --- | --- |
| 32³ | 1，1×1×1 | scalar | 1.0260237104375847e-11 | 1.1870716649366474e-13 |
| 32³ | 1，1×1×1 | full | 1.0260237104375847e-11 | 1.1870716649366474e-13 |
| 32³ | 2，2×1×1 | scalar | 1.2533973858808167e-11 | 1.1218934931910437e-13 |
| 32³ | 2，2×1×1 | full | 1.2533973858808167e-11 | 1.1218934931910437e-13 |
| 32³ | 2，1×2×1 | scalar | 1.2590817277668975e-11 | 1.1630651964832395e-13 |
| 32³ | 2，1×1×2 | full | 1.1738165994756855e-11 | 1.1838243446572157e-13 |
| 33×35×37 | 1，1×1×1 | scalar | 1.4352963262354024e-11 | 1.6892493537685484e-13 |
| 33×35×37 | 2，2×1×1 | full | 1.2079226507921703e-11 | 1.6499161665558718e-13 |

最大绝对差由总能量分量给出；归一化门槛按各自分量分别计算，不让能量量级掩盖动量误差。
五个分量的最终场最大绝对差分别为
6.772360450213455e-14、1.187078653251211e-13、1.689249353768548e-13、
1.240429789729489e-13、1.435296326235402e-11。
同状态滤波与实际CPU函数对照最大绝对差6.224354365258478e-12、归一化差3.470044801606758e-14，
raw solution的原矩阵残差最高1.473281738839069e-15。

首末诊断步三个stage的pre_rhs/post_update也分别比较，六个代表工况的`all_phases.txt`均通过。
已有动能、拟涡能和耗散率统计最大绝对差分别为
4.969635813978357e-14、4.818367926873179e-14、4.672911363412524e-17，
采样step严格相同，时间按CPU原文本精度匹配，未修改2e-10数值门槛。
状态有限、rho/p/T为正，CFL最高0.1780797。

scalar/full的直接GPU比较在NP=1和NP=2 x分解上分别覆盖12/24份物理节点q快照，
均逐值一致，报告为`scalar_full_np1.txt`和`scalar_full_np2_x.txt`。
这不是跨拓扑一致性证明：本轮仍分别对照各拓扑内的同一旧局部算子。

### 13.4 设备检查、回归及证据

除§12的四类108组组件设备检查外，实际10完整步memcheck覆盖：

- 32³ NP=1 scalar：0 errors。
- 32³ NP=2 x分解full：每rank各0 errors，日志按进程分别保存。
- 最终源码清理后重建，再次运行NP=2 full完整步、EOS、统计量及memcheck：每rank各0 errors。

这些host-staged检查显式采用§10.4的ob1/主机MPI配置，不屏蔽CUDA API错误，也不据此宣称CUDA-aware检查通过。
另以exp(4x)补充滤波截断阶数检查，与CPU/GPU实现差分别判断：
旧局部接口型的8→16→32区间误差观测阶为9.93、9.89；到64区间后误差进入约1e-14舍入量级，
不再用该相邻比值声称保持十阶。双物理端点型的8/16/32/64区间误差依次为
3.9785595129244604e-5、9.2890550805867967e-7、2.0531389699808369e-8、3.8557606611345875e-10，
观测阶5.42、5.50、5.73，趋近边界闭合的六阶。
这些是滤波结果相对光滑输入的截断效应，不对它们套用2e-10的同算子CPU/GPU差值门槛。
对应`orders/convergence.csv`，带此项的probe又通过108组组件及8组加密测试的完整memcheck，无泄漏或错误。
最小数值回归均运行10完整步：无滤波643c/643c、NP=2 z分解，最大场绝对差1.4210854715202004e-13；
原显式643e/643e开启滤波、NP=2 x分解scalar，最大场绝对差2.842170943040401e-13。
CPU-only及CUDA完整构建通过；受影响刷新/存储/halo/pipeline源级合同测试65项通过。

证据仍在`testout/compact_cp3_20261019/`，收口约2.5 GiB。
正常TGV目录为表中同名的`tgv32_*`、`tgv_odd_*`，各有输入、日志、诊断快照及`summary.json`。
`tgv32_memcheck_scalar/`、`tgv32_np2_x_memcheck_full/`保存集成工具结果，
最终重建复核在`final_tgv32_np2_x_full/`。
回归目录为`nofilter_np2_z/`和`explicit_filter_np2_x/`。这些诊断证据不加入Git。

```bash
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/tgv32_scalar --filter t --workspace scalar
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/tgv_odd_full --filter t --workspace full --grid 33,35,37 --topology 2,1,1
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/tgv_memcheck --filter t --workspace full --topology 2,1,1 --gpu-memcheck
```

### 13.5 阶段结论

CP3通过。新滤波仍限显式开发身份下的非反应、笛卡尔周期643c/643c TGV。
物理端点闭合的组件检查不等于壁面流动准入；全局周期耦合、跨rank接口PCR、平板/NSCBC及精确续算尚未交付。
本次CPU刷新修复限643c/643c；CP6接入743e/643c等混合组合时仍须核对并扩展相应刷新条件。
下一段是CP4；本轮没有进入CP4，没有暂存、提交、推送或操作平台任务。

## 14. CP4：全局两层分布式线求解

本段于2026-10-19按新的CP4 goal执行，主程序仍保留CP2/CP3的局部开发接线。
新组件位于独立probe目标中，不把组件矩阵验证等同于全局CFD时间推进或物理边界准入。
最终分区复核、受影响设备检查及组件回归通过；带IPC配置的工具链映射残留不在无泄漏通过声明内。

### 14.1 所有权、方向组与固定数据

`src/compact_operator.F90`实现商余数区间划分与半开区间节点所有权。
非周期末端节点额外归最后一个rank，周期末端只保留全域首点的副本。
`compact_direction_communicator`从三维笛卡尔通信域取方向组，再按相应坐标排序；
测试刻意反转父通信域的世界rank顺序，并为不同横向线组赋予不同RHS，检查组内排序和组间隔离。

`compact_line_solver::compact_endpoint_responses`在初始化使用已有CPU PCR预处理，
求两个局部固定端点响应，不采用完整稠密LU处理生产系数。
小型完整矩阵的带主元LU只用于测试参考。
`compact_distributed_gpu.cuf`复用rank内分块PCR作为局部子系统，
对2×2块缩减接口系统预处理PCR的逆与左右系数，运行时按1/2/4……距离传输两个端点RHS。
回代保留完整局部响应，不设小耦合截断阈值。

循环系统采用`A=A0+UV^T`的秩二表达，A0保留所有MPI内部接口耦合，仅去掉全域首尾两项。
初始化用同一分布式基系统求两个固定响应，并形成2×2 Woodbury校正矩阵。
运行时广播全域首尾的缩减端点数据并在GPU完成校正，P=1/2不引入重复循环邻居。
随后向共享及周期存储副本复制最终解，不平均同一个未知量的多份结果。

这些固定数据、私有通信域与设备/主机缓冲在plan初始化时建立，批次间复用。
运行期没有系数预处理、缓冲注册或全局整线聚集，也没有整线RHS回读。
测试observer允许回读本地解并在MPI结束后离线重建全域线，不能据此推断生产路径存在整线聚集。

### 14.2 两种通信与已修正的生命周期问题

`ASTR_GPU_COMPACT_TRANSPORT=pinned|device-aware`仅在新组件入口生效，默认pinned。
每个plan复制有序线通信域，使用私有tag24001/24002与24011/24012。
pinned分配显式核对实际页锁定状态；device-aware显式核对MPI设备缓冲能力，不可用时终止，不换成另一后端。
全部内核及通信边界保留显式同步。MPI请求完成后才复用端点数组。

本轮设备检查先发现并修正两项组件问题，原始失败证据保留：

- P=1的零长度设备PCR系数数组触发每进程512字节未释放报告。现在无接口轮次时不分配该数组，
  最小红/绿对照为`leak_repro/`与`leak_fixed/`，后者0 errors、0 bytes leaked。
- NVFORTRAN数组切片回读会读到完整容量，包括未使用列。容量2、有效1列时观察到32字节回读，
  实际需要16字节；固定周期端点初始化也观察到容量11的176字节回读，而实际只需32字节。
  现在这些接口和初始化拷贝均用显式元素数`cudaMemcpy`，不通过初始化备用列掩盖超范围读取。
  容量1红/绿对照为`final_initcheck/`与`prefix_fixed_initcheck/`，容量11完整矩阵绿证据为`release_initcheck/`。

UCX IPC另有独立现象：本机HPC-X/UCX的`cuIpcOpenMemHandle`路径在结束时报告每进程512字节映射残留，
其分配栈不在ASTR数组分配中；关闭`UCX_CUDA_IPC_CACHE`后仍可复现。
未新增泄漏抑制项，未用设备重置掩盖该报告，也不把带IPC配置宣称为无泄漏通过。
明确指定`UCX_TLS=self,sm,cuda_copy`的device-aware配置已在NP=2/3完整矩阵通过带泄漏检查的memcheck，
应用仍向MPI传设备指针。这不保证MPI库内部不暂存，也不代表P2P或GPUDirect性能验收。
带IPC的完整数值对照与该显式无IPC内存检查分别记录，不能互相替代。

复用的Sanitizer XML仅针对仓库已有的UCX指针类型探测API及特定调用栈，不抑制内核访存、应用分配或IPC泄漏。
工具链路径核对通过`mpirun .../ucx_info -v`确认MPI作业内加载HPC-X自己的UCX 1.20.0；
直接在未加载其环境的shell调用ucx_info会加载系统库，不能据此判断MPI作业发生混库。

### 14.3 最终验收

最终源码及可执行文件身份见`testout/compact_cp4_20261019/source_manifest.json`。
最终probe SHA256为`c159db6efc72ba545abcc1c094e6dc9cfe5a90c724945db88542bada945303ac`。
CPU-only及CUDA主程序均由根CMake构建通过，probe使用独立Fortran模块目录，
采用NVFORTRAN 26.1、`-O2 -cuda -gpu=rdc -gpu=lineinfo`及HPC-X设备缓冲能力查询。
测试没有给组件添加生产分派，也没有启动新的完整CFD算例。

完整V2覆盖区间数127/128/129、255/256/257、511/512/513，
导数及alpha=0.49滤波两种矩阵，非周期物理行及真正循环行，1/5/11列RHS批次。
各列幅值跨10^-3到10^3，误差逐列归一化，不用最大幅值掩盖小列误差。
11列仅验证代数接口，不代表AIR5紧致格式准入。

| 总rank数 | 最终完整矩阵拓扑 |
| --- | --- |
| 1 | 1×1×1 |
| 2 | 2×1×1、1×2×1、1×1×2 |
| 3 | 3×1×1、1×3×1、1×1×3 |
| 4 | 4×1×1、2×2×1、2×1×2、1×2×2 |
| 5 | 5×1×1 |
| 6 | 3×2×1 |
| 8 | 2×2×2、8×1×1 |

两种后端共30次独立MPI运行、38232条逐rank批次记录，全部通过。
P>2均在本地两卡上共享设备，只作为正确性验证，不是多卡扩展效率证据。
全部方向组分别对照独立CPU全矩阵LU，并在原始全局矩阵上通过端点交换计算分布式残差。
另对横向组标识为0的全域线离线重建，确认每个唯一节点恰有一个所有者，再比较不同分区结果。
每个运行目录均有324组完整的全域基准线记录，不接受缺失所有者或空集合比较。

| 最终完整V2指标 | 最大值 |
| --- | --- |
| GPU/独立CPU最大绝对差 | 9.777068044058979e-12 |
| GPU/独立CPU逐列归一化最大差 | 1.0378906776255692e-14 |
| 原全局矩阵归一化残差 | 5.958025774852495e-16 |
| 跨分区最大绝对差 | 5.9117155615240335e-12 |
| 跨分区逐列归一化最大差 | 3.709650846036875e-15 |
| 共享/周期副本对CPU参考的归一化最大差 | 1.0378906776255692e-14 |
| 同拓扑pinned/device-aware逐值最大差，4860组数组比较 | 0 |
| NP=2/3的IPC/明确无IPC设备感知结果逐值最大差 | 0 |

原矩阵残差门槛为1e-12，逐列归一化场差门槛为2e-10，均未放宽。
最大绝对差随列幅值变化，与逐列归一化误差分别报告。
额外的7/8/9区间、容量1、P=1/2/3两后端检查也通过，覆盖两节点局部子系统和周期基响应所需的内部双列容量。
这些短段是代数边缘检查，不证明不足以容纳真实导数/滤波RHS模板的CFD分区可用。

最终设备检查均使用同一可执行文件，各rank日志单独保留：

| 检查 | 配置 | 结果 |
| --- | --- | --- |
| memcheck，含leak-check full | pinned，NP=3，完整矩阵，每rank324组 | 每rank0 errors、0 bytes leaked |
| initcheck | pinned，NP=3，完整矩阵及1/5/11批次 | 每rank0 errors |
| racecheck | pinned，NP=3，7/8/9区间及1/5/11批次 | 每rank0 errors、0 warnings、0 hazards |
| synccheck | pinned，NP=3，7/8/9区间及1/5/11批次 | 每rank0 errors |
| memcheck，含leak-check full | 明确无IPC的device-aware，NP=2及3完整矩阵 | 五份rank日志均0 errors、0 bytes leaked |

受影响的CP1 CPU/GPU组件各270组通过，CP3 GPU滤波108组通过；
CP1在GPU0运行，CP3在GPU1运行，CPU-only和CUDA主程序构建通过。
本段没有重跑完整TGV，也不将这些组件检查称为新的CFD物理或时间推进验收。

完整矩阵中plan受控设备数组最大279220字节；pinned接口暂存为`48×capacity`字节，本轮容量11时528字节。
这不包含调用者RHS/解、测试CPU完整矩阵、CUDA上下文或MPI内部资源，不能作为总显存。
device-aware接口层显式主机暂存计数为0；这不描述UCX内部实现。
传输计数只覆盖运行接口，初始化后清零；累积跨1/5/11三批次，广播计数是逻辑数据复制量，不是链路实测字节或性能计时。

### 14.4 证据与下一段

最终证据目录为`testout/compact_cp4_20261019/`，收口约443 MiB，保留早期失败和工具链诊断，不加入Git。

- `release_v2/summary.json`：30次最终数值运行，含各rank CSV及测试专用二进制解。
- `final_summary.json`：最终误差、4860组同拓扑后端直接比较及IPC/无IPC逐值对照。
- `release_capacity1/`：单列容量及短段退化检查。
- `release_memcheck/`、`release_initcheck/`、`release_racecheck/`、`release_synccheck/`：组件最终设备检查。
- `release_aware_memcheck_no_ipc/`：明确无IPC的设备感知完整矩阵内存检查。
- `release_aware_memcheck/`、`aware_memcheck_no_cache/`：未放行的UCX IPC映射残留证据。
- `cp1_cpu_regression.csv`、`cp1_gpu_regression.csv`、`cp3_filter_regression.csv`：最小组件回归。

```bash
cmake --build build_compact_cp01_gpu --target compact_distributed_gpu_probe -j 8
python3 tests/gpu_validation/run_compact_distributed.py --exe build_compact_cp01_gpu/bin/compact_distributed_gpu_probe --out NEW_DIRECTORY/v2
python3 tests/gpu_validation/run_compact_distributed.py --exe build_compact_cp01_gpu/bin/compact_distributed_gpu_probe --out NEW_DIRECTORY/memcheck --scope full --backends pinned --topologies 3,1,1 --sanitizer memcheck
python3 tests/gpu_validation/run_compact_distributed.py --exe build_compact_cp01_gpu/bin/compact_distributed_gpu_probe --out NEW_DIRECTORY/aware_memcheck --scope full --backends device-aware --topologies 2,1,1 3,1,1 --sanitizer memcheck --suppressions tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml --ucx-tls self,sm,cuda_copy
```

CP4组件通过，CP5尚未开始。下一段须把CPU/GPU主程序接到相同全局矩阵与RHS，
先643c/643c周期TGV关闭滤波，再启用紧致滤波及scalar/full，比较完整步同相位场及已有统计量。
不能让GPU全局解去匹配旧CPU的MPI局部闭合，也不能只放开格式字符串而保留某一显式导数。
平板743e/643c、NSCBC、检查点算子身份及性能仍分别属于CP6—CP9。
当前没有全局CFD生产放行、A800运行或性能加速比结论，没有进行Git写操作。

## 15. CP5：周期TGV主程序全局分派

### 15.1 CPU/GPU使用同一流动算子

`src/compact_runtime.F90`在`steploop`入口配置全局流动算子，
此时`geomcal`及初始流场已完成。CPU通过现有`finite_difference`抽象类型的扩展，
把`fds%central`的流动调用绑定到全局周期导数，不改写既有中央对流、梯度及黏性调用链。
紧致描述符只增加一个方向编号，由`fd_scheme_initiate`设置。
CPU `filterq`在原位置进入新的全局x→y→z滤波，并保留原单项计时入口。

几何初始化不能直接使用周期流场导数：坐标沿周期方向可能有非零全域增量，
不是首尾取值相同的周期标量。因此不在`solvrinit`立即替换几何阶段的导数实例。
此阶段仍使用原坐标导数定义，CPU/GPU读取相同已建立的网格度量。
这一区分不是允许流动MPI接口继续使用旧局部闭合。

ASTR在`parapp`中把余数区间分给高坐标rank，与CP4探针默认先给低rank的夹具不同。
运行期描述符直接使用ASTR的本地尺寸和`ig0/jg0/kg0`，从已有i/j/k通信组按方向坐标排序建立私有线通信域，
不另行重分网格。沿每个方向拥有半开区间节点，周期末端只是全域首点的副本。

CPU采用独立直接消元：先对本地内部节点做Thomas消元，保留两端点的Schur系统；
仅收集每rank两个端点RHS，直接求解2P规模的缩减接口系统，再回代本地内部节点。
系数及缩减矩阵的带主元LU只建立一次，不对每条全域长线做稠密LU。
CPU直接消元和GPU PCR的运算顺序不同，但离散矩阵、RHS及唯一节点定义相同。
GPU导数、中央对流、黏性通量及紧致滤波复用CP4分布式PCR，全部新内核显式同步。
接口解返回后复制共享/周期副本；外围原有ASTR场halo及qswap规则没有在本段重写。

全局周期RHS在所有唯一节点使用同一内点公式，不引入旧MPI接口的显式导数行或滤波1.11系数。
滤波保持x→y→z，每方向先刷新所需已知场halo；一个方向的所有分量完成后才进入下一方向。
scalar/full继续有效，默认scalar；本批输入与RHS已捕获后提交q，不新增完整qwork或3D滤波ping-pong。
仍有私有输入、RHS、解和PCR缓冲，不能称为无工作区滤波。

当前支持的全周期、非反应、FP64 643c/643c TGV默认进入全局路径，
不需要`ASTR_GPU_COMPACT_DEVELOPMENT=local`或`BUILD_TESTING`授权开关。
旧局部组件保留作几何初始化/独立探针参考，不再由主程序选择为该流动算子。
已有显式格式默认值不变，`usegpu`继续只选择执行后端。
非周期物理边界及743e/643c组合仍待CP6，混合精度/依赖同步不在本轮准入内。
全局紧致重启身份待CP7，当前明确拒绝`lrestart=t`，不把只有643c字符串的检查点当作同算子精确续算。

### 15.2 完整步与拓扑矩阵

基础数值矩阵包含28组643c/643c CPU/GPU运行对及1组显式回归。
每对CPU/GPU分别完成10个RK步，Re=1600、Ma=0.1、参考温度273.15 K、dt=1e-3、黏性开启，
开启滤波时alpha=0.49。沿用包含起点的主循环，`maxstep=9`对应从0到9共十步，
CPU/GPU的完整步CFL记录及最终RK相位用于核对推进到t=0.01，不把maxstep直接当作完成步数。

| 项目 | 基础矩阵 |
| --- | --- |
| 32³区间、开启滤波的固定基线 | NP=1；NP=2 x/y/z；NP=3 x/y/z；NP=4的2×2×1、2×1×2、1×2×2；NP=6的3×2×1；NP=8的2×2×2 |
| 33×35×37区间、开启滤波 | NP=1、2×1×1、1×3×1、3×2×1，按ASTR真实不等长分区 |
| 关闭滤波 | 32³ NP=1/2 x分解；33×35×37 NP=1 |
| scalar/full直接对照 | 32³ NP=1及2×1×1 |
| device-aware完整步 | 32³ NP=2的2×1×1及NP=6的3×2×1 |
| 同一CUDA二进制切换usegpu | 32³、NP=2；CPU分支同时对照CPU-only二进制 |
| 完整步设备检查 | NP=1 scalar memcheck；NP=2 full memcheck；NP=2 scalar initcheck；NP=2 full补带leak-check full |
| 显式回归 | 643e/643e、32³、NP=2 x分解、开启显式滤波，10完整步 |

32³与33×35×37在此均指区间数，也是周期唯一节点数；实际存储另含共享/周期端点。
NP=2日志确认rank0/1分别绑定物理GPU0/1；NP>2共享本地两卡，只是正确性测试。
体输出、切片与checkpoint事件全部关闭，仅写必要统计和测试观察器快照。
没有启动DNS长时生产运行或A800任务。

先比较CPU/GPU每rank物理节点的同相位守恒量、前RHS原始量与独立EOS、已有动能/拟涡能/耗散率。
首末诊断步均覆盖三个RK stage的pre_rhs/post_update，不只比较最终一个状态。
直接跨拓扑比较时读取`parallel.info`，去掉共享及周期重复平面，
验证每个唯一三维节点恰有一个所有者，再重建全域q，与同网格NP=1基线对照。
这属于离线测试观察器，不是GPU推进的主机场回读路径。

| 基础矩阵指标 | 最大值 |
| --- | --- |
| 最终q逐分量最大绝对差 | rho=3.5860203695392556e-14；rho*u=5.1514348342607263e-14；rho*v=5.301943878978439e-14；rho*w=4.8012375325479084e-14；总能量=6.59383658785373e-12 |
| 最终q逐分量归一化最大差 | 5.301943878978439e-14 |
| 首末诊断步全部同相位q归一化最大差 | 7.105427357601002e-14 |
| 同后端跨拓扑全域q最大绝对/归一化差 | 4.519051799434237e-12 / 5.918358184309796e-14 |
| 既有动能/拟涡能/耗散率最大绝对差 | 4.934941344458821e-14 / 5.0182080713057076e-14 / 5.648693318649478e-17 |
| 前RHS缓存与独立EOS最大归一化差 | 4.417699177817086e-16 |
| 同状态全局导数对参考RHS的原矩阵归一化残差 | 4.0293205188624245e-16 |
| 同状态全局滤波对参考RHS的原矩阵归一化残差 | 1.5679982311711883e-14 |
| 最大CFL | 0.1780797 |

同状态残差使用独立CPU stencil计算的参考RHS，包含两后端RHS舍入次序差，
不是单独扣除了RHS差异的PCR求解器内部残差；两者另由CP4矩阵探针区分。
GPU生成的RHS还独立与CPU stencil逐列比较。数值门槛仍为1e-12残差和2e-10逐分量归一化场差，未放宽。
每rank归一化检查不弱于全域统一分量尺度，原始量及物理节点均有限，rho/p/T/内能为正。
统计step一致，文本时间按已有输出精度核对，不因文本小数位差要求错误的逐位一致。

直接数组对照的最终唯一节点q最大绝对差均为0：NP=1/2的scalar/full、NP=2/6的pinned/device-aware，
以及CPU-only与CUDA二进制中的CPU分支。CPU参考本身也与各自GPU分支按同相位门槛比较。
device-aware测试显式使用`UCX_TLS=self,sm,cuda_copy`，保留应用设备缓冲MPI接口；
流场halo仍选pinned，与新线系统接口后端独立，不把halo日志中的device-aware=F误认为线系统未使用所选后端。
CP4的带IPC映射残留仍单独保留，未新增抑制或静默回退。

### 15.3 组件、设备与回归

独立CPU直接求解组件对照全矩阵LU：原V0长度、两种矩阵、非周期/循环、1/5/11列，
NP=1/2/3及3×2×1方向组、两种GPU接口传输均通过，共7776条逐rank批次记录。
CPU参考最大绝对差8.185452315956354e-12，逐列归一化差1.0287863734358713e-14。
另补7/8/9区间、容量1的两节点局部段与循环退化检查，通过。

完整时间推进的pinned memcheck覆盖NP=1 scalar、NP=2 full；initcheck覆盖NP=2 scalar，均每rank0 errors。
NP=2 full的完整泄漏检查确认每rank0 bytes leaked，不只验证kernel没有越界。
底层PCR的四类组件设备检查沿用CP4未改设备算法的证据，不把它声称为本轮新RHS内核的独立四工具全矩阵。
原局部导数16组、原局部滤波108组仍通过，分别在GPU0/1运行。
既有显式643e/643e完整步回归最大q绝对差2.8421709430404007e-13，归一化差2.8032967162969353e-15。
受影响刷新、scalar/full存储、halo和pipeline源级合同65项通过。

末轮只补回CPU新滤波分支漏记的原有单项计时，并清理本段无用局部变量，不改离散或时间推进。
基础矩阵身份保留为`numerical_matrix_manifest.json`，汇总保留为`numerical_matrix_summary.json`。
核查发现`utility::timereporter`的实际报告逻辑目前全部被注释，故这里只保留累计量与原调用合同，
不宣称恢复了可用的CPU分项计时报表，也没有启用或重写旧报告器。
`timing_cpu/`的一完整步检查正常结束；未找到单项文本输出与该空实现一致，不作为性能数据使用。

最终CPU-only及CUDA构建均通过，又补跑NP=2 full、十完整步memcheck（含leak-check full），
以及33×35×37、NP=6、显式无IPC的device-aware十完整步。
CPU/GPU的最终唯一节点q分别与改动前同条件运行对照，四组数组比较最大绝对差均为0。
新memcheck两份rank日志均0 errors、0 bytes leaked。
因此复用基础数值矩阵，不为只涉及计时记录与无用变量清理的改动重跑全部拓扑。
最终共30组紧致CPU/GPU运行对及1组显式回归对；另有一个CPU单步计时接口检查。
独立复核全部31组运行对的CPU/GPU日志，均有十条完整步CFL事件，最终状态时间均为0.01。
最大场差、拓扑差及CFL未超出§15.2表值。

### 15.4 证据与下一段

证据在`testout/compact_cp5_20261019/`，最终约6.8 GiB，不加入Git。
`numerical_matrix_summary.json`汇总基础矩阵及直接scalar/full、后端和usegpu检查；
`final_summary.json`汇总最终31组运行对、完整步计数与末轮重建对照。
`source_manifest.json`记录最终源码和CPU/CUDA二进制身份，基础矩阵身份单独保留，不覆盖旧证据。
各算例目录有输入、日志、诊断快照与独立summary。
`line_full/`、`line_tiny/`记录新增CPU直接消元与CP4 GPU组件对照。
`local_derivative_regression.csv`、`local_filter_regression.csv`记录保留的局部组件；
`memcheck_np1/`、`memcheck_np2_full/`、`initcheck_np2/`及`final_memcheck_np2/`记录完整步工具检查。
`release_np2_full_memcheck/`及`release_odd_np6_aware/`记录最终重建后的代表性复核。

```bash
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/tgv32 --filter t
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/odd_np6 --grid 33,35,37 --filter t --topology 3,2,1 --baseline EXISTING_ODD_NP1
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/aware --filter t --topology 2,1,1 --compact-transport device-aware
python3 tests/gpu_validation/run_compact_derivative_tgv.py --cpu build_compact_cp01_cpu/bin/astr --gpu build_compact_cp01_gpu/bin/astr --out NEW_DIRECTORY/memcheck --filter t --workspace full --topology 2,1,1 --gpu-memcheck
```

CP5数值矩阵通过。当前交付是全周期、非反应、FP64 643c/643c TGV的全局流动接线与短测，
不代表一般物理边界、CURVE、AIR5、重启或生产规模紧致格式已准入，也没有紧致格式加速比结论。
GPU工作区按方向预分配，完整大规模方向分批与布局/同步优化仍按CP8完成，不能从32³短测推断256³或更大批次的容量和性能。
下一段为CP6：96×48×32拉伸平板、743e/643c，按计划分常规及22/52 incoming_only边界，先导数后滤波。
本轮没有进入CP6，也没有暂存、提交、推送或操作平台任务。

## 16. CP6平板前置集成：NP=1通过，拓扑门槛未通过

### 16.1 实现与构建

本段goal为CP6/V4，不包含CP7检查点、CP8优化或CP9计时。
`src/compact_operator.F90`新增全局导数矩阵及RHS构造，保留原物理端点三阶偏置、
邻壁四阶、内部六阶关系，内部MPI分区不新增闭合。
非周期方向全域未知量数为区间数加一，仅末rank多拥有末端物理节点；周期z仍只求唯一节点。
CPU独立缩减直接求解与GPU两层PCR使用同一矩阵，主数组布局不变。

`src/compact_runtime.F90`及GPU分派扩展743e/643c的已选平板开发范围。
GPU仍调用MP7-LD对流，不因为黏性格式为紧致就误走中央对流。
黏性导数提交范围对应CPU的is:ie、js:je、ks:ke，不替换专用边界RHS。
设备打包不读取物理面外无效通量halo。同状态诊断检查RHS、原矩阵解及恢复后的物理节点。
平板滤波仍显式拒绝，等待§16.4的缓存修订及实际滤波接线，不能因源码有分派而声称验收。

根CMake的CPU-only及CUDA构建均通过。本次复用`build_compact_cp01_cpu`和
`build_compact_cp01_gpu`，但将`ASTR_USERDEFINE_SOURCE`明确设置为原M12的
`/home/dell/workspace/M12_C13/user_define_module/userdefine_hbl.F90`。
原资源只读，使用现有96×48×32子采样工具，由该UDF读取grid.2d并沿z挤出至Lz=90。
输入、场值及两端仍来自原文件；出口两个物理目标映射到同一原节点时，按严格递增索引选择相邻原节点，
不插值原流场或删除全域端点。原64×32×24选择保持不变。

早期预跑记录`cpu_preflight/`被错误触发的检查点准入拦住；修订后，随机吹吸检查点检查
仅在实际开启检查点或恢复时执行，不开放紧致检查点身份。
`cpu_preflight_r2/`使用通用空UDF，产生零坐标/零Jacobian，不能用于数值结论。
选对原网格UDF后的`cpu_preflight_r3/`通过；这些失败记录未覆盖或删除。

### 16.2 常规边界单rank十完整步

证据目录：`testout/compact_cp6_20261019/basic_np1/`，
逐项数据为`summary.json`，原日志及相位场保存在cpu/gpu子目录。
配置为96×48×32区间、Ma=12、Re=1e4、参考温度594.3 K、Sutherland、
Tw/Tinf=10、dt=0.005、原MP-LD参数、RK3、常规11/prof,21,41,50,1,1边界。
关闭全域滤波及壁面吹吸，保留原海绵层0/50、0/20、0/0；实际覆盖为x索引46:96、y索引28:48。
不写检查点、三维场序列或切片；小型验收观察器另写首/末步三stage快照及初始化几何。
真正完成10个RK步，最终物理时间0.05。

| 指标 | 结果 |
| --- | --- |
| CPU/GPU同相位守恒量最大绝对差 | 5.107025913275720e-15 |
| 同相位逐分量最大归一化差 | 4.374527938470985e-15 |
| 6组同状态梯度/黏性导数原矩阵最大残差 | 2.737596451931355e-15 |
| 前RHS原始量与独立EOS最大归一化差 | 1.577582416078840e-15 |
| massflux/fbcx/wallheatflux/wrms统计最大绝对差 | 1.759704e-13 / 4.710858e-17 / 5.434564e-18 / 9.453456e-21 |
| CPU/GPU初始化几何 | 逐值一致，Jacobian严格为正 |
| 最大CFL | 两后端均0.0257929，小于0.5 |
| 最小物理节点间距x/y/z | 5.7690878749 / 0.0882106033 / 2.8125 |
| dt乘局部扩散率估计 | 0.0062785284591 |

扩散估计使用`nu_eff=max(4mu/(3rho),gamma*mu/(Pr*rho))`及
`nu_eff_max*sum(1/h_min**2)`，其倒数约0.79636495。
它用于报告局部尺度，不能当作已求得紧致矩阵与RK组合的严格稳定边界。
全部所检查物理节点状态有限，rho/p/T/内能为正。本项是数值路径检查，不是Ma12真实气体物理验证。

### 16.3 跨拓扑失败及已确认的前置差异

使用同一CPU二进制、相同原坐标/初场子集、dt和无吹吸输入，比较
`cpu_preflight_r3/`的NP=1与`cpu_topology_x/`的NP=2、2×1×1。
全局装配保留唯一物理节点并去除共享/周期副本。
机器可读记录：`geometry_topology_audit.json`，源码/构建/UDF/二进制身份为`source_manifest.json`。

| 项目 | NP=1与x向NP=2差值 |
| --- | --- |
| 初始化q、首stage前RHS q | 各分量均逐值一致 |
| 初始化dxi11最大绝对差 | 3.700236130944751e-4，最大点在全域i=48接口 |
| 初始化Jacobian最大绝对差/归一化差 | 4.942393202263247 / 2.777525276674867e-3 |
| 首stage后q最大绝对差/归一化差 | 1.088121873715764e-7 / 9.332174585486784e-8 |
| 十步后q最大绝对差/归一化差 | 1.069201149617172e-6 / 9.169441496379346e-7 |
| 十步后最大场差位置 | 全域i/j/k=48/15/1，x动量 |

`geom::gridgeom`在全局流动分派启用前，仍以原`fds%central`计算坐标及保守度量。
它使用`fd_scheme_initiate`的rank局部接口闭合，改变MPI分区会改变非均匀网格的几何导数。
因此当前两次推进并非连几何系数也完全相同的离散问题。
这已违反本轮跨拓扑比较的前提，不能称为舍入误差，不能靠放宽2e-10门槛放行。
现有结果证明几何阶段先产生分区差异；是否解释全部演化场差，须在统一几何后的隔离复测中判断。

### 16.4 待人工决定与恢复顺序

本小节记录首次停点，当时以下两项尚未执行；随后用户明确回复“1,2均批准”，已恢复实施，最新状态见§17。

1. 混合格式滤波缓存：当前`src/mainloop.F90`及GPU两个入口的刷新条件仍只包含e/e和643c/643c，
   743e/643c未包含。建议沿用CP3修订，滤波后用当前q重建原始量，再执行边界/halo/导数通量。
   既有批准明确限643c/643c，本次扩展尚未执行。
2. 平板几何算子：建议将本轮笛卡尔挤出平板的坐标导数和度量生成接入全局耦合，
   保留当前物理端点闭合及周期z坐标的全域增量，不把坐标直接当周期标量。
   此项会改变旧几何阶段的MPI耦合，尚未执行。

获得对应批准后，先完成同坐标NP=1/2/3的几何及原矩阵检查，再恢复流动拓扑矩阵；
随后才进入incoming_only特征组合、scalar/full滤波、固定拓扑原随机吹吸、两种传输和设备内存检查。
当前没有新增CP6 Compute Sanitizer通过结论，也没有CP6性能结果。
夹具及已有M12相关源级检查28项通过；测试工具语法检查和diff检查通过。
CP6未完成，goal未标记完成；本次没有暂存、提交、推送或操作超算任务。

## 17. CP6获批修订、几何与常规流动复测、NSCBC新停点

### 17.1 已批准修订及几何隔离

用户明确批准§16.4两项修订。CPU及GPU三个准备/推进入口的滤波后刷新条件已包含
非反应五变量743e/643c；实际滤波完整步仍未验收，不能仅依据源码条件声称通过。

`compact_runtime`增加平板专用几何相位，由`geom::geomcal`在`gridgeom`前后开启/关闭。
物理x/y使用同一全局紧致矩阵；周期z对每条线取全域增量`f(N)-f(0)`，
先减去对应全域线性趋势，再求周期部分导数，最后加回线性导数。
坐标与原保守度量构造因而不会被误当作首尾相同的周期标量。
该相位仅用于已选笛卡尔挤出平板；TGV及一般CURVE几何不因此变更。

隔离复核先发现本轮新构造的实现错误：`periodic_compact_core`已切掉局部矩阵角项，
非周期构造随后从被清零的位置读取edge，意外丢失MPI接口系数。
修正为先恢复本行真实系数、施加物理闭合，再保存edge和切断局部矩阵角项。
CPU/GPU使用同一构造，GPU计算块及MPI接口均不新增数值闭合。
失败记录保存在`approved_geometry/`，未覆盖。

`approved_geometry_r2/reference_summary.json`记录零推进的独立几何检查：
NP=1、x/y/z三个方向各NP=2/3，共7布局。
`run_compact_flatplate.py::geometry_reference`独立组装全域稠密坐标导数矩阵，
与原节点坐标、仿射z以及实际Jacobian/九个度量比较。
独立坐标矩阵最大残差2.995110629369916e-17，最大跨拓扑归一化几何差2.551446376720389e-13。

`compact_distributed_probe`另新增构造合同：运行时构造的局部矩阵及edge必须逐值等于
独立全域矩阵的切片。`runtime_core_probe/summary.json`覆盖7/8/9区间、
NP=1/2/3 x向、循环/非循环导数与滤波、1/5 RHS及pinned/device-aware两后端，
共6运行、864条rank记录。GPU最大归一化差4.841615649425312e-15、
原矩阵残差2.456831946626159e-16，CPU最大归一化差3.189111755119828e-15。
device-aware明确使用`UCX_TLS=self,sm,cuda_copy`，没有静默回退或宣称IPC泄漏已解决。

### 17.2 常规无滤波完整步

`basic_f_1x1x1/`及`basic_f_2x1x1/`均完成CPU/GPU十完整步，仍为原批准的
96×48×32、dt=0.005、常规边界和零吹吸，不改变海绵层或初场。

| 指标 | NP=1 | NP=2，2×1×1 |
| --- | --- | --- |
| CPU/GPU最大同相位绝对差 | 5.329070518200751e-15 | 6.217248937900877e-15 |
| CPU/GPU最大同相位归一化差 | 4.564724792226291e-15 | 5.325512272921199e-15 |
| 同状态线系统最大残差 | 2.737596451931355e-15 | 2.737596451931355e-15 |
| 最大CFL | 0.0257929 | 0.0257929 |
| 对同后端NP=1最终场的CPU/GPU最大归一化差 | 基准 | 2.852952995141432e-15 / 3.138248294655575e-15 |

这些常规流动记录通过，但尚未覆盖y/z、余数与混合拓扑全矩阵。
后续NSCBC诊断还将全局导数的端点RHS等价改写为差值形式，减少近常量场的相消。
例如左端仍是同一三阶关系，RHS从`-2.5*f0+2*f1+0.5*f2`改为
`2*(f1-f0)+0.5*(f2-f0)`，右端对称改写。
CPU/GPU同步，系数、矩阵、独立稠密几何参考和误差门槛未变。
§17.3的单rank诊断使用改写后的二进制；本表记录来自此前的常规流动运行，不冒称同一二进制。

物理边界全局紧致滤波的矩阵、RHS、端点投影、scalar/full批次及获批的缓存刷新已接线。
bc22出口自身的显式横向滤波保留独立面缓冲，避免因该面操作额外需要全域五分量qwork。
当时这些新增组合尚未做完整滤波/吹吸验收或新增Compute Sanitizer检查；随后仅对独立常规单rank补充的结果见§18。

### 17.3 NSCBC单rank的状态相位及声波定义差异

首次`nscbc_f_1x1x1/`发现GPU边界执行顺序不等于CPU `boucon`。
本轮紧致平板新增明确顺序：入口、出口滤波/RHS、底壁、顶面滤波/RHS，最后刷新halo。
底壁保留已有交接边界的RHS；统计准备后的首stage复用同一准备状态，不重复调用边界。
已有其他格式的分派不改成该新顺序。

`nscbc_f_1x1x1_r2/`在统计梯度检查停下：端点旧写法对近常量场的CPU/GPU相消差
混入独立RHS残差，归一化值3.223324928038303e-12。
上述差值形式改写后，`nscbc_f_1x1x1_r3/`同状态最大残差5.157764716794102e-16，
最大归一化解差2.652017345693825e-15，未改残差算法、参考或1e-12门槛。
首stage前RHS物理q的CPU/GPU最大绝对差4.440892098500626e-16，说明准备后的物理状态已对齐。

完整场仍未通过：首stage后最大q绝对差1.410813180280357e-6。
原因已隔离为CPU/GPU的bc22声波策略不同，证据为`nscbc_outflow_acoustic_policy_audit.json`：

| 同状态出口点(96,42,8) | 数值 |
| --- | --- |
| 法向Mach | 11.757323697622965 |
| 旧CPU使用的全边界最大法向Mach平方 | 143.97314485406304 |
| 旧CPU强制覆盖的LODi(5) | -2.999470461927798e-5 |
| 内部物理计算得到的出射LODi(5) | -6.024420084177081e-7 |
| 该处实测CPU/GPU密度RHS差绝对值 | 0.09540694082253104 |
| 独立R5投影预测与实测向量的最大绝对差 | 2.023520240257426e-13 |

此处所有双曲特征均向外，GPU保留内部出射波；CPU仍无条件覆盖声波，且旧全边界Mach定义给出负松弛系数。
不能以GPU退回该覆盖方式消除数值差。CPU修订尚未执行，须按计划§27.2第4项决定。
CPU的boundary及conv_after轴快照位于`rhscal`整体翻号前，GPU已存实际RHS；
上述声波审计按相反号比较，不把快照符号差当成物理误差。

### 17.4 CPU多rank隔离：界面标记与出口旧halo

`nscbc_cpu_y_diagnostic/`为CPU 1×3×1十步诊断，与单rank CPU相同输入。
首stage前物理q逐值一致，缓存EOS检查误差在约1e-14以下；
初始化几何最大归一化差约4.219e-15。
尽管如此，首stage后q最大绝对差2.110457471449223e-6，十步后为1.122655114071947e-5。

`nscbc_sensor_interface_audit.json`记录首个界面分支矛盾。
在x=22、k=0，全域y节点31/32的已膨胀标记为1/0，原始传感器及物理输入在两布局相同。
单rank界面选择`1 OR 0`，1×3×1的rank2在本地界面j=-1仅选择本侧j=0的0。
CPU `solver::convrsduwd`及GPU `shock_interface_active`均有该本地数组边缘单侧规则。
该点x向RHS一致，从y向对流起出现0.04165350467884876差异，符合错误重构分支的接口位置。
这不是许可改变传感器阈值的依据，须修复MPI邻接标记供给；尚待第3项批准。

独立的`nscbc_outflow_filter_halo_audit.json`还给出bc22旧halo的直接预测。
全域(96,16,0)、1×3×1的rank1在出口横向滤波后，切向差分仍读取未滤波的下侧halo。
由同一初态与滤波后物理态单独计算该界面旧/新通量的差，可预测整个五分量RHS差：
密度分量实测8.093282633163945e-4，预测与实测向量最大差1.301042606982605e-17。
这与传感器分支问题独立，发生在对流重构前的boundary快照。
不得把复现该旧halo时序作为CPU/GPU验收目标；同步修订尚待第5项批准。

### 17.5 当前停止范围

用户先前批准的两项已实施接线，几何隔离和常规无滤波x向双卡回归通过。
新发现的第3项MPI标记，以及第4/5项CPU右x bc22声波/halo修订，已分别提交人工决定，尚未执行。
停止依赖它们的多rank/NSCBC、滤波、吹吸和性能验收，不改变输入、参考或容差绕过。
根CMake CPU/CUDA构建、29项相关工具/源级检查及diff检查通过；没有新增完整设备内存检查结论。
`approved_partial_summary.json`汇总当前证据，不把未通过的NSCBC完整场或未运行矩阵写成通过。
完整CP6 goal保持未完成，未进行Git写操作或超算操作。

## 18. 独立补充：常规单rank滤波与最小显式回归

第3/4/5项CPU/MPI修订仍未获批，未改动其数值实现，也不推进依赖它们的多rank/NSCBC矩阵。
本节仅检查不含这些故障条件的常规边界单rank、无吹吸路径，验证已批准的滤波/缓存接线。
它不能替代完整V4验收。

### 18.1 海绵层工作区缺口及修复

`basic_t_np1_scalar/`首轮运行完成了三个方向的15组滤波同状态检查及导数检查，
但在首stage的`sponge_xmax_to_work_kernel`触发CUDA 700。
原因是紧致滤波已改用私有打包线存储、不分配旧`qwork_d`，
海绵层仍将该完整五分量数组当作隐式暂存使用。
这是本轮GPU工作区集成遗漏，不是CPU数值修订；失败原件保留。

`src_gpu/sponge_gpu.cuf`现在在初始化时核算x/y实际海绵层的五分量暂存需求：
旧qwork存在时继续复用，不增加重复缓冲；不存在时预分配一份足以容纳最大方向层的线性设备缓冲，
在两个方向及所有stage循环复用，结束时释放。
原六邻点公式、更新范围、方向通信及先计算后提交的同步次序不变。
没有增加运行选项、每步分配或主机流场回读。
本夹具新增可控暂存为3102000字节，不是新增完整域五分量ping-pong数组。

### 18.2 scalar/full完整步与设备检查

证据为`basic_t_np1_scalar_r2/`、`basic_t_np1_full_r2/`，
汇总为`independent_filter_np1_summary.json`及`independent_filter_np1_manifest.json`。
仍采用96×48×32区间、Ma12、Sutherland、dt=0.005、原海绵范围、常规边界、零吹吸，
`lfilter=t`，CPU/GPU各完成10个完整步至t=0.05，不写检查点、场序列或切片。

| 指标 | scalar | full |
| --- | --- | --- |
| CPU/GPU同相位q最大绝对差 | 5.329070518200751e-14 | 5.329070518200751e-14 |
| CPU/GPU同相位q最大归一化差 | 4.342983244266397e-14 | 4.342983244266397e-14 |
| 滤波原矩阵最大残差 | 1.588980911965932e-15 | 1.588980911965932e-15 |
| 滤波后前RHS缓存EOS最大归一化差 | 1.702384581359902e-15 | 1.702384581359902e-15 |
| 最大CFL | 0.0258118 | 0.0258118 |
| Compute Sanitizer memcheck | 未运行 | 0错误、0字节泄漏 |

同后端scalar/full的首/末步三stage、共12份物理q快照逐值一致，CPU及GPU最大差均为0。
full路径在实际ASTR十步中执行memcheck及完整泄漏检查，包含本次打包海绵层缓冲的读写及释放。
此处不包含多MPI方向、NSCBC、吹吸或其他Sanitizer工具的新增通过结论。

### 18.3 最小显式回归

`independent_explicit_regression_summary.json`记录两项受影响路径的十完整步回归：

| 工况 | 目录 | CPU/GPU最大绝对差 / 归一化差 |
| --- | --- | --- |
| 常规平板743e/643e、NP=1、无滤波，复用原qwork海绵路径 | `explicit_basic_np1_sponge_regression/` | 3.774758283725532e-15 / 3.233338215041998e-15 |
| 周期TGV 643e/643e、NP=2 x向、显式滤波 | `explicit_tgv_np2_filter_regression/` | 2.842170943040401e-13 / 2.886563621156071e-15 |

根CMake CUDA重建通过，相关海绵/工作区/刷新源级检查28项通过，diff检查通过。
未修改`src/bc.F90`或`src/solver.F90`中的待批准问题，不操作Git或平台任务。
CP6目标仍未完成；下一项仍是计划§27.2第3/4/5的人工决策，而不是放宽其误差门槛。

## 19. 第3/4/5项获批后的修复及新增交点矛盾

用户随后明确“均批准”，计划§27.2三项现已授权并接线：

- CPU/GPU仅在五变量、非反应743e/643c路径中交换一层已膨胀激波标记。
  发送第1及倒数第1个内点，避开重复共享节点；计算界面取两点OR，物理外侧用端点副本。
  不改原始传感器、扩张范围或重构权重，其他格式及AIR5仍走原分支。
- CPU右x bc22采用GPU已有局部法向特征策略：超声速保留出射声波，
  亚声速松弛传入声波并匹配R5归一，倒流报错。
- CPU/GPU出口横向滤波后刷新q的halo及对应物性，再计算切向RHS；CPU所有rank参与通信。

新增GPU通信仅打包字节标记的两个面，使用预分配device及pinned host缓冲并复用，
结束时释放；无新增全场回读。私有固定MPI标签24401/24402，通信按方向顺序执行。
黏性尺度工具现对所有rank取最大值，而不是只取rank0；不是新的离散谱稳定性证明。
相关夹具、工作区、缓存及接口检查30项通过。

### 19.1 首次NSCBC复核停止

`testout/compact_cp6_20261019/approved345_nscbc_f_1x1x1/cpu/run.log`
在推进前的边界准备中报`bc22 right-x outlet backflow`，物理点(96,0,0)，
法向速度-8.357398998820964e-4。未继续GPU或后续NSCBC矩阵。

原初态在该处的壁速为零。出口y向五点端点闭合权重为
`[15/16,1/4,-3/8,1/4,-1/16]`，独立计算滤波后`(rho*u)/(rho)`得到
-8.357398998820964e-4，与CPU报错逐值一致。
该负值发生在滤波之后、重新施加bc41之前，不能直接解释为内部流动的真实倒流。
GPU的既有`nscbc_outflow_filter_y_physical_kernel`已经保留底壁交点输入，CPU则未保留。

这是新的CPU/GPU数值交点规则差异，已按计划§27.3第6项请求人工批准，尚未执行。
不关闭倒流检查，不增加壁面或随机吹吸调用，也不改变冻结的网格、dt与误差门槛。

用户随后明确“确认允许”，第6项现已获批。CPU限定当前743e/643c五变量、
右x bc22与底壁bc41的交点，保留出口横向滤波前的五分量输入状态，
与GPU既有规则一致；其他节点仍使用原滤波值。

### 19.2 第3项修订的常规无滤波拓扑回归

`approved345_basic_matrix_summary.json`记录12组CPU/GPU十步对照：
NP=1，NP=2的x/y/z方向，NP=3的x/y/z方向，
2×2×1、2×1×2、1×2×2、3×2×1及2×2×2。
各布局均与本轮单rank基线比较唯一物理节点，关闭吹吸和滤波，其他物理输入未变。

| 指标 | 12组最大值 |
| --- | --- |
| CPU/GPU同相位q绝对差 | 6.217248937900877e-15 |
| CPU/GPU同相位q归一化差 | 5.346701816224081e-15 |
| 同后端跨拓扑q归一化差 | 5.32551225759734e-15 |
| 原矩阵同状态残差 | 4.522309133726274e-16 |
| CFL | 0.0257929 |

全rank汇总的最大扩散率估计为0.009758903544262348，
最小物理间距为(5.769087874851493,0.08821060328203983,2.8125)，
`dt*max(diffusivity)*sum(1/h²)`为0.006278528459066821，
其倒数速率对应无量纲时间0.7963649496211969；不是紧致/RK组合的离散谱稳定性证明。
此矩阵使用第6项改动前的二进制，记录于`approved345_source_manifest.json`。
它不进入bc22分支，不作为第6项修复或NSCBC通过证据。超过两rank只作共享双卡正确性检查。

### 19.3 第6项后的无滤波NSCBC与常规滤波

`approved6_nscbc_f_*/summary.json`的12种布局全部完成CPU/GPU十步及同后端跨拓扑比较，
布局与§19.2相同。最大同相位q绝对差6.217248937900877e-15，
归一化差5.346701816224081e-15，跨拓扑归一化差5.32551225759734e-15，
最大CFL为0.0257929。第6项导致的初始壁面滤波倒流不再复现。

`approved6_basic_t_*/summary.json`首轮7组滤波联测通过：
NP=1及NP=2 x向各scalar/full，NP=3 y向scalar，NP=3 z向full/device-aware，
NP=6的3×2×1 full/device-aware。
最大同相位归一化场差6.422676793622846e-14，滤波原矩阵残差1.589108526556323e-15。
本轮源与二进制身份为`approved6_source_manifest.json`，CPU/CUDA根CMake构建及38项相关检查通过。

### 19.4 NSCBC滤波暴露的stage交点相位问题

`approved6_nscbc_t_1x1x1_scalar_pinned/cpu/run.log`在下一步的边界准备中停止：
点(96,0,0)，速度-2.4397453457161455e-9，`bc22 right-x outlet backflow`。
第6项保留了端点输入，但该输入并非必然等于本stage壁面目标。
首步三stage快照给出：

| RK stage | pre-RHS交点u | post-update交点u |
| --- | --- | --- |
| 1 | 0 | 5.4220131098027755e-8 |
| 2 | 0 | 5.251075276393315e-8 |
| 3 | 0 | 1.087808202062986e-7 |

这说明壁面目标在RHS前重施加，但RK更新后的边界暂态仍会进入下一stage的先行滤波和出口判断。
底壁post-update的u还包含约1e-7正负分量，滤波后交点符号可能转负。
本次不把该边界处理相位的暂态解释为实际分离或机器噪声，也不关闭倒流检查。
建议的交点分阶段修订见计划§27.4第7项，已请求人工决定，尚未实施。
停止依赖该项的NSCBC滤波、吹吸及完整V4结论。

### 19.5 常规吹吸、设备检查及显式回归补充

`approved6_basic_t_blow_2x1x1_scalar/`及`approved6_basic_t_blow_2x1x1_full/`
保留原legacy_random吹吸参数，在固定2×1×1拓扑上各完成CPU/GPU十步。
最大同相位q绝对差6.439293542825908e-14，归一化差5.653959161415696e-14，CFL为0.0365582。
同后端scalar/full的首末步三stage共24份物理q快照逐值一致，
CPU及GPU最大差均为0，记录于`approved6_blow_workspace_summary.json`。
按原调用路径，每stage一次壁面供给，GPU统计准备后的首stage复用该供给，
本轮第3至第6项均未增加随机抽样入口。此处不包含NSCBC吹吸通过结论。

`approved6_basic_t_2x1x1_full_memcheck/`的两个rank均为memcheck零错误、零字节泄漏。
额外的`approved6_basic_t_3x2x1_full_aware_initcheck/`首次initcheck报错，
栈位于`shock_sensor_gpu.cuf`的验证快照D2H赋值：NVHPC对物理切片隐式回读整个父分配，
其中包含未参与计算的halo边角。首批报告均为该主机拷贝入口，不是计算核访问。

验证快照现先由私有GPU核打包物理节点的raw sensor和字节mask，
再用精确元素数回读两份连续数组，复用打包缓冲并在结束时释放。
该分支仅供已有验证/传感器转储请求使用，不修改计算场、传感器公式或halo值。
`approved6_basic_t_3x2x1_full_aware_initcheck_r2/`同配置复跑，六rank均为initcheck零错误。
修复前后首末步三stage共72份GPU物理q快照逐值一致，见`diagnostic_pack_state_invariance.json`。
此次诊断打包修订的二进制/源码身份另记于`approved6_diagnostic_pack_manifest.json`。

| 最小显式回归 | CPU/GPU最大同相位归一化差 | 设备检查 |
| --- | --- | --- |
| `approved6_explicit_basic_np1/`，743e/643e、常规边界、无滤波 | 3.2333382150419975e-15 | memcheck零错误、零字节泄漏，含新增验证快照缓冲生命周期 |
| `approved6_explicit_tgv_np2/`，32³、643e/643e、显式滤波、2×1×1 | 2.8865636211560708e-15 | 本项未加Sanitizer |

### 19.6 本轮阶段状态

`approved6_partial_summary.json`汇总37组已通过的紧致CPU/GPU十步对照，
包括两组边界各12布局无滤波、7组常规滤波、常规滤波memcheck、
两组固定拓扑吹吸、诊断修复后的initcheck，以及§19.7的两组吹吸动态计数。
另有上述两组显式回归。

| 已通过集合的指标 | 最大值 |
| --- | --- |
| 同相位q绝对差 / 归一化差 | 6.88338275267597e-14 / 6.422676793622846e-14 |
| 同后端跨拓扑q归一化差 | 3.564217228351231e-14 |
| 导数 / 滤波原矩阵残差 | 5.157764716794102e-16 / 1.589108526556323e-15 |
| 缓存EOS归一化差 | 2.1089597830531383e-14 |
| CFL | 0.0365582 |

38项相关工具与接口检查、根CMake CPU/CUDA构建及diff检查通过。
CP6仍未通过完整V4：NSCBC滤波及其吹吸必须先决定并验证计划§27.4第7项。
本轮不将能运行十步作为物理发展或定常验收，不推进CP7，不进行Git写操作或超算操作。

### 19.7 原随机供给的动态调用核查

新增测试专用`--wall-call-check`，通过GDB在原
`wall_blowing_random_sample_legacy_wall_velocity_`入口计数，MPI按rank标记输出。
不修改求解器或随机数生成器，不额外调用采样函数，不将调试器运行时间用于性能结果。
该观察器使用本轮NVHPC二进制的原函数符号，不作为其他编译器的通用接口。

| 原吹吸、紧致滤波十步 | CPU每rank实际调用数 | GPU每rank实际调用数 |
| --- | --- | --- |
| `approved6_basic_t_blow_2x1x1_calls/`，scalar | rank0=30，rank1=30 | rank0=30，rank1=30 |
| `approved6_basic_t_blow_1x2x1_calls/`，full | 底壁rank0=30，非底壁rank1=0 | 底壁rank0=30，非底壁rank1=0 |

两组完整ASTR场/EOS/统计及原矩阵残差比较均通过。
固定种子映射、rank、本地网格、原采样函数及抽样顺序相同，动态计数相同，
首末步三stage的CPU/GPU物理场也通过原门槛。当前常规边界未发现额外抽样。
2×1×1与未启用观察器的同配置比较，CPU/GPU各24份物理q快照逐值一致，
证据为`wall_call_observer_invariance.json`、`wall_call_counts_summary.json`及
`wall_call_observer_manifest.json`。
NSCBC滤波仍停在待批准的第7项，没有因此改动边界时序或放行该组合。

## 20. 第7项获批后的交点阶段准备

用户随后明确“批准”，已恢复goal。仅本轮非反应五变量743e/643c、bc22/bc41平板采用：
入口处理后、出口滤波前，先供给本stage壁面吹吸样本，并对出口/底壁交点应用现有bc41公式。
CPU复用`noslip`的压力外推、壁温、几何法向及守恒量重建，新增可选的已供给速度与交点范围参数；
GPU增加只处理该交线的设备核，调用已有壁面速度及守恒量重建函数。
随后原底壁处理保留原压力外推和完整壁面范围，只复用该stage速度样本，不再抽样。
并非将旧压力缓存冻结到整个stage，也不扩大为所有边界的统一新架构。

旧格式及其他算例仍走原调用。没有改变dt、滤波系数、松弛系数、倒流判据或数值门槛。
`approved7_nscbc_t_1x1x1_scalar/`首先通过原失败工况的十步CPU/GPU检查，
同相位归一化场差3.942920095332304e-14，CFL为0.0258118。
`approved7_nscbc_t_blow_2x1x1_calls/`也通过，
场差4.01564138498987e-14，CFL为0.0365582，CPU/GPU两rank各实际抽样30次。
源及二进制身份为`approved7_source_manifest.json`。

### 20.1 后续矩阵及内存检查停点

第7项后的NSCBC无滤波12布局均通过。随后NP=1 full及NP=2 x向scalar滤波也通过。
`approved7_nscbc_t_2x1x1_full_pinned/`完成数值推进，但memcheck在退出时报告
每rank30份512字节泄漏，共15360字节，验收停止，没有将此运行记为通过。
初始分配栈出现CFL局部数组及顶面Mach归约kernel入口。

尝试将两处工作区改成模块级复用，`approved7r2_nscbc_t_2x1x1_full_memcheck/`
仍报告相同30份泄漏，部分初始分配栈转为统计量归约数组。
这否定了“两个局部工作区各自未释放”的初始归因，不能把GPU分配池首次分配的栈等同于最终持有者。
该复用尝试已撤回，CFL及统计量实现没有保留相关改动。

当前最小候选只将顶面Mach kernel的只按索引访问的`partial(:)`形参改为`partial(*)`，
避免创建不需要的隐式设备形状描述符。不改归约操作、线程网格或分配/释放次序。
下一步须复跑相同memcheck，随后比较修订前后同相位物理场，不能以抑制报告、
关闭leak-check或设备reset替代修复。失败证据为`approved7_reduction_lifetime_audit.json`。

### 20.2 隐式描述符修订的验证

`approved7r3_nscbc_t_2x1x1_full_memcheck/`保持同一输入和数值配置，
仅上述线性输出形参改为`partial(*)`，两rank均为memcheck零错误、零字节泄漏。
与第一次泄漏运行对照，两后端各24份首末步三stage物理q快照逐值一致，
见`approved7_descriptor_state_invariance.json`。该结果支持泄漏来自不必要的隐式设备形状描述符，
而不是CFL或统计归约公式；工作区复用试验没有保留到最终源码。
没有增加抑制规则或调用设备reset，分配/释放及归约操作保持原样。

根CUDA最终重链接后，另用当前可执行文件完成
`approved7final_relink_nscbc_t_2x1x1_full_memcheck/`，两rank仍为零错误、零字节泄漏。
前后可执行文件身份分别保留，不把重链接前的hash冒称为最终hash。

### 20.3 第7项后的完整数值矩阵

NSCBC无滤波在`approved7_nscbc_f_*/`重新完成12种布局：
NP=1，NP=2三方向，NP=3三方向，2×2×1、2×1×2、1×2×2、3×2×1及2×2×2。
均与同轮单rank基线比较唯一物理节点，原随机吹吸关闭。
符号、物理端点及MPI共享节点所有权不变，CPU/GPU同相位场和同后端跨拓扑场均通过原门槛。

NSCBC滤波覆盖单rank scalar/full、NP=2 x向scalar/full、NP=2 y向scalar、
NP=2 z向full、NP=3 y向full/device-aware、NP=3 z向scalar、
NP=6的3×2×1 full/device-aware以及NP=8 full/pinned。
源前缀为`approved7_`、`approved7r3_`和`approved7final_`，仅计入有通过summary的运行，
两次泄漏运行保留为失败证据、不混入通过集合。
`approved7final_nscbc_t_3x2x1_full_device-aware/`的六rank initcheck均零错误。

固定拓扑原随机吹吸通过：NP=2 x向滤波关、scalar/full滤波开，
以及NP=2 y向full滤波开。GDB实际计数为每底壁rank十步30次、非底壁rank零次，
CPU/GPU一致。单rank、NP=2及NP=2吹吸的scalar/full代表比较均逐值一致，
见`approved7_scalar_full_invariance.json`。

常规边界的已验证数学路径没有因第7项扩大修改；最新二进制又通过
`approved7final_basic_t_2x1x1_full/`与旧单rank基线的比较。
已通过的常规12布局、代表滤波、吹吸及工具证据与新NSCBC证据一并列入最终索引，
分别保留源及二进制版本，不声称全部运行来自同一次构建。

### 20.4 CP6最终结论及下一步

最终索引为`cp6_v4_final_summary.json`，源码/二进制身份为`cp6_final_source_manifest.json`。
54组紧致CPU/GPU十步对照包括不同存储/传输、观察器和设备工具重跑，
不等同于54个独立物理工况。主要最大值为：

| 指标 | 最大值 |
| --- | --- |
| 同相位q绝对差 / 归一化差 | 6.88338275267597e-14 / 6.422676793622846e-14 |
| 同后端跨拓扑q归一化差 | 3.564217228351231e-14 |
| 导数 / 滤波原矩阵残差 | 5.157764716794102e-16 / 1.589108526556323e-15 |
| 缓存EOS / 原有统计量归一化差 | 2.1089597830531383e-14 / 1.759703494030873e-13 |
| CFL | 0.0365582 |
| 黏性局部估计 `dt*max(diffusivity)*sum(1/h²)` | 0.009697913272268356 |

3888份首末步pre-RHS/post-update物理q快照严格有限，
最小rho为0.01911521580428372，最小压力为0.0009481753871172483，最小T为0.9630879966254554。
范围仅为实际保存相位，不冒称保存了全部中间stage；黏性估计也不是离散谱稳定性证明。
所有短测保持dt=0.005、十个完整步至t=0.05及原验收门槛。

最终CPU-only及CUDA根CMake构建、38项工具/接口检查通过。
最新显式回归`approved7final_explicit_basic_np1/`和`approved7final_explicit_tgv_np2/`
分别为743e/643e常规平板及32³、NP=2、643e/643e显式滤波TGV，
最大同相位归一化差分别3.2333382150419975e-15和2.8865636211560708e-15。

CP6/V4数值门槛关闭，没有待批准的CP6修订。超过两rank仅作共享双卡正确性检查。
不将本地十步对照升级为真实高焓物理、湍流定常、原生产大网格或A800验收。
CP6完成时下一阶段为CP7。随后独立goal的实际结果见§21，不覆盖本节原证据版本。

## 21. CP7：算子身份与周期TGV续算

日期：2026-10-19。CP7/V5通过；该阶段未启动CP8，也未进行Git写操作或平台任务。
当前证据：`testout/compact_cp7_20261019_v5_final/`、`testout/compact_cp7_v5_final.xml`；
实现和运行器分别为`src/output_runtime.F90::compact_control_file`及
`tests/gpu_validation/test_compact_restart_runtime.py`。

### 21.1 持久化和恢复合同

紧致检查点在已有control.bin尾部追加240字节，不改变原state.h5字段及批次成员：

- 记录标记ASTRCM01及算子身份astr-global-compact-v1，包含格式、halo宽度、维度/分量、重构、黏性/滤波/特征开关、
  均匀方向、边界编号，以及已启用的滤波系数、MP-LD系数和传感器阈值。算法版本同时标识当前全局矩阵、RHS及闭合定义。
- 精确恢复要求身份和上述设置一致；无身份源或不匹配源只能显式采用exact_restart=f恢复，日志报告未知/变更身份。
  这不是旧停用checkpoint格式兼容，也不绕过既有物理/几何/字段/RNG合同。
- PCR预处理、方向通信域、端点响应及私有工作区在新进程中重建，不保存至文件。
  scalar/full仅改变已验证的存储编排；在紧致路径中不把其位标志作为精确身份差异，其他合同位仍保留。
- 显式检查点没有新增尾部，exact_restart=f默认及精确源/非精确源的区别不变。
  没有新增CRC、写后全场回读或附属文件。

### 21.2 测试及结果

主矩阵为32³周期TGV、Re=1600、Ma=0.1、dt=1e-3，连续12完整步与5+7完整步至t=0.012。
CPU/GPU及NP=1/2 x分解、滤波关/开共8组主对照。GPU单rank使用scalar，双rank代表full。
比较完整state.h5的数据集及字节，不只检查五个物理分量：identity、partitions、q0001至q0011和rank_extras均逐值一致。

| 检查 | 结果 |
| --- | --- |
| 8组主矩阵同后端同拓扑精确续算 | 全数据集逐值一致，最大差0 |
| 缺身份、旧版本及不同滤波系数，CPU/GPU各一组 | 共6组精确拒绝、非精确完成恢复 |
| GPU NP=2 scalar源→full恢复 | 完整状态逐值一致 |
| CPU/GPU NP=2 643e/643e显式回归 | 无新增身份尾部，12步与5+7步逐值一致 |
| GPU NP=2 full续算memcheck | 两rank各ERROR SUMMARY: 0 errors；本项未额外启用完整泄漏检查 |
| 运行测试总计 | 18通过，0失败、0跳过 |
| 根构建及受影响工具/接口检查 | CPU-only/CUDA构建通过，另39项检查通过 |

物理节点状态有限，密度为正；日志所见最大CFL为0.1631108。
权威场/cache/rank_extras的精确比较属于本次同后端同拓扑合同，不据此声称CPU/GPU逐位相同。
本轮未启用原位累计统计或渲染，不扩展平板/壁面、CURVE、AIR5及跨拓扑紧致续算。

运行前新增测试在旧可执行文件上明确失败于“compact checkpoint lacks operator identity”；
补齐实现后通过。首轮两个辅助测试误设keep=3，被现有只允许1/2的配置读取器拒绝，
未进入RK；修正测试编排为独立生成5步源，不修改生产保留策略，不将这些失败作为数值失败或通过证据。
前序局部结果保留在compact_cp7_20261019_red/cpu_first/cpu/gpu/extra及v5目录，最终矩阵以v5_final报告为准。

复核命令：设置ASTR_COMPACT_CPU_EXE、ASTR_COMPACT_GPU_EXE为相应根构建可执行文件，
使用已记录的pinned MPI环境，执行`python3 -m pytest -q tests/gpu_validation/test_compact_restart_runtime.py`。
CP7阶段结束后另行启动CP8，其实施结果见§22。

## 22. CP8：同步、复制及批次优化

2026-10-19，V6通过。证据为`testout/compact_cp8_20261019/`，最终运行报告`runtime_final.xml`，
数值与工具结果位于`final_*`目录，成本原始值为`cost_samples.json`，候选取舍为`experiments.json`。
本节不将CP8短窗结果称为CP9中等规模或生产扩展性验收；不操作Git或远程任务。

### 22.1 实施及依赖边界

执行选项集中在`compact_line_solver_gpu`，ASTR_GPU_COMPACT_SYNC默认explicit，可选dependency；
主循环仍为ASTR_GPU_SYNC_MODE=explicit，未修改gpu_check的全局准入。中间kernel在同默认流串行提交，
每次仍检查启动错误；公开解/恢复结果、最终权威场写回及MPI前后保留完成等待，PCR共享内存屏障不变。

`solve_distributed_compact_gpu`不再将输入复制到plan%rhs；`restore_compact_line_nodes_gpu`不再将解复制到plan%y。
端点kernel的输入形状按实际nrhs声明，直接消费原解。plan%rhs只为初始化两个周期基向量保留两列。
固定矩阵、端点响应及系数复用不变。所有改变保持同一逐列算术，没有删除小耦合项或增加局部闭合。

主算子以ASTR_GPU_COMPACT_BATCH_RHS限制每批独立线/分量列，默认65535，范围1:65535，来自CUDA grid.y硬限制。
偏移统一映射回原线号与分量号，尾批仅传实际前缀；方向、分量和滤波source捕获次序不变。
每个方向的批次缓冲预分配、循环复用，目前仍是方向私有缓冲，并非一个跨方向共享arena。
执行同步、批次和既有scalar/full不进入CP7数值身份；跨设置精确续算已实际验证。

### 22.2 数值与设备结果

| 范围 | 结果 |
| --- | --- |
| CP8执行变体与CP7运行矩阵 | 28项通过，0失败、0跳过；x/y/z、scalar/full、默认与1024列分批逐值一致 |
| 跨explicit→dependency、65535→1024、scalar→full精确续算 | 完整state.h5逐值一致 |
| 旧CP7 GPU NP=2 full完整12步与新默认状态 | 全数据集逐值一致，包括缓存和rank_extras |
| TGV NP=3 z、scalar、1024列、device-aware | 同相位CPU/GPU最大归一化差4.994182151163429e-14；initcheck通过 |
| NSCBC平板 NP=2 x、full、4096列、pinned | 最大归一化场差4.1485958810228614e-14；memcheck通过 |
| NSCBC平板 NP=6 3×2×1、full、4096列、device-aware | 最大归一化场差4.8512632087832246e-14；initcheck通过 |
| 128×128×8、NP=1、full，实际超过65535列 | 最大归一化场差1.0130785099704553e-13；最大CFL0.4609456 |
| 分布式组件，NP=1/6，两种接口传输 | 原矩阵残差最大4.146360632583571e-16；独立CPU参考及memcheck通过 |
| 本地PCR共享块 | racecheck与synccheck各270组通过，最大原矩阵残差6.930989818401768e-16 |
| 构建与最小受影响检查 | CPU-only/CUDA根构建通过，另39项工具/接口检查通过 |

所有场差门槛仍为逐分量归一化最大差≤2e-10，原矩阵残差≤1e-12。
精确续算逐值门槛未改为容差门槛。超过两rank只作共享双卡正确性测试。

设备感知组件的首轮无抑制memcheck报告2322个库API探测错误，但数值完成且零泄漏。
调用栈为UCX的uct_cuda_copy_detect_vmm、Open MPI的mca_common_cuda_is_gpu_buffer等，
explicit模式同样复现，并在第一次紧致算子配置前的MPI_Init/Comm_split中出现。
随后仅使用已有且未修改的compute_sanitizer_ucx_cuda_aware.supp.xml，NP=1/6均零剩余错误、零泄漏。
不将无抑制运行记为通过，也不以规则掩盖设备读写、未初始化或泄漏。带IPC配置的旧映射清理问题仍未放行。
原位帧、渲染、AIR5及一般CURVE不在本轮新增紧致准入内。

### 22.3 本地成本与候选取舍

双卡各一rank，NP=2 x-slab，full滤波。32³ TGV与96×48×32特征平板分别12完整步，
前2步排除，后10步按每rank时间之和取最大rank；5种变体×2个算例×3轮，共30次独立运行。
轮次交错执行，关闭可视化/检查点/中间场输出，保留必要统计及边界/通信。
下表单位ms/步，均为未启用profiler的中位数及三轮最小—最大值。

| 变体 | TGV | 特征平板 |
| --- | ---: | ---: |
| CP7基线 explicit+pinned | 18.000 [17.819,18.896] | 52.390 [51.954,52.992] |
| 复制/缓冲修订 explicit+pinned | 17.400 [17.336,17.608] | 51.909 [51.520,52.186] |
| dependency+pinned | 15.706 [15.654,15.736] | 50.460 [49.646,50.725] |
| dependency+device-aware | 14.251 [14.220,14.658] | 45.584 [43.495,45.862] |
| dependency+pinned、4096列 | 18.257 [18.173,18.514] | 55.894 [55.715,55.967] |

本地代表条件下，dependency+pinned相对旧基线耗时降低约12.75%/3.68%；
再选择device-aware时分别约20.83%/12.99%。这不是CPU加速比或单/双卡扩展效率，
也不能外推A800或CP9网格。本地UCX显式使用self,sm,cuda_copy，不宣称已证明IPC/GPUDirect传输。

相同12步的Nsight全进程rank0计数为：TGV设备同步12577→6911、copy_columns kernel 1416→480；
平板12568→7560及1408→472。基线平板最耗GPU时间的是三方向MP-LD特征通量，
紧致模块优化不改变这一部分。API耗时包含GPU等待并与kernel/MPI执行重叠，不能直接求和当作整步收益。
所有启动、模块加载和初始化活动留在原始trace；RK短窗不包含启动，首步的紧致预处理也由预热排除。

平板rank0已计数的导数+滤波设备数组为124685864字节，4096列时为56254120字节，降低约54.88%。
这是这些私有数组的容量计数，不是整个GPU进程峰值，不包含权威q、halo、运行时或上下文。
4096列增加批次/通信并降低本地速度，因此保留为低显存选项，不改性能默认。
默认仍explicit、pinned及scalar滤波；复制移除和初始化缓冲缩小是等价实现，不增加另一个选择开关。

已考虑但未实施：原场获取与RHS融合可能增加映射通量重复计算，跨方向arena需要额外所有权设计，
持久请求面对不同PCR距离与尾批前缀，没有本轮充分收益证据。已有固定系数和pinned缓冲已复用，
不为“优化”重复建设缓存。上述方向保留为未来候选，不冒称已实现。
CP8结束时，下一步为CP9中等规模性能对比；后续执行记录见§23。

## 23. CP9：本地中等规模性能对比

历史状态：TGV及平板无滤波部分完成时，平板滤波组在§23.9失正，CP9/V7尚未通过。后续获批恢复和最终完成记录见§23.11、§24。
基线提交`833dbbf`，本轮只增加计时/只读预检工具和紧致NSCBC滤波阶段标记。
证据目录：`testout/compact_cp9_20261019`。

### 23.1 配置与前置检查

两张RTX 4000 Ada，驱动595.91.07，每卡19195 MiB，拓扑为NODE，无NVLink。
根CMake、NVFORTRAN 26.1，CUDA Fortran `-O2 -cuda -gpu=rdc`，FP64。
本轮TGV冻结主程序SHA256为
`0bfaca04e741abfdfa552bfd981575d6a1a9acbc82582b7d6a6d4f41136e9a89`。
MPI使用无IPC的UCX `self,sm,cuda_copy`，一rank一卡，双rank沿x分解，CPU不绑定。
公共场halo pinned，紧致端点device-aware，紧致私有dependency，外层explicit，批次65535，滤波full。
`full`对紧致路径表示每批处理多个分量，不表示额外完整q ping-pong。

TGV为256³区间，统一`dt=2.5e-4`，显式/紧致各做滤波关/开和NP=1/2。
八组三步预检均通过只读守恒状态可容许域检查，CFL为0.3262163。
该检查读取q，不刷新物性、不修补状态、不回读完整场，正式计时关闭。
32³ NP=2的12份pre_rhs/post_update物理节点q快照在诊断关/开之间逐值一致。
32项针对性计时工具测试通过。额外调用旧无场输出合同测试时，有三项已漂移断言失败：
旧测试未计入output_runtime调用、仍硬编码旧readgrid路径、把AIR5已有归约也计入非反应phase配置。
本轮未修改这些对应生产行为；排除三项后该组与针对性测试合计39项通过。

预检资源为20ms外部采样峰值，含初始化和CUDA上下文，不是所有瞬时分配的严格上界。

| 算子 | 滤波 | NP=1 每卡峰值 GiB | NP=2 每卡最大峰值 GiB |
| --- | --- | ---: | ---: |
| 643e/643e | 关 | 9.26 | 5.05 |
| 643e/643e | 开 | 9.26 | 5.05 |
| 643c/643c | 关 | 11.57 | 6.98 |
| 643c/643c | 开 | 13.16 | 8.54 |

节点进程树RSS峰值约12.28—12.99 GiB。
紧致导数已计数私有缓冲NP=1为2466401544字节/进程，NP=2为2061626028字节/进程；
开启紧致滤波另有2466403200/2061627684字节/进程。
显式full qwork为761366520字节/单rank全域，紧致滤波开启时不分配它。
容量计数与进程采样是不同口径，不直接互相代替。

### 23.2 正式计时与待补范围

本节“滤波关/无滤波”仅表示lfilter=f，即关闭全域十阶滤波。
NSCBC内部仍执行其专用六阶横向滤波：bc22出口保持原处理；bc52顶面在全域滤波关闭时
执行x/z滤波，全域滤波开启时跳过该组顶面滤波。开关RK时间差还包含这条既有条件路径的变化，
不能作为独立全域滤波耗时。边界类型、目标态和这些开关条件未被本轮另行改写。

主表为10步预热、30步采样、3轮交错，先按rank累计窗口，再取最大rank除以30。
无profiler、原位、检查点、中间整场或预检探针。保留小统计。
每轮`launch.json`、`run.log`、`summary.json`在完成后立即留存。
原有启动网格写出不在RK内；`launch_elapsed_seconds`包含它以及MPI启动、初始化、统计、结束，
不能称为纯初始化时间。正式RK不含这些完整步外工作。
不同数值算子的时间比是成本对比，不是同算子优化加速比。
阶段计时另跑，三次滤波及各差分阶段直接累计；不以滤波开/关的RK差代替核时间。

平板冻结为450×130×96区间、Ma12、原入口剖面、Sutherland、743e MP7-LD、
NSCBC [11,22,41,52,1,1] incoming_only、目标(1,1,0,0,1)、dt=0.005、吹吸幅值为零。
审查发现现有显式GPU能力函数未覆盖该模式，也不允许本组合开启滤波。
用户已批准仅复用现有核补准入并先做96×48×32、NP=1/2、10步CPU/GPU对照；
不换边界或格式，不把紧致单边数据称为完整平板对比。平板首组短测未通过，见§23.5。

### 23.3 TGV三轮正式性能结果

24次正式运行完成，三轮中位数，单位ms/完整RK步。
全部使用相同256³网格、dt=2.5e-4、40步中的后30步，保留必要边界和通信。

| 算子 | 滤波 | NP=1 中位数 [最小,最大] | NP=2 中位数 [最小,最大] | 双卡加速比 | 双卡效率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 643e/643e | 关 | 381.020 [380.503,381.195] | 457.098 [386.168,457.441] | 0.834 | 41.68% |
| 643e/643e | 开 | 460.986 [460.857,461.005] | 574.087 [566.746,577.162] | 0.803 | 40.15% |
| 643c/643c | 关 | 2329.140 [2328.426,2330.034] | 1453.943 [1445.982,1459.425] | 1.602 | 80.10% |
| 643c/643c | 开 | 2775.877 [2775.514,2776.014] | 1759.352 [1754.186,1764.133] | 1.578 | 78.89% |

紧致/显式RK成本比：单卡滤波关/开为6.113/6.022，双卡为3.181/3.065。
本轮紧致整步较慢，双卡扩展更好。显式双卡无滤波有明显轮间波动，原始三轮全部保留，
不只挑最快样本。该结论限于本机FP64、外层explicit及pinned公共场halo的配置，
不是A800结论，也不以不同算子的成本比证明精度优劣。

汇总文件：`tgv_performance.json`和`tgv_performance.csv`；
原始文件位于`timing/tgv/`。同一配置均通过3轮完整采样身份与CFL检查，末步CFL为0.3262201。
冻结二进制保存在`frozen/tgv-astr`。

### 23.4 TGV独立阶段诊断

每配置另跑12步，前2步不采样，非正式三轮主表；每项按rank窗口和的最大值除以10。
下表单位ms/完整RK步，阶段含该步全部3个stage。
`filter`为全域三维滤波，不把其他边界内横向滤波混入该名称。

| 阶段 | 显式 NP=1 | 紧致 NP=1 | 显式 NP=2 | 紧致 NP=2 |
| --- | ---: | ---: | ---: | ---: |
| 对流 | 111.710 | 845.711 | 56.630 | 429.778 |
| 梯度及黏性通量 | 55.018 | 770.980 | 28.646 | 397.235 |
| 黏性RHS | 104.522 | 598.512 | 53.145 | 308.902 |
| 全域滤波 | 80.252 | 448.674 | 113.068 | 302.869 |
| 黏性halo | 10.752 | 10.961 | 120.851 | 117.421 |
| 解变量halo | 18.341 | 18.659 | 148.690 | 153.787 |
| RK数组更新 | 31.887 | 31.923 | 16.234 | 16.351 |

私有计时没有拆开pack、PCR、端点MPI及unpack，因此不把整个紧致阶段时间称为纯PCR核时间。
对流、梯度/黏性通量、黏性RHS构成单卡紧致的主要时间，合计约2215.203 ms/步。
显式这三项合计271.250 ms/步。双卡下二者计算阶段均减少，但两类halo阶段明显增加，
独立记录支持公共场通信/同步成本抵消显式计算收益这一判断。
紧致全域滤波的成本约为显式的5.591倍（单卡）和2.679倍（双卡）。
`prepare`含子滤波和halo；各阶段最大值可能来自不同rank，不能把它们机械相加等同主表RK。
原始阶段数据位于`phases/tgv/*/summary.json`及对应run.log。

统计路径仍在GPU计算梯度和归约，只回读标量；完整RK调用明确传入copy_back=false。
本轮没有因紧致格式每步回读完整流场。完整步外的统计梯度是额外计算，
因此不能用RK中位数乘步数代替包含统计、启动和初始化的作业墙钟。

### 23.5 平板显式准入失败及人工决策

首组`admission/643e_filterf_np1/`完成CPU/GPU十步，CFL约0.01898，
所存首末步物理状态均有限且rho/p/T严格为正。但同相位post_update未通过2e-10：

| 首步stage1分量 | 最大绝对差 | 最大缩放差 |
| --- | ---: | ---: |
| rho | 3.248311416298444e-5 | 2.7825461809667366e-5 |
| rho*u | 1.2917412111068671e-6 | 1.1078493212483129e-6 |
| rho*v | 4.2796797271210593e-8 | 4.2796797271210593e-8 |
| rho*w | 4.2268679162022136e-8 | 4.2268679162022136e-8 |
| 总能量 | 4.033086820556461e-6 | 4.033086820556461e-6 |

密度最大差位置为(96,0,0)，即出口/底壁交点。
该失败不是机器舍入，也不是CFL或状态失正；后续三个短测及全部平板性能矩阵未启动。

明确不一致：`src/bc.F90::boucon`的交点阶段准备、`outflow_nscbc`的壁点保留、
滤波后qswap及已修正的声波公式均受紧致格式条件限制。
显式CPU仍以出口面最大法向Mach数作旧的无条件声波覆盖；GPU的bc22核使用局部法向Mach、
仅传入声波及已归一化的压力松弛公式。不能为匹配CPU而恢复这一旧分支。
此外，显式GPU pre_rhs记录在出口NSCBC处理前，CPU记录在boucon后，
因此首个pre_rhs最大差0.014322237688074502不能称为严格同相位误差。
上表post_update在两端均位于RK/海绵更新后，是真实的同相位失败。

已请求批准把CP6既有CPU修正扩到本轮显式平板，并同步GPU边界顺序与观察点。
获批后重编CPU/GPU，先完成原4组小网格，再做8组中等网格预检、24次正式和4次阶段诊断。
保持dt、目标状态、内部差分/滤波公式和误差门槛不变，不重跑已冻结TGV昂贵矩阵。
本次新准入条件已撤销，正常构建目录可执行文件恢复为0bfaca04...验证版本。
失败候选SHA256为`e1d8d070a45dd873429523e53c49b7cc1d0b42ef6192ae991aed02d516a3ecf5`，
保存在`frozen/plate-astr`，仅作失败复核。
没有运行遗留后台任务，没有操作Git暂存/提交/推送或超算。

### 23.6 获批的显式边界修正与短测

2026-10-20用户确认扩展既有CPU修正，恢复本轮工作。
`bc::explicit_profile_nscbc_corrections_enabled`限定非反应五变量、bl/profile、
743e/643e、[11,22,41,52,1,1]且incoming_only非激波目标。
此组合复用已有的交点准备、壁点保留、出口滤波后qswap、局部法向Mach与正确声波幅值。
原紧致分支不变，其他显式组合不变，没有新增用户开关或改变内部离散公式。

GPU共享prepare_profile_nscbc_gpu，在阶段内按格式选择原显式或紧致全域滤波。
边界顺序保持入口、交点准备、出口滤波及物性/halo刷新、出口RHS、底壁、顶面处理；
同一stage壁面供给复用，不增加随机抽样。pre_rhs现在位于全部边界处理之后。
更改前失败记录保留在admission/；新记录位于admission_corrected/。

| 路径 | NP | 滤波 | 最大缩放场差 | 最大绝对场差 | CPU/GPU最大CFL |
| --- | ---: | --- | ---: | ---: | ---: |
| 743e/643e | 1 | 关 | 3.2333382150419975e-15 | 3.774758283725532e-15 | 0.0189798 |
| 743e/643e | 2 | 关 | 3.2333382150419975e-15 | 3.774758283725532e-15 | 0.0189798 |
| 743e/643e | 1 | 开 | 2.71856350150488e-15 | 3.3306690738754696e-15 | 0.0189993 |
| 743e/643e | 2 | 开 | 2.356647125219661e-15 | 2.886579864025407e-15 | 0.0189993 |
| 743e/643c回归 | 2 | 开 | 4.1485958810228614e-14 | 4.8183679268731794e-14 | 0.0258118 |

每项96×48×32、dt=0.005、10步，NP=2为x分解，full滤波，原吹吸幅值为零。
比较首末步各stage的pre_rhs/post_update及输出统计，全部通过2e-10。
所有已存物理节点状态有限且rho/p/T>0；五组联合最小rho=0.02168203677756717，
p=0.0005742820115509266，T=0.9543282420768727，三个最小值不必位于同一工况或节点。
紧致回归还通过已有独立线参考、残差及EOS检查。

根CMake CPU-only/CUDA构建通过。冻结GPU为frozen/plate-corrected-astr，SHA256
`1a4b2590f0949310f07c8acb0cb99f322bc9078a56b825be443af4547dfdf07b`；
CPU为frozen/plate-corrected-cpu-astr，SHA256
`def4bce1c471cec1026219cabe465d367d49d65daaac0e1b28541fa194aad5ca`。
TGV保留原冻结版本及所有正式结果，不混合两种二进制的同一性能组。
完成这些数值门槛后才启动450×130×96的三步预检；正式性能仍按原40步/3轮协议。

### 23.7 平板中等规模预检与无滤波性能

8组三步只读状态/CFL/资源预检通过，dt=0.005不变，最大CFL=0.0462834。
以下设备峰值为每物理GPU的进程显存采样值，含初始化与上下文。

| 梯度/黏性格式 | 滤波 | NP=1峰值 GiB | NP=2每卡最大峰值 GiB |
| --- | --- | ---: | ---: |
| 643e | 关 | 3.547 | 1.998 |
| 643e | 开 | 3.547 | 1.998 |
| 643c | 关 | 5.553 | 3.357 |
| 643c | 开 | 7.332 | 4.607 |

主机进程树RSS峰值约4.443—4.941 GiB。紧致导数私有容量NP=1/2分别
2139403068/1447064844字节/进程，紧致滤波另有2139404724/1447066500字节/进程。
显式full qwork为278204280/142421280字节/进程；紧致滤波开启时不分配该qwork。
容量计数与采样峰值不是同一口径，不把二者相加作为总峰值。

无滤波12次正式运行通过原40步协议（预热10、计时30，3轮）。
首轮4项先完成；滤波组失败并定位后，只补无滤波后两轮，其相对交错顺序保持原计划。
全部使用同一冻结plate-corrected-astr，没有加入阶段/状态/模板探针。
下表为ms/完整RK步，仍固定743e MP7-LD对流。

| 梯度/黏性格式 | NP=1中位数 [最小,最大] | NP=2中位数 [最小,最大] | 双卡加速比 | 双卡效率 |
| --- | ---: | ---: | ---: | ---: |
| 643e | 862.933 [862.679,863.909] | 546.521 [544.272,546.904] | 1.579 | 78.95% |
| 643c | 1468.708 [1467.525,1471.458] | 846.683 [844.381,847.090] | 1.735 | 86.73% |

紧致/显式整步成本比为1.702（单卡）和1.549（双卡）。
这不是完整对流格式的替换：固定的是743e重构/通量，选定梯度也被激波传感器调用。
不能将整个RK差称为纯黏性导数核差。
汇总为plate_unfiltered_performance.json/csv；plate_performance.json/csv当前也仅含这4个无滤波配置，
不代表原目标8个配置已经完成。原始数据在timing/plate/，各配置均有3份成功summary.json。

### 23.8 平板无滤波独立阶段诊断

4次额外12步运行，前2步不采样，不替代原目标的滤波开启阶段诊断。
单位ms/完整RK步，各项含全部3个stage。未启用滤波，不伪造滤波时间为零。

| 阶段 | 显式 NP=1 | 紧致 NP=1 | 显式 NP=2 | 紧致 NP=2 |
| --- | ---: | ---: | ---: | ---: |
| MP7-LD对流 | 671.634 | 671.425 | 370.104 | 370.448 |
| 梯度及黏性通量 | 30.732 | 261.140 | 16.086 | 150.720 |
| 黏性RHS | 47.849 | 195.002 | 26.068 | 96.079 |
| 黏性halo/等待 | 0.996 | 0.975 | 32.117 | 10.253 |
| 首stage边界准备 | 19.824 | 20.150 | 27.627 | 26.134 |
| RK数组更新 | 11.195 | 10.870 | 5.323 | 5.524 |

原始数据在phases_unfiltered/plate/。
传感器在convection计时区间前调用compute_gradcal_gpu，未被单独计时；
其梯度也随643e/643c切换，仍计入完整RK。表中梯度/黏性通量不是全部梯度调用之和。
compact阶段包含其pack/PCR/端点通信/unpack，不是纯PCR核耗时。
MPI阶段含等待，最大值可能来自不同rank；不能据某项较小就断言通信字节数或纯传输时间更低。

### 23.9 滤波失正的原生复现

显式单卡滤波开启在零起算step31（第32个时间步）后触发CFL非有限状态错误。
上一步CFL约0.04628，未达到原0.5门槛。
只读检查未改变q、物性或RHS，也未启用crashfix、截断或额外滤波限制器。

1. 相同冻结版本的complete-RK检查，在step31记录到(1,38,0)动量及能量NaN。
2. 测试构建的阶段检查将首次失正前移定位到step31、stage2、全域滤波之后。
3. 在同一位置(4,23,0)采集滤波前11³×5局部采样及滤波后5个分量，总53312字节。
4. compact_filter_gpu_probe的--explicit-window模式在CUDA初始化前，直接调用原CPU
   filter_coefficient_explicit及spafilter10exp，按相同x/y/z顺序复算该点。
   不运行CPU完整CFD，不使用Python流场推进器。

| 诊断量 | 实际值 |
| --- | ---: |
| 局部输入最小内能 | 1.8627873478249985e-4 |
| GPU滤波后rho | 3.7748882338506679e-2 |
| GPU滤波后总能量 | -1.2800270147928544e-4 |
| CPU滤波后内能 | -1.6793346937688837e-4 |
| CPU/GPU滤波结果最大绝对差 | 1.3877787807814457e-17 |

输入局部状态均有限、rho及内能为正，CPU原滤波得到与GPU相同的负内能。
因此该失效是现有滤波对该状态不保持Euler可容许域，不是此次GPU移植误差，
也不能以机器误差或低CFL放行。背景线性滤波的谱稳定性不等同于保持rho/p/T正值。
没有据此声称完整CPU算例也在相同步数失效；本次CPU证据是同状态局部算子复算。

证据分别位于diagnostic/643e_filtert_np1_state、643e_filtert_np1_stages、643e_filtert_np1_window；
最后目录保存diagnostics/profile_filter_window.bin及cpu_reference.log。
测试专用ASTR_VALIDATION_PROFILE_STAGE_STATE、ASTR_VALIDATION_PROFILE_FILTER_WINDOW均默认关闭，
并受ASTR_BUILD_TESTING控制，不是生产修复开关。未回读完整场或增加检查点。
原性能仍使用plate-corrected-astr，不混入诊断二进制结果。

#### 23.9.1 同状态逐方向定位：x滤波首先失正

按用户要求只定位，不增加保正修复，不改变滤波、物理条件或性能计时矩阵。
扩展compact_filter_gpu_probe的既有--explicit-window诊断，继续直接调用原CPU
spafilter10exp，保留x、y、z各方向的中间守恒量。没有重新推进CFD，
没有修改生产核或先前冻结的性能二进制。

诊断点为全局零起算(4,23,0)。读取同一运行的datin/grid.h5得到
(x,y,z)=(108,1.5453606961027473,0)，不是j=0壁面节点。
正式失败运行及局部捕获运行的datin/wallbs.dat均为幅值0，配置区间20≤x≤60。
因此本次既未启用吹吸，捕获点也不位于该名义吹吸带。

以下是该点的原生CPU逐方向复算，内能密度定义为
`q(5)-0.5*sum(q(2:4)**2)/q(1)`，不是单位质量内能。

| 相位 | 密度 | 总能量密度 | 动能密度 | 内能密度 |
| --- | ---: | ---: | ---: | ---: |
| 滤波前 | 3.8455126914149505e-2 | 2.1911809924700769e-4 | 3.2839364464461481e-5 | 1.8627873478254620e-4 |
| x滤波后 | 3.7849024410667287e-2 | -9.6128922751569611e-5 | 3.9520284286731906e-5 | -1.3564920703830152e-4 |
| y滤波后 | 3.7748882338506685e-2 | -1.2800270147928682e-4 | 3.9930767897600371e-5 | -1.6793346937688718e-4 |
| z滤波后 | 3.7748882338506692e-2 | -1.2800270147928796e-4 | 3.9930767897600405e-5 | -1.6793346937688837e-4 |

捕获的整个11³输入窗口内能密度最小值为1.8627873478249985e-4，所有输入
密度及内能均正且有限。前一RK更新的全场可容许性检查也已通过。
本表在同一输入上将首次失正定位到x滤波；y滤波进一步降低内能，
z滤波仅有约1e-18的数值变化。三方向最终结果与捕获GPU结果最大绝对差仍为
1.3877787807814457e-17。这里的中间方向数据来自CPU算子复算，
不是新采集的GPU中间快照，也没有据此确定全域唯一最先失正节点。

i=4位于入口侧0-6-6-6-8-10闭合中的八阶行，实际模板为i=0..8，
对应x=0,27,54,...,216。CPU spafilter10exp和GPU filter_x_global_kernel
都在该行调用相同的八阶系数。权重按i递增为
`[-1,8,-28,56,186,56,-28,8,-1]/256`，权重之和为1但含负值。
滤波前，总能量密度在i=3、4、5分别为1.6508557313329900e-2、
2.1911809924700769e-4、3.8910758635743475e-2，已有很深的局部低谷。
将捕获的正总能量逐项乘上述权重，直接得到-9.6128922751569611e-5。
因此本次不是仅在扣除动能时因近零相消失正，而是x滤波已经产生负总能量。

直接触发环节已确定为既有八阶闭合对该输入分布产生负向超调。
本诊断不证明闭合系数抄写错误，也未确定此前低能量谷由初值、入口处理、
先前滤波还是RHS演化中的哪个环节首先形成。该上游成因与本次首次越过
可容许域的环节应分开，后续如继续追踪应比较更早的同相位状态。

原始窗口及旧参考不覆盖；新增逐方向输出保存在
`diagnostic/643e_filtert_np1_window/directional_cpu_reference.log`。
根CMake的compact_filter_gpu_probe目标重编通过，原生局部复算再次通过。
没有启动完整算例、修改保正算法或执行Git操作。

### 23.10 当时未完成项（由§24后续关闭）

| 原目标项 | 状态 |
| --- | --- |
| TGV 24次正式、滤波开/关、NP=1/2 | 完成，见23.3 |
| TGV 4次独立滤波开启阶段诊断 | 完成，见23.4 |
| 平板CPU边界修正及4组显式短测、紧致回归 | 获批并通过，见23.6 |
| 平板8组三步预检 | 通过，但不能代替40步滤波可容许性 |
| 平板无滤波12次正式、NP=1/2 | 完成，见23.7 |
| 平板无滤波4次独立阶段诊断 | 额外完成，见23.8 |
| 平板滤波开启12次正式 | 未完成；首个显式NP1运行失败，其他组合未作正式计时 |
| 平板滤波开启4次阶段诊断 | 未完成；没有用提前截短失败窗口冒充验收 |

该历史停点曾请求五变量滤波保正与守恒一致性修订设计。随后用户选择直接复用
现有可选crashfix，恢复检查结果见§23.11；没有实施新的保正限制器，
没有改变滤波系数、步长、网格、初始/边界状态或采样长度。
既不将36次成功正式运行称为48次全部通过，也不将补齐无滤波子集定义为整个CP9完成。

最终工具检查45项通过，另3项旧断言漂移仍单列，见23.1，不把排除项称为通过。
根CPU/CUDA主程序及新增原生局部参考分支构建通过，局部参考复算误差1.39e-17。

### 23.11 用户批准的crashfix恢复路径

用户要求“加crashfix修复，能通过就行”。本轮复用既有非反应五变量邻点平均修复，
不改滤波公式或增加截断归零。仅在controller的lcracon=t时启用；
关闭时保留原推进与失效行为。GPU调用crashfix_gpu::repair_crashfix_state_gpu，
修复在设备上完成，不回读整场调用CPU。保留原RK后修复，并在全域滤波后、
边界/RHS消费该状态前补充同一恢复步骤。修复前后各刷新一次解halo和相应物性，
避免以旧缓存判断邻点资格或把修复结果留在过期halo中。
CPU同步在非反应五变量滤波后的物性刷新处调用既有crashfix，前后执行qswap。
global compact仅为已有profile平板放开lcracon，未扩大到反应流或紧致TGV修复。
既有crinod历史与MP通量耗散处理不变。

新增工具选项--crashfix映射到原controller开关，默认不启用；
summary显式记录crashfix、crashfix_nonzero_calls、crashfix_repaired_node_visits。
修复次数是所有rank物理数组上的访问计数，包含共享/周期副本，不能称为唯一网格点数。

4组GPU恢复检查均保持450×130×96、dt=0.005、40个完整步、原边界及零吹吸，
使用full工作区、pinned场halo和无IPC的device-aware紧致端点通信。
启用逐阶段及完整步可容许性检查和20ms资源采样，所有完整步状态有限且rho/p/T>0。

| 导数/滤波路径 | NP | 完整步 | 非零修复调用 | 节点修复访问 | 最大CFL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 743e/643e，显式滤波 | 1 | 40 | 3 | 291 | 0.0462830 |
| 743e/643e，显式滤波 | 2 | 40 | 3 | 291 | 0.0462830 |
| 743e/643c，紧致滤波 | 1 | 40 | 0 | 0 | 0.0462726 |
| 743e/643c，紧致滤波 | 2 | 40 | 0 | 0 | 0.0462726 |

显式修复发生在零起算step31、34、37的stage2，每次97个节点访问。
单卡最大总能量密度改变量约2.9702541720567546e-2；这不是机器噪声级修正。
crashfix的邻点替换不保证全域守恒，持续存在的滤波低谷也没有因此被证明消失。
紧致本轮未触发修复，只能说明该40步工况通过，不能作为一般保正证明。

两组96×48×32、10步的CPU/GPU滤波开启对照：
显式NP=1最大绝对/缩放场差3.3306690738754696e-15/2.71856350150488e-15；
紧致NP=2为4.8183679268731794e-14/4.1485958810228614e-14。
两组统计检查通过，紧致另通过既有EOS检查。
这两组小网格不触发实际修复，不能代替中等网格修复后的CPU/GPU逐场误差验收。

证据位于testout/compact_cp9_20261019/recovery/，4组中等网格各有launch.json、
run.log、resources.json和summary.json；小网格为admission_643e及admission_643c_np2。
GPU二进制SHA256为25ec768b0ab425fa97a17b17393210f51ce9bec1393a80385a8ead567642fa88，
CPU为9b0ebc1070757a69bf5864ef9f6ed85ccae8062505574b15591fd3c8712689ac。
根CPU/CUDA构建及15项针对性工具/结构检查通过，git diff --check干净。
另尝试的旧通用滤波回归14项因夹具缺少强制input.output而在初始化退出，
未进入科学计算，未算通过，也未为此修改不相关旧驱动。

本轮是恢复能力验证，不是正式性能验收。状态扫描、资源采样及并发小网格对照
使其计时不能进入此前三轮表。后续需在相同实现和lcracon模式下重采平板滤波开/关的
三轮正式计时及阶段诊断；旧关闭修复数据保留，不混合。当时CP9仍未全部完成，后续完成见§24。
没有操作Git、启动远程任务或修改检查点格式。
新诊断只改变测试观测，不是数值修复，不将预期失正的复现测试当作物理准入成功。
没有Git暂存/提交/推送、超算操作、CUDA架构手工更改或大规模CPU性能测试。

## 24. CP9统一crashfix的正式性能结果（2026-10-20）

### 24.1 配置、身份与完整RK时间

本节替代§23的平板部分完成状态，不覆盖旧失败或关闭修复的性能记录。
新证据根目录为`testout/compact_cp9_20261020_crashfix/`。
平板8种配置全部使用同一个controller `lcracon=t`，包括全域滤波关闭组；
没有增加第二个生产修复开关。测试入口`--crashfix`仅填写该既有开关。
其余网格、dt、边界、初始剖面、零吹吸及同步/通信选择均沿用§23.2。
主程序冻结于`frozen/astr`，SHA256为
`25ec768b0ab425fa97a17b17393210f51ce9bec1393a80385a8ead567642fa88`。

8组三步预检及24次正式运行全部通过。每次40个完整步，前10步预热，
后30步按每rank窗口累计再取最大rank，交错3轮。正式轮无profiler、
阶段计时、状态扫描、外部资源采样、可视化、检查点或中间整场输出。
RK内必要通信、边界及实际修复均计入。少量完整步间统计和启动工作不计入纯RK。

| 黏性/梯度 | 全域滤波 | NP=1 中位数 [最小,最大] ms/步 | NP=2 中位数 [最小,最大] ms/步 | S2 | E2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 643e | 关 | 882.046 [879.295,883.546] | 555.517 [555.215,556.567] | 1.588 | 79.39% |
| 643e | 显式 | 938.395 [937.319,940.185] | 608.526 [606.370,609.207] | 1.542 | 77.10% |
| 643c | 关 | 1487.591 [1482.695,1489.714] | 847.937 [846.199,848.908] | 1.754 | 87.72% |
| 643c | 紧致 | 1672.230 [1670.864,1672.675] | 963.807 [962.350,971.591] | 1.735 | 86.75% |

同滤波开关下，紧致/显式成本比：无滤波NP=1/2为1.687/1.526，
有滤波为1.782/1.584。紧致路径本地整步较慢，双卡效率较高。
这是不同空间算子的成本对比，不是同算子优化收益，也不是CPU加速比。
原始三轮值及派生比值在`plate_performance.json/.csv`，每次输入与原始日志在`timing/plate/`。

每次显式滤波运行均在step31、34、37的stage2发生非零修复，合计291次节点修复访问；
显式无滤波和紧致两种配置均无实际修复。全部完整步CFL≤0.0462834。
计数含存储副本，不是唯一全局节点数。恢复后的40步完成不证明邻点替换守恒或滤波一般保正。

TGV继续采用§23.3的冻结二进制和24次正式结果，`lcracon=f`，不与平板构成同工况比较。
滤波关/开时，643e单/双卡RK中位数分别为381.020/457.098、460.986/574.087 ms；
643c为2329.140/1453.943、2775.877/1759.352 ms。TGV显式双卡在本机未加速，不能只选其最快轮掩盖波动。
本轮另以当前冻结程序与旧TGV二进制比较32³、NP=2、643c/643c、滤波开启的
12份pre_rhs/post_update物理q快照，逐值一致。证据在`transparency/tgv32_np2/`。
该小测支持本次计时观测不改变该路径的数值结果，不声称两个二进制在所有配置上性能相同。

### 24.2 独立滤波开启阶段计时

另外4次12步诊断，排除前2步，三次RK stage按完整步累计；不进入正式主表。
同样全部`lcracon=t`，保持相同冻结二进制。单位ms/完整步。

| 阶段 | 显式NP=1 | 紧致NP=1 | 显式NP=2 | 紧致NP=2 |
| --- | ---: | ---: | ---: | ---: |
| 全域滤波 | 30.624 | 158.668 | 22.506 | 87.905 |
| MP7-LD对流 | 675.188 | 676.073 | 372.501 | 373.274 |
| 梯度及黏性通量 | 30.866 | 261.337 | 15.807 | 151.229 |
| 黏性RHS | 47.940 | 194.859 | 26.436 | 96.379 |
| 黏性halo及等待 | 0.932 | 0.979 | 31.339 | 11.539 |
| 首stage准备 | 37.513 | 80.320 | 43.772 | 64.186 |
| RK数组更新 | 10.749 | 10.966 | 5.435 | 5.563 |

紧致的额外成本主要出现在梯度/黏性及滤波路径，固定MP7-LD对流时间接近。
传感器还在convection计时区间前调用梯度，其成本包含在完整RK而未独立列入本表。
紧致阶段包括布局、RHS、局部PCR、接口处理和写回，不将其命名为纯PCR核。
各阶段取最大rank，可能不是同一rank，且prepare含首stage子滤波，不能直接求和还原RK。
filter区间结束后才调用滤波后crashfix helper，故该项不含修复检查及额外halo。
12步诊断尚未达到首次实际修复的step31，不能用于估计非零修复成本。
也不能用滤波开/关整步之差替代全域滤波时间，因为NSCBC的横向滤波分支随lfilter变化。
原始分rank窗口在`phases/plate/*/summary.json`。

### 24.3 资源观测

下表来自8组独立三步预检的20ms NVML/进程树RSS采样，包括初始化和上下文。
设备列为本作业每卡峰值，NP=2取两卡较大者，不是整个节点现有显存占用。

| 路径 | NP=1 每卡峰值GiB | NP=2 每卡最大峰值GiB |
| --- | ---: | ---: |
| 显式、滤波关/开 | 3.580 | 2.016 |
| 紧致、滤波关 | 5.586 | 3.375 |
| 紧致、滤波开 | 7.365 | 4.625 |

进程树RSS峰值4.442—4.941 GiB。紧致导数私有设备计数NP=1/2为
2139403068/1447064844字节每rank；紧致滤波另有2139404724/1447066500字节。
这类full工作区是多分量批处理，不是额外完整q ping-pong；开启紧致滤波不分配显式qwork。
20ms采样不是所有瞬时分配的严格上界，私有容量计数也不是总进程显存。
原始观测在`preflight/plate/*/resources.json`，不得混入正式计时。

### 24.4 初始化与缩减接口的独立观测

使用`run_compact_initialization.py`及`compact_initialization.gdb`，从对应预检输入
复制后仅执行一个完整步。TGV仍用原冻结二进制，平板仍用本节冻结二进制。
主机GDB包围方向组配置、三方向导数/滤波计划首次建立，并观察既有私有传输计数器的差值。
没有修改数值核，未使用CUDA profiler；本表含主机调试器开销，仅为初始化分项诊断，
不是纯初始化总时间或可用于优化收益比较的正式计时。

| 工况 | 滤波 | NP | 方向组 ms | 导数计划 ms | 滤波计划 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| TGV | 关 | 1 | 0.977 | 19.182 | — |
| TGV | 关 | 2 | 80.627 | 67.646 | — |
| TGV | 开 | 1 | 0.693 | 24.204 | 18.828 |
| TGV | 开 | 2 | 3.516 | 26.736 | 64.366 |
| 平板 | 关 | 1 | 37.369 | 10.439 | — |
| 平板 | 关 | 2 | 20.347 | 10.584 | — |
| 平板 | 开 | 1 | 36.826 | 9.198 | 11.988 |
| 平板 | 开 | 2 | 25.269 | 8.829 | 11.046 |

每项取各rank最大值，不能把各列相加当成启动总时间。计划建立包括三方向
固定系数、局部/缩减接口预处理及其私有工作区准备，不包括整个ASTR初始化。

以下传输计数合计所有rank，范围为启动加一个完整步及其统计，不是正式30步RK窗口。

| 工况 | 滤波 | NP | exchange/broadcast例程调用 | 逻辑发送字节 | 应用层主机暂存字节 |
| --- | --- | ---: | ---: | ---: | ---: |
| TGV | 关 | 1 | 645 | 272650272 | 0 |
| TGV | 关 | 2 | 1030 | 636891424 | 0 |
| TGV | 开 | 1 | 870 | 367760832 | 0 |
| TGV | 开 | 2 | 1374 | 859062976 | 0 |
| 平板 | 关 | 1 | 243 | 75623680 | 0 |
| 平板 | 关 | 2 | 306 | 124586240 | 0 |
| 平板 | 开 | 1 | 336 | 103982560 | 0 |
| 平板 | 开 | 2 | 426 | 171306080 | 0 |

例程次数包括单rank方向组及自通信，不等于跨卡MPI轮次；逻辑字节包含周期同rank消息，
不是实际网络或PCIe流量。host_stage_bytes=0仅说明所选device-aware私有传输没有应用层
显式D2H/H2D暂存，不覆盖公共pinned场halo、MPI/UCX内部拷贝或其他主机传输。
本轮没有新的等待时间或网络计数器profile；小规模归因沿用CP8已保留的Nsight证据。
不能从这些调用/字节数推算真实链路带宽或GPUDirect收益。
原始数据在`initialization/`、`initialization_tgv/`，各配置保留调试日志及summary。

CP9/V7至此完成：两个工况各24次正式运行、4次独立滤波阶段诊断、8组资源预检，
另完成本节8组初始化/传输观测。平板统一开启crashfix，旧TGV基线保持关闭，
不要求两个不同工况的恢复模式相同。不设固定性能收益门槛，不外推为物理或生产规模验收。

## 25. CP10受影响回归与交付（2026-10-20）

### 25.1 必需input.output与旧测试相位

旧通用滤波夹具未生成新输出入口必需的`datin/input.output`，此前14项在初始化退出。
本轮修复TGV和S1输入生成器：补入明确禁用checkpoint/volume/slices的新式配置，
TGV来源已有配置则保留，不在主程序中添加旧输出回退。调用者仍可写入自己的输出配置。

相应滤波回归改为CPU/GPU均读取step3、rk03的post_update二进制q快照，
按输入Ma和理想气体EOS得到rho/u/v/w/p/T；比较物理节点及全部5个守恒分量。
不再把CPU旧RK快照与GPU已废弃的flowfield.h5当成同相位场。
统计仍通过原驱动检查，绝对门槛保持2e-10。负向准入测试只检查原定错误与未发布输出帧，
不再要求出现已停用的旧初始场文件。

第一次复测使用M12专用CPU UDF，LDC统计失败。该UDF的udf_bc为空，不能作为移动顶盖参考；
改用根构建的仓库默认UDF，原有移动顶盖条件即恢复。未修改LDC边界或降低门槛。
该失败保存在`filter_refresh_regression/`，最终17项结果在`filter_refresh_current/`及同名XML。

### 25.2 实际验收

所有可执行程序均通过根CMake构建；本轮没有新的数值核改动，沿用§23.11的恢复实现。
M12专用CPU、默认UDF CPU和CUDA主程序均构建通过。
通用短测使用`build_release_restart_cpu/bin/astr`及`build_compact_cp01_gpu/bin/astr`，
平板同状态参考仍使用匹配M12 UDF的CPU构建。当前GPU重新链接后的SHA256为
`6abd580bce0d35d26a08e8e0c5037cdde091a81a71d37655465193248d2c8dd8`；
§24正式性能始终使用独立冻结副本25ec768b，不混合两个文件身份。

| 验收组 | 结果 |
| --- | --- |
| 计时、输入、选点及无场I/O合同 | 31项通过；拒绝混合二进制/修复模式及不完整三轮矩阵；旧路径断言已同步实际新输出合同 |
| 滤波后刷新回归 | 17项通过；bc41 x/y/z、bc42 x/y、411/421、对称/外推、LDC、RTI、静态CURVE、曲线profile及负向准入 |
| 显式TGV CPU/GPU同相位 | NP=1及NP=2 x/y/z共4项通过，状态最大绝对差2.5579538487363607e-13，统计最大差2.7755575615628914e-15 |
| 显式TGV CPU/GPU精确续算 | 2项通过，NP=2、12步对5+7步，state.h5全部数据集逐值一致，不新增紧致身份 |
| 紧致执行选项续算 | 1项通过，NP=2、scalar→full及explicit→dependency、batch=1024，同拓扑状态逐值一致 |

总计55项本轮针对性工具/运行检查通过，另有§24的12份TGV透明性快照逐值一致。
17项滤波回归中最大物理场差4.547473508864641e-13，曲线profile含缓存数据集最大差
4.973799150320701e-14。TGV区域累计统计最大差8.477982411428698e-18。
对应XML为`tools_current.xml`、`filter_refresh_current.xml`、`cp10_runtime.xml`。
Python编译检查及`git diff --check`通过。CP0—CP8未受影响的线残差、解析阶数、设备检查和拓扑
复用既有不可变证据，不虚报本轮重新跑过所有Sanitizer或全部组合。

### 25.3 交付与保留边界

同步本计划、结果、GPU验证矩阵、当前状态、顶层路线/架构、发布能力说明、USER_GUIDE及测试README。
CP10/V8在既定本地范围内关闭，不把条件紧致能力写成任意工况的全格式支持。
现有显式默认、运行时usegpu及原lcracon开关不变；GPU恢复保持设备执行。
紧致TGV新式同拓扑续算已有验证，紧致profile平板仍只验收新初态启动。
一般CURVE、AIR5、紧致迎风、五对角、目标平台生产规模、长期物理及修复后的守恒性仍未扩展。
没有新增线求解库、CUDA C++或Catalyst依赖，没有Git写操作、远程任务或大规模CPU性能测试。
