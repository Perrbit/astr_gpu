# 可直接启动的 ASTR 示例

本目录提供五套小网格、短时间启动示例。所有输入、启动脚本及附加数据生成代码
都在本目录内，不调用 tests/ 下的文件，不需要 Cantera 或独立化学参考求解器。
它们用于检查安装、边界调用及基本推进，不是已充分发展的湍流初场，也不是生产 DNS 参数。

## 1. 示例一览

网格列是主输入的 ia,ja,ka；非周期方向包括两端节点，不能直接当成存储数组长度。
默认从 step=0 做 **3 次完整更新**，controller 的 maxstep=2。

| 目录 | 物理设置 | 网格 | deltat | 六面边界（x-/x+/y-/y+/z-/z+） | 滤波 |
|---|---|---|---|---|---|
| tgv | 单组分，Re=1600、Ma=0.1 | 32,32,32 | 1e-3，无量纲 | 1/1/1/1/1/1 | 十阶显式 |
| channel | 单组分，Re=3000、Ma=0.3、壁温=1、固定驱动力=1e-4 | 32,32,32 | 5e-4，无量纲 | 1/1/41/41/1/1 | 十阶显式 |
| flatplate | 单组分，Re=1000、Ma=0.3、壁温=1 | 32,64,16 | 1e-5，无量纲 | 11(prof)/21/41/51/1/1 | 十阶显式 |
| air5_tgv | 高温五组分双温周期涡，化学与振动能交换开启 | 16,16,16 | 2e-10 s | 1/1/1/1/1/1 | 关闭 |
| air5_flatplate | Ma=4、p=20000 Pa、T=Tv=1500 K、壁温=3000 K | 31,63,15 | 2.5e-10 s | 11(free)/50/41/51/1/1 | 十阶显式，补偿开启 |

**不提供 AIR5 槽道模板**：当前没有已完成的 AIR5 槽道驱动力、双壁热化学处理及验证组合。
不能在普通 channel 输入中只把 lcomb 改为 t 就认为获得反应槽道。
AIR5 编译开启的同一个二进制可以运行前三个非反应算例，lcomb 仍为 f。

## 2. 构建

先按 [使用说明](../../USER_GUIDE.md) 配置 nvfortran、MPI、HDF5 Fortran/HL 和动态库环境。
在项目根目录运行：

```bash
# 默认：CUDA 非反应流构建，运行前三套单组分示例。
bash examples/GPU_Quickstart/build.sh

# 可选实验功能：另外构建 AIR5，支持全部五套示例。
AIR5=ON BUILD_DIR="$PWD/build_air5" bash examples/GPU_Quickstart/build.sh

# 可选：纯 CPU 二进制。运行时还需 MODE=cpu。
CUDA=OFF BUILD_DIR="$PWD/build_cpu" bash examples/GPU_Quickstart/build.sh
```

脚本通过根 CMakeLists.txt 构建，默认 AIR5=OFF、CUDA=ON、BUILD_TESTING=OFF、CHEMISTRY=OFF、
Release、JOBS=2。FC 默认为 nvfortran，BUILD_DIR 默认为项目 build_prod。
更换编译器使用新构建目录。运行 AIR5 必须使用 AIR5=ON 产生的二进制。
AIR5 源码与示例保留，但属于实验功能，不代表真实反应流生产物理验收已完成。
本目录脚本不修改系统软件、不申请调度资源、不提交平台任务。

Python 3 用于准备输入；只有普通 flatplate 网格生成需要 NumPy 和 h5py。
可在已有 Python 环境中检查：

```bash
python3 -c 'import numpy, h5py; print(numpy.__version__, h5py.__version__)'
```

## 3. 完整启动命令

以下命令均在项目根目录执行。默认读取 build_prod/bin/astr：

```bash
bash examples/GPU_Quickstart/tgv/run.sh
bash examples/GPU_Quickstart/channel/run.sh
bash examples/GPU_Quickstart/flatplate/run.sh
# 先完成上面的 AIR5=ON 构建，再显式选择该可执行文件。
EXE="$PWD/build_air5/bin/astr" bash examples/GPU_Quickstart/air5_tgv/run.sh
EXE="$PWD/build_air5/bin/astr" bash examples/GPU_Quickstart/air5_flatplate/run.sh
```

每条命令会建立不同的新目录，复制完整主输入与 controller，生成必要的附加文件，
再调用 MPI 启动器。目录已存在时直接报错，不清空或覆盖旧结果。

默认单 rank、GPU、FP64、显式同步、pinned 主机中转通信。
脚本默认不设置 ASTR_GPU_FILTER_WORKSPACE，使用更新后求解器的 scalar 默认值。

双卡示例（对五套模板均采用流向二分）：

```bash
NP=2 TOPOLOGY=2,1,1 bash examples/GPU_Quickstart/tgv/run.sh
NP=2 TOPOLOGY=2,1,1 bash examples/GPU_Quickstart/channel/run.sh
NP=2 TOPOLOGY=2,1,1 bash examples/GPU_Quickstart/flatplate/run.sh
EXE="$PWD/build_air5/bin/astr" NP=2 TOPOLOGY=2,1,1 bash examples/GPU_Quickstart/air5_tgv/run.sh
EXE="$PWD/build_air5/bin/astr" NP=2 TOPOLOGY=2,1,1 bash examples/GPU_Quickstart/air5_flatplate/run.sh
```

CPU 使用同一套输入，只改变 usegpu；CUDA 编译的二进制也可这样运行：

```bash
MODE=cpu bash examples/GPU_Quickstart/tgv/run.sh
MODE=cpu bash examples/GPU_Quickstart/channel/run.sh
MODE=cpu bash examples/GPU_Quickstart/flatplate/run.sh
EXE="$PWD/build_air5/bin/astr" MODE=cpu bash examples/GPU_Quickstart/air5_tgv/run.sh
EXE="$PWD/build_air5/bin/astr" MODE=cpu bash examples/GPU_Quickstart/air5_flatplate/run.sh
```

修改步数、输出目录或显式选择旧的全分量工作区：

```bash
EXE="$PWD/build_single/bin/astr" STEPS=20 RUN_DIR=/absolute/new/tgv_run \
  bash examples/GPU_Quickstart/tgv/run.sh

FILTER_WORKSPACE=full NP=2 TOPOLOGY=2,1,1 \
  bash examples/GPU_Quickstart/channel/run.sh
```

## 4. 启动脚本参数

这些是脚本参数，不是求解器原生环境变量。

| 变量 | 默认 | 说明 |
|---|---|---|
| EXE | 项目 build_prod/bin/astr | 已编译的二进制；脚本不自动重编译 |
| MPIEXEC | mpirun | 必须与链接 MPI 匹配；只填可执行文件名/路径，不在其中拼接命令选项 |
| MODE | gpu | cpu 或 gpu，写入运行副本的 usegpu |
| NP | 1 | MPI 进程数 |
| TOPOLOGY | 1,1,1 | 三方向进程数，乘积必须等于 NP；模板要求最小局部索引跨度至少为 6 |
| STEPS | 3（由模板给定） | 从零开始的更新次数，转换为 maxstep=STEPS-1；不是重启追加步数 |
| DELTAT | 各模板数值 | 可填 Fortran d 指数；必须有限且为正 |
| FILTER_WORKSPACE | default | default 不设置求解器变量；scalar/full 显式设置；CPU 不设置该变量 |
| HALO_TRANSPORT | pinned | pageable、pinned、device-aware；最后一种需自行确认 MPI 支持，默认示例不依赖它 |
| RUN_DIR | 本目录 runs 下的唯一目录 | 必须不存在，可指定外部目录 |
| PYTHON | python3 | 数据准备解释器 |
| OMP_NUM_THREADS | 1 | 不会自动将 MPI 求解改成多线程 |

脚本在自己的进程中清除继承的 ASTR_* 环境变量，再设置本算例需要的值。
这样不会误继承另一个任务的无输出、混合精度、反应源项或特征顶面配置；
**也意味着在命令前直接设置 ASTR_* 不会覆盖本脚本**。
需要自定义原生开关时，采用第 7 节的手动启动方式，而不是删除保护检查。

不会清除调度器/MPI、CUDA_VISIBLE_DEVICES、PATH、LD_LIBRARY_PATH 等平台环境。
在 Slurm 等平台先申请资源，使用平台规定的 MPI 启动方法。
本地双卡示例不证明多节点通信性能。

## 5. 初场与附加文件

### TGV 与槽道

TGV 网格和初场由求解器生成。普通 TGV 的域长由内置参考长度生成。
槽道使用内置网格和随机扰动初场，当前随机种子含 MPI rank：
同一拓扑内 CPU/GPU 可对照，但不同拓扑不保证相同的随机初场。
做跨拓扑精确比较时应从同一个 checkpoint 开始。

槽道脚本显式设置固定驱动力 1.d-4，避免短启动中反馈力变化。
这是力，不是指定流速，也不证明已有目标 Reynolds 数下的充分发展统计。
方向由原槽道实现决定；不要只交换网格轴就认为驱动会自动旋转。

### 单组分平板

采用 543e/MP7 对流、643e 六阶显式黏性差分和十阶滤波，
这是当前该物理边界组合已实现的格式搭配；不是中心对流模板。

prepare.py 在运行目录生成：
- datin/grid.h5：x 在 [0,10]、y 在 [0,1]、z 在 [0,0.25]；y 按指数映射加密。
- datin/inlet.prof：每个 y 节点的 rho/u/v/T；rho=T=1，u=1-exp[-(y/0.08)^2]，v=0。

这是沿用现有启动测试的解析剖面，不是可压缩相似解或湍流入口。
输入使用 ninit=0，求解器按 bl 初始化；入口按 prof 读取剖面。
没有把 Python 用作流动推进器。后续物理研究需替换为匹配工况的初场/入口。

### AIR5 TGV

内置初场 air5tgv 采用 rho=0.05 kg/m³、速度幅值 100 m/s、
基准 T=6000 K、Tv=1000 K，质量分数按 N2/O2/N/O/NO 为
0.55/0.15/0.10/0.12/0.08。实际 T 随 TGV 压力扰动变化。
域边长 2*pi*0.01 m。主输入的参考温度不是这里内置基准温度的自由调节接口。

源项 coupled，组分对流 symmetric_species、扩散 layered。
本模板关闭滤波和补偿，不意味着 AIR5 滤波不支持；下一套示例展示相应设置。

### AIR5 平板

采用已有 Mach-4 前驱的解析启动剖面参数：
域长 0.03003857142511096 × 0.004505785713766645 × 0.0007509642856277741 m，
入口剖面厚度 0.0009764153158718681 m。
生成器提供完整 13 列组分/双温剖面及 air5_hbl_domain.dat；
rho 由 p、T 和当前 AIR5 的混合气体常数计算，五组分和为 1。
初始 Y_N2=0.767、Y_O2=0.233，其他三种为零。

流向速度使用四次多项式，T=Tv 从壁面 3000 K 过渡至来流 1500 K，
上方延伸为恒定自由来流。该剖面复制到所有流向截面作为启动状态，
**不是已发展的边界层解**，需由 ASTR 继续发展。

11,free 是此 AIR5 专用边界编号写法，实际由 AIR5 处理读取专用剖面，
不能按单组分 free 入口理解。顶面为直接给定状态 prescribed，出口为 50。
没有入射激波，也没有启用特征顶面。

本模板显式启用 symmetric_species、layered、COMPENSATION=on 和
FILTER_VALIDATION=on。最后一个值表示用户允许进入已实现的组合，
不表示程序已替用户完成物理验证。

## 6. 输出与结果检查

各次运行保留：
- datin/input.dat、controller 和全部生成的数据。
- case.json：步数、时间步、网格、拓扑和预期终点。
- runtime.env：实际 ASTR 开关及 GPU 可见设备设置。
- libraries.txt、launch.txt：依赖库和启动二进制信息。
- run.log、flowstate.dat、outdat/ 等求解器输出。

lwsequ/lwslic/lavg 默认关闭，但 checkpoint 和网格输出仍保留。
短跑最后一次更新安排 checkpoint；CPU/GPU 精确场差必须另行核对写出相位。
checkpoint 的步号、文件 time 与程序最终完成的更新时间需要分别核对。
case.json 的 expected_final_time 表示计划完成全部更新后的积分时长；程序可能在
写出 checkpoint 后继续推进，不能据此要求二者始终相等。
同相位场比较应匹配 checkpoint 状态及边界处理阶段，而非仅比较文件名或步号。
非反应流 GPU 精确续算还需同代的全部 restart_q 文件，不能只复制展示场。
AIR5 平板每步记录专用监测量，不等于已进行 Favre 统计或展向平均。

检查正常结束、实际 usegpu、绑定、CFL、温度/组分和滤波存储日志。
增大步数或修改 dt 前先观察 CFL；本脚本不自动搜索稳定时间步。
比较 full/scalar 应使用相同初场、拓扑、步数及物理选项。
标量工作区减少的是滤波辅助数组，不保证所有硬件上耗时更短。

## 7. 只准备文件，手动启动

```bash
python3 examples/GPU_Quickstart/prepare.py flatplate \
  --destination /absolute/new/flatplate --mode gpu --np 2 --topology 2,1,1 --steps 20

cd /absolute/new/flatplate
export ASTR_FORCE_MPI_TOPOLOGY=2,1,1
export ASTR_GPU_FILTER_WORKSPACE=scalar
export ASTR_GPU_PRECISION_MODE=fp64
export ASTR_GPU_SYNC_MODE=explicit
export ASTR_GPU_HALO_TRANSPORT=pinned
export ASTR_PROFILE_INFLOW_MODE=complete_state
mpirun -np 2 /absolute/path/to/astr run datin/input.dat >run.log 2>&1
```

手动启动须自行清除不相关环境变量。AIR5 和槽道所需开关完整列在 run_case.sh，
不得只复制上面的单组分平板设置。

这些脚本只处理全新启动，不自动恢复 checkpoint。
重启应按 [使用说明](../../USER_GUIDE.md) 成套复制状态和辅助文件，
设置 lrestart=t，并核对补偿与边界配置。不要用新启动脚本覆盖旧运行目录。

## 8. 已验证范围

在 NVHPC、MPI 和 HDF5 的兼容环境中，五套模板完成过 CPU NP1、
GPU NP1 和 GPU NP2（2×1×1）三次更新的启动检查。
启用滤波的四套模板还完成 full/scalar 对照，对应输出数值字段最大绝对差为零。
AIR5 TGV 模板关闭滤波，不计入滤波对照。

这些结果用于说明模板可以启动及受测存储方式一致，不代表长时物理收敛、
所有 CPU/GPU 输出同相位，或其他平台的通信与性能已经验收。
非反应流的较长短窗、同相位场和重启结果见
[验证摘要](../../documents/ASTR_RELEASE_LOCAL_REGRESSION_20260929.md)。
