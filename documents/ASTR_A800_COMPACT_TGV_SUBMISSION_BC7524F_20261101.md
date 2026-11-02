# A800 紧致TGV分开提交记录

日期：2026-11-01。用户确认关闭物理场I/O后分开提交CPU与GPU任务。

状态：已于2026-11-01 21:27分别提交，尚未运行计算节点预检。

| 作业 | ID | 提交后状态 |
| --- | ---: | --- |
| GPU | 508963 | PENDING / Priority |
| CPU | 508964 | PENDING / Priority（首次检查为Resources） |

两任务无调度依赖，Slurm时限各48小时，不设置额外单项timeout。排队开始时间未知。

## 源码与目录

- 源码基点：`bc7524f2fad767387b1f67aa33eeaa46b1842ee6`。
- 另附本轮未提交的分拆任务驱动／平台脚本补丁，不将其冒称纯bc7524f版本。
- Fortran数值源码与该提交一致；新增后端筛选、结果合并、独立环境／构建／节点预检和Slurm包装。
- 部署目录：`/data/user/hd56000/weiph/compact_tgv_bc7524f_split_20261101_210437`。
- `deployment.json`记录源码包和部署补丁包的身份；旧任务500619及其目录不修改。

## 已确认关闭的I/O

正式入口明确设置`ASTR_GPU_BENCHMARK_NO_FIELD_IO=1`，生成关闭checkpoint／volume／slices的`input.output`。
解析生成网格，不复制网格HDF5；`gridgeneration`跳过网格写出，`flowinit`跳过初始场写出。
正式调用`validation=False`，不启用`ASTR_VALIDATION_RHS_PREFIX`，因此不保存诊断流场快照。
Catalyst、渲染、三维场序列、切片、均值场及重启文件关闭。保留标量统计、CFL和计时日志。
控制器`feqchkpt=1`在该测试中用于控制器／CFL事件，不会覆盖新输出的关闭设置去写每步检查点。

## 分拆任务

| 项目 | GPU任务 | CPU任务 |
| --- | --- | --- |
| 队列 | A800-N | Intel-8358 |
| 资源 | 4张A800，4 MPI rank，每rank绑定2物理核 | 32物理核上限，无GPU，实际按NP1／2／4／8／16／32运行 |
| 主机内存 | 192GiB | 192GiB |
| 正式配置 | 两格式强扩展20种＋弱扩展6种 | 两格式、六种核数，共12种 |
| 重复 | 各5轮，130次启动 | 各5轮，60次启动 |
| 步数 | 每轮22步，丢弃前2步 | 每轮7步，丢弃前2步 |
| 场I/O | 关闭 | 关闭 |

数值设置不变：FP64，Re1600，Ma0.1，dt=1e-4，RK3，两种六阶导数与对应十阶滤波。
两个队列会使用不同节点，CPU型号和NUMA／绑定在运行日志中留存。
GPU请求为4任务×2核；平台因192GiB内存另显示`MinCPUsNode=25`，实际CPU分配和计费口径需待`AllocTRES`形成后确认，不能预先把请求核数当作最终分配核数。

## 预检与容错

1. 登录节点按根CMake分别编译CPU-only和cc80 CUDA版本，关闭Catalyst／AIR5／Cantera，使用默认仓库UDF，无M12目录依赖。
2. 解析实际动态库，检查MPI、NVHPC数学库、HDF5及压缩依赖；真实导入本次Python入口，不要求无关的h5py包。
3. 提交前短CPU登录测试确认输入、数学库和无HDF5输出，不在登录节点尝试完整GPU计算。
4. 每任务在自身计算节点先跑小网格检查：CPU NP1／32对照；GPU NP1／4与同拓扑CPU参考对照，保留2e-10门槛。
5. 正式轮不启用场快照、逐场扫描或profiler。单配置失败保留日志并继续独立配置；两个任务无调度依赖。
6. 提交脚本不增加单项timeout；Slurm时限按队列和已审阅计划确定。

## 已完成的提交前检查

- 本地分拆、清单、通信和汇总针对性单测32项通过，Bash／Python语法及diff检查通过。
- 本地计算节点预检包装额外完成两格式CPU NP1／32短测。
- 登录节点CPU-only与CUDA目标均由根CMake完成`--parallel 8`构建。
- 实际GPU编译参数含`-cuda -gpu=rdc,cc80 -O2`，未使用设备调试`-G`或快速数学选项。
- CPU／GPU的`ldd -r`没有缺失库或未解析符号；HDF5来自无szip的1.14.5，数学库明确来自NVHPC25.5 CUDA12.9配套目录。
- 实际Python入口导入通过，不依赖当前venv未安装的h5py。
- 登录节点两格式各4步CPU短跑通过，确认没有HDF5场输出和诊断快照；保存于部署目录`login-smoke/result.json`。
- 两类资源请求通过`sbatch --test-only`；试提交编号不作为实际作业编号。
- 提交后检查确认500619仍为PENDING；未取消、修改或复用原任务。

GPU输出日志为`logs/gpu-508963.out`／`.err`，CPU为`logs/cpu-508964.out`／`.err`。
正式结果目录预计为`results/gpu_508963`和`results/cpu_508964`；它们在任务实际启动后才创建。
计算节点的小网格预检会保存32³诊断快照，独立于正式无场I/O测量，不混入性能表。

## 代码变动

分拆使用`run_compact_a800_campaign.py --backend cpu/gpu`，之后可在GPU结果目录以
`--mode summarize --merge-from <CPU结果目录>`合并同算子加速比。
平台包装为`run_zhongke_a800_compact_split.sbatch`，只运行指定后端，不自动提交其它任务。
环境和构建脚本只写本部署目录；不覆盖旧源码、二进制、HDF5或Python环境。
