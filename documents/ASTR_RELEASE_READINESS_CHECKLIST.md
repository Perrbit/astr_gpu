# ASTR GPU 发布执行清单

更新日期：2026-09-29。状态：含重启修复的精简源码独立构建、安装及安装版代表性回归通过；修复未提交，目标平台及最终版本冻结未完成。

## 执行约束

在 feature/gpu_dev 当前目录逐步完成发布准备。不创建 worktree，不自动
创建 release/prod，不执行 Git 暂存、提交、推送或目录裁剪。不干预远程任务。
物理模型、数值方法、验收标准或交付范围需要人工决定时停止。
历史验证只有在源码、输入和环境适用时才复用；不以历史通过替代最终发布包验收。

## 初始基线盘点（R2 提交前）

- 当前 HEAD：`bd5207b521787c3c2074e538b9e00d40b6210a23`。
- HEAD 说明：`feat(air5): support compensated filtering with characteristic top boundaries`。
- 当前工作区不干净，不能把已有构建和运行结果全部归属于这个 HEAD。
- 跟踪文件变化包括根及 src CMake、src/test.F90、scalar 滤波默认值、README
  和验证文档/测试；USER_GUIDE、GPU_Quickstart、发布说明及诊断脚本还未跟踪。
- 未跟踪的手册、演示材料及 USER_GUIDE.md.save 不自动列入交付，不删除。
- 原有许可证文件存在；尚未完成全部第三方来源和交付许可审查。

## 顺序与状态

| 顺序 | 项目 | 当前状态 | 完成条件 |
|---|---|---|---|
| R1 | 冻结发布能力和默认配置 | 方案 A 已批准并落实 | 非反应流基础发布；AIR5 显式启用且标为实验功能 |
| R2 | 源码基线与变更归属 | 候选已提交为 6952dc0 | 尚非通过验收的正式发布版本 |
| R3 | 发布文件清单 | 已生成约 70 MB 精简源码候选，不含 tests/chem/历史结果 | 最终文档及交付版本仍需冻结 |
| R4 | 最终包独立构建 | 含重启模块的 242 文件精简副本与工作区一致；CPU/CUDA 构建、双前缀安装通过 | 修复提交后固定最终版本；不等同于全部物理验收 |
| R5 | 目标平台依赖 | A800 参考平台登录节点只读检查完成；实际部署平台待定 | 每个部署环境单独验证工具链、计算节点运行库、驱动及启动方式 |
| R6 | 最小数值回归 | 本地六组代表性 CPU/GPU 场、统计和 x/y/z 分解检查通过 | 仅限已记录的短窗组合，最终版本仍需固定 |
| R7 | 输出与重启 | 非反应流 GPU 实际推进状态独立保存；OpenSBLI CPU/GPU 重启及错误元数据拒绝通过 | 同拓扑、同配置；AIR5 不由本轮放行 |
| R8 | 默认 scalar 滤波 | 修复后 TGV、槽道和 CURVE 100 步分段重启逐场差为零，full/scalar 连续场一致 | 本地指定组合通过；最终交付版本和目标平台仍需冻结 |
| R9 | CUDA-aware MPI | 作为条件能力，未完成本轮平台准入 | 目标软件栈和拟交付边界组合通过，提供明确主机中转配置 |
| R10 | 使用说明 | 启动示例、恢复、网格/入口及 AIR5 激波/OpenSBLI 数据合同已说明 | 其他外部初场及最终交付文件仍需核对 |
| R11 | 发布清单及已知问题 | 来源核查按用户决定暂缓；保留根许可证及源码声明，其余发布元数据待冻结 | 版本、依赖、证据、能力限制及部署恢复方法齐全；暂缓项如实记录 |

上述状态不等于全部能力从未验证，而是区分研发历史与本轮最终交付验收。
本轮完成源码/文档核对及 30 项针对性测试，没有重新执行构建、GPU 算例或平台测试。

## R1 已批准方案 A

修改前核查发现：

- 根 CMake 的 `ASTR_WITH_AIR5_CHEMISTRY` 默认 OFF。
- `examples/GPU_Quickstart/build.sh` 的 `AIR5` 默认 ON，为五套示例共同构建。
- 发布能力说明建议基础部署默认关闭 AIR5，按需启用受限验证版。

这不是求解器错误，但会让“默认发布配置”产生歧义。用户已选择方案 A：

**A，推荐：非反应流稳定版为默认。** AIR5 源码保留，启动包构建默认改为
OFF。两个 AIR5 示例仍保留，但明确要求先 `AIR5=ON` 构建，并标注实验范围。
同步脚本、指南和相关测试，不更改物理模型或删除 AIR5 源码。

共同基础配置：FP64、scalar 滤波、explicit 同步；主机中转作为
无需 CUDA-aware MPI 的基础通信选择，device-aware 按平台和组合单独准入。
混合精度、流水线通信和 fused 通量不自动成为默认。

启动包默认已改为 OFF，README 和 USER_GUIDE 同步显式 AIR5 构建及 EXE
选择方法。新增构建脚本参数测试，覆盖未设置、空值、OFF 和 ON。测试通过：

```bash
python -m pytest -q tests/gpu_validation/test_gpu_quickstart.py \
  tests/gpu_validation/test_scalar_filter_workspace_contract.py
# 30 passed
```

参数测试使用模拟 CMake 记录实际 shell 参数，不冒充真实编译检查。
未改变根 CMake 默认、求解器物理模型或 AIR5 源码。

## R2 提交范围已获授权

用户已授权 git add、git commit，未授权 push 或创建分支。本次提交将下列
发布准备内容固定为候选基线；实际提交号以 Git 记录为准，不在提交内部自引用。

- CMakeLists.txt、src/CMakeLists.txt、src/test.F90：测试构建解耦。
- src_gpu/commarray_gpu.cuf：此前批准的 scalar 滤波默认值。
- examples/GPU_Quickstart/：完整启动包，不包含 runs 和缓存。
- README.md、USER_GUIDE.md、发布能力说明、本清单和当前状态文档。
- tests/gpu_validation/README.md 及三个对应测试：
  test_cmake_production_decoupling.py、test_gpu_quickstart.py、
  test_scalar_filter_workspace_contract.py。
- 当前状态文档所链接的 ASTR_AIR5_PRECURSOR_ADMISSION_20260929.md，
  及其 check_air5_precursor_admission.py、test_air5_precursor_admission.py，
  保持已记录诊断可追溯；不纳入稳定版物理验收结论。

不使用 git add 全目录，不纳入手册、演示文件、USER_GUIDE.md.save、
运行结果或其他未审查脚本。提交前再核查暂存差异。
本次只提交上述范围，不创建 release/prod，不删除文件，不推送。
下一步是 R3 文件清单与 R4 最终包独立构建。版本被提交不等于发布验收完成。

## 依据

- `documents/ASTR_RELEASE_PROD_CAPABILITY_SCOPE.md`
- `documents/ASTR_AIR5_PRECURSOR_ADMISSION_20260929.md`
- `CMakeLists.txt`
- `examples/GPU_Quickstart/build.sh`
- `tests/gpu_validation/test_cmake_production_decoupling.py`

## R3/R4 原安装路径阻塞记录（已修复）

候选基线 `6952dc0`，从 Git 导出至 `/tmp/astr-release-6952dc0-zTGf8U/source`，
仅包含 CMakeLists、LICENSE、README、USER_GUIDE、src、src_gpu、
user_define_module、examples 及两份发布说明。没有复制 tests、chem、
历史输出或开发工作区未跟踪文件。这是检查用候选，不是可直接对外交付的最终包。

NVHPC 26.1、HPC-X MPI、HDF5 1.14.6 下，CPU/AIR5 OFF、BUILD_TESTING OFF
配置成功。显式设置安装前缀为临时目录中的 `install`，检查发现：

1. 根 CMake 的 `set(CMAKE_INSTALL_PREFIX ${ASTR_ROOT})` 覆盖用户前缀。
   CMakeCache 仍显示 install，生成的 cmake_install.cmake 默认却是 source。
2. examples 安装规则将 `${CMAKE_INSTALL_PREFIX}/examples/...` 展开成绝对路径，
   即使后续用 `cmake --install --prefix` 覆盖前缀，示例目的地仍固定为原路径。
3. Taylor_Green_Vortex_2D/CMakeLists.txt 使用 Taylor_Green_Vortex 作为目的目录，
   二维和三维示例 datin 被合并到相同位置，存在同名文件覆盖风险。

证据：临时目录 `configure.log`、`build_cpu/CMakeCache.txt`、根及 src/examples
下生成的 `cmake_install.cmake`。没有执行编译或安装，没有写回源码目录。
上述问题属于构建/部署行为，不是 CPU 数值算法缺陷。

已获人工授权并实施的最小修复：

- 去掉强制覆盖安装前缀，保留已有默认 build/opt 和显式用户选择。
- 示例安装目的地改为相对于安装前缀的路径，保持根 CMake 无条件进入 examples。
- 二维 TGV 使用独立的 Taylor_Green_Vortex_2D 安装目录。
- 增加临时前缀安装检查，检查无源码写入及无二维/三维示例冲突。

本轮没有修改求解器数值实现，没有执行 Git 暂存/提交/推送或远程任务。

## 安装修复验证结果

- CPU/CUDA 两种非反应流构建和安装检查：2 passed，耗时 313.71 s。
- 配置时指定前缀、安装时 --prefix 覆盖、二维/三维 TGV 数据分离通过。
  安装前后源树文件清单及内容哈希一致。默认前缀为 build/opt 的配置检查通过。
- CPU/CUDA、AIR5 OFF/ON、开发模式和生产 trace 拒绝检查：7 passed，30.74 s。
  这一组为配置检查，不是 AIR5 全量编译或运行检查。
- 两套安装后二进制在当前本机环境执行 ldd，无 not found；未验证远程节点。
- git diff --check 通过。没有重新运行流场、物理或性能测试。

构建安装证据：`/tmp/astr-release-install-20260929a/`。
各测试目录保存 configure.log、build.log、configured-install.log、
override-install.log、default-configure.log。此次安装测试的旧助手复制了
额外的本地 examples 文件；该副本不作为发布包。随后已将助手改为仅复制
Git 跟踪的生产输入，以当前工作区内容包含待提交修复。
新助手的配置检查证据：`/tmp/astr-release-configure-20260929b/`。

独立精简包：`/tmp/astr-release-package-6952dc0-l2QCxx/astr-source.tar.gz`。
SHA256：`b2d4359cd797fc447f62f55afbf838e9756fc8936e6a22033657eda5fc760e54`。
同目录 files.nul、base-commit.txt、worktree.patch 记录文件清单、基准和修复。
该包是 6952dc0 加未提交安装修复，不是纯 6952dc0，也不是最终发布版本。

精简包独立解包后的 CUDA 配置通过。包内全部 src/src_gpu/UDF、根及 examples
构建文件、所选示例输入和启动脚本均与刚构建的副本逐文件比较一致，因此
复用该构建证据，未再重复编译。包中的内部状态文档是打包时快照，不随此后
清单更新改变；最终交付前仍需更新发布元数据。

下一步可继续本地最小数值回归和输出/重启检查。目标平台登录或运行、创建
release/prod、提交本轮修复均未自动授权，需要执行时停下确认。

## 本地运行回归更新

R6/R7 代表性检查已执行，R8 增加周期 TGV 100 步窗口。详细设置、容差、
逐场误差及三次检查失败的原因见
[本地回归记录](ASTR_RELEASE_LOCAL_REGRESSION_20260929.md)。所有计算均已结束。
仅修改测试助手接口和比较相位，没有修改求解器数值代码。

## R5 参考环境检查（2026-09-29）

用户批准 A800 只读检查，并明确未来部署未必使用该平台。检查仅在登录节点
`ln01` 进行，远程工作目录为 `/data/user/hd56000/weiph`。没有上传或修改
文件，没有编译、运行求解器、申请计算节点或干预任何作业。

检查对象为历史 `build_gpu_scaling_2744795/bin/astr`，不是本轮发布候选。
构建依据为该目录 `CMakeCache.txt`，HDF5 依据为
`opt/hdf5-1.14.5-nvhpc25.5-noszip/lib/libhdf5.settings`。

- CMake 3.26.5 可见。历史构建使用 NVHPC 25.5、其 CUDA 12.9 配套
  HPC-X 2.22.1 和并行 HDF5 1.14.5；HDF5 的 C/Fortran 编译包装器来自同套 HPC-X。
- 默认登录环境加载 Open MPI 5.0.5 和 GCC 11.5。旧二进制的 `ldd` 没有
  缺失库，却将 MPI 解析到 Open MPI 5.0.5，与构建环境不一致。因此仅检查
  `not found` 不足以确认部署正确，也不能据此认定已有生产作业混用了 MPI。
- 对照历史作业脚本
  `a800_tgv_scaling_matrix_20260913/run_zhongke_a800_tgv_scaling_matrix.sbatch`，
  在一次性 SSH shell 中执行 `module purge`、加载 `gcc/11.5` 和 `nvhpc/25.5`，
  后者自动加载 HPC-X，再设置上述自建 HDF5。启动器、编译包装器和实际 MPI
  库路径匹配，`ldd` 未出现缺失库。没有改变持久环境或作业脚本。
- 额外隔离库路径检查曾遗漏 GCC 运行库而出现 `libatomic.so.1` 缺失；加入
  GCC 11.5 运行库后消失。实际作业模块环境也能正确解析该库。

结论限定为历史二进制在参考平台登录节点的依赖解析检查。没有验证当前发布
版本在该平台的构建、GPU 驱动、计算节点动态组件、设备绑定或 CUDA-aware MPI。
R5/R9 不据此关闭。其他部署平台采用 USER_GUIDE 第 14 节的同一验收顺序，
但自行确定软件路径、GPU 架构、模块和调度器配置。

下一人工决定点：明确实际部署环境及允许的计算节点检查范围后，再进行该环境
的短跑验收。本次授权不包含远程提交、交互分配或生产任务变更。

## R11 来源初查与用户决定

生产源码包含 LGPL、MIT 及外部 FFT/xcompact 来源注释，不能只依据根
Apache-2.0 许可证认定全部文件来源已核验。相关文件均在主程序编译列表内。
具体位置见 [源码来源初查](ASTR_RELEASE_SOURCE_PROVENANCE_REVIEW.md)。
用户已决定跳过本轮来源追溯，维持根目录 Apache-2.0 许可证。该项不再阻塞
当前发布准备，状态为暂缓核查，不是许可审查通过。保留源码已有第三方
版权、许可和来源注释，不进行统一重新授权或删除实现。
本轮不改许可证或数值代码，不执行 Git；其他数值、部署及版本冻结门槛不变。

## R10 部署操作说明补齐

已依据 readcheckpoint、readmeanflow、主循环 readcont 调用位置及 GPU
紧凑统计恢复检查，补充 USER_GUIDE 10.4/10.5：序列场辅助文件改名规则、
累计统计的同代文件要求、紧凑统计不可直接跨拓扑恢复，以及失败回退步骤。
此处只记录现有行为，不修改输出相位、读写实现或数值算法。

默认值应区分三个入口：根 CMake 默认 CUDA=OFF、BUILD_TESTING=ON；
GPU_Quickstart 明确 CUDA=ON、BUILD_TESTING=OFF、AIR5=OFF；求解器运行时
仍由 usegpu 选择 CPU/GPU。求解器通信默认 pageable，Quickstart 显式选 pinned。
两者均不可被概括为“所有入口默认 GPU/pinned”。

剩余工作不因文档补充自动关闭：R8 壁面/CURVE 长窗口、R10 数据字典及组合
范围核对、最终版本与交付包冻结；R5/R9 需要实际目标部署环境。

## R10 输入数据合同更新

新增 [部署输入数据字典](ASTR_DEPLOYMENT_INPUT_CONTRACT.md)，明确全局网格
HDF5 排布、单组分静态剖面四行头与状态标记、动态切片字段及时间窗口、
AIR5 平板 13 列完整状态和五套启动模板的受限组合。

纠正 USER_GUIDE 原来“动态切片使用绝对状态”的错误说明：现有受支持的
无量纲 intp 路径在 CPU/GPU 均将插值增量叠加到静态剖面。仅修改说明，
未更改入口算法，也未将其他有量纲/化学入口推定为同一语义。

检查：Quickstart、动态切片生成器及动态入口源码合同测试共 36 passed。
git diff --check 通过。没有运行新的流场、物理或性能测试，没有 Git 操作。
该数据字典应随使用说明进入最终白名单，否则使用说明链接不完整。

继续补充正激波 12 列状态、入射激波几何/11 分量状态及固定 OpenSBLI
namelist 的区分，记录单位、能量定义、状态一致性及顶面几何约束。
USER_GUIDE 不再要求随包携带整份研发历史状态文档；能力说明的建议白名单
已明确加入 USER_GUIDE 和输入数据字典。最终包仍需检查所有相对链接闭包，
当前不是已冻结发布包，也没有完成全部外部初场格式说明。

## R10 外部初场和文档交付检查

输入数据字典已补充 ninit=1/2/3 的固定文件名、实际读取字段、压力重建和
时间归零行为，以及 AIR5 专用二维初场。没有将这些遗留读取器推广为任意
反应流初始化，也不以文档核对代替新数值验证。

README 的默认安装路径改为构建目录下 opt/bin/astr，区分构建产物与安装
产物，并明确精简包不提供旧 Make/Cantera/script 工具流程。

`documents/ASTR_RELEASE_DOCUMENT_MANIFEST.txt` 列出候选包的文档白名单，
路径相对项目根目录。它不是全部源码清单。保留发布清单及其链接的本地回归
和来源初查记录，不复制整份研发历史或运行产物。文中源码位置可在源包中
核对；开发测试路径和临时结果路径属于证据索引，不是随包依赖或启动命令。
最终归档时仍需把此文档清单、LICENSE 和生产源码清单合并并验证。

本轮使用 Markdown 解析器核对清单中的 8 份文档：12 个相对文件链接均指向
清单内存在的文件。只检查相对文件链接，不验证外部网址或代码块中的示例路径。
git diff --check 通过；README 的历史 CRLF 有 Git 换行提示，无空白错误。
未重新构建或运行算例，没有 Git 暂存、提交或推送。

## 最新本地安装产物验收

在新增 checkpoint_gpu.cuf 后重新执行隔离构建。复制器只纳入已跟踪生产
文件及明确列出的该新增文件，不以 git add 代替验收，也不复制其他未跟踪内容。
临时源码含 242 个文件，逐一 SHA256 与当前工作区相同，不含 tests/ 或 chem/。

- 配置矩阵：7 passed，2 个安装测试在配置轮跳过，随后单独完整执行。
- 独立 CPU/CUDA 非反应流构建安装：2 passed。分别检查配置安装前缀和
  --prefix 覆盖、2D/3D TGV 输入隔离、默认 build/opt，以及安装不修改源码。
- Quickstart、scalar 合同及 OpenSBLI 输入生成器：31 passed。
- 安装版依赖检查：CPU/GPU 均无缺失动态库；本机 NVHPC 26.1、HPC-X 2.25.1、
  HDF5 1.14.6。不是其他平台的运行库准入结果。
- 安装版 TGV 32³、NP2 2×1×1、100 步：CPU/GPU 最大场差 4.831690603168681e-13，
  full/scalar 与 50→100 重启场差均为零，统计通过。
- 安装版槽道 32³、NP2 2×1×1、100 步：与修复前连续参考的 full 场差为零，
  scalar 50→100 重启场差及对应统计差为零。

独立构建/安装日志和源码清单：/tmp/astr-release-final-install-20260929/。
计算证据：tests/gpu_validation/out/release_6952dc0_installfix_20260929/
下 installed_exact_tgv 和 installed_exact_channel。二进制指纹见本地回归记录。
本轮未重做昂贵物理长算例、性能或 AIR5 数值矩阵。

输出范围明确为 iomode=h；s/n 的主程序路径没有完整分派，未作为发布能力
放行，也没有擅自修改这些遗留实现。详见 USER_GUIDE 第 10 节。

接下来需要人工确认的事项：冻结安装/重启修复和文档的 Git 提交；确定实际
部署环境及允许的计算节点检查范围。R5/R9 仍按平台准入，不因本地通过关闭。
本轮未执行 Git 写操作、创建分支、裁剪当前目录或操作远程作业。
