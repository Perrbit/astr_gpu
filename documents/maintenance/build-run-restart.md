# ASTR 构建、运行与重启

本指南给出主求解器的共同操作契约。站点账号、主机名、私有目录和凭据不属于共享维护文档。任何构建成功只证明依赖和语法闭合，不证明数值、物理或性能验收。

## 前置依赖

| 路径 | 必需依赖 | 说明 |
|---|---|---|
| CPU | CMake、MPI Fortran compiler、parallel HDF5 Fortran/HL | GNU、Intel、NVHPC 等 compiler flags 由根 `CMakeLists.txt` 分支选择 |
| CUDA Fortran | NVHPC Fortran、CUDA runtime、MPI、parallel HDF5 Fortran/HL | 通过 `ASTR_WITH_CUDA=ON` 将 `src_gpu/*.cuf` 编入同一个 `astr` binary |
| GPU 运行 | 可见 NVIDIA GPU、driver/runtime compatibility | 输入中的 `use_gpu=t` 才进入 backend |
| 验证与 profile | Python 3、验证脚本依赖、Compute Sanitizer、Nsight Systems/Compute | 每项工具证明不同层面，不应互相替代 |

根 `CMakeLists.txt` 是唯一主构建入口，`src/CMakeLists.txt` 定义 `astr` target 成员。不要从 `src_gpu/` 建立另一套脱离主程序的 production build。

## CPU 构建

```bash
ASTR_ROOT="$PWD"
BUILD_DIR="$ASTR_ROOT/build_cpu"
MPI_FC="${MPI_FC:-mpifort}"

cmake -S "$ASTR_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_Fortran_COMPILER="$MPI_FC" \
  -DASTR_WITH_CUDA=OFF \
  -DBUILD_TESTING=OFF
cmake --build "$BUILD_DIR" -j4
```

必须检查 CMake 输出中的 compiler ID、MPI 和 HDF5 路径。`mpifort` 可能包装 gfortran，也可能包装 nvfortran；需要 GNU CPU oracle 时应显式确认 `mpifort --showme:command` 或站点等价命令，不能仅根据 executable 名称判断 compiler。

## CUDA-capable 构建

```bash
ASTR_ROOT="$PWD"
BUILD_DIR="$ASTR_ROOT/build_gpu"
NVHPC_MPI_FC="${NVHPC_MPI_FC:-mpif90}"

cmake -S "$ASTR_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_Fortran_COMPILER="$NVHPC_MPI_FC" \
  -DASTR_WITH_CUDA=ON \
  -DBUILD_TESTING=OFF
cmake --build "$BUILD_DIR" -j4
```

`ASTR_WITH_CUDA=ON` 只表示 binary 包含 CUDA backend。运行时 `use_gpu=f` 仍执行 CPU 路径，`use_gpu=t` 才执行 GPU capability gate。非 CUDA binary 接收 `use_gpu=t` 时必须停止，不能静默退回 CPU。

## 平台差异

| 项目 | 本地 NVHPC 工作站 | A800/HPC 平台 |
|---|---|---|
| 构建位置 | 本地 shell，可直接使用已安装 NVHPC/MPI/HDF5 | 通常在 login/build node 加载 compiler、MPI、HDF5 module 后构建 |
| 运行位置 | 本地 GPU，可直接 `mpirun` | 必须通过 scheduler 获得 compute node/GPU，不在 login node 执行 GPU case |
| Rank binding | 用 `CUDA_VISIBLE_DEVICES` 和运行日志核对 | scheduler GPU allocation、node-local rank 与 device ID 必须一致 |
| 多 rank 解释 | 两卡 oversubscription 可做 topology correctness smoke | production scaling 应采用一个 MPI rank 对应一个物理 GPU |
| 观测 | `nvidia-smi`, `nvitop`, Nsight CLI | 遵守站点 profiler permission 和作业时限 |
| 文件系统 | build、case、evidence 分目录 | 所有路径服从站点允许的 project/work directory policy |

HPC job script 应在提交前完成 shell syntax、输入文件、executable、HDF5 dynamic library 和 environment variable 检查。login node 无 GPU 导致的 device error 不能作为 compute job 失败证据。

## 运行

```bash
ASTR_ROOT="$PWD"
EXE="$ASTR_ROOT/build_gpu/bin/astr"
CASE_DIR="$ASTR_ROOT/examples/Taylor_Green_Vortex"
INPUT="$CASE_DIR/datin/input.tgv"
NP=1

cd "$CASE_DIR"
mpirun -np "$NP" "$EXE" run "$INPUT"
```

实际 case 路径和 binary 路径由本次验证目录决定。生产或验证运行前检查：

1. `input.*` 和 `controller` 为 LF，不含 CRLF。
2. `flowtype`、scheme、`rkscheme`、physics flags、boundary codes 和 `use_gpu` 属于当前 capability。
3. `NP` 与 topology product 一致，每个 local active dimension 足以容纳 stencil halo。
4. 新 evidence directory 不覆盖已有结果，source checkpoint 保持只读。
5. GPU 运行日志报告 device binding 和 synchronization mode，`nvitop` 或 `nvidia-smi` 能看到对应 process。

CRLF 检查：

```bash
find examples -type f \( -name 'input.*' -o -name 'controller' \) -print0 \
  | xargs -0 file | rg 'CRLF'
```

预期无输出。NVHPC Fortran runtime 可能把末尾 `\r` 保留在字符串中，使 `select case(trim(flowtype))` 匹配失败，并污染 grid/HDF5 文件名。发现 CRLF 时先修输入，不继续用 NaN 或 CFL 崩溃反推数值 kernel。

## Restart 契约

restart 入口为 `src/initialisation.F90::flowinit`。`lrestart=t` 时调用 `src/readwrite.F90::readcheckpoint` 读取 `outdat/auxiliary.txt` 和 HDF5 flowfield，然后由 `src/fludyna.F90::updateq` 从 primitive fields 重建 conservative state。

### 必须一致的状态

| 状态 | 来源 | 检查 |
|---|---|---|
| `nstep` | `auxiliary.txt` namelist 与 HDF5 dataset/attribute | 两者必须相等，否则 `Checkpoint step mismatch` 并 `error stop` |
| `time` | HDF5 | 与 controller 中继续步数和输出节奏一致 |
| Primitive fields | `ro,u1,u2,u3,p,t`，可选 species/turbulence | dataset、shape、topology 和当前 model flags 一致 |
| Conservative fields | restart 后 `updateq` 重建 | 不直接假定旧 host `q` 可继续使用 |
| Statistics | `nsamples` 及启用的 meanflow state | 连续统计必须读取对应文件；不需要时明确重新开始 |
| Boundary auxiliary state | inlet profile、conservative boundary config、sponge | 从当前输入/config 重建，不能依赖旧 halo |

### Restart 顺序

```text
保留原 checkpoint 副本
-> 复制到新的 restart case directory
-> 设置 lrestart=t 与新的 maxstep/feqchkpt
-> 读取 auxiliary.txt 和 flowfield HDF5
-> 校验 nstep 一致并重建 q
-> 重建 physical boundary、MPI halo 和 backend auxiliary state
-> GPU case 上传完整 host state
-> 从 checkpoint 的下一 step 继续
-> 与同终止 step 的 continuous run 比较 field 和 statistics
```

GPU 初始化仍先在 host 完成 restart read，再由 `src_gpu/commarray_gpu.cuf::copy_flow_to_gpu` 上传。当前 HDF5/checkpoint 不是 device-side I/O。GPU 时间循环在显式 checkpoint 边界调用 `src_gpu/gpu_runtime.cuf::gpu_sync_flow_to_host`，写出上一完整 RK 后 state。

## Restart 验证

`tests/gpu_validation/run_opensbli_restart_equivalence.sh` 是当前完整 restart contract gate。它分别验证 CPU NP=1 与 GPU NP=2：

- continuous 四步与 fresh 两步加 restart 两步的最终 field 一致；
- restart 后 `flowstate.dat` step 序列连续；
- `auxiliary.txt` 与 HDF5 的故意 `nstep` mismatch 必须被拒绝；
- source checkpoint 不被原地重用或覆盖。

该 gate 证明指定 OpenSBLI 路径的 restart 等价，不代表所有 flowtype、statistics accumulation 或 topology change 都已验证。改变 `readcheckpoint`、`writechkpt`、GPU upload、boundary initialization 或 statistics state 后必须重跑对应 restart gate。

## 停止条件

出现以下任一情况时不继续提交 production job：compiler/MPI/HDF5 组合与目标 binary 不一致，输入含 CRLF，GPU capability 明确拒绝，checkpoint step/time 不闭合，local domain 小于 stencil 需求，rank 未绑定预期 device，或启动 smoke 已出现 NaN、Inf、负 density/pressure。此时先报告逻辑或配置缺陷，不通过延长运行寻找“是否自行恢复”。

