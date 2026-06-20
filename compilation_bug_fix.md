# ASTR 项目 NVHPC 编译/运行问题修复记录

将 ASTR 项目从 gfortran 编译迁移至 NVIDIA NVHPC 26.1 (`nvfortran`) 编译器过程中遇到的源代码问题、编译问题、运行时崩溃及对应修复。

## 编译环境

| 组件 | 版本/路径 |
|------|-----------|
| NVHPC SDK | 26.1 (`/opt/nvidia/hpc_sdk/Linux_x86_64/26.1`) |
| MPI | HPC-X OpenMPI (`hpcx/bin/mpif90`) |
| HDF5 | 1.14.6 (`/opt/hdf5-1.14.6`, 已用 NVHPC 编译) |
| 编译器 | `h5pfc` → `nvfortran` |

CMake 编译标志：`-Mpreprocess -Mextend -traceback -g -O2 -DHDF5 -Mnostack_arrays`

---

## 问题 2: nvfortran 解析器 bug — 字符串 `\'` 被误解为转义引号

**文件**: `src/strings.F90` 第 2659–2705 行

**问题**: `visible` 函数内定义了一个 256 元素的字符参数数组 `chars(0:255)`，其中包含 `'M-^\'` 这样的字符串字面量。nvfortran 将字符串内的 `\'` 序列误解析为转义的单引号，导致引号匹配错误。

**报错信息**:
```
NVFORTRAN-S-0034-Syntax error at or near identifier m (strings.F90: 2669)
NVFORTRAN-S-0023-Syntax error - unbalanced brackets (strings.F90: 2687)
NVFORTRAN-S-0038-Symbol, chars, has not been explicitly declared
```

**根因分析**: 经过逐元素二分隔离测试，确认触发条件是 `'M-^\'` 这个特定的字符串——反斜杠 `\` 后紧跟单引号 `'` 被 nvfortran 解析器识别为类 C 语言的转义序列 `\'`（转义单引号），导致后续所有字符解析错乱。

`visible` 函数并未被项目中任何其他模块调用（仅 `strings` 模块的 `split` 被 `utility.F90` 使用），因此直接删除该函数是安全的。

**修复**:
1. 删除整个 `visible` 函数（第 2626–2705 行，含注释块）
2. 移除对应的 `PUBLIC visible` 声明（原第 52 行）

---

## 问题 3: `integer(kind=16)` — 128 位整数类型不支持

**文件**: `src/vtkio.F90` 第 38, 190, 358, 535 行

**问题**: NVHPC 不支持 `integer(kind=16)`（16 字节 / 128 位整数）。该变量 `ioff` 用作文件偏移量，64 位整数已足够。

**报错信息**:
```
NVFORTRAN-S-0081-Illegal selector - KIND parameter has unknown value for data type
```

**修复**: 模块级引入 `int64`，4 处局部声明改用标准可移植类型：
```diff
 module WriteVTK
+  use iso_fortran_env, only: int64
   implicit none
   ...
-     integer(kind=16) :: ioff
+     integer(int64) :: ioff
```

> `int64` (KIND=8) 上限 9.2 EB，对文件偏移量完全够用。

---

## 问题 4: `isnan()` — gfortran 扩展函数

**文件**: `src/fludyna.F90`, `src/geom.F90`, `src/solver.F90`, `src/pp.F90`

**问题**: `isnan()` 是 gfortran 的内置扩展函数，不是标准 Fortran。NVHPC 不提供该函数。标准 Fortran 2008 的等价函数是 `ieee_is_nan()`，需通过 `use ieee_arithmetic` 引入。

**报错信息**:
```
NVFORTRAN-S-0038-Symbol, isnan, has not been explicitly declared
```

**涉及子程序** (共 8 个):

| 文件 | 子程序 | 行号范围 |
|------|--------|---------|
| `src/fludyna.F90` | `updateq` | 251–293 |
| `src/geom.F90` | `gridgeom` | 99–956 |
| `src/geom.F90` | `pointintriangle_tri` | 4143–4246 |
| `src/geom.F90` | `pointintriangle_nodes` | 4248–4321 |
| `src/solver.F90` | `src_tbl` | 401–487 |
| `src/pp.F90` | `solidgen_sphere_tri` | 1353–1577 |
| `src/pp.F90` | `solidgen_sphere` | 1590–1727 |
| `src/pp.F90` | `solidgen_circle` | 1739–1872 |

**修复** (分两步):

**步骤 A** — 全局替换函数名:
```bash
sed -i 's/isnan(/ieee_is_nan(/g' src/fludyna.F90 src/geom.F90 src/solver.F90 src/pp.F90
```

**步骤 B** — 在每个子程序的 `use` 声明块中添加:
```fortran
use ieee_arithmetic, only: ieee_is_nan
```

---

## 问题 5: `real(kind=real128)` — 128 位浮点类型不支持

**文件**: `src/strings.F90` 第 5373 行

**问题**: NVHPC 不支持四精度浮点数 `real128`。`print_generic` 子程序中的 `SELECT TYPE` 分支引用了 `real(kind=real128)`，导致编译错误。

**报错信息**:
```
NVFORTRAN-S-0081-Illegal selector - KIND value must be non-negative
NVFORTRAN-S-0155-Duplicate TYPE IS real
```

**修复**: 用预处理器条件包裹，仅在非 NVHPC 编译器下编译该分支:
```diff
+ #ifndef __NVCOMPILER
      type is (real(kind=real128));     write(line(istart:),'(1pg0)') generic
+ #endif
```

> 注：NVHPC 预处理器同时定义 `__NVCOMPILER` 和 `__PGI` 宏。

---

## 问题 6: 链接时缺少 HDF5 High-Level 库

**文件**: `build_nvhpc.sh`

**问题**: `h5pfc` 默认只链接 `-lhdf5_fortran -lhdf5`，但 ASTR 使用了 HDF5 Light (`h5lt`) API（`use h5lt`），需要额外链接 `-lhdf5_hl_fortran -lhdf5_hl`。

**报错信息**:
```
undefined reference to `h5lt_h5ltread_dataset_integer_kind_4_rank_1_'
undefined reference to `h5lt_h5ltmake_dataset_real_kind_8_rank_1_'
... (等 13 个未定义引用)
```

**修复**:
```bash
# 链接命令中显式添加 HL 库
LIBS="-lhdf5_hl_fortran -lhdf5_hl -lz -lm"
```

---

## 问题 7: `present(timerept) .and. timerept` — NVHPC 求值顺序导致段错误

**文件**: `src/parallel.F90`, `src/comsolver.F90`, `src/commcal.F90`, `src/mainloop.F90`, `src/readwrite.F90`, `src/solver.F90`, `src/statistic.F90`, `src/bc.F90` (共 6 个源文件, 56 处)

**问题**: 代码中大量使用 `if(present(timerept) .and. timerept)` 模式。Fortran 标准不保证 `.and.` 操作符的左到右短路求值。gfortran 恰好先求值 `present(timerept)`，但 **NVHPC 选择先求值 `timerept`（解引用可选参数的指针）**。当 `timerept` 未传入时该指针为 NULL → 段错误。

**报错信息**:
```
Segmentation fault (11) at address (nil)
backtrace: parallel_array3d_sync_+0x124 → geom_gridgeom_ → geom_geomcal_ → MAIN_
           comsolver_gradcal_+0x42 → mainloop_time_integration_rk_ → mainloop_steploop_
```

**根因分析**: 反汇编确认 NVHPC 生成的代码在 `array3d_sync` 中先执行 `mov (%rax),%eax`（解引用 `timerept` 指针），再执行 `cmp $0x0,%rax`（检查指针是否为 NULL）。当 `timerept` 未传入（如在 `geom.F90:680` 调用 `call datasync(dxi(0:im,0:jm,0:km,i,j))` 时不传 `timerept`），指针为 NULL，解引用导致段错误。

**涉及子程序**: 所有包含 `present(timerept) .and. timerept` 模式的子程序，包括 `array3d_sync`, `array3d_sendrecv`, `array4d_sync`, `array4d_sendrecv`, `array5d_sendrecv`, `gradcal`, `rhscal`, `diffrsdcal6`, `convrsdcal6`, `src_tbl`, `readwrite` 中的 I/O 子程序等。

**修复**: 将所有 `present(timerept) .and. timerept` 拆分为嵌套 `if` 结构，强制编译器先检查 `present()`：

```diff
-    if(present(timerept) .and. timerept) time_beg=ptime()
+    if(present(timerept)) then
+      if(timerept) time_beg=ptime()
+    endif

-    if(present(timerept) .and. timerept) then
-      ...
-    endif
+    if(present(timerept)) then
+      if(timerept) then
+        ...
+      endif
+    endif
```

| 文件 | 修复处数 |
|------|---------|
| `src/parallel.F90` | 23 |
| `src/comsolver.F90` | 4 |
| `src/commcal.F90` | 2 |
| `src/solver.F90` | 11 |
| `src/readwrite.F90` | 6 |
| `src/statistic.F90` | 4 |
| `src/bc.F90` | 5 |
| `src/mainloop.F90` | 1 |
| **合计** | **56** |

---

## 问题 8: CRLF 换行符导致 NVHPC 读取字符串错误 → 流场未初始化 → NaN 崩溃

**文件**: `examples/Taylor_Green_Vortex/datin/input.dat`, `controller`, `input.tgv`, `input.tgv2d`, `grid.h5`

**问题**: `examples/Taylor_Green_Vortex/datin/` 目录下的文本文件使用 Windows 风格 CRLF (`\r\n`) 换行符，且网格文件名 `grid.h5` 末尾包含回车符 `\r`（实际文件名为 `grid.h5\r`）。NVHPC 的 Fortran 运行时**将 `\r` 视为字符串标记的一部分**，而 gfortran 将其视为空白字符。

**报错信息**:
```
flowtype 显示为: tgv: (后跟乱码字节)
current CFL: 0.0000000
time step for CFL=1: Inf
!! COMPUTATION CRASHED, JOB STOPS !!
Warning: ieee_invalid is signaling
Warning: ieee_divide_by_zero is signaling
```

**根因分析**:
1. 输入文件中 `tgv\r\n` 被 NVHPC 读为 `flowtype = "tgv\r"`（末尾带回车符）
2. `trim("tgv\r")` 长度为 4，与 `case('tgv')`（长度 3）不匹配
3. `select case` 匹配失败，TGV 初始化 `tgvini` **从未被执行**
4. 流场数组 (`rho`, `vel`, `prs`, `tmp`) 保持未初始化状态（垃圾值或零值）
5. 垃圾值经过 `fvar2q` 转换为守恒变量 `q`，再经过时间步进产生 NaN
6. `crashcheck` 检测到 NaN 密度 → 报告 `COMPUTATION CRASHED`
7. `controller` 文件同样有 CRLF 问题，可能导致 `deltat` 等参数读取错误

**验证**: 反汇编确认同一输入文件在 gfortran 下运行正常（gfortran 的 list-directed I/O 将 `\r` 视为空白，正确读取 `"tgv"`）。

**修复**:
1. 移除所有文本文件中的 `\r` 字符（转为 Unix LF 换行）:
   ```bash
   sed -i 's/\r$//' datin/controller datin/input.tgv datin/input.tgv2d
   tr -d '\r' < datin/input.dat > /tmp/input_fixed.dat && cp /tmp/input_fixed.dat datin/input.dat
   ```

2. 重命名网格文件，移除文件名中的 `\r` 字符:
   ```bash
   mv "datin/grid.h5"$'\r' datin/grid.h5
   ```

**修复后验证**:
```
** 3-D TGV initialised.            ← TGV 初始化成功执行
current CFL: 0.6524326             ← CFL 正常
time step for CFL=1: 0.15327E-02   ← 合理的时间步长
** time cost at step0: 1.27s       ← 时间步进正常
nstep 1-10 全部计算成功，密度保持 1.0，无 NaN/崩溃
** The job is done!                ← 程序正常完成
```

---

## 编译标志对照表

| gfortran 标志 | nvfortran 标志 | 用途 |
|---------------|----------------|------|
| `-cpp` | `-Mpreprocess` | C 风格预处理 |
| `-ffree-line-length-none` | `-Mextend` | 132 列源代码行 |
| `-fbacktrace` | `-traceback` | 运行时错误回溯 |
| `-J <dir>` | `-module <dir>` | 模块文件输出目录 |
| `-fallow-argument-mismatch` | 不需要 | nvfortran 默认允许 |

---

## 验证结果

```bash
# 编译 (CMake)
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
cmake --build . -j$(nproc)
# 输出: [100%] Built target astr，56/56 文件编译通过 (含 user_define_module)

# 编译 (build_nvhpc.sh)
bash build_nvhpc.sh
# 输出: Build succeeded! 41/41 文件编译通过
```

### TGV 算例运行验证

```bash
cd examples/Taylor_Green_Vortex
mpirun -np 8 ../../build/bin/astr run datin/input.dat
```

输出:
```
** 3-D TGV initialised.              ← 流场初始化成功
** flowfield initialised.
=========== CFL Condition===========
   current time step:    0.10000E-02
         current CFL:      0.6524326
 time step for CFL=1:    0.15327E-02
====================================
** time cost at step0:      1.26960544s
    nstep=1..10 全部计算成功，密度保持在 1.0
** The job is done!                  ← 正常完成，无崩溃
```

可执行文件: `bin/astr` (8.5 MB, ELF 64-bit, 动态链接, 带调试符号)

### 已修复的 NVHPC 特定问题汇总

| 问题 | 类别 | 严重程度 | 症状 |
|------|------|---------|------|
| 问题 2: `\'` 转义 | 编译器解析器 | 编译失败 | Syntax error |
| 问题 3: `integer(kind=16)` | 类型不支持 | 编译失败 | Illegal selector |
| 问题 4: `isnan()` | 非标准扩展 | 编译失败 | Symbol not declared |
| 问题 5: `real(kind=real128)` | 类型不支持 | 编译失败 | Illegal selector |
| 问题 6: HDF5 HL 库缺失 | 链接 | 链接失败 | undefined reference |
| **问题 7**: `present().and.` | **求值顺序** | **段错误** | **Segfault at NULL** |
| **问题 8**: CRLF 换行符 | **I/O 行为差异** | **数值崩溃** | **NaN/负密度** |
