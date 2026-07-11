# ASTR CPU Numerical Schemes

## 1. Purpose And Audit Scope

This document records the numerical schemes present in the CPU implementation under `src/`. It is an implementation audit, not only an input-file user guide.

The following status labels are used throughout:

| Status | Meaning |
|---|---|
| **Main path** | Selected by the normal runtime input and used by the CPU time-integration path. |
| **Library only** | A function exists in `src/`, but the current main-flow dispatcher does not select it end to end. |
| **Conditional** | Requires a particular compile option, physics model, sensor, boundary type, or reconstruction switch. |
| **Disabled** | The input/reporting branch exists, but the numerical implementation is commented out or reaches `stop`. |
| **GPU ported** | A corresponding CUDA Fortran path has passed a documented CPU/GPU oracle. |

The audit covers:

- convection discretization;
- Steger-Warming flux splitting;
- explicit and compact interface reconstruction;
- gradient and diffusion discretization;
- explicit and compact filtering;
- RK3 and RK4 time integration;
- boundary-specific derivative and filter operators;
- sponge-layer relaxation and chemistry ODE coupling;
- shock-sensor coupling;
- physical-boundary closure and MPI-interface behavior;
- current CPU/GPU format correspondence.

Primary source modules:

| File | Numerical responsibility |
|---|---|
| `src/readwrite.F90` | Runtime input and human-readable scheme reporting. |
| `src/solver.F90` | Convection and diffusion RHS dispatch and assembly. |
| `src/comsolver.F90` | Scheme initialization, gradients, filtering, and solver orchestration. |
| `src/derivative.F90` | Explicit and compact central first derivatives. |
| `src/riemann.F90` | Steger-Warming flux-vector splitting. |
| `src/flux.F90` | Explicit reconstruction, MP/WENO/ROUND functions, compact flux reconstruction. |
| `src/filter.F90` | Explicit and compact low-pass filter coefficients and line solvers. |
| `src/mainloop.F90` | RK stage ordering and state update. |
| `src/commcal.F90` | Ducros-style shock sensor. |
| `src/commfunc.F90` | One-sided boundary derivatives and extrapolation helpers. |
| `src/bc.F90` | NSCBC-specific low-order derivatives and tangential filtering. |
| `src/sponge_layer.F90` | Local six-neighbor sponge relaxation. |
| `src/thermchem.F90` | Conditional implicit chemistry ODE update. |

## 2. Runtime Scheme Encoding

The principal input fields are:

```text
conschm, difschm, rkscheme
recon_schem, lchardecomp, bfacmpld, shkcrt
```

Typical values found under `examples/` are:

```text
643e, 643e, rk3
643c, 643c, rk3
543e, 643e, rk3
543c, 643c, rk3
543c, 642e, rk3
```

For a four-character spatial-scheme token such as `643e`:

- characters 1-3 are the format-family code;
- character 4 is `e` for explicit or `c` for compact;
- the first digit is also used as a dispatch flag: even means central and odd means upwind-biased.

`src/readwrite.F90:200-270` reports `643e` as an explicit central `3-4-6...6-4-3` family and `543e` as an explicit upwind `3-4-5...5-4-3` family. This printed label is not a complete statement of the actual stencil at every node.

Two distinctions are important:

1. `conschm='543e'` selects the explicit upwind framework, but `recon_schem` selects the actual interface reconstruction. For example, `recon_schem=1` uses WENO7 in a periodic interior, not WENO5.
2. The active explicit derivative object currently calls `diff6ec` directly. Therefore changing the first three digits of an explicit central token does not automatically activate the 2nd-, 4th-, or 8th-order utility functions.

## 3. Governing Discrete RHS Structure

For the conservative state

$$
\mathbf{Q}=(\rho,\rho u,\rho v,\rho w,\rho E)^T,
$$

the mapped-grid flow equation is assembled schematically as

$$
\frac{\partial (J\mathbf{Q})}{\partial t}
=\mathbf{R}_{\mathrm{conv}}+\mathbf{R}_{\mathrm{diff}}+\mathbf{R}_{\mathrm{src}}.
$$

The CPU stage order in `src/mainloop.F90:423-510` is:

```text
optional filter
immersed-boundary preparation
physical boundary conditions
solution halo exchange
gradient calculation
save RK reference state on stage 1
convection/diffusion/source RHS
RK state update
sponge filter
primitive-variable refresh
```

Consequently, when `lfilter=t`, filtering is applied at every RK substage, not once per complete time step.

## 4. Central Convection Schemes

### 4.1 CPU dispatch

`src/solver.F90:185-245` reads the first digit of `conschm`. An even first digit calls `convrsdcal6`; an odd first digit selects the upwind path.

For central convection, `src/comsolver.F90:35-115` requires

```text
trim(conschm) == trim(difschm)
```

and stops if the two central-scheme tokens differ. The derivative backend is selected by the suffix:

| Suffix | Backend type | Main CPU implementation |
|---|---|---|
| `e` | `explicit_central` | `df_explicit -> diff6ec` |
| `c` | `compact_central` | `df_compact -> tridiagonal_thomas_solver` |

### 4.2 Curvilinear convective flux

For computational direction $\xi_l$, `convrsdcal6` forms the contravariant velocity

$$
U_l=\xi_{l,x}u+\xi_{l,y}v+\xi_{l,z}w,
$$

and mapped inviscid flux

$$
\mathbf{F}_l=J
\begin{bmatrix}
\rho U_l\\
\rho u U_l+\xi_{l,x}p\\
\rho v U_l+\xi_{l,y}p\\
\rho w U_l+\xi_{l,z}p\\
(\rho E+p)U_l
\end{bmatrix}.
$$

The selected central derivative is applied independently to every conservative flux component and in every active spatial direction. Species and modal-equation flux components, when present, are transported as $JQ_mU_l$.

Code path:

```text
mainloop:time_integration_rk
  -> solver:rhscal
    -> solver:convrsdcal6
      -> derivative:fds%central
```

## 5. Explicit Central First Derivative

### 5.1 Interior sixth-order formula

The active explicit central derivative uses

$$
D_6 f_i=
\frac{3}{4}(f_{i+1}-f_{i-1})
-\frac{3}{20}(f_{i+2}-f_{i-2})
+\frac{1}{60}(f_{i+3}-f_{i-3}).
$$

This is implemented by `derivative:diff6ec` in `src/derivative.F90:350-413`.

### 5.2 Physical-boundary closure actually used

At a lower physical boundary, the code uses

$$
D f_0=-\frac{3}{2}f_0+2f_1-\frac{1}{2}f_2,
$$

$$
D f_1=\frac{1}{2}(f_2-f_0),
$$

$$
D f_2=\frac{2}{3}(f_3-f_1)-\frac{1}{12}(f_4-f_0),
$$

then switches to $D_6$. The upper boundary uses the mirrored formulas.

Therefore the actual explicit closure is second order at the boundary node, second order at the first interior node, fourth order at the second interior node, and sixth order afterward. This does not exactly match the human-readable `3-4-6` label printed for `643e`.

### 5.3 MPI interface behavior

For `ntype=3`, including periodic or MPI-internal blocks with valid halos, the sixth-order centered stencil is applied through every local endpoint. No one-sided physical formula is used at an internal MPI interface.

### 5.4 Main-path status

| Token | Status | Effective implementation |
|---|---|---|
| `643e` | **Main path** | Sixth-order explicit central interior with the boundary closure above. |
| Other `...e` central tokens | **Not independent main formats** | `df_explicit` still calls `diff6ec`; digits do not select another derivative implementation. |

## 6. Compact Central First Derivative

### 6.1 Interior equation

For `643c`, `fd_scheme_initiate` sets the tridiagonal coefficients to $1/3$. The interior equation is

$$
\frac{1}{3}f'_{i-1}+f'_i+\frac{1}{3}f'_{i+1}
=\frac{7}{9}(f_{i+1}-f_{i-1})
+\frac{1}{36}(f_{i+2}-f_{i-2}).
$$

The right-hand side is assembled by `derivative:compact_fd_rhs`, and the derivative is obtained from

```text
tridiagonal_thomas_preprocess
tridiagonal_thomas_solver
```

### 6.2 Physical-boundary equations

At the first physical node, the right-hand side is

$$
d_0=-\frac{5}{2}f_0+2f_1+\frac{1}{2}f_2.
$$

The corresponding compact matrix row uses the boundary coefficient set by `fd_scheme_initiate`. The first interior compact closure uses

$$
d_1=\frac{3}{4}(f_2-f_0),
$$

with neighboring derivative coefficient $1/4$. The upper-boundary equations are mirrored.

### 6.3 Status and limitation

`src/derivative.F90:45-145` only initializes the compact derivative when the hundreds digit is 6 and the numeric token is `643`. Other compact central tokens reach `stop`.

`643c` is therefore the active CPU compact central derivative. It requires a linewise tridiagonal solve in every direction and for every differentiated field.

## 7. Gradient And Diffusion Discretization

### 7.1 Gradient backend

`gradcal` uses the same `fds%central` backend selected from `difschm`. Velocity, temperature, species, and model-variable gradients therefore inherit the explicit or compact central derivative choice.

### 7.2 Viscous and heat flux

`solver:diffrsdcal6` constructs the strain tensor

$$
S_{ij}=\frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}
+\frac{\partial u_j}{\partial x_i}\right),
$$

the Newtonian stress

$$
\tau_{ij}=2\mu\left(S_{ij}-\frac{1}{3}\nabla\cdot\mathbf{u}\,\delta_{ij}\right),
$$

and Fourier heat flux using the runtime Reynolds and Prandtl scaling. The mapped viscous flux divergence is then evaluated with the same central derivative backend.

For combustion builds, the same routine also includes species diffusion and species-enthalpy energy flux. For turbulence builds it can add model fluxes and source terms. Those physics branches do not create a different base finite-difference family.

### 7.3 Effective CPU choices

| `difschm` | Status | Main behavior |
|---|---|---|
| `643e` | **Main path** | Explicit sixth-order central gradient and flux divergence. |
| `643c` | **Main path** | Compact sixth-order central gradient and flux divergence. |
| `642e` | **Accepted in some example inputs** | Active explicit object still calls `diff6ec`; it does not select a distinct `642` implementation. |

## 8. Explicit Derivative Utility Library

`derivative:ddfc_basic` provides additional explicit functions selected from the hundreds digit:

| Utility family | Function | Interior formula/status |
|---|---|---|
| 2xx | `diff2ec` | Second-order centered. **Library only** for the main flow solver. |
| 4xx | `diff4ec` | Fourth-order centered; physical closure explicitly recognizes `422`. **Library only**. |
| 6xx | `diff6ec` | Sixth-order centered. Also used by the main explicit backend. |
| 8xx | `diff8ec` | Eighth-order centered on periodic/internal-halo blocks. **Library only**. |

For `ntype=3` (an internal block with valid halos), the eighth-order formula is

$$
D_8 f_i=\frac{1}{840}\left[
672(f_{i+1}-f_{i-1})
-168(f_{i+2}-f_{i-2})
+32(f_{i+3}-f_{i-3})
-3(f_{i+4}-f_{i-4})
\right].
$$

There are several source-level limitations in these utility functions:

- `diff8ec` enters its one-sided physical-boundary branches for `ntype=1/2`, but those branches require `ns==642`, which is impossible when `ddfc_basic` selected `diff8ec` from an 8xx token. They also use sixth-order, rather than eighth-order, interior stencils after the boundary closure.
- `diff8ec` treats `ntype=4` as a halo-backed eighth-order block even though `ntype=4` means both ends are physical boundaries in the surrounding solver setup.
- `diff4ec` and `diff2ec` do not implement `ntype=4`; that boundary type reaches their error branch.

Therefore only the halo-backed `ntype=3` eighth-order utility path is internally consistent as written. None of the 2nd-, 4th-, or 8th-order utility families should be advertised as a general physical-boundary main-flow format without source repair and dedicated validation.

### 8.1 Boundary-specific low-order operators

The NSCBC routines in `src/bc.F90` do not exclusively reuse the selected `difschm`. They contain dedicated boundary numerics:

- `commfunc:deriv` provides first- through fourth-order one-sided derivatives at a boundary;
- tangential NSCBC flux derivatives call `ddfc(...,'222e',...)`, which selects `diff2ec`;
- selected farfield/outflow NSCBC faces call `spafilter6exp` along tangential directions.

For example, the one-sided derivative utilities are

$$
D_1 f_0=-f_0+f_1,
$$

$$
D_2 f_0=-\frac32f_0+2f_1-\frac12f_2,
$$

$$
D_3 f_0=-\frac{11}{6}f_0+3f_1-\frac32f_2+\frac13f_3,
$$

$$
D_4 f_0=-\frac{25}{12}f_0+4f_1-3f_2+\frac43f_3-\frac14f_4.
$$

These operators are **Conditional** boundary implementations, not globally selectable `conschm`/`difschm` families. Their presence is relevant to future GPU farfield/outflow/NSCBC work because a GPU port must reproduce these embedded boundary operators in addition to the main spatial scheme.

### 8.2 `lfftk` and the FFT implementation status

The input and runtime report expose `lfftk` and `kcutoff`, and `src/singleton.F90` contains a general FFT library. However, the active flow derivative object does not call that FFT library:

- `df_explicit` always calls `diff6ec`;
- `df_compact` always calls the compact finite-difference solver;
- the optional `lfft` argument and FFT work variables in `ddfc_basic` are not used;
- `lfftk` mainly restricts MPI decomposition in k and changes several geometry formulas to constant k increments.

Consequently, `lfftk` must not currently be described as a spectral flow-derivative format merely because `infodisp` prints `FFT used at the k direction`. The repository contains an FFT library and an FFT-facing input flag, but no connected end-to-end FFT flow discretization was found in `src/`.

## 9. Explicit Upwind Convection Framework

### 9.1 Dispatch

An odd first digit in `conschm` selects upwind convection. For suffix `e`, the path is

```text
solver:rhscal
  -> optional commcal:ducrossensor
  -> solver:convrsduwd
    -> riemann:flux_steger_warming
    -> optional characteristic projection
    -> flux:recons_exp
    -> interface-flux difference
```

The main explicit upwind token used by the codebase is `543e`.

### 9.2 Steger-Warming flux-vector splitting

For one mapped direction, define

$$
U=\boldsymbol{\xi}\cdot\mathbf{u},\qquad
s=\frac{1}{\|\boldsymbol{\xi}\|},\qquad
c_a=\frac{c}{s}.
$$

The Euler eigenvalues are

$$
\lambda_{1,2,3}=U,\qquad
\lambda_4=U+c_a,\qquad
\lambda_5=U-c_a.
$$

ASTR regularizes the positive eigenvalues with $\epsilon=0.04$:

$$
\lambda_r^+=\frac{1}{2}\left(\lambda_r+\sqrt{\lambda_r^2+\epsilon^2}\right),
\qquad
\lambda_r^-=\lambda_r-\lambda_r^+.
$$

For mapped Mach number $M_a=U/c_a\ge 1$, the complete Euler flux is assigned to $\mathbf{F}^+$ and $\mathbf{F}^-=0$. For $M_a\le-1$, it is assigned to $\mathbf{F}^-$ and $\mathbf{F}^+=0$. In the subsonic branch, `src/riemann.F90:119-153` reconstructs each split conservative flux from the split eigenvalues.

The numerical interface flux is

$$
\widehat{\mathbf{F}}_{i+1/2}
=\mathcal{R}(\mathbf{F}^+)_i
+\mathcal{R}(\mathbf{F}^-)_i,
$$

where $\mathcal{R}$ is selected by `recon_schem`.

### 9.3 Physical-space and characteristic-space reconstruction

If `lchardecomp=f`, every conservative split-flux component is reconstructed directly.

For the explicit upwind path, `lchardecomp=t` first triggers `ducrossensor`. At each interface, ASTR enters Roe characteristic space only when the adjacent sensor flag `lshock` is true. It then projects the first five split fluxes, reconstructs each characteristic component, and projects the result back. Outside the flagged region, reconstruction remains in physical space. Additional species or modal components are reconstructed without the five-equation characteristic transform.

The compact upwind path differs: when `lchardecomp=t`, `convrsdcmp` performs characteristic projection at every interface. Its shock flag is subsequently used by the MP limiter, not as a gate around the characteristic transform.

## 10. Explicit Reconstruction Families

### 10.1 Reconstruction selector

| `recon_schem` | Reported name | Interior implementation | Near-boundary implementation | Status |
|---:|---|---|---|---|
| `-1` | First order | `f(4)` | `f(4)` | **Main path** |
| `0` | Linear | SUW7 | SUW5, then SUW3/average | **Main path** |
| `1` | WENO | WENO7 | WENO5, then SUW3/average | **Main path** |
| `2` | WENO-Z | WENO7-Z | WENO5-Z, then SUW3/average | **Main path** |
| `3` | MP | MP7 | MP5, then SUW3/average | **Main path** |
| `4` | WENO-SYM | Branch commented out | Branch commented out | **Disabled** |
| `5` | MP-LD | MP7-LD | MP5-LD, then SUW3/average | **Conditional** |
| `6` | ROUND | Three-point ROUND | Three-point ROUND where available | **Main path** |

The `recon_schem=4` warning is important: `src/readwrite.F90` prints `WENO-SYM construction`, but both `recons_exp` branches are commented out. Selecting it falls into the default error branch and stops.

### 10.2 Boundary degradation in `recons_exp`

For a block touching a physical boundary, reconstruction is reduced according to distance from the face:

| Interface distance from physical face | `recon_schem=-1` | Other selectors |
|---|---|---|
| Outermost interface | First order `f(4)` | Arithmetic average `[f(4)+f(5)]/2` |
| Next interface | First order `f(4)` | SUW3 |
| Third interface | First order | Selector-specific 5-point/6-point form |
| Remaining interior | First order | Selector-specific 7-point/8-point form |

MPI-internal and periodic interfaces have halo data and therefore use the interior reconstruction rather than the physical-boundary degradation.

The boundary table above is only implemented for `ntype=1` (lower physical face) and `ntype=2` (upper physical face). `recons_exp` has no `ntype=4` degradation branch. In addition, `convrsduwd` accepts `ntype=4` in x but omits it from the y/z line-range dispatch. Thus the explicit upwind path is not a validated general solution for a local block that owns both physical faces in y or z; even the x `ntype=4` path does not apply the intended near-boundary reconstruction hierarchy.

### 10.3 Linear SUW formulas

The linear reconstructions are

$$
\widehat f^{\mathrm{SUW3}}
=-\frac{1}{6}f_1+\frac{5}{6}f_2+\frac{1}{3}f_3,
$$

$$
\widehat f^{\mathrm{SUW5}}
=\frac{2f_1-13f_2+47f_3+27f_4-3f_5}{60},
$$

$$
\widehat f^{\mathrm{SUW7}}
=\frac{-3f_1+25f_2-101f_3+319f_4+214f_5-38f_6+4f_7}{420}.
$$

### 10.4 WENO5

ASTR uses three candidate polynomials:

$$
p_0=\frac{1}{3}f_1-\frac{7}{6}f_2+\frac{11}{6}f_3,
$$

$$
p_1=-\frac{1}{6}f_2+\frac{5}{6}f_3+\frac{1}{3}f_4,
$$

$$
p_2=\frac{1}{3}f_3+\frac{5}{6}f_4-\frac{1}{6}f_5.
$$

The linear weights are

$$
(C_0,C_1,C_2)=(0.1,0.6,0.3),
$$

and the nonlinear weights are

$$
\alpha_r=\frac{C_r}{(\beta_r+10^{-6})^2},\qquad
\omega_r=\frac{\alpha_r}{\sum_s\alpha_s},
$$

$$
\widehat f=\sum_{r=0}^{2}\omega_rp_r.
$$

The Jiang-Shu-style smoothness indicators are implemented explicitly in `flux:WENO5`.

### 10.5 WENO7

The four candidate polynomials are

$$
p_0=-\frac14 f_1+\frac{13}{12}f_2-\frac{23}{12}f_3+\frac{25}{12}f_4,
$$

$$
p_1=\frac{1}{12}f_2-\frac{5}{12}f_3+\frac{13}{12}f_4+\frac14 f_5,
$$

$$
p_2=-\frac{1}{12}f_3+\frac{7}{12}f_4+\frac{7}{12}f_5-\frac{1}{12}f_6,
$$

$$
p_3=\frac14 f_4+\frac{13}{12}f_5-\frac{5}{12}f_6+\frac{1}{12}f_7.
$$

The linear weights are

$$
(C_0,C_1,C_2,C_3)=\left(\frac1{35},\frac{12}{35},\frac{18}{35},\frac4{35}\right).
$$

ASTR uses four smoothness indicators assembled from first-, second-, and third-difference combinations, then

$$
\alpha_r=\frac{C_r}{(\beta_r+10^{-6})^2},\qquad
\widehat f=\frac{\sum_r\alpha_rp_r}{\sum_r\alpha_r}.
$$

For a periodic `543e`, `recon_schem=1` run, this WENO7 operator is the actual interior reconstruction.

### 10.6 WENO-Z

WENO5-Z reuses the WENO5 candidates and smoothness indicators with

$$
\tau_5=|\beta_2-\beta_0|.
$$

WENO7-Z uses

$$
\tau_7=|\beta_3-\beta_0|.
$$

The code computes the unnormalized weights as

$$
\alpha_r=C_r+C_r\frac{\tau}{(\beta_r+10^{-6})^2}.
$$

This formula is recorded exactly as implemented; it should not be silently replaced in documentation by a different textbook WENO-Z exponent convention.

### 10.7 MP5 and MP7

MP reconstruction begins from the SUW5 or SUW7 linear value $f^{L}_{i+1/2}$. It forms

$$
f^{MP}=f_i+\operatorname{minmod}\left(f_{i+1}-f_i,4(f_i-f_{i-1})\right).
$$

If

$$
(f^L_{i+1/2}-f_i)(f^L_{i+1/2}-f^{MP})\ge10^{-10},
$$

the code constructs limited second differences, upper/lower admissible interface values, and applies a final minmod correction. Otherwise it keeps the linear value.

`MP5` optionally uses a discontinuity flag; `MP7` applies the limiter condition directly.

### 10.8 MP-LD

MP-LD modifies the linear reconstruction through `bfacmpld` before applying MP limiting.

For MP5-LD,

$$
v_{adp}=\frac{\texttt{bfacmpld}}{60},
$$

with a six-point linear combination whose coefficients interpolate between dissipative fifth-order upwind behavior and a lower-dissipation central tendency.

For MP7-LD,

$$
v_{adp}=\frac{\texttt{bfacmpld}}{280},
$$

and an eight-point linear combination is used. In `MP7LD`, the nonlinear limiter is applied only when the shock flag is true; otherwise the modified linear reconstruction is returned.

`recon_schem=5` also requests the Ducros-style sensor before explicit upwind convection.

### 10.9 ROUND

ROUND uses three points and normalized variable

$$
z=\frac{f_2-f_1+10^{-16}}{f_3-f_1+10^{-16}}.
$$

The code builds nonlinear blending functions $p_1(z)$, $p_2(z)$, $p_3(z)$ and weights from

$$
a_1=1+12z^2,\qquad a_2=1+5(z-1)^2,
$$

including localized polynomial corrections around specified normalized-variable intervals. The final value is

$$
\widehat f=g(z)(f_3-f_1)+f_1.
$$

The exact piecewise-polynomial construction is in `flux:round`.

## 11. Compact Upwind Scheme

### 11.1 Available token

`flux:compact_flux_initiate` only supports a scheme with hundreds digit 5 and specifically accepts numeric token `543` for boundary setup. The runtime token is therefore `543c`.

### 11.2 Split flux and line solve

The Euler flux is first split by Steger-Warming. Separate compact systems are initialized for positive and negative wind directions.

For positive wind, the compact matrix coefficients are

$$
a^+=\frac12-\frac{\texttt{bfacmpld}}6,
\qquad
c^+=\frac16+\frac{\texttt{bfacmpld}}6,
$$

and the negative-wind coefficients are mirrored.

The positive-wind RHS is

$$
d_i=\left(\frac1{18}-\frac{b}{36}\right)f_{i-1}
+\left(\frac{19}{18}-\frac{b}{4}\right)f_i
+\left(\frac59+\frac{b}{4}\right)f_{i+1}
+\frac{b}{36}f_{i+2},
$$

where $b=\texttt{bfacmpld}$. The negative-wind formula is its mirror. A tridiagonal Thomas solve produces the interface flux.

### 11.3 Limiting and characteristic behavior

`convrsdcmp` combines compact interface flux reconstruction with MP limiting and optional all-interface characteristic decomposition. The runtime report describes the compact upwind reconstruction as MP-LD.

This scheme requires linewise tridiagonal solves and is deliberately outside the current GPU migration scope.

## 12. Ducros-Style Shock Sensor

The sensor in `src/commcal.F90:185-335` defines

$$
\chi_{\mathrm{Ducros}}=
\frac{(\nabla\cdot\mathbf{u})^2}
{(\nabla\cdot\mathbf{u})^2+|\nabla\times\mathbf{u}|^2+10^{-30}},
$$

and multiplies it by the maximum normalized pressure curvature over the three directions:

$$
\chi_p=\max_l
\frac{\left|p_{i+1}-2p_i+p_{i-1}\right|}
{p_{i+1}+2p_i+p_{i-1}}.
$$

Thus

$$
\mathrm{ssf}=\chi_{\mathrm{Ducros}}\chi_p.
$$

The pressure denominator is not protected by an epsilon or absolute value in the source; the formula therefore assumes physically positive pressure. The sensor is halo-exchanged and expanded along the three coordinate axes over offsets `-hm+1:hm` (an axial cross, not a full three-dimensional box). A node is marked as shock-region when that local maximum exceeds `shkcrt`.

Sensor dispatch:

| Path | Sensor condition |
|---|---|
| Explicit upwind | `recon_schem==5` or `lchardecomp=t` |
| Compact upwind | `lchardecomp=t` |

The sensor itself does not define a numerical flux. In the explicit path it gates characteristic decomposition and MP7-LD limiting; in the compact path it supplies the MP limiter flag while characteristic decomposition, when enabled, remains global.

## 13. Explicit Tenth-Order Central Filter

### 13.1 Main explicit filter

When both convection and diffusion suffixes are `e`, `comsolver:filterq` calls `filterq_explicit10`.

The interior operator is

$$
\widetilde f_i=
\frac{193}{256}f_i
+\frac{105}{512}(f_{i-1}+f_{i+1})
-\frac{15}{128}(f_{i-2}+f_{i+2})
+\frac{45}{1024}(f_{i-3}+f_{i+3})
-\frac{5}{512}(f_{i-4}+f_{i+4})
+\frac{1}{1024}(f_{i-5}+f_{i+5}).
$$

The implementation stores `coef10e(0)=193/512` and evaluates the $m=0$ symmetric pair twice, giving the center coefficient $193/256$.

The filter is explicit: it does not solve a linear system. The CPU implementation applies it direction by direction with temporary line arrays and performs `dataswap` before each direction.

### 13.2 Other explicit filter coefficients

`filter_coefficient_explicit` also defines 2nd-, 4th-, 6th-, and 8th-order coefficient sets and several biased boundary sets. The main conservative-variable filter path nevertheless calls only `spafilter10exp`.

Most lower-order coefficient sets are therefore **library only**, but two exceptions matter:

1. `spafilter6exp` is called conditionally by NSCBC boundary routines. Its halo-backed interior formula is

$$
\widetilde f_i=
\frac{11}{16}f_i
+\frac{15}{64}(f_{i-1}+f_{i+1})
-\frac{3}{32}(f_{i-2}+f_{i+2})
+\frac{1}{64}(f_{i-3}+f_{i+3}),
$$

with one-sided 4th-order coefficients at the outer two nodes and a centered 4th-order formula at the third node. It implements `ntype=1/2/3`, but not `ntype=4`.
2. `comsolver:filter2e` is called for `q(:,:,:,7)` in the compact-filter k-omega branch. Its effective y-direction update is

$$
\widetilde f_j=0.995f_j+0.0025(f_{j-1}+f_{j+1}).
$$

`filter8exp` and `comsolver:filter4e` have no active caller in `src/`. Moreover, `filter4e` contains an asymmetric `j-2/j+3` pair and copies uninitialized edge entries from its temporary array; it should be treated as a defective dormant utility, not a supported filter.

## 14. Compact Filter

### 14.1 Dispatch

If filtering is enabled and either convection or diffusion is compact, `filterq` uses `compact_filter` instead of `filterq_explicit10`.

The compact filter solves

$$
\alpha\widetilde f_{i-1}+\widetilde f_i+\alpha\widetilde f_{i+1}=d_i,
$$

where $\alpha=\texttt{alfa_filter}$ and $d_i$ is a symmetric low-pass RHS.

### 14.2 Interior tenth-order coefficients

Writing

$$
d_i=2a_0f_i+\sum_{m=1}^{5}a_m(f_{i-m}+f_{i+m}),
$$

the coefficients are

$$
a_0=\frac{193+126\alpha}{512},\quad
a_1=\frac{105+302\alpha}{512},\quad
a_2=\frac{-15+30\alpha}{128},
$$

$$
a_3=\frac{45-90\alpha}{1024},\quad
a_4=\frac{-5+10\alpha}{512},\quad
a_5=\frac{1-2\alpha}{1024}.
$$

### 14.3 Boundary policy

The source describes the physical-boundary sequence as approximately

```text
0-6-6-6-8-10 ... 10-8-6-6-6-0
```

with one-sided physical-boundary coefficients, special halo/interface coefficients, and boundary coupling parameters `beter_bouond=0.98` and `beter_halo=1.11` during initialization.

The compact filter requires a tridiagonal solve in each filtered direction and is outside the current GPU migration target.

### 14.4 Sponge-layer relaxation

`mainloop:time_integration_rk` calls `spongefilter` after every RK state update. When a local sponge coefficient $\sigma_{i,j,k}$ is active, `sponge_layer.F90` applies

$$
\widetilde{Q}_{i,j,k}
=(1-\sigma_{i,j,k})Q_{i,j,k}
+\frac{\sigma_{i,j,k}}{6}
\left(Q_{i+1,j,k}+Q_{i-1,j,k}+Q_{i,j+1,k}+Q_{i,j-1,k}
+Q_{i,j,k+1}+Q_{i,j,k-1}\right).
$$

The update uses a temporary array, so all six neighbors come from the pre-sponge state. `spg_def='layer'` applies it only in configured face layers; `spg_def='circl'` uses a geometry-defined global/circular region. This is a conditional second-order smoothing/relaxation operator, separate from `lfilter` and separate from the explicit or compact filter selection.

## 15. Time Integration

### 15.1 RK3

The CPU RK3 path uses a three-stage strong-stability-preserving form. In mapped conservative variables $\mathbf{U}=J\mathbf{Q}$:

$$
\mathbf{U}^{(1)}=\mathbf{U}^n+\Delta t\,\mathbf{R}(\mathbf{Q}^n),
$$

$$
\mathbf{U}^{(2)}=\frac34\mathbf{U}^n+\frac14\mathbf{U}^{(1)}
+\frac14\Delta t\,\mathbf{R}(\mathbf{Q}^{(1)}),
$$

$$
\mathbf{U}^{n+1}=\frac13\mathbf{U}^n+\frac23\mathbf{U}^{(2)}
+\frac23\Delta t\,\mathbf{R}(\mathbf{Q}^{(2)}).
$$

Runtime token: `rkscheme='rk3'`.

### 15.2 RK4

The CPU also implements classical four-stage RK4 using intermediate factors

$$
\left(\frac12,\frac12,1,\frac16\right)
$$

and final RHS weights

$$
(1,2,2,1).
$$

Runtime token: `rkscheme='rk4'`.

Filtering, boundary application, halo exchange, gradients, and RHS assembly occur inside the stage loop for both methods.

### 15.3 Chemistry time integration under `COMB`

Chemistry introduces a second runtime selector, `odetype`, which must not be confused with the flow `rkscheme`:

| `odetype` | Implemented behavior | Status |
|---|---|---|
| `rk3` | `srccomb` adds species production rates to `qrhs` at every flow RK stage. The actual time weights therefore come from the selected flow `rkscheme`. | **Conditional** |
| `ime` | After the final flow RK stage, `thermchem:imp_euler_ode` performs a pointwise iterative production/destruction implicit-Euler update, normalizes mass fractions, and stops after 1000 failed iterations. | **Conditional** |
| `dnn` | Hybrid path choosing between repeated implicit-Euler updates and a neural-network step from a heat-release criterion. | **Conditional/experimental** |

For `ime`, each species iterate has the implemented form

$$
Y_s^{(m+1)}=Y_s^{(m)}
\frac{Y_s^n+\Delta t\,\dot\omega_{s,c}^{(m)}W_s/\rho}
{Y_s^{(m)}+\Delta t\,\dot\omega_{s,d}^{(m)}W_s/\rho},
$$

with a special creation-only branch for very small mass fraction and destruction rate. Convergence uses a logarithmic relative measure with threshold $10^{-6}$, followed by nonnegative normalization.

No chemistry time-integration path is currently GPU ported. This section records CPU behavior only; it does not change the project decision to defer chemistry migration.

## 16. CPU Main-Path Capability Matrix

| Numerical feature | CPU status | Notes |
|---|---|---|
| `643e` central convection | **Main path** | Explicit sixth-order interior derivative. |
| `643c` central convection | **Main path** | Compact sixth-order derivative and tridiagonal solve. |
| `543e` Steger-Warming upwind | **Main path** | Uses `recon_schem`. |
| `543c` compact upwind | **Main path** | Compact flux solve plus limiting. |
| `recon_schem=-1` | **Main path** | First order. |
| `recon_schem=0` | **Main path** | SUW7/SUW5/SUW3 hierarchy. |
| `recon_schem=1` | **Main path** | WENO7/WENO5 hierarchy. |
| `recon_schem=2` | **Main path** | WENO7-Z/WENO5-Z hierarchy. |
| `recon_schem=3` | **Main path** | MP7/MP5 hierarchy. |
| `recon_schem=4` | **Disabled** | Reporting exists, reconstruction calls are commented out. |
| `recon_schem=5` | **Conditional** | MP-LD plus shock sensor. |
| `recon_schem=6` | **Main path** | ROUND. |
| Characteristic decomposition | **Conditional** | Roe-based local transform for first five equations. |
| Explicit diffusion | **Main path** | Active backend fixed to `diff6ec`. |
| Compact diffusion | **Main path** | `643c` only. |
| Explicit 10th-order filter | **Main path** | Used when both spatial suffixes are `e`. |
| Compact filter | **Main path** | Used when a spatial suffix is `c`. |
| NSCBC 2nd-order derivative / 6th-order filter | **Conditional** | Embedded in selected farfield/outflow boundary routines. |
| k-omega `filter2e` | **Conditional** | Reached only after the compact conservative-variable filter path. |
| Sponge relaxation | **Conditional** | Called after every RK stage; acts only where sponge coefficients are active. |
| `lfftk` spectral derivative | **Not connected** | Input/reporting and FFT library exist, but the flow derivative path remains finite difference. |
| Chemistry `odetype` | **Conditional** | `COMB` build: stage-coupled source, implicit Euler, or DNN hybrid. |
| Explicit 2nd/4th/8th derivative utilities | **Library only** | Not selected by the active `fds` explicit object. |
| RK3 | **Main path** | Three-stage third order. |
| RK4 | **Main path** | Four-stage fourth order. |

## 17. Current CPU/GPU Format Correspondence

This section records the project state at the time of this audit. “GPU ported” means a controlled CUDA Fortran path exists and has CPU/GPU validation evidence; it does not imply every boundary, MPI topology, or physics model is supported.

| Format | CPU | GPU | Current GPU scope |
|---|---|---|---|
| `643e` explicit central convection | Yes | **Ported** | Single/multi-rank, supported periodic and physical boundaries. |
| `643e` explicit gradients/diffusion | Yes | **Ported** | Non-reacting `numq=5` path and supported boundaries. |
| Explicit 10th-order filter | Yes | **Ported** | Full-field ping-pong implementation with halo exchange. |
| RK3 | Yes | **Ported** | GPU-resident compute loop. |
| RK4 | Yes | No | GPU capability gate rejects it. |
| `543e`, `recon_schem=-1` | Yes | **Ported** | Phase S0-A1, single-rank periodic forced-3D Sod, no diffusion/filter. |
| `543e`, `recon_schem=1` WENO7 interior | Yes | **Ported** | Phase S0-A2, same controlled single-rank Sod scope. |
| `543e`, `recon_schem=0` SUW | Yes | No | Not yet ported. |
| `543e`, `recon_schem=2` WENO-Z | Yes | No | Not yet ported. |
| `543e`, `recon_schem=3` MP7 interior | Yes | **Ported** | Phase S0-A3, same controlled single-rank periodic Sod scope; physical-boundary MP5 fallback is not ported. |
| `543e`, `recon_schem=5` MP-LD | Yes | No | Requires sensor architecture. |
| `543e`, `recon_schem=6` ROUND | Yes | No | Not yet ported. |
| Characteristic decomposition | Yes | No | Explicitly rejected by current shock gates. |
| Ducros shock sensor | Yes | **Sensor-only ported** | S0-A4 validates the complete single-rank raw field and expanded mask; S0-A5 validates `2x1x1` raw-sensor `hm` halo. Flux consumption is not ported. |
| `643c` compact central | Yes | No | Deliberately outside current GPU scope. |
| `543c` compact upwind | Yes | No | Deliberately outside current GPU scope. |
| Compact filter | Yes | No | Deliberately outside current GPU scope. |
| NSCBC embedded numerics | Yes | No | Open-boundary/NSCBC phase remains future work. |
| Sponge relaxation | Yes | No | Planned only after the explicit shock-format path is validated. |
| Chemistry ODE coupling | Yes, under `COMB` | No | Chemistry migration is currently deferred. |

The mature general GPU format remains:

```text
sixth-order explicit central convection
+ sixth-order explicit central gradients/diffusion
+ tenth-order explicit central filter
+ RK3
```

The controlled S0-A1/S0-A2/S0-A3 paths establish CPU/GPU equivalence for first-order Steger-Warming, periodic WENO7, and periodic MP7. S0-A2/A3 also pass the shared finite-threshold Sod exact-solution oracle. S0-A4 validates the complete single-rank raw sensor and expanded mask independently of the flux path; S0-A5 extends that validation to a `2x1x1` MPI raw-sensor halo. This does not yet establish production SBLI readiness because physical/open boundaries, characteristic reconstruction, and shock-wall coupling remain outside these gates.

## 18. Known Source-Level Caveats

1. `recon_schem=4` is reported but not implemented; selecting it stops.
2. The active explicit derivative object ignores the numeric token and always calls `diff6ec`.
3. The printed `643e` boundary-order label does not match the actual second/second/fourth/sixth explicit closure exactly.
4. `diff8ec` contains a physical-boundary token check against `642`, inconsistent with entry through an 8xx selector; its `ntype=4` branch also applies a halo-backed stencil to a both-physical-faces block.
5. WENO-Z weights should be documented from the implementation, not assumed from another textbook variant.
6. MP5-LD and MP7-LD do not use identical limiter gating: MP7-LD explicitly gates limiting on the shock flag.
7. Compact schemes and compact filters require global line solves and should not be treated as local stencil kernels.
8. `diff4ec` and `diff2ec` do not implement the both-physical-faces `ntype=4` branch.
9. Explicit upwind reconstruction has no `ntype=4` boundary degradation, and its y/z line-range dispatch rejects `ntype=4`.
10. `lfftk` is reported as an FFT option, but no active flow derivative calls the FFT library.
11. Dormant `filter4e` uses an asymmetric `j-2/j+3` stencil and propagates uninitialized temporary-array edges.
12. In the lower-y `farfield_nscbc` metric correction, the third `deriv` call samples `j`, `j+1`, and `j+3`, while the adjacent two components sample `j`, `j+1`, and `j+2`. This is an apparent three-point-stencil indexing error and should be reviewed before that boundary path is used as a GPU oracle.
13. `ducrossensor` clamps lower and upper physical neighbors for `ntype=1` and `ntype=2`, respectively, but has no equivalent both-face handling for `ntype=4`.
14. Presence of species/turbulence/chemistry branches in CPU diffusion does not imply those physics are GPU ported.

## 19. Source Function Index

| Topic | Source function |
|---|---|
| Runtime reporting | `readwrite:infodisp` scheme-reporting block |
| Solver initialization | `comsolver:solvrinit` |
| Central convection | `solver:convrsdcal6` |
| Explicit upwind convection | `solver:convrsduwd` |
| Compact upwind convection | `solver:convrsdcmp` |
| Diffusion RHS | `solver:diffrsdcal6` |
| Explicit central derivative | `derivative:df_explicit`, `derivative:diff6ec` |
| Compact central derivative | `derivative:df_compact`, `derivative:compact_fd_rhs` |
| General derivative utility | `derivative:ddfc_basic` |
| One-sided boundary derivatives | `commfunc:deriv_1o`, `deriv_2o`, `deriv_3o`, `deriv_4o` |
| Steger-Warming split | `riemann:flux_steger_warming` |
| Explicit reconstruction dispatcher | `flux:recons_exp` |
| SUW reconstruction | `flux:suw3`, `flux:suw5`, `flux:suw7` |
| WENO reconstruction | `flux:WENO5`, `flux:WENO7` |
| WENO-Z reconstruction | `flux:WENO5Z`, `flux:WENO7Z` |
| MP reconstruction | `flux:MP5`, `flux:MP7` |
| MP-LD reconstruction | `flux:MP5LD`, `flux:MP7LD` |
| ROUND reconstruction | `flux:round` |
| Compact flux reconstruction | `flux:compact_flux_initiate`, `flux:flux_compact` |
| Shock sensor | `commcal:ducrossensor` |
| Explicit filter | `filter:filter_coefficient_explicit`, `filter:spafilter10exp` |
| NSCBC explicit filter | `filter:spafilter6exp`, `bc:farfield_nscbc`, `bc:outflow_nscbc` |
| Compact filter | `filter:compact_filter_initiate`, `filter:compact_filter` |
| Filter orchestration | `comsolver:filterq`, `comsolver:filterq_explicit10` |
| Sponge relaxation | `sponge_layer:spongefilter_layer`, `spongefilter_global` |
| Time integration | `mainloop:time_integration_rk` |
| Chemistry source/ODE | `solver:srccomb`, `thermchem:imp_euler_ode` |

## 20. Recommended Interpretation

For CPU production cases, scheme support should be claimed at the full-path level:

```text
input token
-> dispatch branch
-> numerical operator
-> boundary closure
-> MPI halo behavior
-> RK/filter ordering
-> case-level validation
```

A coefficient function existing in `src/` is not sufficient evidence of full case support. This distinction is especially important for the explicit 2nd/4th/8th derivative utilities, disabled WENO-SYM selector, compact line solvers, and optional shock-sensor branches.
