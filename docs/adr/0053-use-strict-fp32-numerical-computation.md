# Use strict FP32 numerical computation

On 2026-10-20 the user initially selected native FP32 advancement with explicit
FP64 auxiliary exceptions. After reviewing the full-volume geometry and
statistics storage costs, the user changed question D1 to option B. This
supersedes the initial exception-list choice before implementation.

The proposed mode uses FP32 for numerical state and arithmetic, including
geometry, physical time, CFL, statistics and auxiliary solves. It must not
retain an FP64 numerical initialization, state mirror or hidden fallback.
Performance clocks may retain their FP64 interfaces and elapsed-time values,
as explicitly approved by the user. Those values must not feed physical time,
CFL, statistical weights or flow accumulation. Third-party interfaces and file
conversions do not authorize FP64 numerical advancement. Case coverage and
acceptance remain open in
`documents/ASTR_NATIVE_FP32_DESIGN.md`.

For this delivery, retain the existing time and statistical accumulation
algorithms while adapting their types and interfaces. Preserve existing stable
weighted-mean/central-moment recurrences, but do not add compensation arrays or
new blocked accumulation. If time stagnates or statistical coverage or
scheduling becomes incorrect, report the demonstrated range and propose a
subsequent revision rather than silently promoting it to FP64 or shortening
acceptance windows to obtain a pass.

The user subsequently limited this delivery to non-reacting five-variable
flow. AIR5 two-temperature reacting flow is a separate future stage, not a
completion requirement or a capability inferred from non-reacting tests.
Specific case, scheme and boundary combinations remain to be selected.
The user included both already supported explicit and compact format families
in this delivery, validated separately in that order. This does not add
previously unported numerical schemes or reuse FP64 line-solver tolerances.
Check the reduction/back-substitution algorithm and report FP32 residuals.
Representative numerical paths are fixed to TGV 643e/643e and 643c/643c, and
Mach 12 plate 743e/643e and 743e/643c, each with filtering off and on. Use the
existing paired explicit or compact filter path, without adding independent
derivative/filter selectors. Compare each precision against its matched
discretization, not explicit fields against compact fields. This does not
adopt the earlier performance campaign's grid sizes or run lengths as native
FP32 acceptance settings.
The delivery also includes bounded validation of the existing non-reacting
Mach 12 plate with MP7-LD convection and NSCBC boundaries, after low-Mach TGV
validation. Use already supported derivative/filter combinations and examine
internal-energy cancellation, boundaries and limiter behavior; this does not
authorize changing the Mach number or boundary targets to obtain a pass, nor
require production-scale or statistically stationary runs. Existing positivity
limiting and the lcracon crashfix switch may be retained with matched FP32/FP64
configurations. Report repair activity and conservative-state changes separately
from precision errors; do not impose an arbitrary repair-count ceiling or treat
a repaired run completing as numerical acceptance. Repair sequences need not be
identical across precisions. This does not approve stronger clipping, new floors
or changes to the energy equation. Detailed acceptance remains to be selected.
Representative static curvilinear validation is included through existing
explicit 643e/643e paths: periodic curved TGV and the y-wavy bc41 wall fixture.
Check FP32 geometry, compatible free-stream preservation, physical gradients
and corresponding in-situ products. This does not extend the compact solver
to general CURVE or establish arbitrary curved-boundary coverage. Existing
FP64 capabilities remain unchanged.

The user also selected compile-time precision through the root CMake build:
one shared source implementation and precision definition, separate FP32/FP64
build directories and executables, and the existing runtime CPU/GPU choice.
This does not require hand-maintained duplicate solvers, global real-conversion
compiler flags, or recompilation each time an already built mode is selected.
Existing mixed-workspace candidates remain available in the FP64 build;
the native FP32 build uses its uniform numerical path without stacking those
candidates or allocating duplicate same-precision workspaces. Their historical
evidence is preserved but does not establish native FP32 acceptance.

Numerical payloads in checkpoints, volume sequences and slices follow the
build precision: FP32 for the native FP32 executable, FP64 for the FP64
executable. The files identify both advancement precision and payload type.
This does not downsize integer identities, imply that all files halve in size,
or authorize an FP64 whole-field output mirror.

The user approved cross-precision restore in both directions as explicitly
non-exact continuation using exact_restart=false. Convert the required state
at I/O boundaries and rebuild derived caches in the destination precision;
do not retain a source-precision flow mirror. Exact continuation remains
limited to matched precision and execution conditions. This does not reopen
retired checkpoint formats or simultaneously permit a change of physical
model or discretization.

On a precision change, start a new online statistical segment rather than
inheriting the source segment's accumulated means and moments. Preserve source
segments and provide explicit offline weighted merging after the run, including
the between-segment mean correction for second moments. Record actual time
coverage and precision provenance; do not silently bridge missing observations,
label a mixed-precision window pure FP32, or promise bitwise equivalence with
one-pass accumulation. Same-precision exact continuation still restores its
complete statistics. The user selected FP64 arithmetic and statistical output
for the separate offline merge tool, with blockwise processing. This does not
change online accumulation precision, recover lost source bits, or make the
merged window a uniformly FP64-advanced calculation.

The user included adaptation of existing non-reacting in-situ products in this
delivery: slices, wall quantities, Q surfaces, instantaneous/mean streamlines
and related mean-field products. Validate numerical advancement before adding
products one at a time. This does not add new products or require an exhaustive
product-by-case Cartesian test matrix. Numerical postprocessing, including
gradients, Q, wall quantities, geometry extraction, interpolation and streamline
integration, follows the build precision. Pure graphical rendering internals
such as camera, projection and lighting may retain the library's own types;
numerical analysis is not exempt merely because a third-party library executes
it. Preserve existing FP32 display buffers without adding a full FP64 numerical
field or geometry mirror. Existing FP64 numerical products and their display
conversion remain unchanged; define separate FP32 numerical and display
checks rather than reusing the FP64 field-difference tolerance.

The user rejected fixed FP32-versus-FP64 deviation limits. Delivery is based on
matched algorithms and correct implementation, not a 1% or 0.1% difference cap.
Preserve the equations, discrete coefficients, boundary/filter rules, RK
ordering, communication ownership and limiter/crashfix logic while adapting
types and interfaces. Report field, residual, time-accumulation and diagnostic
differences without a cross-precision pass/fail ceiling. Retain finite/positive
state checks, data-transfer identity and same-precision exact restart checks;
do not treat matching source algorithms as proof of physical accuracy or
long-time statistical reliability. Existing FP64 acceptance is unchanged.

The delivery closes on the local two-GPU system, not A800. After bounded
CPU/GPU and product checks, compare FP32/FP64 on 256-cubed TGV and the
450-by-130-by-96 interval Mach 12 plate at GPU NP=1/2. Report RK timing,
memory and communication volume without a minimum speedup requirement;
static CURVE remains bounded to representative small cases. Engineering
interfaces, phase steps and the proposed test matrix are recorded in the
design for whole-document review, not yet authorized execution.

This is a design decision, not authorization to modify solver code or replace
the existing FP64 production default. The earlier FP64-state/mixed-workspace
design remains the contract of that existing mode.
