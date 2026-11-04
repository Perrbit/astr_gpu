# Temporal Products

This is the TB4 local implementation, not a production or statistical-stationarity
claim. Current evidence and remaining gates are in
`documents/ASTR_INSITU_TURBULENT_BOUNDARY_LAYER_PRODUCTS_PROGRESS.md`.

## Session And Catalog

Use the existing `input.output` product catalog. The current temporal history
provider requires an enabled `catalog_stats` session with `statistics=t`. This
selects compact requested-node statistics, not an implicit full-volume mean
array. Numerical output works without Catalyst. Images require the existing
Catalyst device-rendering build and `render=t`, with the configured processing,
derivative and rendering backend kept consistent.

Examples for a 32-cubed TGV grid:

- `input.products.tgv.temporal.example.jsonl`: numerical line history and probe PSD.
- `input.products.tgv.temporal.images.example.jsonl`: line time-space images and
  probe PSD images, JPEG only. Fixed color/axis ranges are example settings,
  not universal physical thresholds.
- `input.products.tgv.temporal.mean.example.jsonl`: a Reynolds fluctuation line
  and a Favre mean-velocity line, numerical CSV. Set the requested common mean
  window in `input.insitu`; no product implicitly changes that window.
- `input.products.flatplate.temporal.example.jsonl`: streamwise and spanwise
  line histories at fixed global `j=1` on the 64-by-48-by-16 local fixture.
  For a different grid, explicitly choose the actual indices. `j=1` is not a
  universal wall-unit height and must not be labelled as constant local y-plus.

Current real-field evidence is FP64, nonreacting Cartesian on the existing small
flat-plate/SBLI fixture: instantaneous velocity/pressure, and numerical line
histories of Reynolds velocity/pressure/temperature means and fluctuations plus
Favre mean/fluctuating velocity. The mean reference is the accumulated local
mean up to the sample time, not the unknown future full-window mean. Its window
freezes at the requested end time; fluctuation values continue to use the actual
instantaneous field minus that frozen mean. These numerical tests do not qualify
mean/fluctuation PSD or their chart presentation.
No new AIR5, CURVE or FP32 qualification is implied.

## Sampling And Storage

| Setting | Meaning |
| --- | --- |
| `kind=time_space` | Keep the complete enabled line history in time blocks. |
| `sampling.mode=grid_index` | One fixed plane index and two index ranges; exactly one range varies, so this is a grid line, not a downloaded plane. |
| `kind=probe_spectrum` | A grid-node probe with streaming Welch PSD. |
| `sampling.mode=grid_node` | Explicit global `[i,j,k]`; shared MPI nodes have exactly one owner. |
| `temporal.sample_steps` | Independent complete-step sampling stride, default 1. |
| `temporal.block_samples` | Samples per line-history block. For probes, required only when raw-history archiving is explicitly enabled. |
| `temporal.segment_length` | Probe Welch segment length: explicit, even, at least two; not restricted to powers of two. |
| `temporal.archive_samples` | Probe full raw-history archiving; defaults to false. A time-space line always archives its enabled history. |
| `schedule` | Controls PSD CSV/image publication and product clocks, not the independent sampling stride or full-block publication. |

Every sample has its actual physical timestamp. Fixed-step Welch uses periodic
Hann, segment demeaning, 50% overlap and one-sided density scaling. The native
Singleton FFT is unitary; the PSD includes the corresponding length factor.
Incomplete final segments are neither shortened nor zero-padded. CSV metadata
reports estimate availability, valid segments and frequency-grid generation.
An unavailable estimate has no fabricated PSD rows.

Welch uses only actual complete-step samples whose physical times satisfy
`statistics_window(1) <= time <= statistics_window(2)` in the common session.
No boundary sample is interpolated, snapped or added. The product's integer-step
sampling stride remains independent of its publication cadence. Raw probe
archiving, when explicitly enabled, retains actual samples over the enabled
history; time-space lines likewise keep their enabled history. These raw
histories are not trimmed to the frequency-analysis window. Outside the window
the spectrum is frozen, but its existing estimate can still be published.

Spectrum metadata separates the raw `sampled_endpoints` from the window-admitted
`welch_sampled_endpoints`, with the first/last admitted step/time and requested
window. Admitted inputs include an incomplete segment; valid segment count and
availability determine whether a PSD exists. Variable-step inputs still do not
produce a Welch estimate. Raw endpoints outside the window are never counted as
frequency-analysis inputs.
Select fixed stepping explicitly through `input.run`'s `&timestep mode='fixed'`
and a case-appropriate `fixed_dt`; publishing a PSD product does not change the
solver's default automatic timestep mode or its stability control.

Automatic time stepping produces no first-version Welch estimate. Actual times
and values remain in the current bounded probe block and latest sample; full
long-term probe history still requires explicit `archive_samples=true`.
No interpolation, resampling or average sampling-frequency substitution occurs.
CPU/GPU automatic-step 12 versus 5+7 continuation now has real runtime evidence:
timestamps genuinely change spacing, state/history restoration is exact, and no
Welch rows or valid segments are fabricated.

Sampling is from the complete state. GPU transfer is the requested one-dimensional
field packet; coordinates use the existing geometry and one-dimensional
reduction. No new volume/final-geometry readback is introduced. The current
image consumer draws these reduced histories with the existing host Matplotlib
path inside Catalyst; it is not a new GPU rasterizer or a claim of zero host
analysis.

## Reading Blocks

Numerical products use `outdat/.../temporal_products/<id>/version*/segment*/`.
For image-channel temporal products, numeric blocks and PSD CSV are colocated
under the corresponding `render_products` segment.

- `block.g*.n*.complete.csv`: a complete time block.
- `block.g*.n*.partial.step*.csv`: an immutable partial snapshot at normal finish
  or a sampling/storage-definition transition.
- `spectrum.step*.csv`: an estimate/status publication at the product cadence.
- `frame.step*.jpeg` and receipt JSON: requested images.

Partial snapshots do not clear the checkpoint's current block. Consequently a
prefix's final partial snapshot can overlap the resumed run's complete block.
Do not blindly concatenate every CSV. For one known continuation lineage, use:

```bash
python scripts/insitu/temporal_archive.py /path/to/segments/block.*.csv
```

The reader coalesces matching product/history/sampling/block identities, checks
the overlapping values for exact equality and keeps the longest block. It
rejects conflicting snapshots or duplicated samples across committed blocks.
Different sampling definitions remain separate groups. It does not silently
merge unrelated trajectories.

Blocks and spectra include field/time/frequency units, PSD units, a fixed
reference state and first/last actual sample identities. Publication writes a
`.writing` file, closes it, then publishes the final name without overwriting an
existing result. A failed product is disabled, not automatically retried at
finalize. Other products and the solver continue. Explicitly increasing that
product's `enable_request` requests another activation; it is not a silent retry.

Per-row columns are `step,time,gap_before,node,x,y,z,<fields>`. The row's time is
not a step-number axis. Images split the mesh at true gap markers; an isolated
time row is drawn only as points, not assigned an invented time width. Spatial
coordinates can use the approved fixed origin/reference-length display mapping.
No background grid or lighting modifies the scalar colors.

Mean/fluctuation blocks additionally carry `mean_window_start`, `mean_window_end`
and `mean_measure` on every node row. They describe the mean used for that sample,
not the entire block or the image cadence. A positive covered duration is
required before the first mean-dependent sample. Only these products allocate
the three-real coverage array per block slot. Coverage is checkpointed with the
block, including a partially filled block and its last-sample identity; current
time-block subrecords are `ASTRTB02`. No additional sidecar is introduced.

Matching active consumers can supply already acquired Welch buffers and sums to
a new or temporarily absent consumer. Matching includes the integer-step phase,
not just the stride. No unavailable old raw archive is reconstructed. During a
long absence, bounded raw blocks may be dropped while matching shared sampling
keeps the spectrum continuous; `gap_before` then describes a visible archive gap,
not necessarily a missing sample in Welch. Without a continuing sampler, a true
gap drops incomplete Welch segments instead of bridging the gap. Changing only
`block_samples` preserves accumulated spectra; changing segment length starts a
new sampling identity.

Inactive mean histories do not absorb another product's changing mean packets:
matching fields alone do not establish identical accumulator coverage. An active
same-step donor must also match the recorded coverage before a mean history is
reused. Broader dynamic mean-history reuse needs its own accumulator-identity
evidence; instantaneous shared-sampling tests are not evidence for that case.

## Checkpoints And Validation Boundary

Temporal state is appended to the existing selected-statistics word payload in
`statistics.h5`, not a new sidecar file. The existing `checkpoint_statistics`
switch applies. Exact continuation retains line blocks, current probe values,
Welch overlap, incomplete segments and accumulated PSD. The checkpoint timestep
is not sampled twice. A fresh history's generation is at least its effective
step plus one; exact restoration retains the saved generation.

The current temporal word collection is `ASTRTH03`, including the window-admitted
sample counters and first/last identities; no compatibility reader was
added for the earlier experimental TB4 payload. Non-temporal statistics keep
their existing payload. The CPU/GPU topology matrix, both device image pipelines,
dynamic gaps/shared sampling, late activation, live block/segment changes,
variable stepping, injected publication failure and explicit reactivation now
have local evidence. `ASTR_INSITU_TIMING=1` reports per-product sample/publication/
archive timings, transfers and retained-array bytes; these inclusive phases must
not be added as disjoint costs. Diagnostics-off equivalence is tested. The local
off/data/images cost pair has no field/checkpoint HDF5, no artificial resource
limit, and no production-performance claim.

The first local TB4 scope now has implementation and runtime evidence. The
common Welch window was approved on 2026-11-04; `window_and_gaps.xml` checks actual
boundary selection, pre/in/post-window restoration and real/shared gaps, while
`window_integration.xml` covers the current topology and continuation paths.
Earlier activation-continuous spectrum results are not evidence for this rule.
A checkpoint without statistics starts
an actual new history at the restored state; a CPU/GPU fixed-step representative
test confirms no old Welch segment is invented. Numerical mean/fluctuation line
coverage and its exact restoration now have CPU/GPU evidence; their image/PSD
and broader dynamic-identity combinations are not covered by those tests.
See the progress document for evidence
paths and the unresolved choices; do not infer qualification from parsing.
