#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Geometry is supplied by the existing Fortran grid/metric initializer.
 * These operations do not rebuild geometry or infer dimensional scaling. */
int astr_device_air5_geometry_prepare(void *handle);
int astr_device_air5_rk_prepare(void *handle);
int astr_device_air5_rk_update(void *handle,const int first[3],const int last[3],
    double dt,double a,double b,double c,int first_stage);
/* Sensor-only preparation on a fresh AIR5 context; no diffusion fields. */
int astr_device_air5_velocity_gradients_prepare(void *handle);
/* Physical-node gradients and fluxes; halo exchange remains the caller's
 * explicit next phase. Only layered diffusion allocates the six energy channels.
 * low/high identify true physical ends, not MPI interfaces. */
int astr_device_air5_transport_prepare(void *handle,int layered);
int astr_device_air5_gradients(void *handle,const int low[3],const int high[3],int zero_species_flux_lower_y);
/* Failure is read through astr_device_air5_check; failed nodes do not commit. */
int astr_device_air5_diffusive_flux(void *handle);
typedef struct AstrAir5DiffusionReport {
    int claimed,location[4],limited;
    uint64_t minimum_bits;
    double values[5];
} AstrAir5DiffusionReport;
int astr_device_air5_diffusion_prepare(void *handle);
int astr_device_air5_diffusion_begin(void *handle);
/* Reuse the existing inclusive RK boxes (interior and optional top plane).
 * energy=1 is the second pass of layered diffusion, after species ratio halo.
 * Begin clears the budget status once; the caller releases it across ranks
 * before exchanging ratios or applying any divergence. */
int astr_device_air5_diffusion_budget(void *handle,const int ntypes[3],const int first[3],const int last[3],
    const double floor[5],double a,double b,double cdt,int first_stage,int energy,int diagnostics);
int astr_device_air5_diffusion_rhs(void *handle,const int ntypes[3],const int first[3],const int last[3],int axis);
int astr_device_air5_diffusion_report(void *handle,AstrAir5DiffusionReport *report);
/* The non-symmetric limiters share the scalar ratio workspace with diffusion.
 * Each pass includes all owned nodes; the caller releases status collectively
 * before exchanging a ratio or committing the limited RHS. */
typedef struct AstrAir5ConvectionReport {
    int claimed,location[4],active[10]; /* total, four constraints, five species */
    uint64_t minimum_bits[10];
    double values[5];
} AstrAir5ConvectionReport;
int astr_device_air5_raw_convection(void *handle,const int ntypes[3],const int first[3],const int last[3],
    int capturing,int symmetric);
int astr_device_air5_convection_budget_prepare(void *handle,int consistent);
int astr_device_air5_convection_budget_begin(void *handle,int phase);
int astr_device_air5_convection_budget(void *handle,const int ntypes[3],const int first[3],const int last[3],
    const double floor[5],double a,double b,double cdt,int first_stage,int capturing,int species_budget,int phase,int collect);
int astr_device_air5_convection_rhs(void *handle,const int ntypes[3],const int first[3],const int last[3],
    int capturing,int phase);
int astr_device_air5_convection_read_report(void *handle,AstrAir5ConvectionReport *report);
/* Reusable per-axis 11-component candidate storage, with component 12 reserved
 * for the later limiter. No budget, divergence or MPI exchange is applied.
 * kind: 0 selective centered/MP7, 1 LLF, 2 diffusion, 3 species-carried energy.
 * axis is 1..3; ntype is the original 1 lower/2 upper/3 neither/4 both closure.
 * The inclusive box spans -1..upper along axis, physical transverse nodes only.
 * Input face halos (including metrics/fluxes) must already be current. */
int astr_device_air5_faces_prepare(void *handle);
int astr_device_air5_reconstruct_faces(void *handle,int axis,int ntype,int kind,int shock_capturing,int symmetric,
    const int first[3],const int last[3]);
/* Original first failing face diagnostic: claimed/i/j/k/axis/code/top,
 * six eleven-component vectors, and c*dt. Check the scalar status separately. */
typedef struct AstrAir5TransportFailure {
    int failure[7];
    double values[66];
    double cdt;
} AstrAir5TransportFailure;
/* The original host-computed vibrational floor values are copied once.
 * with_top allocates an optional (i,0,k,component) high-normal RHS plane. */
int astr_device_air5_symmetric_prepare(void *handle,const double floor[5],int with_top);
/* Base clears this phase's status/diagnostic and budget, computes the original
 * low-order RK budget, and clears active RHS only outside top-only mode.
 * The caller exchanges the base halo before owned faces are evaluated. */
int astr_device_air5_symmetric_base(void *handle,const int ntypes[3],double a,double b,double cdt,int first_stage,int top_only);
int astr_device_air5_symmetric_top_normal(void *handle,const int ntypes[3],double cdt,int preserve_all);
int astr_device_air5_symmetric_faces(void *handle,const int ntypes[3],int axis,int capturing,double cdt,int top_only,int preserve_all);
/* Pack order: lower-face0, upper-face(dim-1); receive order: plus, minus.
 * Unpack writes face dim and -1 respectively, including coefficient12.
 * The explicit local copy is only for a one-rank periodic direction. */
int astr_device_air5_symmetric_plane(void *handle,int axis,int unpack,int receive_plus,int receive_minus);
int astr_device_air5_symmetric_buffers(void *handle,void *buffers[4],size_t *capacity);
int astr_device_air5_symmetric_copy_local_planes(void *handle,int axis);
/* axis is 1..3. Use the prepared halo topology/backend, but exchange only
 * the two owned face planes, including coefficient12, without averaging.
 * Pinned mode reuses fixed host slots; device mode passes device pointers.
 * Call on the MPI thread after owned faces and before symmetric_apply. */
int astr_device_air5_symmetric_exchange(void *handle,int axis);
/* Existing ASTR_AIR5_FLUX_HALO choices. They select packet organization,
 * independently of the prepared pinned/device MPI backend. */
enum AstrAir5FluxHaloMode { ASTR_AIR5_FLUX_SEPARATE=0,ASTR_AIR5_FLUX_PACKED=1,ASTR_AIR5_FLUX_PACKED_OVERLAP=2 };
int astr_device_air5_flux_transport_prepare(void *handle,int mode);
/* All physical-transverse hm face layers, no shared-node averaging.
 * The optional six carried-energy channels follow their allocated field. */
int astr_device_air5_flux_exchange(void *handle);
/* Original per-axis compensation component, axis=1..3. normalize=0 exchanges
 * q/carry face packets and updates carry only. The caller then performs the
 * q shared average, followed by normalize=1 (TwoSum on exchanged faces).
 * Reuses prepared transport slots; pinned/device choices remain explicit.
 * This does not implement the complete solution-halo controller. */
int astr_device_air5_compensation_axis(void *handle,int axis,int normalize);
int astr_device_air5_symmetric_apply(void *handle,const int ntypes[3],int axis,double cdt,int top_only);
int astr_device_air5_symmetric_final(void *handle,const int ntypes[3],double a,double b,double cdt,int first_stage,int top_only);
int astr_device_air5_symmetric_failure(void *handle,AstrAir5TransportFailure *failure);
/* Original conservative filter: caller must have established its uniform
 * Cartesian 643e/643e geometry contract. global/offset count mesh intervals.
 * scalar selects 1 versus11 workspace components, not another filter formula. */
int astr_device_air5_filter_prepare(void *handle,const double tr_floor[5],const double v_floor[5],
    const int global[3],const int offset[3],const int periodic[3],int scalar);
int astr_device_air5_filter_buffers(void *handle,void *buffers[8],size_t *strip_capacity,size_t *area);
int astr_device_air5_filter_pack(void *handle,int axis);
int astr_device_air5_filter_faces(void *handle,int axis,int has_upper);
int astr_device_air5_filter_commit(void *handle,int axis,int has_lower,int has_upper);
int astr_device_air5_filter_alias(void *handle,int axis,int unpack);
int astr_device_air5_filter_check(void *handle,int preserve_all);
/* MPI thread only. phase0 strips, phase1 owned face, phase2 high-end alias.
 * apply performs the original per-axis sequence and globally releases physics
 * status before commit and after model checking. It does not roll back axes.
 * q/carry and RHS scratch change, but primitive/transport caches do not: the
 * caller must rebuild spatial halo data and caches from this q/carry before
 * subsequent spatial work. */
int astr_device_air5_filter_exchange(void *handle,int axis,int phase);
int astr_device_air5_filter_apply(void *handle,int preserve_all,int *physics_status);
/* Legacy explicit filter only: save rho and species into RHS before the high
 * field stencil; run repair after any high/carry normalization. Check the
 * device physics status collectively before using the result. No halo exchange
 * or cache refresh here. floor is the original host vibrational floor. */
int astr_device_air5_legacy_filter_save(void *handle);
int astr_device_air5_legacy_filter_repair(void *handle,const double floor[5]);
/* Ordinary endpoint rows. Axis is zero-based; component zero filters all eleven. */
int astr_device_air5_legacy_linear_prepare(void *handle,int scalar);
int astr_device_air5_legacy_linear_axis(void *handle,int axis,int component,int lower,int upper);
int astr_device_air5_legacy_linear_wall(void *handle,int axis,int component,int lower,int upper,int family);
int astr_device_air5_legacy_linear_commit(void *handle,int component);
int astr_device_air5_legacy_linear_y_ghosts(void *handle,int component);
int astr_device_air5_legacy_filter_compensation(void *handle,int swap);
/* wall_axis is zero-based, or -1 when ordinary endpoint closure is in use. */
int astr_device_air5_legacy_filter_apply(void *handle,const double floor[5],int wall_axis,int family,int *physics_status);
#ifdef __cplusplus
}
#endif
