#pragma once
#include "device_state.h"
#include "astr_precision.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Private legacy y-wall sample, not the seven-node Cartesian wall diagnostics.
 * geometry is host SoA (nx,nz,nwalls,9): xyz, inward unit normal, +x-projected
 * unit tangent. Static geometry is uploaded once; flow and metrics are borrowed.
 * first_y is 0 or local jm; nwalls may be zero on interior y ranks.
 * Endpoint ownership is next-rank lower -> local upper, before q conversion.
 * Curved walls require the existing periodic x/z tangential halo contract.
 * layers is 3 (legacy closure) or 8 (existing high-order Cartesian closure).
 */
int astr_insitu_wall_create(void *parent,int communicator,const int lower[3],const int upper[3],
    const int sizes[3],const int periodic[3],int mode,int first_y,int nwalls,int layers,int curved,
    const astr_numerical_real material[7],const astr_numerical_real *geometry,void **wall);
int astr_insitu_wall_capture(void *wall);
/* Views are SoA (nx,nz,nwalls,ncomponents), valid through the next capture or
 * destroy. geometry has 9 components; fields has p, shear_x, heat_into_gas, ny. */
int astr_insitu_wall_view(void *wall,int geometry,AstrDeviceView *view);
/* A selected line only: host Fortran (6,nx), xyz then the three physical quantities. */
int astr_insitu_wall_line(void *wall,int k,int side,astr_numerical_real *host,size_t bytes);
/* Explicit component-test/host-consumer observation, never implicit capture. */
int astr_insitu_wall_download(void *wall,astr_numerical_real *host,size_t bytes);
int astr_insitu_wall_air5_span(void *wall,const astr_numerical_real *weights,int owned_x,astr_numerical_real *host);
/* device current/peak, pinned, sent, staging D2H/H2D, scalar D2H,
 * explicit observation D2H, captures; setup geometry H2D is reported separately. */
int astr_insitu_wall_metrics(void *wall,int64_t values[10]);
/* Existing unit-density three-scalar wall-time moments. State is allocated only
 * on first push/restore; the 34-value wire layout is unchanged. The Fortran
 * owner supplies clipped trapezoid weights and owns the window/sample clock. */
int astr_insitu_wall_statistics_push(void *wall,const astr_numerical_real weights[2],int previous);
int astr_insitu_wall_statistics_pack(void *wall,astr_numerical_real *host,size_t bytes,
    const astr_numerical_real window[2],astr_numerical_real time);
int astr_insitu_wall_statistics_restore(void *wall,const astr_numerical_real *host,size_t bytes);
/* Replace only a destination capture's three quantities by wall-time means,
 * preserving its geometry/normal, then complete the existing endpoint owners. */
int astr_insitu_wall_statistics_mean(void *destination,void *statistics,astr_numerical_real duration);
int astr_insitu_wall_destroy(void **wall);
/* Original stationary Cartesian lower-wall SBLI quantities at a physical
 * spanwise plane. Capture two bracketing k layers, complete their x endpoints,
 * apply the original seven-node wall derivative and normalize per layer, then
 * interpolate the three quantities. Only Fortran (4,nx) is downloaded.
 * reference is rho_inf, U_inf, p_inf; material uses the legacy seven constants.
 * This private observer never changes solver q/metrics. */
int astr_insitu_sbli_wall_create(void *parent,int communicator,int lower,int upper,int parts,int mode,
    int first_k,astr_numerical_real fraction,const astr_numerical_real material[7],
    const astr_numerical_real reference[3],const astr_numerical_real *x,void **sample);
int astr_insitu_sbli_wall_capture(void *sample,astr_numerical_real *line,size_t bytes);
/* device, pinned, sent, staging D2H/H2D, status D2H, curve D2H, setup H2D. */
int astr_insitu_sbli_wall_metrics(void *sample,int64_t values[8]);
int astr_insitu_sbli_wall_destroy(void **sample);
/* Existing spanwise-periodic profile-wall diagnostic. Reads the current q/halo
 * and metric views without endpoint averaging. Wall-normal derivatives use
 * three physical layers; shear includes d(v_normal)/dx with the original
 * 2/4/6-order physical-end closures. weights is host SoA (nx,unique_nz),
 * reference is rho_inf,U_inf. Output Fortran (3,nx): span,Cf,heat_into_wall.
 * This normalization is distinct from the physical-midspan SBLI product. */
int astr_insitu_profile_wall_create(void *parent,int offset,int global_x,
    const astr_numerical_real material[7],const astr_numerical_real reference[2],
    const astr_numerical_real *weights,void **sample);
int astr_insitu_profile_wall_capture(void *sample,astr_numerical_real *line,size_t bytes);
/* device bytes, setup-weight H2D, captures, status D2H, curve D2H */
int astr_insitu_profile_wall_metrics(void *sample,int64_t values[5]);
int astr_insitu_profile_wall_destroy(void **sample);
#ifdef __cplusplus
}
#endif
