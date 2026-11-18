#pragma once
#include <cuda_runtime.h>
#include "device_state.h"
#include "astr_precision.h"

struct AstrAir5DiffusionReport;
struct AstrAir5ConvectionReport;
namespace astr {
constexpr int block_threads = 512;
struct Buffer {
    AstrDeviceView view{};
    size_t bytes = 0;
};
struct HaloTransport;
struct KeepPlan;
int release_keep(KeepPlan *plan);
struct SpongePlan;
int release_sponge(SpongePlan *plan);
struct ProductionStatisticsPlan;
int release_production_statistics(ProductionStatisticsPlan *plan);
struct CrashfixPlan;
int release_crashfix(CrashfixPlan *plan);
struct MpPositivityPlan;
int release_mp_positivity(MpPositivityPlan *plan);
struct Me6Plan;
int release_me6(Me6Plan *plan);
struct IndexPlan;
int release_insitu_index(IndexPlan *plan);
struct Air5ChemistryPlan;
int release_air5_chemistry(Air5ChemistryPlan *plan);
struct Air5ConvectionPlan;
int release_air5_convection(Air5ConvectionPlan *plan);
struct Air5FilterPlan;
int release_air5_filter(Air5FilterPlan *plan);
struct Air5BoundaryPlan;
int release_air5_boundary(Air5BoundaryPlan *plan);
struct Air5MonitorPlan;
int release_air5_monitor(Air5MonitorPlan *plan);
int release_transport(HaloTransport *transport);
bool transport_in_flight(const HaloTransport *transport);
struct State {
    cudaStream_t stream = nullptr;
    Buffer fields[ASTR_DEVICE_FIELD_COUNT];
    Buffer nscbc_outlet_plane;
    Buffer nscbc_top_target;
    Buffer nscbc_top_plane;
    Buffer nscbc_top_prediff;
    Buffer lower_wall_velocity;
    Buffer inflow_slices;
    CrashfixPlan *crashfix=nullptr;
    MpPositivityPlan *mp_positivity=nullptr;
    int8_t *critical_mask=nullptr;
    bool nscbc_top_prediff_ready = false;
    numerical_real *nscbc_top_mach = nullptr;
    int *nscbc_incident_status = nullptr;
    int8_t *slip_lower_mask = nullptr;
    int upper[3]{};
    int halo = 0;
    bool debug_sync = false;
    bool fluid_ready = false;
    bool air5_ready = false;
    int *air5_failure = nullptr;
    AstrAir5DiffusionReport *air5_diffusion_report = nullptr;
    AstrAir5ConvectionReport *air5_convection_report = nullptr;
    int *air5_mean_failure = nullptr;
    bool gradients_ready = false;
    bool rhs_ready = false;
    bool viscous_ready = false;
    int upwind_storage = 0; // 0 build kind, 1 MP1 split rounding, 2 MP3 final physical-flux store.
    int compact_plans = 0;
    int sample_plans = 0;
    int moment_plans = 0;
    bool compact_sync = false;
    int *fluid_failure = nullptr;
    int8_t *shock_mask = nullptr;
    int8_t *shock_faces = nullptr;
    bool shock_faces_ready = false;
    numerical_real *cfl_partial = nullptr;
    numerical_real *cfl_values = nullptr;
    numerical_real *statistics_partial = nullptr;
    numerical_real *statistics_values = nullptr;
    void *cfl_records = nullptr;
    numerical_real *cfl_diagnostic_values = nullptr;
    int64_t *cfl_diagnostic_keys = nullptr;
    numerical_real *cfl_profile = nullptr;
    HaloTransport *transport = nullptr;
    KeepPlan *keep = nullptr;
    SpongePlan *sponge = nullptr;
    ProductionStatisticsPlan *production = nullptr;
    Me6Plan *me6 = nullptr;
    IndexPlan *index = nullptr;
    Air5ChemistryPlan *air5_chemistry = nullptr;
    Air5ConvectionPlan *air5_convection = nullptr;
    Air5FilterPlan *air5_filter = nullptr;
    Air5BoundaryPlan *air5_boundary = nullptr;
    Air5MonitorPlan *air5_monitor = nullptr;
};
int me6_inner_bounds(const State &state,int first[3],int last[3]);
int allocate_field(Buffer &buffer, const int upper[3], int halo, int rank,
                   int components = 1, int directions = 1,int element_bytes = sizeof(numerical_real));
inline int finish_launch(State &state)
{
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    return state.debug_sync ? static_cast<int>(cudaStreamSynchronize(state.stream)) : 0;
}
__device__ inline size_t offset(const AstrDeviceView &view, int i, int j, int k,
                               int c = 1, int d = 1)
{
    return static_cast<size_t>(i - view.lower[0]) * view.stride[0]
         + static_cast<size_t>(j - view.lower[1]) * view.stride[1]
         + static_cast<size_t>(k - view.lower[2]) * view.stride[2]
         + static_cast<size_t>(c - view.lower[3]) * view.stride[3]
         + static_cast<size_t>(d - view.lower[4]) * view.stride[4];
}
__device__ inline numerical_real &at(const AstrDeviceView &view, int i, int j, int k,
                            int c = 1, int d = 1)
{
    return static_cast<numerical_real *>(view.data)[offset(view, i, j, k, c, d)];
}
__device__ inline numerical_real load_value(const AstrDeviceView &view,int i,int j,int k,int c=1,int d=1)
{
    const size_t index=offset(view,i,j,k,c,d);
    return view.element_bytes==sizeof(float)?static_cast<numerical_real>(static_cast<const float *>(view.data)[index])
        :static_cast<numerical_real>(static_cast<const double *>(view.data)[index]);
}
__device__ inline void store_value(const AstrDeviceView &view,int i,int j,int k,numerical_real value,int c=1,int d=1)
{
    const size_t index=offset(view,i,j,k,c,d);
    if(view.element_bytes==sizeof(float)) static_cast<float *>(view.data)[index]=static_cast<float>(value);
    else static_cast<double *>(view.data)[index]=value;
}
}
