#include "device_state_internal.cuh"
#include "device_halo.h"

namespace {
__global__ __launch_bounds__(astr::block_threads)
void periodic_kernel(AstrDeviceView field, int axis, int upper, int halo,
                     int transverse1, int transverse2, int n1, int n2, size_t count, int reconcile)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (id < count) {
        int p[3] = {0,0,0};
        p[transverse1] = id%n1;
        p[transverse2] = (id/n1)%n2;
        const int c = (id/(static_cast<size_t>(n1)*n2))%field.extent[3]+1;
        const int d = id/(static_cast<size_t>(n1)*n2*field.extent[3])+1;
        const size_t first = astr::offset(field,p[0],p[1],p[2],c,d);
        const size_t stride = field.stride[axis];
        double *v = static_cast<double *>(field.data);
        if (reconcile) {
            const double mean = .5*(v[first]+v[first+upper*stride]);
            v[first] = mean;
            v[first+upper*stride] = mean;
        }
        for (int h = 1; h <= halo; ++h) {
            v[first-h*stride] = v[first+(upper-h)*stride];
            v[first+(upper+h)*stride] = v[first+h*stride];
        }
    }
    return;
}
}

extern "C" int astr_device_periodic_axis(void *handle, int field, int axis, int reconcile)
{
    if (!handle || field < 0 || field >= ASTR_DEVICE_FIELD_COUNT || axis < 0 || axis > 2)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s = *static_cast<astr::State *>(handle);
    const AstrDeviceView view = s.fields[field].view;
    if (!view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if (view.lower[axis] != -s.halo || s.upper[axis] < s.halo || s.halo == 0)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const int a = (axis+1)%3, b = (axis+2)%3, n1 = s.upper[a]+1, n2 = s.upper[b]+1;
    const size_t count = static_cast<size_t>(n1)*n2*view.extent[3]*view.extent[4];
    const int threads = astr::block_threads;
    periodic_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(
        view,axis,s.upper[axis],s.halo,a,b,n1,n2,count,reconcile);
    return astr::finish_launch(s);
}
