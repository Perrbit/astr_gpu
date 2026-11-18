#include <cuda_runtime.h>
#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_coupling.h"
#include "chemistry_ros2_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int im=7,jm=5,km=3,halo=5;
constexpr int cells=(im+1)*(jm+1)*(km+1);
constexpr double duration=5.e-11;
int node(int i,int j,int k) {return i+(im+1)*(j+(jm+1)*k);}
double density_scale(int id) {return 1.+double(id)/1024.;}
double history_seed(int id) {return 1.e-15*(1.+.01*id);}
void check(int status) {if(status) {std::fprintf(stderr,"device status %d\n",status);std::exit(1);}}
size_t count(const AstrDeviceView &v) {return v.stride[4]*v.extent[4];}
size_t index(const AstrDeviceView &v,int i,int j,int k,int c=1)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3];}
bool same(const std::vector<double> &a,const std::vector<double> &b)
{return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;}
struct Fields {
    void *handle=nullptr;
    AstrDeviceView q{},carry{},history{},rho{},vel{},t{},tv{},p{},y{};
    AstrAir5ChemistryOptions options{};
};
void prepare(Fields &f,const double input[17],AstrAir5ChemistryOptions options,int debug)
{
    f.options=options;check(astr_device_state_create(0,im,jm,km,halo,debug,&f.handle));
    check(astr_device_air5_prepare(f.handle));check(astr_device_air5_chemistry_prepare(f.handle,&options));
    const int fields[]={ASTR_DEVICE_Q,ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_TEMPERATURE,
        ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_PRESSURE,ASTR_DEVICE_AIR5_SPECIES};
    AstrDeviceView *views[]={&f.q,&f.rho,&f.vel,&f.t,&f.tv,&f.p,&f.y};
    for(int a=0;a<7;++a) {
        check(astr_device_state_view(f.handle,fields[a],views[a]));
        std::vector<double> values(count(*views[a]),-777.);
        if(a==0) for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i)
            for(int c=1;c<=11;++c) values[index(f.q,i,j,k,c)]=input[c-1]*density_scale(node(i,j,k));
        check(astr_device_state_upload(f.handle,fields[a],values.data(),values.size()*sizeof(double)));
    }
    int status=astr_device_state_view(f.handle,ASTR_DEVICE_AIR5_CARRY,&f.carry);
    if(options.compensated) {
        check(status);std::vector<double> values(count(f.carry),0.);
        for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i)
            for(int c=1;c<=11;++c) values[index(f.carry,i,j,k,c)]=c<=5?double(c)*1.e-20*(node(i,j,k)+1):input[11+c-6]*density_scale(node(i,j,k));
        check(astr_device_state_upload(f.handle,ASTR_DEVICE_AIR5_CARRY,values.data(),values.size()*sizeof(double)));
    } else if(status!=ASTR_DEVICE_NOT_ALLOCATED) std::exit(2);
    status=astr_device_state_view(f.handle,ASTR_DEVICE_AIR5_HISTORY,&f.history);
    if(options.history_enabled) {
        check(status);std::vector<double> values(count(f.history));
        for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i)
            values[index(f.history,i,j,k)]=history_seed(node(i,j,k));
        check(astr_device_state_upload(f.handle,ASTR_DEVICE_AIR5_HISTORY,values.data(),values.size()*sizeof(double)));
    } else if(status!=ASTR_DEVICE_NOT_ALLOCATED) std::exit(2);
}
std::vector<double> download(const Fields &f,int field,const AstrDeviceView &view)
{
    std::vector<double> result(count(view));
    check(astr_device_state_download(f.handle,field,result.data(),result.size()*sizeof(double)));return result;
}
void upload(const Fields &f,int field,const std::vector<double> &values)
{check(astr_device_state_upload(f.handle,field,values.data(),values.size()*sizeof(double)));}
bool active(int i,int j,int k)
{return i>=1 && i<=im-1 && j>=1 && j<=jm-1 && k>=1 && k<=km-1 ||
        i>=1 && i<=im && j==jm && k>=1 && k<=km-1;}
AstrAir5ChemistryReport regions(Fields &f,int collect)
{
    const int first[]={1,1,1},last[]={im-1,jm-1,km-1},top_first[]={1,jm,1},top_last[]={im,jm,km-1};
    check(astr_device_air5_chemistry_begin(f.handle));
    check(astr_device_air5_chemistry_region(f.handle,first,last,duration,collect));
    check(astr_device_air5_chemistry_region(f.handle,top_first,top_last,duration,collect));
    int status;check(astr_device_air5_chemistry_status(f.handle,&status));
    AstrAir5ChemistryReport report;check(astr_device_air5_chemistry_report(f.handle,&report));
    if(status || report.status || report.total!=62) std::exit(3);
    return report;
}
__global__ void point_reference(const double *input,AstrAir5ChemistryOptions options,double *out)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=cells) return;
    input+=18*id;out+=32*id;
    double final_state[6],carry[6],atol[6],history=input[17],suggested;
    for(int c=0;c<6;++c) {carry[c]=input[11+c];atol[c]=options.atol_factor*(c<5?input[0]:fmax(fabs(input[10]),1.));}
    astr::air5::Ros2Counts counts;bool skipped;
    const int status=astr::air5::ros2_advance(input[0],input+1,input[4],input+5,duration,duration,
        options.rtol,atol,200000,final_state,suggested,counts,options.source_mode,options.source_only!=0,
        options.compensated?carry:nullptr,options.history_enabled?&history:nullptr,&options.activity_threshold,&skipped);
    out[30]=status;
    for(int c=0;c<11;++c) out[c]=c<5?input[c]:final_state[c-5];
    for(int c=0;c<6;++c) out[11+c]=carry[c];out[17]=history;
    int decode=astr::air5::conservative_to_primitive(out,options.preserve_all!=0,out[18],out+19,out[22],out+23,out[28],out[29]);
    out[31]=decode;
}
}

int main(int argc,char **argv)
{
    if(argc!=2) return 2;
    FILE *file=std::fopen(argv[1],"r");if(!file) return 2;
    int n,mode;double span,input[17];
    if(std::fscanf(file,"%d%lf%d",&n,&span,&mode)!=3 || n!=1) return 2;
    for(double &v:input) if(std::fscanf(file,"%lf",&v)!=1) return 2;
    std::fclose(file);
    std::vector<double> nodal(18*cells);
    for(int i=0;i<cells;++i) {
        for(int c=0;c<17;++c) nodal[18*i+c]=input[c]*density_scale(i);
        nodal[18*i+17]=history_seed(i);
    }
    double *id,*answer;check(cudaMalloc(&id,nodal.size()*sizeof(double)));check(cudaMalloc(&answer,32*cells*sizeof(double)));
    check(cudaMemcpy(id,nodal.data(),nodal.size()*sizeof(double),cudaMemcpyHostToDevice));
    std::vector<double> primary_q,primary_c,primary_h;
    for(int test=0;test<8;++test) {
        AstrAir5ChemistryOptions options{1,1,0,1,1,32,test==0?0:1,1.e-9,1.e-13,0.};
        const int collect=test!=2;
        if(test==3) {options.compensated=0;options.history_enabled=0;options.source_only=0;options.block_size=128;options.preserve_all=0;}
        if(test==4) options.compensated=0;
        if(test==5) {options.history_enabled=0;options.source_mode=2;}
        if(test==6) options.source_mode=1;
        if(test==7) {options.source_mode=3;options.activity_threshold=1.e-4;}
        Fields f;prepare(f,input,options,test==0?1:0);
        const auto before_q=download(f,ASTR_DEVICE_Q,f.q);
        const auto before_c=options.compensated?download(f,ASTR_DEVICE_AIR5_CARRY,f.carry):std::vector<double>{};
        const auto before_h=options.history_enabled?download(f,ASTR_DEVICE_AIR5_HISTORY,f.history):std::vector<double>{};
        point_reference<<<(cells+31)/32,32>>>(id,options,answer);check(cudaGetLastError());
        std::vector<double> reference(32*cells);check(cudaMemcpy(reference.data(),answer,reference.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for(int i=0;i<cells;++i) if(reference[32*i+30]!=0 || reference[32*i+31]!=0) return 4;
        const auto report=regions(f,collect);
        const auto q=download(f,ASTR_DEVICE_Q,f.q);
        const auto c=options.compensated?download(f,ASTR_DEVICE_AIR5_CARRY,f.carry):std::vector<double>{};
        const auto h=options.history_enabled?download(f,ASTR_DEVICE_AIR5_HISTORY,f.history):std::vector<double>{};
        for(int k=-halo;k<=km+halo;++k) for(int j=-halo;j<=jm+halo;++j) for(int i=-halo;i<=im+halo;++i)
            for(int v=1;v<=11;++v) {
                const size_t a=index(f.q,i,j,k,v);
                const double expected=active(i,j,k)?reference[32*node(i,j,k)+v-1]:before_q[a];
                if(std::memcmp(&q[a],&expected,sizeof(double))) return 5;
            }
        for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i) {
            if(options.compensated) for(int v=1;v<=11;++v) {
                const size_t a=index(f.carry,i,j,k,v);
                const double expected=active(i,j,k) && v>=6?reference[32*node(i,j,k)+11+v-6]:before_c[a];
                if(std::memcmp(&c[a],&expected,sizeof(double))) return 6;
            }
            if(options.history_enabled) {
                const size_t a=index(f.history,i,j,k);const double expected=active(i,j,k)?reference[32*node(i,j,k)+17]:before_h[a];
                if(std::memcmp(&h[a],&expected,sizeof(double))) return 7;
            }
        }
        const int fields[]={ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_TEMPERATURE,
                            ASTR_DEVICE_AIR5_SPECIES,ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_PRESSURE};
        const AstrDeviceView views[]={f.rho,f.vel,f.t,f.y,f.tv,f.p};const int starts[]={18,19,22,23,28,29};
        for(int a=0;a<6;++a) {
            const auto values=download(f,fields[a],views[a]);
            for(int k=-halo;k<=km+halo;++k) for(int j=-halo;j<=jm+halo;++j) for(int i=-halo;i<=im+halo;++i)
                for(int v=1;v<=views[a].extent[3];++v) {
                    const double expected=active(i,j,k)?reference[32*node(i,j,k)+starts[a]+v-1]:-777.;
                    if(values[index(views[a],i,j,k,v)]!=expected) return 8;
                }
        }
        if(!collect) for(int v:report.max_counts) if(v!=0) return 9;
        if(collect) {
            double mins[3]={DBL_MAX,DBL_MAX,DBL_MAX},maxs[2]={0,0};
            for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i) if(active(i,j,k)) {
                const double *r=reference.data()+32*node(i,j,k);
                double s=r[5];for(int c=6;c<=9;++c) s=fmin(s,r[c]);
                mins[0]=fmin(mins[0],s);mins[1]=fmin(mins[1],r[22]);mins[2]=fmin(mins[2],r[28]);
                maxs[0]=fmax(maxs[0],r[22]);maxs[1]=fmax(maxs[1],r[28]);
            }
            if(std::memcmp(mins,report.minimum_bits,sizeof(mins)) || std::memcmp(maxs,report.maximum_bits,sizeof(maxs))) return 16;
        }
        if(test==7 && (report.skipped!=62 || report.max_counts[2]!=1 || report.max_counts[3]!=0)) return 15;
        if(test==0) {primary_q=q;primary_c=c;primary_h=h;}
        if(test==1 || test==2) if(!same(q,primary_q)||!same(c,primary_c)||!same(h,primary_h)) return 10;
        if(test==0) {
            regions(f,1);const auto continued=download(f,ASTR_DEVICE_Q,f.q);
            const auto continued_c=download(f,ASTR_DEVICE_AIR5_CARRY,f.carry),continued_h=download(f,ASTR_DEVICE_AIR5_HISTORY,f.history);
            Fields restored;prepare(restored,input,options,0);upload(restored,ASTR_DEVICE_Q,q);
            upload(restored,ASTR_DEVICE_AIR5_CARRY,c);upload(restored,ASTR_DEVICE_AIR5_HISTORY,h);regions(restored,1);
            if(!same(continued,download(restored,ASTR_DEVICE_Q,restored.q)) ||
               !same(continued_c,download(restored,ASTR_DEVICE_AIR5_CARRY,restored.carry)) ||
               !same(continued_h,download(restored,ASTR_DEVICE_AIR5_HISTORY,restored.history))) return 11;
            check(astr_device_state_destroy(&restored.handle));
            auto bad=q;bad[index(f.q,1,1,1)]=-1.;upload(f,ASTR_DEVICE_Q,bad);
            const auto bad_c=download(f,ASTR_DEVICE_AIR5_CARRY,f.carry),bad_h=download(f,ASTR_DEVICE_AIR5_HISTORY,f.history);
            check(astr_device_air5_chemistry_begin(f.handle));const int location[]={1,1,1};
            check(astr_device_air5_chemistry_region(f.handle,location,location,duration,1));
            const int tf[]={1,jm,1},tl[]={im,jm,km-1};check(astr_device_air5_chemistry_region(f.handle,tf,tl,duration,1));
            int status;check(astr_device_air5_chemistry_status(f.handle,&status));
            AstrAir5ChemistryReport failure;check(astr_device_air5_chemistry_report(f.handle,&failure));
            // The original half-step forms species atol from rho: a negative
            // rho reaches ROS-2's invalid-tolerance check before state validation.
            if(status!=10 || failure.status!=10 || failure.failure[0]!=1 || failure.failure[1]!=1 || failure.failure[2]!=1 ||
               failure.failure[3]!=1 || failure.failure[4]!=10 || failure.failure[5]!=1 || failure.total!=15) {
                std::fprintf(stderr,"failure status=%d report_status=%d location/status/phase=%d,%d,%d,%d,%d,%d total=%lld\n",
                    status,failure.status,failure.failure[0],failure.failure[1],failure.failure[2],failure.failure[3],
                    failure.failure[4],failure.failure[5],static_cast<long long>(failure.total));
                return 12;
            }
            const auto failed=download(f,ASTR_DEVICE_Q,f.q),fc=download(f,ASTR_DEVICE_AIR5_CARRY,f.carry),fh=download(f,ASTR_DEVICE_AIR5_HISTORY,f.history);
            for(int v=1;v<=11;++v) if(failed[index(f.q,1,1,1,v)]!=bad[index(f.q,1,1,1,v)] ||
                fc[index(f.carry,1,1,1,v)]!=bad_c[index(f.carry,1,1,1,v)]) return 13;
            if(fh[index(f.history,1,1,1)]!=bad_h[index(f.history,1,1,1)]) return 14;
        }
        check(astr_device_state_destroy(&f.handle));std::printf("AIR5_CONTEXT test=%d active=62 field_mapping_exact=1\n",test);
    }
    check(cudaFree(id));check(cudaFree(answer));
    return 0;
}
