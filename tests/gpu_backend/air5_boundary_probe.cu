#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_coupling.h"
#include "chemistry_boundary.h"
#include "chemistry_boundary_math.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void check(int status) {if(status){std::fprintf(stderr,"device operation %d\n",status);std::exit(1);}}
__global__ void evaluate(const double *x,double *out,int *status,int n)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;x+=32*i;out+=11*i;
    switch(static_cast<int>(x[31])) {
    case 0:status[i]=astr::air5::boundary::pressure_outlet(x,x[22],x[23]!=0,out);break;
    case 1:status[i]=astr::air5::boundary::isothermal_wall(x,x+11,x[22],x[23]!=0,out);break;
    case 2:status[i]=astr::air5::boundary::hbl_outflow(x,x+11,x[23]!=0,out);break;
    case 3:status[i]=astr::air5::boundary::similarity_farfield(x,x[24],x[22],out);break;
    }
}
size_t size(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t offset(AstrDeviceView v,int i,int j,int k,int c)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+c*v.stride[3];}
void fields(const std::vector<double> &input,const std::vector<int> &reference_status,const std::vector<double> &reference)
{
    int covered[2][2]{};
    for(int row=0;row<reference_status.size();++row) {
        const double *x=input.data()+32*row;const int kind=static_cast<int>(x[31]),failed=reference_status[row]!=0;
        if(covered[kind][failed])continue;covered[kind][failed]=1;
        void *handle=nullptr;check(astr_device_state_create(0,8,7,6,5,0,&handle));check(astr_device_air5_prepare(handle));
        const AstrAir5ChemistryOptions options{1,0,0,1,1,32,0,1e-9,1e-13,0.};check(astr_device_air5_chemistry_prepare(handle,&options));
        AstrDeviceView q,carry;check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CARRY,&carry));
        std::vector<double> initial(size(q),-777.),low(size(carry),1e-30),actual(size(q)),after_low(size(carry));
        for(int k=0;k<=6;++k)for(int j=0;j<=7;++j)for(int i=0;i<=8;++i)for(int c=0;c<11;++c) {
            if(kind==0 && i==7)initial[offset(q,i,j,k,c)]=x[c];
            if(kind==1 && (j==1 || j==2))initial[offset(q,i,j,k,c)]=x[(j-1)*11+c];
        }
        check(astr_device_state_upload(handle,ASTR_DEVICE_Q,initial.data(),initial.size()*8));
        check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_CARRY,low.data(),low.size()*8));
        check(kind==0?astr_device_air5_pressure_outlet_upper_x(handle,x[22],x[23]!=0):astr_device_air5_wall_lower_y(handle,x[22],x[23]!=0));
        int status;check(astr_device_air5_check(handle,&status));if(status!=reference_status[row])std::exit(4);
        check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual.data(),actual.size()*8));
        check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CARRY,after_low.data(),after_low.size()*8));
        if(std::memcmp(low.data(),after_low.data(),low.size()*8))std::exit(5);
        for(int k=-5;k<=11;++k)for(int j=-5;j<=12;++j)for(int i=-5;i<=13;++i)for(int c=0;c<11;++c) {
            const size_t at=offset(q,i,j,k,c);
            const bool target=kind==0?i>=8 && j>=0 && j<=7 && k>=0 && k<=6:j<=0 && i>=0 && i<=8 && k>=0 && k<=6;
            if(!failed && target) {
                const double expected=reference[11*row+c],tolerance=1e-13*fmax(1.,fabs(expected))+5e-12*fabs(expected);
                if(fabs(actual[at]-expected)>tolerance)std::exit(6);
            } else if(std::memcmp(&actual[at],&initial[at],8))std::exit(7);
        }
        check(astr_device_state_destroy(&handle));
    }
    for(int k=0;k<2;++k)for(int f=0;f<2;++f)if(!covered[k][f])std::exit(8);
    std::puts("AIR5_BOUNDARY_FIELDS_PASS wall/outlet successful ghosts and failed words; carry unchanged");
}
void hbl_fields(const std::vector<double> &input,const std::vector<int> &rs,const std::vector<double> &rv)
{
    int calls=0,resets=0;
    for(int preserve=0;preserve<2;++preserve)for(int debug=0;debug<2;++debug) {
        int selected[5]={-1,-1,-1,-1,-1};
        for(int row=0;row<rs.size();++row) {
            const double *x=input.data()+32*row;if(static_cast<int>(x[23])!=preserve)continue;
            if(x[31]==2) {
                double high[11];for(int c=0;c<11;++c)high[c]=(4.*x[c]-x[11+c])/3.;
                if(rs[row]==0 && selected[0]<0 && std::memcmp(high,rv.data()+11*row,88)==0)selected[0]=row;
                if(rs[row]==0 && selected[1]<0 && high[0]<0.)selected[1]=row;
                if(rs[row]!=0 && selected[2]<0)selected[2]=row;
            } else if(x[31]==3) {
                if(rs[row]==0 && x[22]>0. && selected[3]<0)selected[3]=row;
                if(rs[row]!=0 && selected[4]<0)selected[4]=row;
            }
        }
        for(int row:selected)if(row<0)std::exit(10);
        const double *base=input.data()+32*selected[0];
        void *handle=nullptr;check(astr_device_state_create(0,8,7,6,5,debug,&handle));check(astr_device_air5_prepare(handle));
        const AstrAir5ChemistryOptions options{1,0,0,1,1,32,0,1e-9,1e-13,0.};check(astr_device_air5_chemistry_prepare(handle,&options));
        AstrDeviceView q,carry;check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CARRY,&carry));
        std::vector<double> initial(size(q),-777.),low(size(carry)),after_low(size(carry)),actual(size(q)),expected(size(q));
        for(size_t a=0;a<low.size();++a)low[a]=1e-30*(a+1);
        for(int k=0;k<=6;++k)for(int j=0;j<=7;++j)for(int i=0;i<=8;++i)for(int c=0;c<11;++c)
            initial[offset(q,i,j,k,c)]=base[c]*(1.+(i+2*j+3*k)/1024.);
        auto upload=[&](const std::vector<double> &before) {
            check(astr_device_state_upload(handle,ASTR_DEVICE_Q,before.data(),before.size()*8));
            check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_CARRY,low.data(),low.size()*8));
        };
        auto verify=[&](int physics,const std::vector<double> &before,int kind,int row,bool dynamic,bool top) {
            int status;check(astr_device_air5_check(handle,&status));if(status!=physics)std::exit(11);
            check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual.data(),actual.size()*8));
            check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CARRY,after_low.data(),after_low.size()*8));
            if(std::memcmp(low.data(),after_low.data(),low.size()*8))std::exit(12);
            for(int k=-5;k<=11;++k)for(int j=-5;j<=12;++j)for(int i=-5;i<=13;++i)for(int c=0;c<11;++c) {
                const size_t at=offset(q,i,j,k,c);
                bool numerical=kind==2?i>=8 && j>=0 && j<=7 && k>=0 && k<=6 && !(dynamic && top && j==7):
                    kind==3 && i>=0 && i<=8 && j>=7 && k>=0 && k<=6;
                numerical=numerical && physics==0;
                if(numerical) {
                    double scale=1.;for(int d=0;d<11;++d)scale=fmax(scale,fabs(rv[11*row+d]));
                    const double ref=rv[11*row+c],tolerance=1e-13*scale+5e-12*fabs(ref);
                    if(!std::isfinite(actual[at]) || fabs(actual[at]-ref)>tolerance)std::exit(13);
                } else if(std::memcmp(&actual[at],&expected[at],8))std::exit(14);
            }
            ++calls;
        };
        for(int which=0;which<3;++which)for(int dynamic=0;dynamic<2;++dynamic)for(int top=0;top<2;++top) {
            const int row=selected[which];const double *x=input.data()+32*row;auto before=initial;
            for(int k=0;k<=6;++k)for(int j=0;j<=7;++j)for(int c=0;c<11;++c) {
                before[offset(q,7,j,k,c)]=x[c];before[offset(q,6,j,k,c)]=x[11+c];
            }
            expected=before;
            if(dynamic && top)for(int k=0;k<=6;++k)for(int c=0;c<11;++c)for(int i=9;i<=13;++i)
                expected[offset(q,i,7,k,c)]=before[offset(q,8,7,k,c)];
            upload(before);check(astr_device_air5_hbl_outflow_upper_x(handle,top,dynamic,preserve));
            verify(rs[row],before,2,row,dynamic,top);
        }
        std::vector<double> x_top(9*7);double incident[22];for(int c=0;c<22;++c)incident[c]=base[c%11];
        for(int which=3;which<5;++which) {
            const int row=selected[which];const double *x=input.data()+32*row;
            for(double &v:x_top)v=x[22];
            check(astr_device_air5_top_boundary_prepare(handle,x,incident,x[24],.5,0,x_top.data(),x_top.size()));
            upload(initial);expected=initial;check(astr_device_air5_farfield_upper_y(handle,0,preserve));
            verify(rs[row],initial,3,row,false,true);
        }
        // Constant target selection includes the exact incident location; no shock physics is asserted.
        for(int c=0;c<11;++c) {incident[c]=base[c]*.875;incident[11+c]=base[c]*1.125;}
        const std::vector<double> saved_incident(incident,incident+22);
        for(int k=0;k<=6;++k)for(int i=0;i<=8;++i)x_top[i+9*k]=i*.125;
        check(astr_device_air5_top_boundary_prepare(handle,base,incident,1.,.5,1,x_top.data(),x_top.size()));
        for(double &v:x_top)v=NAN;for(double &v:incident)v=NAN;
        expected=initial;for(int k=0;k<=6;++k)for(int i=0;i<=8;++i)for(int j=7;j<=12;++j)for(int c=0;c<11;++c)
            expected[offset(q,i,j,k,c)]=saved_incident[(i<4?0:11)+c];
        upload(initial);check(astr_device_air5_farfield_upper_y(handle,0,preserve));verify(0,initial,4,0,false,true);
        for(int bad=0;bad<2;++bad) {
            auto before=initial;if(bad)before[offset(q,4,7,3,0)]=-1.;expected=before;
            for(int k=0;k<=6;++k)for(int i=0;i<=8;++i)if(!bad || i!=4 || k!=3)
                for(int j=8;j<=12;++j)for(int c=0;c<11;++c)expected[offset(q,i,j,k,c)]=before[offset(q,i,7,k,c)];
            upload(before);check(astr_device_air5_farfield_upper_y(handle,1,preserve));verify(bad?2:0,before,4,0,true,true);
        }
        for(int mask=0;mask<16;++mask)for(int dynamic=0;dynamic<2;++dynamic) {
            const int lower[3]={mask&1,(mask>>2)&1,1},upper[3]={(mask>>1)&1,(mask>>3)&1,1};
            upload(initial);check(astr_device_air5_hbl_boundary_carry_reset(handle,lower,upper,dynamic));
            check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual.data(),actual.size()*8));
            if(std::memcmp(actual.data(),initial.data(),actual.size()*8))std::exit(15);
            check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CARRY,after_low.data(),after_low.size()*8));
            for(int k=0;k<=6;++k)for(int j=0;j<=7;++j)for(int i=0;i<=8;++i)for(int c=0;c<11;++c) {
                const size_t at=offset(carry,i,j,k,c);
                const bool fixed=(lower[0] && i==0)||(upper[0] && i==8 && !(dynamic && upper[1] && j==7))||
                    (lower[1] && j==0)||(upper[1] && j==7 && !dynamic);
                const double ref=fixed?0.:low[at];if(std::memcmp(&ref,&after_low[at],8))std::exit(16);
            }
            ++resets;
        }
        check(astr_device_state_destroy(&handle));
    }
    std::printf("AIR5_HBL_FIELDS_PASS calls=%d carry_masks=%d dynamic_top/incident/similarity/outflow; untouched words exact\n",calls,resets);
}
int main(int argc,char **argv)
{
    if(argc!=3 && argc!=4)return 2;FILE *f=std::fopen(argv[1],"rb");if(!f)return 2;int n;
    if(std::fread(&n,4,1,f)!=1)return 2;std::vector<double> input(32*n),output(11*n);std::vector<int> status(n);
    if(std::fread(input.data(),8,input.size(),f)!=input.size())return 2;std::fclose(f);
    double *x,*out;int *s;check(cudaMalloc(&x,input.size()*8));check(cudaMalloc(&out,output.size()*8));check(cudaMalloc(&s,n*4));
    check(cudaMemcpy(x,input.data(),input.size()*8,cudaMemcpyHostToDevice));evaluate<<<(n+31)/32,32>>>(x,out,s,n);
    check(cudaGetLastError());check(cudaDeviceSynchronize());check(cudaMemcpy(output.data(),out,output.size()*8,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(status.data(),s,n*4,cudaMemcpyDeviceToHost));check(cudaFree(x));check(cudaFree(out));check(cudaFree(s));
    f=std::fopen(argv[2],"wb");if(!f)return 2;
    bool written=std::fwrite(&n,4,1,f)==1 && std::fwrite(status.data(),4,n,f)==static_cast<size_t>(n) && std::fwrite(output.data(),8,output.size(),f)==output.size();
    if(std::fclose(f)!=0 || !written)return 2;
    if(argc==4) {
        f=std::fopen(argv[3],"rb");if(!f)return 2;int nr;if(std::fread(&nr,4,1,f)!=1 || nr!=n)return 2;
        std::vector<int> rs(n);std::vector<double> rv(11*n);
        if(std::fread(rs.data(),4,n,f)!=n || std::fread(rv.data(),8,rv.size(),f)!=rv.size())return 2;std::fclose(f);
        if(input[31]>=2)hbl_fields(input,rs,rv);else fields(input,rs,rv);
    }
    return 0;
}
