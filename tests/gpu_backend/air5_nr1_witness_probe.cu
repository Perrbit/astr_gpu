#include <cuda_runtime.h>
#include "chemistry_flux_math.cuh"
#include "air5_nr1_budget_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Fixture order: frozen face input (80), beta, projection input (22), witness (5).
__global__ void verify(const double *data,double *results,int count,bool construct)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count)return;
    using namespace astr::air5;
    const double *h=data+108*id,*l=h+11,*base_l=h+22,*base_r=h+33,*ql=h+44,*qr=h+55;
    const double cdt=h[66],beta=h[80],*x=h+81,*flux=h+103;
    double *out=results+10*id;
    out[4]=0.;
    double trial[11],gas[5],lower[5],upper[5],c[5],f[5],g[5],ev[5],energy[5],delta[5],left[11],right[11];
    double sf=fabs(x[15]),sw=0.;
    for(int s=0;s<5;++s) {
        gas[s]=gas_constant(s);
        const double rl=128.*DBL_EPSILON*(fabs(base_l[5+s])+6.*cdt*(2.*fabs(l[5+s])+fabs(h[5+s])));
        const double rr=128.*DBL_EPSILON*(fabs(base_r[5+s])+6.*cdt*(2.*fabs(l[5+s])+fabs(h[5+s])));
        lower[s]=l[5+s]-fmax(0.,base_r[5+s]-rr)/(6.*cdt);
        upper[s]=l[5+s]+fmax(0.,base_l[5+s]-rl)/(6.*cdt);
    }
    // All rows in this trace are shared interior faces.
    if(h[67]!=1. || h[68]!=1.)return;
    for(int j=0;j<11;++j)trial[j]=l[j]+beta*(h[j]-l[j]);
    for(int s=0;s<5;++s) {
        if(h[5+s]==0. && l[5+s]==0. && base_l[5+s]==0. && base_r[5+s]==0. && ql[5+s]==0. && qr[5+s]==0.)
            lower[s]=upper[s]=0.;
        if(trial[5+s]!=x[s] || lower[s]!=x[5+s] || upper[s]!=x[10+s] || gas[s]!=x[16+s])return;
        sf=fmax(sf,fmax(fabs(x[s]),fmax(fabs(lower[s]),fabs(upper[s]))));
        sw=fmax(sw,fabs(gas[s]));
    }
    if(trial[0]!=x[15] || dot5(gas,trial+5)!=x[21])return;
    if(construct) {
        const int status=nr1_budget::project(x,lower,upper,x[15],gas,x[21],out+5);
        if(status!=ok) {out[4]=-status;return;}
        flux=out+5;
    }
    for(int s=0;s<5;++s)if(!astr::air5::finite(flux[s]) || flux[s]<lower[s] || flux[s]>upper[s])return;
    const double mass=x[15]/sf,moment=(x[21]/sw)/sf,k=64.*DBL_EPSILON;
    double mass_abs=0.,moment_abs=0.;
    for(int s=0;s<5;++s) {
        c[s]=x[s]/sf;f[s]=flux[s]/sf;g[s]=gas[s]/sw;
        mass_abs+=fabs(f[s]);moment_abs+=fabs(g[s]*f[s]);
    }
    const double t=k*fmax(fmax(fabs(mass),fabs(moment)),abs_sum5(c));
    out[0]=fabs(sum5(f)-mass)/(t+k*mass_abs);
    out[1]=fabs(dot5(g,f)-moment)/(t+k*moment_abs);
    double density,u[3],temp,y[5],tv,p;
    if(conservative_to_primitive(ql,false,density,u,temp,y,tv,p))return;
    double tf=.5*temp,tvf=.5*tv,uf[3];
    for(int d=0;d<3;++d)uf[d]=.5*u[d];
    if(conservative_to_primitive(qr,false,density,u,temp,y,tv,p))return;
    tf+=.5*temp;tvf+=.5*tv;
    for(int d=0;d<3;++d)uf[d]+=.5*u[d];
    const double kinetic=.5*(uf[0]*uf[0]+uf[1]*uf[1]+uf[2]*uf[2]);
    for(int s=0;s<5;++s) {
        ev[s]=species_ev(s,tvf);energy[s]=data::formation_energy[s]+cv_tr(s)*tf+ev[s]+kinetic;
        delta[s]=flux[s]-trial[5+s];trial[5+s]=flux[s];
    }
    trial[10]+=dot5(ev,delta);trial[4]+=dot5(energy,delta);
    for(int j=0;j<11;++j) {left[j]=base_l[j]-6.*cdt*(trial[j]-l[j]);right[j]=base_r[j]+6.*cdt*(trial[j]-l[j]);}
    out[2]=state_admissible(left,h+70);out[3]=state_admissible(right,h+70);
    out[4]=1.;
}

void check(cudaError_t error)
{
    if(error!=cudaSuccess) {std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);}
}

int main(int argc,char **argv)
{
    const bool construct=argc==3 && std::strcmp(argv[2],"--construct")==0;
    if(argc!=2 && !construct)return 2;
    FILE *file=std::fopen(argv[1],"rb");if(!file)return 2;
    std::vector<double> input;
    double row[108];
    size_t n;
    while((n=std::fread(row,sizeof(double),108,file))==108)input.insert(input.end(),row,row+108);
    if(n!=0 || std::ferror(file) || std::fclose(file)!=0 || input.empty())return 2;
    const int count=input.size()/108;
    std::vector<double> results(10*count);
    double *di,*dr;
    check(cudaMalloc(&di,input.size()*8));check(cudaMalloc(&dr,results.size()*8));
    check(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    verify<<<(count+31)/32,32>>>(di,dr,count,construct);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(results.data(),dr,results.size()*8,cudaMemcpyDeviceToHost));
    check(cudaFree(dr));check(cudaFree(di));
    int status=0;
    for(int j=0;j<count;++j) {
        const double *r=results.data()+10*j;
        if(r[4]<0.) {std::fprintf(stderr,"candidate construction failed: row %d status=%d\n",j,int(-r[4]));return 1;}
        if(r[4]!=1.) {std::fprintf(stderr,"invalid fixture/candidate: row %d\n",j);return 1;}
        std::printf("AIR5_WITNESS %.17g %.17g %d %d\n",r[0],r[1],int(r[2]),int(r[3]));
        if(construct)std::printf("AIR5_CONSTRUCTED %.17g %.17g %.17g %.17g %.17g\n",r[5],r[6],r[7],r[8],r[9]);
        if(!(r[0]<=1. && r[1]<=1.)) {
            std::fprintf(stderr,"witness rejected: row %d\n",j);status=1;
        }
    }
    return status;
}
