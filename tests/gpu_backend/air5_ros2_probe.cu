#include <cuda_runtime.h>
#include "chemistry_ros2_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void transaction_cases(const double *input,int *records)
{
    const int id=threadIdx.x;
    if(id>24) return;
    using namespace astr::air5;
    if(id==24) {
        double high=1.,carry=0.;const double quarter_ulp=DBL_EPSILON*.25;
        compensated_add(high,carry,quarter_ulp);
        bool add=high==1. && carry==-quarter_ulp;
        for(int i=0;i<3;++i) compensated_add(high,carry,quarter_ulp);
        records[192]=add && high==1.+DBL_EPSILON && carry==0.;
        const double tiny=__longlong_as_double(1LL);
        high=1.;carry=0.;compensated_add(high,carry,tiny);
        records[193]=high==1. && __double_as_longlong(carry)==__double_as_longlong(-tiny);
        high=1.;carry=0.;compensated_rk(high,carry,1.,0.,.75,.25,tiny);
        records[194]=high==1. && __double_as_longlong(carry)==__double_as_longlong(-tiny);
        return;
    }
    const int test=id%12;const bool source_only=id/12!=0;
    double state[6],seed[6],carry[6],atol[6],final_state[6];
    for(int v=0;v<6;++v) {
        state[v]=input[5+v];seed[v]=.125*DBL_EPSILON*state[v];
        atol[v]=1.e-13*(v<5?input[0]:fmax(fabs(state[5]),1.));
    }
    seed[1]=-seed[1];seed[2]=-0.;
    const double nan=__longlong_as_double(0x7ff8000000000025LL);
    if(test==6) seed[2]=nan;
    if(test==7) state[2]=nan;
    for(int v=0;v<6;++v) carry[v]=seed[v];
    double duration=2.e-8,first=duration,history=duration,threshold=0,suggested;
    int attempts=200000,mode=coupled;
    if(test==0) { mode=frozen;history=duration*.125; }
    if(test==1) duration=-1.;
    if(test==2) attempts=0;
    if(test==3) attempts=1;
    if(test==4) { attempts=1;history=1.e-15; }
    if(test==5) threshold=1.e250;
    if(test==8) history=nan;
    if(test==9) mode=4;
    if(test==10) atol[0]=0;
    if(test==11) duration=nan;
    const double old_history=history;
    Ros2Counts counts;bool skipped;
    const int status=ros2_advance(input[0],input+1,input[4],state,duration,first,1.e-9,atol,attempts,
        final_state,suggested,counts,mode,source_only,carry,&history,&threshold,&skipped);
    bool exact=true;
    for(int v=0;v<6;++v) exact=exact && __double_as_longlong(final_state[v])==__double_as_longlong(state[v])
        && __double_as_longlong(carry[v])==__double_as_longlong(seed[v]);
    exact=exact && (test==0?history==duration:__double_as_longlong(history)==__double_as_longlong(old_history));
    const int expected[12]={ok,invalid_timestep,step_limit,step_limit,step_limit,ok,nonfinite,
        nonfinite,invalid_timestep,invalid_source_mode,invalid_tolerance,invalid_timestep};
    bool correct=status==expected[test] && exact;
    if(test==3) correct=correct && counts.rejected>0;
    if(test==4) correct=correct && counts.accepted>0;
    if(test==5) correct=correct && skipped && counts.rhs==1 && counts.jacobian==0;
    if(test==0) correct=correct && counts.accepted==0 && counts.rejected==0 && counts.rhs==0 && counts.jacobian==0;
    int *row=records+8*id;
    row[0]=status;row[1]=expected[test];row[2]=correct;row[3]=counts.accepted;
    row[4]=counts.rejected;row[5]=counts.rhs;row[6]=counts.jacobian;row[7]=skipped;
}

__global__ void advance_batch(int n,const double *input,double *output,int *counts,double duration,
    int mode,double rtol,double atol_factor,int segments,bool history_enabled,bool source_only,double threshold)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    using namespace astr::air5;
    const double *x=input+18*i;double *out=output+14*i;int *total=counts+5*i;
    double state[6],carry[6],history=x[17],suggested=0;
    for(int v=0;v<6;++v) { state[v]=x[5+v];carry[v]=x[11+v]; }
    for(int v=0;v<5;++v) total[v]=0;
    int skipped_calls=0;
    const double interval=duration/double(segments);
    for(int segment=0;segment<segments;++segment) {
        double atol[6],candidate[6];
        for(int s=0;s<5;++s) atol[s]=atol_factor*x[0];
        atol[5]=atol_factor*fmax(fabs(state[5]),1.);
        Ros2Counts local;bool skipped;
        total[0]=ros2_advance(x[0],x+1,x[4],state,interval,interval,rtol,atol,200000,
            candidate,suggested,local,mode,source_only,carry,history_enabled?&history:nullptr,&threshold,&skipped);
        total[1]+=local.accepted;total[2]+=local.rejected;total[3]+=local.rhs;total[4]+=local.jacobian;
        for(int v=0;v<6;++v) state[v]=candidate[v];
        if(skipped) ++skipped_calls;
        if(total[0]!=0) break;
    }
    for(int v=0;v<6;++v) { out[v]=state[v];out[6+v]=carry[v]; }
    out[12]=suggested;out[13]=double(skipped_calls);
}
static void check(cudaError_t status)
{
    if(status!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1); }
}
static double number(const char *text)
{
    char *end;const double value=std::strtod(text,&end);
    if(end==text || *end!='\0') { std::fprintf(stderr,"invalid real argument: %s\n",text);std::exit(2); }
    return value;
}
static int contracts(const char *input_path,const char *output_path)
{
    FILE *f=std::fopen(input_path,"r");if(!f) return 2;
    int n,mode;double duration;
    if(std::fscanf(f,"%d%lf%d",&n,&duration,&mode)!=3 || n!=1) return 2;
    std::vector<double> x(18,0.);
    for(int i=0;i<17;++i) if(std::fscanf(f,"%lf",&x[i])!=1) return 2;
    std::fclose(f);
    double *xd;int *rd;std::vector<int> results(24*8+3);
    check(cudaMalloc(&xd,18*sizeof(double)));check(cudaMalloc(&rd,results.size()*sizeof(int)));
    check(cudaMemcpy(xd,x.data(),18*sizeof(double),cudaMemcpyHostToDevice));
    transaction_cases<<<1,32>>>(xd,rd);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(results.data(),rd,results.size()*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaFree(xd));check(cudaFree(rd));
    f=std::fopen(output_path,"w");if(!f) return 2;
    std::fprintf(f,"# case source_only actual expected transaction_exact_and_path_exercised accepted rejected rhs jac skipped\n");
    bool passed=true;
    for(int i=0;i<24;++i) {
        std::fprintf(f,"%d %d",i%12,i/12);
        for(int j=0;j<8;++j) std::fprintf(f," %d",results[8*i+j]);
        std::fprintf(f,"\n");passed=passed && results[8*i+2]!=0;
    }
    std::fprintf(f,"# AIR5_IEEE compensated_add=%d fp64_subnormal_carry=%d compensated_rk=%d\n",results[192],results[193],results[194]);
    passed=passed && results[192] && results[193] && results[194];
    return std::fclose(f)==0 && passed?0:1;
}
int main(int argc,char **argv)
{
    if(argc==4 && std::strcmp(argv[1],"contracts")==0) return contracts(argv[2],argv[3]);
    if(argc!=8 || std::strcmp(argv[1],"batch")!=0) return 2;
    const int segments=std::atoi(argv[6]);
    const bool history_enabled=std::strcmp(argv[7],"reset")!=0;
    const int width=std::strcmp(argv[7],"history-from-input")==0?18:17;
    double rtol=number(argv[4]),atol_factor=number(argv[5]),duration;
    int n,mode;
    FILE *f=std::fopen(argv[2],"r");if(!f) return 2;
    if(std::fscanf(f,"%d%lf%d",&n,&duration,&mode)!=3 || n<=0 || segments<=0) return 2;
    std::vector<double> input(18*n,0.),output(14*n);std::vector<int> counts(5*n);
    for(int i=0;i<n;++i) for(int v=0;v<width;++v) if(std::fscanf(f,"%lf",&input[18*i+v])!=1) return 2;
    std::fclose(f);
    const char *evaluation=std::getenv("ASTR_AIR5_SOURCE_EVALUATION");
    if(evaluation && std::strcmp(evaluation,"joint")!=0 && std::strcmp(evaluation,"source-only")!=0) return 2;
    const bool source_only=evaluation && std::strcmp(evaluation,"source-only")==0;
    const char *eta=std::getenv("ASTR_AIR5_LOW_ACTIVITY");
    const double threshold=eta && std::strcmp(eta,"off")!=0?number(eta):0.;
    double *xd,*od;int *cd;
    check(cudaMalloc(&xd,input.size()*sizeof(double)));check(cudaMalloc(&od,output.size()*sizeof(double)));
    check(cudaMalloc(&cd,counts.size()*sizeof(int)));
    check(cudaMemcpy(xd,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice));
    advance_batch<<<(n+31)/32,32>>>(n,xd,od,cd,duration,mode,rtol,atol_factor,segments,history_enabled,source_only,threshold);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(),od,output.size()*sizeof(double),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(counts.data(),cd,counts.size()*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaFree(xd));check(cudaFree(od));check(cudaFree(cd));
    f=std::fopen(argv[3],"w");if(!f) return 2;
    bool failed=false;
    for(int i=0;i<n;++i) {
        std::fprintf(f,"%d",i+1);
        for(int v=0;v<5;++v) std::fprintf(f," %d",counts[5*i+v]);
        for(int v=0;v<13;++v) std::fprintf(f," %.17e",output[14*i+v]);
        std::fprintf(f,"\n# AIR5_ACTIVITY index=%d skipped_calls=%d total_calls=%d\n",i+1,int(output[14*i+13]),segments);
        failed=failed || counts[5*i]!=0;
    }
    const int closed=std::fclose(f);
    return failed || closed!=0?1:0;
}
