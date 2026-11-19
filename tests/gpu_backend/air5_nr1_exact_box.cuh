#pragma once
#include <cstdint>

namespace astr { namespace air5 { namespace nr1_budget {
// A finite binary64 product is an integer multiple of 2^-2148. The largest
// residual below sums eleven products; 66 limbs leave room for sign and carry.
struct ExactSum {
    std::uint64_t words[66];

    __device__ void add_limb(int i,std::uint64_t word,bool negative)
    {
        for(;i<66 && word;++i) {
            const auto old=words[i];
            words[i]=negative?old-word:old+word;
            word=negative?old<word:words[i]<old;
        }
    }

    __device__ void product(double a,double b,bool subtract=false)
    {
        const auto x=static_cast<std::uint64_t>(__double_as_longlong(a));
        const auto y=static_cast<std::uint64_t>(__double_as_longlong(b));
        const int ex=(x>>52)&2047,ey=(y>>52)&2047;
        const std::uint64_t implicit=std::uint64_t{1}<<52;
        const std::uint64_t sa=(x&(implicit-1))|(ex?implicit:0);
        const std::uint64_t sb=(y&(implicit-1))|(ey?implicit:0);
        if(sa==0 || sb==0)return;
        const std::uint64_t al=static_cast<std::uint32_t>(sa),ah=sa>>32;
        const std::uint64_t bl=static_cast<std::uint32_t>(sb),bh=sb>>32;
        const std::uint64_t p0=al*bl,p1=al*bh+ah*bl;
        const std::uint64_t low=p0+(p1<<32),high=ah*bh+(p1>>32)+(low<p0);
        const int offset=(ex?ex-1075:-1074)+(ey?ey-1075:-1074)+2148;
        const int i=offset/64,shift=offset%64;
        const bool negative=((x^y)>>63)!=subtract;
        if(!shift) {
            add_limb(i,low,negative);add_limb(i+1,high,negative);
        } else {
            add_limb(i,low<<shift,negative);
            add_limb(i+1,(low>>(64-shift))|(high<<shift),negative);
            add_limb(i+2,high>>(64-shift),negative);
        }
    }

    __device__ int sign() const
    {
        if(words[65]>>63)return -1;
        for(auto word:words)if(word)return 1;
        return 0;
    }
};

__device__ inline bool moment_brackets(const double lo[5],const double hi[5],double mass,
    const double gas[5],double moment,const int order[5],bool upper)
{
    for(int k=0;k<5;++k) {
        ExactSum residual{};
        residual.product(mass,1.);
        for(int s=0;s<5;++s)residual.product(s<=k?hi[order[s]]:lo[order[s]],1.,true);
        if(residual.sign()>0)continue;
        // At the pivot only one component remains free. Expand its eliminated
        // moment without first rounding the free mass or the weight differences.
        const double gp=gas[order[k]];
        ExactSum bound{};
        bound.product(mass,gp);
        bound.product(moment,1.,true);
        for(int s=0;s<5;++s)if(s!=k) {
            const int i=order[s];
            const double fixed=s<k?hi[i]:lo[i];
            bound.product(gas[i],fixed);
            bound.product(gp,fixed,true);
        }
        return upper?bound.sign()>=0:bound.sign()<=0;
    }
    return false;
}

__device__ __noinline__ bool strict_feasible(const double lo[5],const double hi[5],double mass,
    const double gas[5],double moment)
{
    ExactSum low{},high{};
    low.product(mass,1.);high.product(mass,1.,true);
    for(int s=0;s<5;++s) {low.product(lo[s],1.,true);high.product(hi[s],1.);}
    if(low.sign()<0 || high.sign()<0)return false;
    int order[5]={0,1,2,3,4};
    for(int i=1;i<5;++i)for(int j=i;j>0 && gas[order[j]]<gas[order[j-1]];--j) {
        const int t=order[j];order[j]=order[j-1];order[j-1]=t;
    }
    if(!moment_brackets(lo,hi,mass,gas,moment,order,false))return false;
    for(int s=0;s<2;++s) {const int t=order[s];order[s]=order[4-s];order[4-s]=t;}
    return moment_brackets(lo,hi,mass,gas,moment,order,true);
}
}}}
