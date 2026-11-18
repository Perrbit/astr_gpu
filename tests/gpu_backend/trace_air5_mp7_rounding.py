"""Replay one MP7 diagnostic component from a frozen old-RHS trace (Python 3.13+)."""
import argparse
from functools import lru_cache
from itertools import product
import json
from math import fma, sqrt
from pathlib import Path

import numpy as np

from run_air5_filter_fields import fixture3
from run_air5_convection_budget import read as read_budget


COEFFICIENTS={3:(-1.,5.,2.),5:(2.,-13.,47.,27.,-3.),7:(-3.,25.,-101.,319.,214.,-38.,4.)}
DIVISOR={3:6.,5:60.,7:420.}


def minmod(*v):
    return min(v) if min(v)>0. else (max(v) if max(v)<0. else 0.)


def polynomial(v, order, fused, reciprocal):
    c=COEFFICIENTS[order]
    total=c[0]*v[0]
    for weight,value in zip(c[1:],v[1:]):
        total=fma(weight,value,total) if fused else total+weight*value
    d=DIVISOR[order]
    lin=total*(1./d) if reciprocal else total/d
    if order==3:
        return lin
    mid=order//2
    center=v[mid];m=v[mid-1];p=v[mid+1]
    mp=center+minmod(p-center,4.*(center-m))
    if min(center,mp)<=lin<=max(center,mp):
        return lin
    dm=v[mid-2]-2.*m+center
    d0=m-2.*center+p
    dp=center-2.*p+v[mid+2]
    dl=minmod(4.*dm-d0,4.*d0-dm,dm,d0)
    dr=minmod(4.*d0-dp,4.*dp-d0,d0,dp)
    upper=center+4.*(center-m)
    med=.5*(center+p)-.5*dr
    base=center+.5*(center-m)
    curv=fma(4./3.,dl,base) if fused else base+(4./3.)*dl
    lo=max(min(center,p,med),min(center,upper,curv))
    hi=min(max(center,p,med),max(center,upper,curv))
    return lin+minmod(lo-lin,hi-lin)


def read_fixture(path):
    with path.open("rb") as f:
        header=np.fromfile(f,"<i4",6)
        _=np.fromfile(f,"<f8",5)
        q=np.fromfile(f,"<f8").reshape((20,*(header[:3][::-1]+1)))
    return header,q


def run(fixture, prefix, limited_prefix=None):
    header,q=read_fixture(fixture)
    whole=header[:3]
    periodic=header[3:]
    model,_,_=fixture3("smooth")
    g=tuple(map(float,model.gas_constant));cv=tuple(map(float,model.cv_tr))

    @lru_cache(None)
    def state(p, axis, radius_fused):
        xyz=tuple(int(x%n if h else max(0,min(x,n))) for x,n,h in zip(p,whole,periodic))
        v=q[:,xyz[2],xyz[1],xyz[0]]
        m,c=0.,0.
        for y,r,d in zip(v[15:20],g,cv):
            m=fma(float(y),r,m) if radius_fused else m+float(y)*r
            c=fma(float(y),d,c) if radius_fused else c+float(y)*d
        normal,k,prs=float(v[11+axis]),float(v[4]),float(v[14])
        radius=abs(normal)+sqrt((1.+m/c)*prs/float(v[0]))
        return radius,(k+prs)*normal,k,k+prs,normal

    @lru_cache(None)
    def face(p,axis,n,ntype,last,variant,low=False):
        radius_fused,split_plus,split_minus,poly_fused,recip,polish=variant
        bound=0
        if ntype in (1,4) and 0<=n<=2:
            bound=1+2*n
        elif ntype in (2,4) and 1<=last-n<=3:
            bound=2*(last-n)-1
        order=1 if low else (bound or 7)
        left=n-(order-1)//2
        right=n+(order+1)//2
        nodes=[]
        for i in range(left,right+1):
            child=list(p);child[axis]=p[axis]+i-n
            nodes.append(state(tuple(child),axis,radius_fused))
        alpha=max(v[0] for v in nodes)
        plus=[];minus=[]
        for i in range(order):
            _,f,k,enthalpy,normal=nodes[i]
            trial=fma(normal,enthalpy,alpha*k) if split_plus==2 else (fma(alpha,k,f) if split_plus==1 else f+alpha*k)
            plus.append(.5*trial)
            _,f,k,enthalpy,normal=nodes[-1-i]
            trial=fma(normal,enthalpy,-alpha*k) if split_minus==2 else (fma(-alpha,k,f) if split_minus==1 else f-alpha*k)
            minus.append(.5*trial)
        if order==1:
            return plus[0]+minus[0]
        return polynomial(plus,order,poly_fused,recip)+polynomial(minus,order,poly_fused,recip or polish)

    variants=list(product((False,True),range(3),range(3),(False,True),(False,True),(False,True)))
    scores={d:[0]*len(variants) for d in range(3)}
    magnitudes={d:[0.]*len(variants) for d in range(3)}
    # The old compiler sees both faces in the caller. Check separately
    # whether preserving the same contraction on either face is valid.
    choices=[x for x in variants if x[0] and x[3] and not x[4] and not x[5]]
    per_face={d:np.zeros((len(choices),len(choices)), dtype=int) for d in range(3)}
    per_face_peak={d:np.zeros((len(choices),len(choices))) for d in range(3)}
    pp=np.random.default_rng(8)
    summary={}
    hotspot_terms={}
    limited_choices=list(product(range(3),repeat=4))
    limited_scores=np.zeros((len(limited_choices),)*2,dtype=int) if limited_prefix else None
    limited_peak=np.zeros((len(limited_choices),)*2) if limited_prefix else None
    for rank in (0,1):
        with Path(f"{prefix}.axes.rank{rank}.bin").open("rb") as f:
            dims=np.fromfile(f,"<i4",3)
            rhs=np.fromfile(f,"<f8").reshape((3,11,*(dims[::-1]+1)))
        offset=np.array([rank*dims[0],0,0])
        ntypes=(1 if rank==0 else 2,4,3)
        lo=[1 if rank==0 else 0,1,0];hi=[int(dims[0]-(rank==1)),int(dims[1]-1),int(dims[2])]
        hotspots=((2,5,9),(7,8,6),(5,11,9),(2,5,3))
        nodes=[tuple(map(int,pp.integers(lo,np.asarray(hi)+1))) for _ in range(160)]+list(hotspots)
        for d in range(3):
            for node in nodes:
                p=tuple(int(k+x) for k,x in zip(node,offset));i=node[d]
                prev=0. if d==0 else rhs[d-1,4,node[2],node[1],node[0]]
                wanted=rhs[d,4,node[2],node[1],node[0]]
                p0=list(p);p0[d]-=1
                for iv,var in enumerate(variants):
                    l=face(tuple(p0),d,i-1,ntypes[d],int(dims[d]),var)
                    r=face(p,d,i,ntypes[d],int(dims[d]),var)
                    delta=abs(float(prev+l-r-wanted))
                    scores[d][iv]+=delta!=0.
                    magnitudes[d][iv]=max(magnitudes[d][iv],delta)
                l=np.asarray([face(tuple(p0),d,i-1,ntypes[d],int(dims[d]),v) for v in choices])
                r=np.asarray([face(p,d,i,ntypes[d],int(dims[d]),v) for v in choices])
                difference=abs((prev+l[:,None])-r[None,:]-wanted)
                per_face[d]+=difference!=0.
                per_face_peak[d]=np.maximum(per_face_peak[d],difference)
        if limited_prefix:
            _,_,ratios,values=read_budget(limited_prefix,rank)
            assert np.all(ratios==1.), "diagnostic addition assumes the frozen unit limiter state"
            limited=values[1,4].reshape(tuple(dims[::-1]+1))
            ll,lr,ml,mr=np.asarray(limited_choices).T
            fixed=[(True,a,b,True,False,False) for a,b in product(range(3),repeat=2)]
            for node in nodes:
                p=tuple(int(k+x) for k,x in zip(node,offset))
                result=np.zeros_like(limited_peak)
                for d in range(3):
                    left=list(p);left[d]-=1;left=tuple(left)
                    common=(ntypes[d],int(dims[d]))
                    h0=np.asarray([face(left,d,node[d]-1,*common,v) for v in fixed])[3*ll+ml,None]
                    h1=np.asarray([face(p,d,node[d],*common,v) for v in fixed])[3*lr+mr,None]
                    l0=np.asarray([face(left,d,node[d]-1,*common,v,True) for v in fixed])[None,3*ll+ml]
                    l1=np.asarray([face(p,d,node[d],*common,v,True) for v in fixed])[None,3*lr+mr]
                    result=((result+l0)-l1)+(h0-l0)-(h1-l1)
                    if node==(5,11,9):
                        h=limited_choices.index((1,2,1,2))
                        l=limited_choices.index((1,2,0,2))
                        hotspot_terms[f"{rank}/{d}"]=[float(x) for x in
                            (h0[h,0],h1[h,0],l0[0,l],l1[0,l],result[h,l])]
                difference=abs(result-limited[node[2],node[1],node[0]])
                limited_scores+=difference!=0.
                limited_peak=np.maximum(limited_peak,difference)
    for d in range(3):
        values=sorted(zip(scores[d],magnitudes[d],variants))
        combined=sorted((int(per_face[d][i,j]),float(per_face_peak[d][i,j]),choices[i],choices[j])
                        for i,j in product(range(len(choices)),repeat=2))
        summary[d+1]={"best":values[:5],"prior":next(v for v in values if v[2]==(True,True,1,True,False,False)),
                      "current":next(v for v in values if v[2]==(True,True,2,True,False,False)),
                      "separate":combined[:5]}
    if limited_prefix:
        summary["limited"]=sorted((int(limited_scores[i,j]),float(limited_peak[i,j]),limited_choices[i],limited_choices[j])
                                  for i,j in product(range(len(limited_choices)),repeat=2))[:8]
        h=limited_choices.index((1,2,1,2))
        l=limited_choices.index((1,2,0,2))
        summary["limited_current"]=[int(limited_scores[h,l]),float(limited_peak[h,l])]
        summary["hotspot_terms"]=hotspot_terms
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("fixture","prefix","output"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    parser.add_argument("--limited-prefix",type=Path)
    a=parser.parse_args()
    report=run(a.fixture,a.prefix,a.limited_prefix)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
