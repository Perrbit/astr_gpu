"""Identify old NVIDIA Jacobian-pair contraction on frozen faces (Python 3.13+)."""
import argparse
from itertools import product
import json
from math import fma, prod

import numpy as np

from run_air5_diffusion import read


PAIRS = tuple(product(range(3), repeat=3))


def pair(a, b, ja, jb, mode):
    if mode == 1:
        return fma(jb,b,ja*a)
    if mode == 2:
        return fma(ja,a,jb*b)
    return ja*a+jb*b


def anchor(v, j, mode):
    outer,near,middle=(pair(v[d],v[5-d],j[d],j[5-d],m) for d,m in enumerate(mode))
    return fma(1./60.,outer,fma(37.*(1./60.),middle,(-2./15.)*near))


def arrays(path, dims):
    padded=tuple(int(n)+11 for n in dims[::-1])
    n=prod(padded)
    physical=prod(dims+1)
    data=np.fromfile(path,"<f8")
    widths=(11*n,11*physical,11*physical,n,9*n,6*n,3*n,15*n,3*n,6*n)
    assert sum(widths)==data.size
    assert np.isfinite(data).all()
    blocks=np.split(data,np.cumsum(widths)[:-1])
    return [v.reshape((-1,*padded)) for v in blocks[3:]], blocks[2].reshape(11,-1)


def solve(dims, ratios, current, axes, rank, inputs):
    fields,rhs=arrays(inputs,dims)
    jac,metric,_,heat,_,_,carried=fields
    halo=dims[::-1]+11
    ratios=ratios.reshape((2,*halo))
    shape=dims[::-1]+1
    scores={}
    rng=np.random.default_rng(85)
    for d in range(3):
        # Only the centered kernel is inferred here. Physical closures are
        # intentionally left to the separate directional trace.
        start=[1,1,0];stop=list(dims)
        stop[0]-=1
        if d!=2:
            start[d]=3;stop[d]=int(dims[d])-3
        nodes=[tuple(int(rng.integers(l,h+1)) for l,h in zip(start,stop)) for _ in range(128)]
        _,_,_,ref=read(axes/f"old_a{d+1}",rank)
        left=[];right=[];target=[];seed=[];low=[];high=[]
        for p in nodes:
            at=np.ravel_multi_index(p[::-1],shape)
            target.append(ref[4,at]);seed.append(rhs[4,at])
            for side,values,theta in ((-1,left,low),(0,right,high)):
                center=p[d]+side
                points=[]
                for n in range(center-2,center+4):
                    x=list(p);x[d]=n
                    points.append(tuple(s+5 for s in x[::-1]))
                t0=list(p);t0[d]=center
                t1=t0.copy();t1[d]+=1
                a=tuple(s+5 for s in t0[::-1]);b=tuple(s+5 for s in t1[::-1])
                species=min(ratios[0][a],ratios[0][b])
                theta.append(min(ratios[1][a],ratios[1][b]))
                jj=[jac[0][x] for x in points]
                proj=[heat[d][x]*metric[4*d][x] for x in points]
                carry=[carried[d][x]*metric[4*d][x] for x in points]
                values.append([fma(species-1,anchor(carry,jj,mode),anchor(proj,jj,mode)) for mode in PAIRS])
        attempts=[]
        for ll,rr in product(range(len(PAIRS)),repeat=2):
            got=[s+fma(h,r[rr],-l*n[ll]) for s,h,r,l,n in zip(seed,high,right,low,left)]
            diff=np.abs(np.array(got)-target)
            attempts.append((int(np.count_nonzero(diff)),float(diff.max()),PAIRS[ll],PAIRS[rr]))
        scores[d+1]=dict(nodes=len(nodes),best=sorted(attempts)[:3],
                         current=next(v for v in attempts if v[2:]==(current,current)))
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    from pathlib import Path
    for arg in ("input-directory","axis-directory","output"):
        p.add_argument(f"--{arg}",type=Path,required=True)
    a=p.parse_args()
    report={}
    for rank in (0,1):
        hd,_,r,_=read(a.axis_directory/"old_a0",rank)
        report[rank]=solve(hd[:3],r,(1,1,1),a.axis_directory,rank,a.input_directory/f"old.input.rank{rank}.bin")
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
