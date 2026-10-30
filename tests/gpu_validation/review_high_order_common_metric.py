"""Exact necessary conditions for a frozen common energy metric, not CFD."""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from test_high_order_adiabatic_pressure_candidate import pressure_rows
from test_high_order_boundary_closures import derivative_rows


def exact_blocks(n,kind):
    if kind not in (41,42,421):raise ValueError('Only the current 41/42/421 models are covered')
    d=sp.zeros(n+1)
    for row,weights in enumerate(derivative_rows(True)):
        for column,value in enumerate(weights):
            d[row,column]=n*value;d[n-row,n-column]=-n*value
    for row in range(3,n-2):
        for shift,value in zip(range(-3,4),(-1,9,-45,0,45,-9,1)):
            d[row,row+shift]=sp.Rational(n*value,60)
    velocity=sp.zeros(n+1,n-1);velocity[1:n,:]=sp.eye(n-1)
    thermal=velocity.copy()
    if kind!=41:
        weights=sp.finite_diff_weights(1,list(range(8)),0)[1][-1]
        for j in range(7):
            thermal[0,j]=-weights[j+1]/weights[0]
            thermal[n,n-2-j]=-weights[j+1]/weights[0]
    gradient=sp.zeros(n-1,n+1)
    for row in range(3,n-2):
        for shift,value in zip(range(-3,4),(-1,9,-45,0,45,-9,1)):
            gradient[row-1,row+shift]=sp.Rational(n*value,60)
    degree=7 if kind==41 else 6
    for row,weights in enumerate(pressure_rows(degree),start=1):
        for j in range(degree):
            gradient[row-1,j]=n*weights[j];gradient[n-row-1,n-j]=-n*weights[j]
    b=d*velocity;c=sp.Rational(2,5)*b[1:n,:]
    dv=sp.Rational(4,3)*(d*d*velocity)[1:n,:]
    dt=sp.Rational(35,18)*(d*d*thermal)[1:n,:]
    return b,c,gradient,thermal,dv,dt


def assemble(blocks):
    b,c,g,t,dv,dt=blocks;r,v=b.shape
    zero_r=sp.zeros(r);zero_v=sp.zeros(v)
    a=sp.BlockMatrix([[zero_r,-b,sp.zeros(r,v)],[-g,zero_v,-g*t],
        [sp.zeros(v,r),-c,zero_v]]).as_explicit()
    diffusion=sp.diag(zero_r,dv,dt)
    return a,diffusion


def parity_certificate(metrics,n):
    v=n-1;symmetric=sp.zeros(v,(v+1)//2)
    for i in range(symmetric.cols):
        symmetric[i,i]=symmetric[v-i-1,i]=1
    restrictions=[symmetric.T*h[n+1:2*n,n+1:2*n]*symmetric for h in metrics]
    span=DomainMatrix.from_Matrix(sp.Matrix([list(c) for c in restrictions])).rank()
    report=dict(velocity_symmetric_dimension=symmetric.cols,restriction_span_dimension=span,
        indefinite_one_dimensional_restriction=False)
    if span!=1:return report
    template=next(c for c in restrictions if c!=sp.zeros(c.rows))
    template=template/max(abs(x) for x in template)
    pivot=next((i,j) for i in range(template.rows) for j in range(template.cols) if template[i,j])
    scales=[c[pivot[0],pivot[1]]/template[pivot[0],pivot[1]] for c in restrictions]
    assert all(c==scale*template for c,scale in zip(restrictions,scales))
    witnesses={}
    def record(vector):
        value=(vector.T*template*vector)[0]
        if value:
            witnesses['positive' if value>0 else 'negative']=dict(
                vector=[str(x) for x in vector],quadratic_value=str(value))
    for i in range(template.rows):record(sp.eye(template.rows)[:,i])
    for i in range(template.rows):
        for j in range(i+1,template.rows):
            if template[i,i] and template[i,i]*template[j,j]-template[i,j]**2<0:
                vector=sp.zeros(template.rows,1)
                vector[i]=template[i,j];vector[j]=-template[i,i]
                record(vector)
    report.update(template=[[str(x) for x in row] for row in template.tolist()],
        restriction_scales=[str(x) for x in scales],exact_scalar_multiple_check=True,witnesses=witnesses,
        indefinite_one_dimensional_restriction=set(witnesses)=={'positive','negative'})
    return report


def exact_common_space(n,kind):
    # A zero density block in D forces the corresponding dissipation rows to
    # vanish. Full-rank Dv/G then remove rho-v and v-T metric blocks.
    blocks=exact_blocks(n,kind);b,c,g,t,dv,dt=blocks;r,v=b.shape
    rank_g=g.rank();rank_dv=dv.rank();left=dt.T.nullspace()
    if rank_g!=v or rank_dv!=v or len(left)!=(0 if kind==41 else 1):
        raise ValueError('The reduced exact constraints need full-rank G/Dv and the declared thermal null modes')
    zero_r=sp.zeros(r);zero_v=sp.zeros(v);zero_x=sp.zeros(r,v)
    basis=[];equations=[]
    for block,size in enumerate((r,v,v)):
        for i in range(size):
            for j in range(i,size):
                hr=zero_r.copy();hv=zero_v.copy();ht=zero_v.copy();x=zero_x.copy()
                target=(hr,hv,ht)[block];target[i,j]=target[j,i]=1
                basis.append((hr,hv,ht,x))
    for thermal_null in left:
        for i in range(r):
            x=zero_x.copy();x[i,:]=thermal_null.T
            basis.append((zero_r.copy(),zero_v.copy(),zero_v.copy(),x))
    for hr,hv,ht,x in basis:
        equations.append(sp.Matrix(list(hr*b+x*c+g.T*hv)+list(x.T*b+ht*c+(g*t).T*hv)))
    system=sp.Matrix.hstack(*equations)
    kernel=DomainMatrix.from_Matrix(system).nullspace().to_Matrix()
    assert system*kernel.T==sp.zeros(system.rows,kernel.rows)
    a,diffusion=assemble(blocks)
    metrics=[]
    for row in kernel.tolist():
        hr=zero_r.copy();hv=zero_v.copy();ht=zero_v.copy();x=zero_x.copy()
        for coefficient,parts in zip(row,basis):
            if coefficient==0:continue
            hr+=coefficient*parts[0];hv+=coefficient*parts[1]
            ht+=coefficient*parts[2];x+=coefficient*parts[3]
        h=sp.BlockMatrix([[hr,sp.zeros(r,v),x],[sp.zeros(v,r),hv,zero_v],
            [x.T,zero_v,ht]]).as_explicit()
        assert a.T*h+h*a==sp.zeros(a.rows)
        assert (diffusion.T*h+h*diffusion)[:r,:]==sp.zeros(r,a.rows)
        metrics.append(h)
    stacked=sp.Matrix.vstack(*metrics)
    common_kernel=DomainMatrix.from_Matrix(stacked).nullspace().to_Matrix()
    for row in common_kernel.tolist():
        assert all(h*sp.Matrix(row)==sp.zeros(a.rows,1) for h in metrics)
    parity=parity_certificate(metrics,n)
    infeasible=bool(common_kernel.rows or parity['indefinite_one_dimensional_restriction'])
    report=dict(intervals=n,bctype=kind,space='P7' if kind==41 else 'P6/T7',
        scope='necessary common-metric conditions for the frozen normal block; not a filter or CFD admission',
        exact_arithmetic='rational, no singular-value cutoff',dimension=a.rows,rank_G=rank_g,rank_Dv=rank_dv,
        thermal_left_nullity=len(left),unknowns=len(basis),equations=system.rows,
        solution_dimension=kernel.rows,common_kernel_dimension=common_kernel.rows,
        common_kernel=[[str(value) for value in row] for row in common_kernel.tolist()],
        parity_certificate=parity,positive_definite_common_metric_impossible=infeasible,
        verdict=('no positive-definite common metric under these exact necessary conditions' if infeasible else
            'no obstruction certified; positive definiteness and viscous dissipation remain unproved'))
    return metrics,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for n in (8,9):
        for kind in (41,42):
            print('begin',n,kind,flush=True)
            metrics,report=exact_common_space(n,kind)
            with (args.output/f'N{n}_bc{kind}.json').open('x') as stream:
                json.dump(report,stream,indent=2);stream.write('\n')
            normalized=[np.array(h/max(abs(x) for x in h),float) for h in metrics]
            np.savez(args.output/f'N{n}_bc{kind}.npz',necessary_metric_basis=np.array(normalized))
            print(n,kind,report['solution_dimension'],report['common_kernel_dimension'],report['verdict'],flush=True)


if __name__=='__main__':main()
