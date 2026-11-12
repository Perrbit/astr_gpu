"""Prepare the explicit KEEP SBP6/3 weak-wall example; does not run ASTR."""
import argparse
from pathlib import Path
import shutil
from string import Template

import h5py
import numpy as np


def prepare(destination,backend,intervals=16):
    if intervals<11:
        raise ValueError('SBP6/3 needs disjoint six-row boundary bands')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    datin=destination/'datin';datin.mkdir()
    source=Path(__file__).resolve().parent/'datin'
    (datin/'input.case').write_text(Template((source/'input.case.in').read_text()).substitute(
        use_gpu='t' if backend=='gpu' else 'f',intervals=intervals))
    for filename in ('input.run','input.output'):
        shutil.copyfile(source/filename,datin/filename)
    x,y,z=np.meshgrid(np.linspace(0,2*np.pi,intervals+1),np.linspace(0,np.pi,intervals+1),
                      np.linspace(0,2*np.pi,intervals+1),indexing='ij')
    with h5py.File(datin/'grid.h5','x') as grid:
        for name,value in zip(('x','y','z'),(x,y,z)):
            grid[name]=value.transpose(2,1,0)
    fields=dict(ro=np.ones_like(x),t=np.ones_like(x),
        u1=.05*np.sin(y)*(1+.1*np.cos(x)*np.cos(z)),
        u2=.002*np.sin(x)*np.sin(y)**2,u3=.001*np.cos(z)*np.sin(y)**2)
    for name in ('u1','u2','u3'):
        fields[name][:,0,:]=0;fields[name][:,-1,:]=0
    with h5py.File(datin/'flowini3d.h5','x') as initial:
        for name,value in fields.items():
            initial[name]=value.transpose(2,1,0)
    (destination/'outdat/new').mkdir(parents=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['cpu','gpu'],required=True)
    parser.add_argument('--intervals',type=int,default=16)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.output,args.backend,args.intervals)
