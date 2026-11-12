"""Prepare deterministic 16-interval KEEP boundary examples without a test-tree dependency."""
import argparse
from pathlib import Path
import shutil
from string import Template

import h5py
import numpy as np


def prepare(destination,kind,backend,viscosity='standard',inlet_mode='prof',blowing='off',extruded=False,convection='keep'):
    if kind in ('incident','incident-compact') and (viscosity!='standard' or convection!='keep'):
        raise ValueError('The inviscid incident examples use KEEP without ME6 viscosity')
    if convection=='standard' and viscosity!='me6_opti':
        raise ValueError('Original convection in this new-form example requires ME6 viscosity')
    if (inlet_mode!='prof' or blowing!='off' or extruded) and kind not in ('open','plate'):
        raise ValueError('Dynamic inlet, wall forcing and HBL extrusion recipes apply to open or plate examples')
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    datin=destination/'datin';datin.mkdir()
    source=Path(__file__).resolve().parent/'datin'
    (datin/'input.case').write_text(Template((source/(kind+'.case.in')).read_text()).substitute(
        use_gpu='t' if backend=='gpu' else 'f',viscous_form=viscosity,convection_form=convection,inlet_mode=inlet_mode,
        wall_mode='legacy_random' if blowing=='legacy_random' else 'current',
        read_grid='f' if extruded else 't',
        grid_name='datin/grid.generated.h5' if extruded else 'datin/grid.flatplate.h5',
        initial_dimension=2 if extruded else 3,
        incident_pout=(2*1.4*(2*np.sin(np.deg2rad(40.)))**2-.4)/2.4))
    for name in ('input.run','input.output'):
        shutil.copyfile(source/name,datin/name)
    (destination/'outdat/new').mkdir(parents=True)
    if kind in ('open','plate'):
        yline=np.expm1(3*np.linspace(0,1,17))/np.expm1(3.)
        x,y,z=np.meshgrid(np.linspace(0,10,17),yline,np.linspace(0,.25,17),indexing='ij')
        with h5py.File(datin/('grid.2d' if extruded else 'grid.flatplate.h5'),'x') as grid:
            for name,value in zip(('x','y','z'),(x,y,z)):
                grid[name]=value[:,:,0].T if extruded else value.transpose(2,1,0)
        profile=1-np.exp(-(y/.08)**2);ramp=1-np.exp(-x/10);phase=2*np.pi*z/.25
        temperature=1+1e-4*ramp*profile**2*np.cos(phase)
        fields=dict(ro=1/temperature,t=temperature,u1=profile*(1+1e-4*ramp*np.cos(phase)),
                    u2=1e-4*ramp*profile**2*np.sin(phase),u3=1e-4*ramp*profile**2*np.cos(phase))
        with (datin/'inlet.prof').open('w') as stream:
            stream.write('KEEP low-order open example\nrho u v temperature\n.08 .026 .018 .01\ny rho u v temperature\n')
            for yy in yline:
                stream.write(f'1.0000000000000000e+00 {(1-np.exp(-(yy/.08)**2)):.16e} '
                             '0.0000000000000000e+00 1.0000000000000000e+00\n')
    elif kind in ('incident','incident-compact'):
        x,y,z=np.meshgrid(np.linspace(0,1,17),np.linspace(0,1,17),
                          np.linspace(0,.125,17),indexing='ij')
        with h5py.File(datin/'grid.flatplate.h5','x') as grid:
            for name,value in zip(('x','y','z'),(x,y,z)):
                grid[name]=value.transpose(2,1,0)
        fields=dict(ro=np.ones_like(x),t=np.ones_like(x),u1=np.ones_like(x),
                    u2=np.zeros_like(x),u3=np.zeros_like(x))
        with (datin/'inlet.prof').open('w') as stream:
            stream.write('Uniform incident-boundary fixture density=provided\nrho u v temperature\n1 1 1 1\n'
                         'y rho u v temperature\n')
            stream.write(('1.0000000000000000e+00 1.0000000000000000e+00 '
                          '0.0000000000000000e+00 1.0000000000000000e+00\n')*17)
    elif kind=='y-farfield':
        x,eta,z=np.meshgrid(*([np.linspace(0,2*np.pi,17)]*3),indexing='ij')
        y=eta+.15*np.sin(x)*np.sin(z)
        with h5py.File(datin/'grid.yfarfield.h5','x') as grid:
            for name,value in zip(('x','y','z'),(x,y,z)):
                grid[name]=value.transpose(2,1,0)
        envelope=(eta/(2*np.pi))**2
        temperature=1+.01*envelope*np.sin(x)*np.cos(z)
        fields=dict(ro=1/temperature,t=temperature,
            u1=.02*envelope*(1+.1*np.cos(x)*np.cos(z)),
            u2=.003*envelope*np.sin(x)*np.cos(z),u3=.005*envelope*np.cos(x)*np.sin(z))
    elif kind=='inflow':
        x,y,z=np.meshgrid(np.linspace(0,2*np.pi,17),np.linspace(0,np.pi,17),
                          np.linspace(0,2*np.pi,17),indexing='ij')
        with h5py.File(datin/'grid.inflow.h5','x') as grid:
            for name,value in zip(('x','y','z'),(x,y,z)):
                grid[name]=value.transpose(2,1,0)
        wave=np.cos(2*y)*np.cos(z)
        fields=dict(ro=1+.001*np.cos(x)*wave,t=1+.0002*np.sin(x)*wave,
                    u1=1+.001*np.cos(x)*wave,u2=.001*np.sin(2*y)*np.cos(z),u3=.001*np.cos(2*y)*np.sin(z))
    elif kind=='ldc':
        x,y,z=np.meshgrid(*([np.linspace(0,1,17)]*3),indexing='ij')
        with h5py.File(datin/'grid.ldc.h5','x') as grid:
            for name,value in zip(('x','y','z'),(x,y,z)):
                grid[name]=value.transpose(2,1,0)
        envelope=np.sin(np.pi*x)**2*np.sin(np.pi*y)**2
        fields=dict(ro=1+.001*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(2*np.pi*z),
                    t=1+.001*envelope*np.cos(2*np.pi*z),
                    **{f'u{d+1}':.01*(d+1)*envelope*np.cos(2*np.pi*z) for d in range(3)})
    else:
        x,y,z=np.meshgrid(np.linspace(0,2*np.pi,17),np.linspace(0,np.pi,17),
                          np.linspace(0,2*np.pi,17),indexing='ij')
        profile=np.sin(y)
        fields=dict(ro=np.ones_like(x),t=np.ones_like(x),
            u1=.05*profile*(1+.1*np.cos(x)*np.cos(z)),u2=.002*np.sin(x)*profile**2,
            u3=.001*np.cos(z)*profile**2)
        for name in ('u1','u2','u3'):
            fields[name][:,0,:]=0;fields[name][:,-1,:]=0
    if extruded:
        fields={name:value[:,:,0] for name,value in fields.items() if name!='u3'}
    with h5py.File(datin/f'flowini{2 if extruded else 3}d.h5','x') as initial:
        for name,value in fields.items():
            initial[name]=value.T if extruded else value.transpose(2,1,0)
    if inlet_mode=='intp':
        stream=destination/'inflow';stream.mkdir()
        wave=np.sin(np.pi*np.linspace(0,1,17))[:,None]*np.cos(2*np.pi*np.linspace(0,1,17))[None,:]
        amplitudes=dict(ro=.001,u1=.01,u2=.002,u3=.003,t=.0005)
        for index in range(12):
            modulation=1+.35*np.sin(.73*index)+.10*np.cos(.31*index**2)
            with h5py.File(stream/f'islice{index:05d}.h5','x') as plane:
                plane['time']=index*1e-5
                for name,amplitude in amplitudes.items():
                    plane[name]=(amplitude*modulation*wave).T
    if blowing!='off':
        (datin/'wallbs.dat').write_text('KEEP wall fixture\namplitude beta xa xb xc\n'
            f'1.e-3 2.0 2.0 5.0 8.0\nnmod_t nmod_z\n{2 if blowing=="modal" else 0} 2\n')
        if blowing=='modal':
            (datin/'wallbs_phase.dat').write_text(' '.join(f'{v:.17e}' for v in np.linspace(.05,.75,15))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['wall','open','ldc','inflow','plate','y-farfield','incident','incident-compact'],required=True)
    parser.add_argument('--backend',choices=['cpu','gpu'],required=True)
    parser.add_argument('--viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--inlet-mode',choices=['prof','intp'],default='prof')
    parser.add_argument('--blowing',choices=['off','modal','counter','legacy_random'],default='off')
    parser.add_argument('--extruded',action='store_true',help='Requires the existing userdefine_hbl.F90 root-CMake build')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.output,args.case,args.backend,args.viscosity,args.inlet_mode,args.blowing,args.extruded,args.convection)
