"""Read-only spectra and periodic two-point correlations of stored TGV fields."""
from pathlib import Path
import json
import struct
import time

import h5py
import numpy as np
from scipy import fft


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT/'tests/gpu_validation/out/native_fp32_tgv_checkpoints_20261022'
OUTPUT_DIR = ROOT/'tests/gpu_validation/out/native_fp32_tgv_spectra_20261028'
CASES = ('643e_fp64','643e_fp32','643c_fp64','643c_fp32')
STEPS = (35000,40000)
PAIRS = (('643e_fp64','643e_fp32'),('643c_fp64','643c_fp32'),
         ('643e_fp64','643c_fp64'),('643e_fp32','643c_fp32'))


def spectral_grid(shape):
    """Unit-width radial bins and Hermitian weights for an unpadded real FFT."""
    if len(shape)!=3 or len(set(shape))!=1 or shape[0]<4:
        raise ValueError('The diagnostic requires a cubic periodic uniform grid')
    n=shape[0]
    k=fft.fftfreq(n)*n
    kz=fft.rfftfreq(n)*n
    radius=np.sqrt(k[:,None,None]**2+k[None,:,None]**2+kz[None,None,:]**2)
    bins=np.floor(radius+.5).astype(np.int32)
    weights=np.full(len(kz),2.)
    weights[0]=1
    if n%2==0:
        weights[-1]=1
    counts=np.bincount(bins.ravel(),weights=np.broadcast_to(weights,bins.shape).ravel())
    return bins,weights,counts


def shell_sum(values,bins,weights):
    return np.bincount(bins.ravel(),weights=(values*weights[None,None,:]).ravel(),
                       minlength=int(bins.max())+1)


def forward_velocity(velocity,workers=4):
    """Return mean-removed Fourier coefficients; the forward FFT divides by N^3."""
    if velocity.shape[0]!=3 or not np.isfinite(velocity).all():
        raise ValueError('Expected three finite velocity components')
    means=np.mean(velocity,axis=(1,2,3),dtype=np.float64)
    transformed=np.empty((3,velocity.shape[1],velocity.shape[2],velocity.shape[3]//2+1),complex)
    for component in range(3):
        transformed[component]=fft.rfftn(velocity[component]-means[component],norm='forward',workers=workers)
        transformed[component,0,0,0]=0
    return transformed,means


def velocity_diagnostics(transformed,shape,bins,weights):
    """Shell-summed E_u and all nine component/separation-axis correlations."""
    n=shape[0]
    spectrum=np.zeros(int(bins.max())+1)
    correlations=np.empty((3,3,n))
    one_dimensional_spectra=np.empty((3,3,n//2+1))
    for component in range(3):
        power=np.abs(transformed[component])**2
        spectrum+=.5*shell_sum(power,bins,weights)
        for axis in range(3):
            if axis==2:
                density=power.sum(axis=(0,1))
                correlations[component,axis]=fft.irfft(density,n=n,norm='forward')
                one_dimensional_spectra[component,axis]=density*weights
            else:
                axes=tuple(d for d in range(3) if d!=axis)
                density=(power*weights[None,None,:]).sum(axis=axes)
                correlations[component,axis]=fft.ifft(density,norm='forward').real
                folded=density[:n//2+1].copy()
                stop=(n+1)//2
                folded[1:stop]+=density[-1:-(stop):-1]
                one_dimensional_spectra[component,axis]=folded
    variances=correlations[:,0,0]
    normalized=np.full_like(correlations,np.nan)
    np.divide(correlations,variances[:,None,None],out=normalized,where=variances[:,None,None]>0)
    return dict(energy=spectrum,correlation=correlations,normalized_correlation=normalized,
                directional_power=one_dimensional_spectra,variance=variances)


def density_weighted_spectrum(density,velocity,bins,weights,workers=4):
    """sqrt(rho)*u is not demeaned: the shell sum is the full kinetic energy."""
    result=np.zeros(int(bins.max())+1)
    for component in range(3):
        field=np.sqrt(density)*velocity[component]
        transformed=fft.rfftn(field,norm='forward',workers=workers)
        result+=.5*shell_sum(np.abs(transformed)**2,bins,weights)
    return result


def compare_transforms(reference,candidate,bins,weights):
    """Cross-shell alignment and difference energy; no division by tiny cell values."""
    length=int(bins.max())+1
    ea,eb,error,cross=(np.zeros(length) for _ in range(4))
    for component in range(3):
        a,b=reference[component],candidate[component]
        ea+=.5*shell_sum(np.abs(a)**2,bins,weights)
        eb+=.5*shell_sum(np.abs(b)**2,bins,weights)
        error+=.5*shell_sum(np.abs(b-a)**2,bins,weights)
        cross+=.5*shell_sum((a*np.conj(b)).real,bins,weights)
    denominator=np.sqrt(ea*eb)
    alignment=np.full(length,np.nan)
    np.divide(cross,denominator,out=alignment,where=denominator>0)
    return dict(reference_energy=ea,candidate_energy=eb,error_energy=error,
                cross_energy=cross,spectral_alignment=alignment)


def word_real(word,width):
    raw=struct.pack('<Q',int(word)&((1<<64)-1))
    return float(struct.unpack('<f' if width==4 else '<d',raw[:width])[0])


def read_checkpoint(case,step):
    source=case/'outdat/output/checkpoints'/f'step{step:012d}'/'state.h5'
    with h5py.File(source) as f:
        identity=f['identity'][:]
        width=int(f['numerical_kind_bytes'][0])
        if int(identity[8])!=step or f['q0001'].dtype.itemsize!=width:
            raise ValueError(f'Checkpoint identity/precision mismatch: {source}')
        selection=np.s_[:-1,:-1,:-1]
        # HDF5 uses z,y,x. Exclude all three duplicated periodic endpoints.
        density=np.asarray(f['q0001'][selection],dtype=np.float64).transpose(2,1,0).copy()
        velocity=np.empty((3,*density.shape))
        for component in range(3):
            momentum=np.asarray(f[f'q{component+2:04d}'][selection],dtype=np.float64).transpose(2,1,0)
            velocity[component]=momentum/density
        total_energy=np.asarray(f['q0005'][selection],dtype=np.float64)
        info=dict(source=str(source),source_bytes=source.stat().st_size,precision_bytes=width,
            step=step,native_time=word_real(identity[9],width),nominal_time=step*.00025,
            native_dt=word_real(identity[10],width),shape_xyz=list(density.shape),
            rho_min=float(density.min()),rho_mean=float(density.mean()),
            total_energy_mean=float(total_energy.mean()),
            velocity_means=np.mean(velocity,axis=(1,2,3)).tolist())
    if not np.isfinite(density).all() or np.any(density<=0) or not np.isfinite(velocity).all():
        raise ValueError(f'Invalid stored state: {source}')
    with h5py.File(case/'outdat/output/resources/geometry.h5') as f:
        coordinates=(f['q0001'][0,0,:],f['q0002'][0,:,0],f['q0003'][:,0,0])
        for axis,line in enumerate(coordinates):
            tolerance=32*np.finfo(line.dtype).eps*max(1,float(np.max(np.abs(line))))
            expected=np.linspace(0,2*np.pi,len(line))
            if not np.allclose(line,expected,rtol=0,atol=tolerance):
                raise ValueError(f'Not the admitted uniform 2pi grid, axis {axis}: {source}')
    return density,velocity,info


def band_summary(energy,n):
    k=np.arange(len(energy))
    return {name:float(energy[mask].sum()) for name,mask in (
        ('k_1_7',(k>=1)&(k<8)),('k_8_31',(k>=8)&(k<32)),
        ('k_32_63',(k>=32)&(k<64)),('k_64_nyquist',(k>=64)&(k<=n//2)),
        ('shell_centers_above_nyquist',k>n//2))}


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    report=dict(source=str(INPUT_DIR),analysis_precision='offline FP64, source arrays unchanged',
        normalization='DFT forward divides by N^3; integer radial shells [k-.5,k+.5); shell sums, not mode means',
        correlation='instantaneous periodic volume mean of mean-removed velocity products; not a time or ensemble mean',
        density_weighted_energy='FFT of sqrt(rho)*u without demeaning; sum E_rho=.5<rho|u|^2>',
        comparison='same completed step; native FP32 time drift is retained, no time interpolation of fields',
        axis_order='component u/v/w, separation x/y/z',snapshots={},pairs={})
    for step in STEPS:
        transforms={}
        diagnostics={}
        for case in CASES:
            started=time.monotonic()
            density,velocity,info=read_checkpoint(INPUT_DIR/case,step)
            bins,weights,counts=spectral_grid(density.shape)
            transformed,means=forward_velocity(velocity)
            values=velocity_diagnostics(transformed,density.shape,bins,weights)
            weighted=density_weighted_spectrum(density,velocity,bins,weights)
            physical_u=.5*sum(np.mean((velocity[c]-means[c])**2) for c in range(3))
            physical_rho=.5*sum(np.mean(density*velocity[c]**2) for c in range(3))
            errors=[abs(values['energy'].sum()-physical_u),abs(weighted.sum()-physical_rho)]
            if max(errors)>1e-12*max(1,physical_u,physical_rho):
                raise ValueError(f'Parseval normalization failed for {case}: {errors}')
            summary=dict(**info,velocity_fluctuation_energy=float(physical_u),
                density_weighted_kinetic_energy=float(physical_rho),parseval_absolute_errors=errors,
                velocity_energy_bands=band_summary(values['energy'],density.shape[0]),
                density_weighted_energy_bands=band_summary(weighted,density.shape[0]),
                max_abs_normalized_correlation_error_at_zero=float(np.nanmax(np.abs(values['normalized_correlation'][:,:,0]-1))))
            key=f'{case}_step{step}'
            np.savez_compressed(OUTPUT_DIR/f'{key}.npz',k=np.arange(len(weighted)),
                mode_counts=counts,weighted_energy=weighted,**values)
            np.savetxt(OUTPUT_DIR/f'{key}_spectra.csv',np.column_stack((np.arange(len(weighted)),counts,values['energy'],weighted)),
                delimiter=',',header='k,mode_count,E_u_fluctuation,E_sqrt_rho_u',comments='')
            rows=[]
            for component in range(3):
                for axis in range(3):
                    for lag in range(density.shape[0]//2+1):
                        rows.append([component,axis,lag,lag*2*np.pi/density.shape[0],
                                     values['correlation'][component,axis,lag],values['normalized_correlation'][component,axis,lag]])
            np.savetxt(OUTPUT_DIR/f'{key}_correlations.csv',rows,delimiter=',',
                header='component_uvw,axis_xyz,lag_nodes,r_over_Lref,R_ii,normalized_R_ii',comments='')
            report['snapshots'][key]=summary
            transforms[case]=transformed
            diagnostics[case]=values
            del density,velocity
            print(f'{key}: Parseval {errors}, elapsed {time.monotonic()-started:.1f}s',flush=True)
        for reference,candidate in PAIRS:
            values=compare_transforms(transforms[reference],transforms[candidate],bins,weights)
            pair=f'{candidate}_minus_{reference}_step{step}'
            denominator=values['reference_energy'].sum()
            err=values['error_energy'].sum()
            correlations=diagnostics[candidate]['normalized_correlation']-diagnostics[reference]['normalized_correlation']
            n=transforms[reference].shape[1]
            aenergy=diagnostics[reference]['energy'].sum()
            benergy=diagnostics[candidate]['energy'].sum()
            fractions={}
            k=np.arange(len(values['error_energy']))
            for low,high in [(1,8),(8,32),(32,64),(64,n//2+1)]:
                select=(k>=low)&(k<high)
                ea=values['reference_energy'][select].sum()
                eb=values['candidate_energy'][select].sum()
                cp=values['cross_energy'][select].sum()
                fractions[f'k_{low}_{high-1}']=dict(reference_energy=float(ea),candidate_energy=float(eb),
                    candidate_over_reference=float(eb/ea) if ea>0 else None,
                    error_over_reference=float(values['error_energy'][select].sum()/ea) if ea>0 else None,
                    alignment=float(cp/np.sqrt(ea*eb)) if ea>0 and eb>0 else None)
            report['pairs'][pair]=dict(reference=reference,candidate=candidate,step=step,
                relative_velocity_L2=float(np.sqrt(err/denominator)),
                velocity_energy_relative_change=float(benergy/aenergy-1),
                normalized_correlation_Linf_components_axes=np.nanmax(np.abs(correlations),axis=2).tolist(),
                bands=fractions,
                difference_identity_absolute_error=float(np.max(np.abs(values['error_energy']-
                    (values['reference_energy']+values['candidate_energy']-2*values['cross_energy'])))))
            np.savez_compressed(OUTPUT_DIR/f'{pair}.npz',**values)
        del transforms,diagnostics
    (OUTPUT_DIR/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(OUTPUT_DIR/'report.json',flush=True)


if __name__=='__main__':
    main()
