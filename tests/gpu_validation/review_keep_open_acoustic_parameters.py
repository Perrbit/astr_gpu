"""Analytic preparation review for a proposed KEEP acoustic gate; never launch ASTR."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    gamma=1.4;mach=.3;rho0=1.;p0=1/(gamma*mach**2);a0=1/mach
    y0=.45;width=.06;probe=.875;amplitude=1e-4;stop=.23
    incident=[(probe-y0-width)/a0,(probe-y0+width)/a0]
    reflected=[(2-y0-probe-width)/a0,(2-y0-probe+width)/a0]
    reference_return=(3-y0-probe-width)/a0
    incident_window=[.105,.150];reflected_window=[.180,.225]
    if not incident_window[0]<=incident[0]<incident[1]<=incident_window[1]<reflected_window[0]<=reflected[0]<reflected[1]<=reflected_window[1]<=stop:
        raise AssertionError('Proposed acoustic windows do not separate the compact packets')
    if reference_return<=stop or probe/a0<=stop:
        raise AssertionError('Linear reference-return/wall-transit estimate is not separated from the observation window')
    rows=[]
    for ny,dt in ((128,1e-4),(256,5e-5)):
        y=np.linspace(0,1,ny+1)
        distance=(y-y0)/width
        pulse=np.zeros_like(y);support=abs(distance)<1
        pulse[support]=amplitude*np.cos(.5*np.pi*distance[support])**4
        log_ratio=np.log1p(pulse/p0)
        rho=np.exp(log_ratio/gamma)
        temperature=np.exp((gamma-1)*log_ratio/gamma)
        velocity=(2*a0/(gamma-1))*np.expm1((gamma-1)*log_ratio/(2*gamma))
        pressure=p0+pulse;sound=a0*np.sqrt(temperature)
        riemann=velocity-2*sound/(gamma-1)+2*a0/(gamma-1)
        entropy=np.log(pressure/p0)-gamma*np.log(rho/rho0)
        if max(np.max(abs(riemann)),np.max(abs(entropy)))>2e-10:
            raise AssertionError('Proposed isentropic initialization identities failed')
        upper_dt=dt*((1+a0)/(10/64)+a0*ny+a0/(.25/16))
        rows.append(dict(intervals=[64,ny,16],extended_intervals=[64,3*ny//2,16],
                         dt=dt,steps=round(stop/dt),probe_j=7*ny//8,
                         compact_support_intervals=2*width/(1/ny),
                         background_c_upper_estimate=upper_dt,
                         background_nu_dt_over_dy2=.001*dt*ny**2,
                         riemann_residual=float(np.max(abs(riemann))),
                         isentropic_residual=float(np.max(abs(entropy)))))
    sample_dt=1e-5
    t=(np.arange(65536)-32768)*sample_dt
    signal=np.zeros_like(t);mask=abs(t)<width/a0
    signal[mask]=np.cos(.5*np.pi*a0*t[mask]/width)**4
    spectrum=abs(np.fft.rfft(signal))**2
    spectrum[1:-1]*=2
    frequency=np.fft.rfftfreq(len(t),sample_dt)
    f95=float(frequency[np.searchsorted(np.cumsum(spectrum),.95*np.sum(spectrum))])
    report=dict(status='proposed_parameters_reviewed_not_executed',
        normalization='dimensionless space/time; frequency is inverse dimensionless time, not Hz',
        background=dict(rho=rho0,p=p0,t=1.,u_outer=1.,v=0.,w=0.,gamma=gamma,mach=mach,
                        reynolds=1000,prandtl=.72,reference_temperature=273.15,
                        sutherland_constant=110.3,sound_speed=a0),
        pulse=dict(absolute_pressure_amplitude=amplitude,relative_amplitude=amplitude/p0,
                   center_y=y0,half_support=width,shape='cos^4(pi*(y-y0)/(2*width)) inside support',
                   state='isentropic positive-y simple wave, using log1p/expm1',
                   approximate_95_percent_temporal_energy_band=[0.,f95]),
        domain=[10.,1.,.25],extended_domain=[10.,1.5,.25],stop_time=stop,levels=rows,
        incident_support=incident,reflected_support=reflected,incident_window=incident_window,
        reflected_window=reflected_window,first_extended_top_return=reference_return,
        first_wall_to_probe_acoustic_transit=probe/a0,
        x_roi=[4.5,5.5],x_plateau=[3.,7.],x_support=[2.,8.],
        side_envelope_first_roi_transit_estimate=(4.5-3)/(1+a0),
        qualification='transit estimates are linear acoustic estimates, not rigorous finite-speed bounds for a discrete operator; incident/reference agreement must also be checked',
        proposed_gates=dict(nonreflecting_probe_return_ratio_max=.05,incident_energy_relative_difference_max=.01,
                            nonreflecting_refinement_ratio_absolute_difference_max=.01,
                            cpu_gpu_first_four_steps_max_scaled=2e-10,
                            other_two_mode_reflection='report, not forced equal to nonreflecting'),
        approval='requires user confirmation before any native physical window')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
