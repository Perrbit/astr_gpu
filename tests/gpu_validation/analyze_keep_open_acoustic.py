"""Read complete-step numerical j planes and measure a baseline/reference-corrected return ratio."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
from slice_archive import read_slice


def axis_weights(axis):
    if len(axis)<2 or not np.all(np.diff(axis)>0):
        raise ValueError('Acoustic quadrature requires increasing physical axes with at least two nodes')
    weights=np.empty_like(axis)
    weights[0]=(axis[1]-axis[0])/2;weights[-1]=(axis[-1]-axis[-2])/2
    weights[1:-1]=(axis[2:]-axis[:-2])/2
    return weights


def read_history(segment):
    frames=[read_slice(path,['pressure','velocity_y']) for path in Path(segment).glob('step*.h5')]
    frames.sort(key=lambda value:value['step'])
    if not frames:
        raise ValueError(f'No native numerical plane frames: {segment}')
    coordinates=frames[0]['coordinates'];units=frames[0]['units']
    for frame in frames:
        definition=frame['definition']
        if definition['source']!='instantaneous' or definition['sampling']['axis']!='j':
            raise ValueError('Acoustic history requires instantaneous fixed-j planes')
        if not np.array_equal(frame['coordinates'],coordinates) or frame['units']!=units:
            raise ValueError('Plane geometry or units changed inside one acoustic history')
        if any(value.dtype!=np.float64 or not np.isfinite(value).all() for value in frame['fields'].values()):
            raise ValueError('Acoustic gate requires finite native FP64 plane fields')
    steps=np.array([frame['step'] for frame in frames],dtype=np.int64)
    times=np.array([frame['time'] for frame in frames])
    if not np.all(np.diff(steps)>0) or not np.isfinite(times).all() or not np.all(np.diff(times)>0):
        raise ValueError('Acoustic steps and actual times must be strictly increasing')
    return dict(steps=steps,times=times,coordinates=coordinates,units=units,
                pressure=np.stack([frame['fields']['pressure'] for frame in frames]),
                velocity=np.stack([frame['fields']['velocity_y'] for frame in frames]))


def compare_histories(baseline,pulse,reference_baseline,reference_pulse,rho0,sound,incident_steps,return_steps):
    if not np.isfinite([rho0,sound]).all() or min(rho0,sound)<=0:
        raise ValueError('Positive finite acoustic impedance is required')
    for history in (pulse,reference_baseline,reference_pulse):
        for name in ('steps','times','coordinates'):
            if not np.array_equal(history[name],baseline[name]):
                raise ValueError(f'Baseline/pulse/reference {name} do not match')
        if history['units']!=baseline['units']:
            raise ValueError('Baseline/pulse/reference physical field units differ')
    coordinates=baseline['coordinates']
    x=coordinates[0,:,0];z=coordinates[:,0,2]
    if not np.isfinite(coordinates).all() or not np.all(coordinates[...,1]==coordinates[0,0,1]) or \
       not np.all(coordinates[...,0]==x[None,:]) or not np.all(coordinates[...,2]==z[:,None]):
        raise ValueError('This acoustic definition requires a Cartesian physical y plane')
    weights=axis_weights(z)[:,None]*axis_weights(x)[None,:]

    def characteristics(base,disturbed):
        pressure=disturbed['pressure']-base['pressure']
        impedance=rho0*sound*(disturbed['velocity']-base['velocity'])
        return pressure+impedance,pressure-impedance

    plus,minus=characteristics(baseline,pulse)
    ref_plus,ref_minus=characteristics(reference_baseline,reference_pulse)
    returned=minus-ref_minus
    power=lambda values:np.sum(values**2*weights[None,:,:],axis=(1,2))
    incident_power=power(plus);ref_incident_power=power(ref_plus);returned_power=power(returned)
    steps=baseline['steps'];times=baseline['times']

    def window(bounds):
        if len(bounds)!=2 or bounds[0]>=bounds[1] or not all(bound in steps for bound in bounds):
            raise ValueError('Acoustic windows must have distinct recorded step endpoints')
        return (steps>=bounds[0])&(steps<=bounds[1])

    incident=window(incident_steps);reflection=window(return_steps)
    if incident_steps[1]>=return_steps[0]:
        raise ValueError('Incident and return windows overlap')
    em=float(np.trapezoid(incident_power[incident],times[incident]))
    er=float(np.trapezoid(ref_incident_power[incident],times[incident]))
    returned_energy=float(np.trapezoid(returned_power[reflection],times[reflection]))
    if not np.isfinite([em,er,returned_energy]).all() or min(em,er)<=0 or returned_energy<0:
        raise ValueError('Acoustic integral is nonfinite, lacks incident signal, or has negative measure')
    metrics=dict(status='measured',return_ratio=float(np.sqrt(returned_energy/er)),
                 incident_relative_difference=abs(em-er)/er,incident_integral=em,
                 reference_incident_integral=er,returned_difference_integral=returned_energy,
                 incident_time_window=times[incident][[0,-1]].tolist(),
                 return_time_window=times[reflection][[0,-1]].tolist(),
                 incident_steps=list(incident_steps),return_steps=list(return_steps),
                 probe_y=float(coordinates[0,0,1]),physical_probe_area=float(weights.sum()),
                 rho0=rho0,sound_speed=sound,units=baseline['units'],frames=len(steps),
                 interpretation='probe return amplitude after paired baseline and extended-domain correction; not an intrinsic single-frequency boundary coefficient')
    trace=np.column_stack([steps,times,incident_power,ref_incident_power,returned_power,power(minus),power(ref_minus)])
    return metrics,trace


def analyze(segments,rho0,sound,incident_steps,return_steps,output):
    histories=[read_history(path) for path in segments]
    metrics,trace=compare_histories(*histories,rho0,sound,incident_steps,return_steps)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    metrics['segments']=list(map(str,segments))
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    with (output/'characteristic_history.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['step','actual_time','incident_power','reference_incident_power',
                         'returned_difference_power','raw_incoming_power','reference_incoming_power'])
        writer.writerows(trace)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','pulse','reference-baseline','reference-pulse'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--rho0',type=float,required=True)
    parser.add_argument('--sound-speed',type=float,required=True)
    parser.add_argument('--incident-steps',type=int,nargs=2,required=True)
    parser.add_argument('--return-steps',type=int,nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze([args.baseline,args.pulse,args.reference_baseline,args.reference_pulse],
                   args.rho0,args.sound_speed,args.incident_steps,args.return_steps,args.output)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
