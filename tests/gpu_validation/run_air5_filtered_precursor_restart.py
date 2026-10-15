#!/usr/bin/env python3
"""Filtered precursor restart: short gate then an explicitly bounded GPU window."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from air5_radau_reference import Air5RadauReference
from check_air5_compensation_checkpoint import read
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics
from run_air5_characteristic_backend_gate import fields
from run_air5_sbli_preflight import ROOT, environment, set_value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, restart_from=None, end_time=1e-5, allow_interrupted=False):
    output = output.resolve()
    source = (restart_from.resolve() if restart_from else
              ROOT/'tests/gpu_validation/out/air5_mach4_precursor_10us_20260928_v2')
    checkpoint = source/'outdat/flowfield.h5'
    original = digest(checkpoint)
    _, _, (step, time), _ = read(checkpoint)
    if restart_from is None and (step != 39000 or abs(time-9.75e-6) > 1e-15):
        raise ValueError('incorrect restart checkpoint')
    if not np.isfinite(end_time) or not time < end_time <= 50e-6:
        raise ValueError('end time must exceed checkpoint time and be at most 50 us')
    window_updates=round((end_time-time)/2.5e-10)
    if window_updates<2 or abs(time+window_updates*2.5e-10-end_time)>1e-15:
        raise ValueError('target time does not match the fixed time step')
    output.mkdir(parents=True, exist_ok=False)
    executable = output/'astr'
    shutil.copy2(ROOT/'build_gpu_probe/bin/astr', executable)
    matrix = json.loads((ROOT/'tests/gpu_validation/out/air5_compensated_filter_20260928/result.json').read_text())
    if matrix['executable_sha256'] != digest(executable) or matrix['status'] != 'short-matrix-pass-not-production-admission':
        raise ValueError('executable does not match the qualified filter matrix')
    previous = json.loads((source/'launch.json').read_text())
    if restart_from is not None:
        previous_result=json.loads((source.parent/'result.json').read_text())
        admitted=previous_result.get('stages',{}).get(source.name,{}).get('status')=='passed'
        if allow_interrupted:
            admitted=admitted or (previous_result.get('status')=='user-stopped' and
                previous_result.get('stages',{}).get('short',{}).get('status')=='passed')
        if (not admitted or
            previous['executable_sha256']!=digest(executable) or not previous.get('lfilter')):
            raise ValueError('filtered continuation requires a qualified or explicitly interrupted window')
    report = dict(status='running', checkpoint=str(checkpoint), checkpoint_sha256=original,
                  executable_sha256=digest(executable), start_step=step, start_time=time,
                  end_time=end_time,allow_interrupted=allow_interrupted,production_ready=False, stages={})
    def save():
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for name, updates, stride in [('short',2,1),('window',window_updates,100 if window_updates>1000 else 10)]:
            case = output/name
            shutil.copytree(source/'datin',case/'datin')
            (case/'outdat').mkdir()
            (case/'validation').mkdir()
            for filename in ('flowfield.h5','auxiliary.txt'):
                shutil.copy2(source/'outdat'/filename,case/'outdat'/filename)
            set_value(case/'datin/input.air5_c4','lrestar','t')
            set_value(case/'datin/input.air5_c4',
                      'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                      'f,t,t,f,f,f,t,t,t')
            last = step+updates-1
            set_value(case/'datin/controller','deltat','2.5d-10')
            set_value(case/'datin/controller','maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                      f'{last},{min(updates-1,1000)},1000000,1000000,10,1000000')
            env = environment('2,1,1')
            env.update(previous['environment'])
            env.update(ASTR_AIR5_FILTER_VALIDATION='on',ASTR_AIR5_TOP_RESTART_VALIDATION='on',
                       ASTR_AIR5_COMPENSATION_RESTART='restore',ASTR_AIR5_FLOW_MONITOR_STRIDE=str(stride),
                       ASTR_VALIDATION_RHS_STEP=str(last),ASTR_VALIDATION_RHS_STEP_SECONDARY=str(step))
            command = ['timeout','--kill-after=10s','86400s',previous['command'][3],
                       '--oversubscribe','-np','2',str(executable),'run','datin/input.air5_c4']
            launch = dict(command=command, environment={k:v for k,v in env.items() if k.startswith('ASTR_')},
                          executable_sha256=digest(executable), checkpoint_sha256=original,
                          updates=updates,dt=2.5e-10,start_step=step,start_time=time,
                          end_time=time+updates*2.5e-10,lfilter=True,scope='bounded filtered precursor')
            (case/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
            report['stages'][name] = dict(status='running',updates=updates)
            save()
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            log = (case/'run.log').read_text()
            cfl = [float(x.replace('D','E')) for x in re.findall(r'current CFL:\s*([\d.EeDd+\-]+)',log)]
            if 'The job is done!' not in log or 'ieee_invalid' in log.lower() or not cfl:
                raise ValueError('unclean completion or missing CFL')
            if not np.isfinite(cfl).all() or max(cfl)>=1:
                raise ValueError('CFL gate failed')
            state = state_gate(case,Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'))
            for label,q in fields(case,(step,last)).items():
                row=metrics(q,(0,0,0))
                if row['transport_sum_violations'] or row['min_species_density']<0:
                    raise ValueError('composition gate failed: '+label)
            for filename in ('air5_profiles.dat','air5_wall.dat'):
                monitor=np.loadtxt(case/'monitor'/filename)
                if not monitor.size or not np.isfinite(monitor).all():
                    raise ValueError('invalid monitor '+filename)
            read(case/'outdat/flowfield.h5')
            if digest(checkpoint)!=original:
                raise ValueError('original checkpoint changed')
            report['stages'][name].update(status='passed',max_cfl=max(cfl),state=state)
            save()
        report['status']='bounded-window-complete-not-steady-admission'
        save()
    except BaseException as error:
        report.update(status='failed',error=str(error))
        save()
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--restart-from',type=Path)
    parser.add_argument('--end-time',type=float,default=1e-5)
    parser.add_argument('--allow-interrupted',action='store_true')
    args=parser.parse_args()
    run(args.output,args.restart_from,args.end_time,args.allow_interrupted)
