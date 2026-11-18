"""Compute-node body for the proposed MB6/MB7 DCU consumer batch.

No scheduler operations. Execute only inside the separately approved allocation.
Python uses its standard library; the root-built comparator reads HDF5.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess


FINAL=Path('outdat/new/checkpoints/step000000000004')
CLOCK=re.compile(r'ASTR_TIMESTEP complete_step=(\d+)\s+time_start=\s*(\S+)\s+time_end=\s*(\S+)'
                 r'\s+dt_used=\s*(\S+)\s+dt_next=\s*(\S+)')


def clocks(case,first=1,count=4):
    rows=[tuple(map(float,m.groups())) for m in CLOCK.finditer((case/'run.log').read_text())]
    if len(rows)!=count or [r[0] for r in rows]!=list(range(first,first+count)):
        raise AssertionError(('complete-step clock identities',str(case)))
    if not all(math.isfinite(v) for row in rows for v in row):raise AssertionError('nonfinite clock')
    return rows


def archive_schedule(path):
    # Same current ABI as run_output_restart_validation.archive_schedule_payload:
    # the final48 bytes are intentionally different segment receipts, not clocks.
    data=path.read_bytes()
    if len(data)<56 or data[:8] not in (b'ASTROA02',b'ASTROA03'):
        raise AssertionError('archive control layout')
    return data[:-48]


def snapshot(case,rank,precision):
    path=case/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
    with path.open('rb') as stream:
        header=struct.unpack('<5i',stream.read(20))
        upper=header[:3];halo,components=header[3:]
        if min(upper)<1 or halo<0 or components!=5:raise AssertionError('terminal snapshot layout')
        values=[v[0] for v in struct.iter_unpack('<f' if precision=='fp32' else '<d',stream.read())]
    widths=[n+2*halo+1 for n in upper];nodes=widths[0]*widths[1]*widths[2]
    if len(values)!=nodes*5:raise AssertionError('terminal snapshot numerical kind/size')
    rho_min=math.inf;energy_min=math.inf
    for k in range(upper[2]+1):
        for j in range(upper[1]+1):
            for i in range(upper[0]+1):
                index=((k+halo)*widths[1]+j+halo)*widths[0]+i+halo
                q=[values[index+c*nodes] for c in range(5)]
                if not all(math.isfinite(v) for v in q) or q[0]<=0:raise AssertionError('nonpositive/nonfinite density/state')
                internal=q[4]-.5*((q[1]*q[1]+q[2]*q[2])+q[3]*q[3])/q[0]
                if not math.isfinite(internal) or internal<=0:raise AssertionError('nonpositive/nonfinite internal energy')
                rho_min=min(rho_min,q[0]);energy_min=min(energy_min,internal)
    return path,dict(density_min=rho_min,internal_energy_min=energy_min)


def command(argv,cwd,env,log):
    with log.open('x') as output:
        subprocess.run([str(v) for v in argv],cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                       stdout=output,stderr=subprocess.STDOUT,check=True)


def backend_identity(case,backend,ranks,transport):
    text=(case/'run.log').read_text()
    records=re.findall(r'ASTR_GPU_CPP rank=(\d+) backend=(\S+) device=(\d+)',text)
    if sorted(records)!=[(str(i),backend,str(i)) for i in range(ranks)]:
        raise AssertionError(('actual backend/device mapping',str(case),records))
    if 'ASTR_GPU_CPP transport='+transport+';' not in text:raise AssertionError('solver transport differs')
    compact=re.search(r'ASTR_CPP_COMPACT transport/batch/explicit=(\d+)\s+(\d+)\s+(\d+)',text)
    if not compact or int(compact[1])!=int(transport=='device'):
        raise AssertionError('private compact-interface transport differs')


def compare(comparator,mode,a,b,output):
    command([comparator,mode,a,b],output.parent,os.environ,output)


def only(paths,label):
    paths=list(paths)
    if len(paths)!=1:raise AssertionError((label,paths))
    return paths[0]


def inspect_family(template,paths,comparator,backend,report_root,original):
    family=json.loads((template.parent.parent/'manifest.json').read_text())['families'][template.name]
    dims=family['topology'];ranks=dims[0]*dims[1]*dims[2];precision=family['precision']
    full,quiet,tail=paths['full'],paths['quiet'],paths['tail']
    baseline=clocks(full)
    reference=[tuple(map(float,m.groups())) for m in CLOCK.finditer((template/'reference.log').read_text())]
    if baseline!=reference or clocks(quiet)!=baseline or clocks(tail,3,2)!=baseline[2:]:
        raise AssertionError('reference/observation/continuation clock differs')
    if clocks(paths['prefix'],count=2)!=baseline[:2]:raise AssertionError('prefix clocks differ')
    report_root.mkdir(parents=True,exist_ok=False)
    physical={key:[] for key in ('full','quiet','tail')}
    for name in ('full','quiet','tail'):
        backend_identity(paths[name],backend,ranks,family['transport'])
    for rank in range(ranks):
        base,metrics=snapshot(full,rank,precision);physical['full'].append(metrics)
        words=base.read_bytes()
        for name in ('quiet','tail'):
            file,metrics=snapshot(paths[name],rank,precision);physical[name].append(metrics)
            if file.read_bytes()!=words:
                raise AssertionError(('terminal q/halo words differ',name,rank))
    backend_identity(paths['prefix'],backend,ranks,family['transport'])
    mode='field32' if precision=='fp32' else 'field'
    compare(comparator,mode,template/'reference.h5',full,report_root/'reference.log')
    if precision=='fp64':compare(comparator,'field',original,full,report_root/'original_gpu.log')
    compare(comparator,'exact',full/FINAL/'state.h5',tail/FINAL/'state.h5',report_root/'state_exact.log')
    if 'EXACT datasets=15 ' not in (report_root/'state_exact.log').read_text():
        raise AssertionError('full exact state must include the current15 datasets')
    flags=struct.unpack('<8q',(full/FINAL/'control.bin').read_bytes()[8:72])
    mp=bool(re.search(r"gpu_mp_positivity\s*=\s*'flux'",(template/'datin/input.case').read_text()))
    if bool(flags[6]&2)!=mp or bool(flags[6]&32)!=mp:raise AssertionError('MP mode/budget restart identity')
    for name in ('control.bin','insitu_control.bin'):
        if (full/FINAL/name).read_bytes()!=(tail/FINAL/name).read_bytes():raise AssertionError('exact control: '+name)
    if archive_schedule(full/FINAL/'archives.bin')!=archive_schedule(tail/FINAL/'archives.bin'):
        raise AssertionError('exact archive clocks')
    if list((quiet/'outdat/new').glob('checkpoints/step*/state.h5')) or list((quiet/'outdat/new').rglob('data.h5')) or \
            list((quiet/'outdat/new/slice_products').rglob('step*.h5')):
        raise AssertionError('disabled output wrote a field/checkpoint')
    volume=only((full/'outdat/new/fields').glob('*/step000000000004/data.h5'),'volume4')
    resumed=only((tail/'outdat/new/fields').glob('*/step000000000004/data.h5'),'restored volume4')
    if sorted(p.parent.name for p in (full/'outdat/new/fields').glob('*/step*/data.h5'))!= \
            ['step000000000002','step000000000004']:
        raise AssertionError('completed-step volume inventory')
    compare(comparator,'exact',volume,resumed,report_root/'volume_exact.log')
    for axis in 'ijk':
        if sorted(p.name for p in (full/'outdat/new/slice_products').glob('plane-'+axis+'/*/segment*/step*.h5'))!= \
                ['step000000000000.h5','step000000000002.h5','step000000000004.h5']:
            raise AssertionError('initial/completed-step slice inventory')
        a=only((full/'outdat/new/slice_products').glob('plane-'+axis+'/*/segment*/step000000000004.h5'),'slice4')
        b=only((tail/'outdat/new/slice_products').glob('plane-'+axis+'/*/segment*/step000000000004.h5'),'restored slice4')
        compare(comparator,'exact',a,b,report_root/('slice_'+axis+'_exact.log'))
    if 'nonexact' in paths:
        other=paths['nonexact'];backend_identity(other,backend,ranks,family['transport'])
        if clocks(other,3,2)!=baseline[2:]:raise AssertionError('non-exact fixed clock differs')
        physical['nonexact']=[snapshot(other,r,precision)[1] for r in range(ranks)]
        compare(comparator,'field32',full/FINAL/'state.h5',other,report_root/'lean_physical_raw.log')
    report=dict(precision=precision,topology=family['topology'],backend=backend,clocks=baseline,
        terminal_q_halo_isolation=True,exact_state_controls_and_archive_schedule=True,
        exact_volume_and_slices=True,physical=physical,difference_limit=2e-10 if precision=='fp64' else None)
    report['MP_budget_identity']=int(flags[6])
    (report_root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def environment(transport,topology=None):
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    if 'UCX_MODULE_DIR' in os.environ:env['UCX_MODULE_DIR']=os.environ['UCX_MODULE_DIR']
    if 'TMPDIR' in env:env['OMPI_MCA_orte_tmpdir_base']=env['TMPDIR']
    env.update(OMPI_MCA_pml='ucx' if transport in ('device','device-aware') else 'ob1',
        OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',
        OMPI_MCA_opal_cuda_support='0',UCX_MEMTYPE_CACHE='n',OMP_NUM_THREADS='1')
    if topology:
        env.update(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_HALO_TRANSPORT=transport,
            ASTR_GPU_COMPACT_TRANSPORT='device-aware' if transport=='device' else 'pinned',
            ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_COMPACT_SYNC='explicit',ASTR_GPU_FILTER_WORKSPACE='scalar',
            ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='4')
    return env


def assignment(text,key,value):
    text,count=re.subn(r'(?m)^(\s*'+re.escape(key)+r'\s*=).*$',lambda m:m[1]+' '+value,text)
    if count!=1:raise AssertionError('one fixture assignment required: '+key)
    return text


def prepare_case(template,target,steps,writing,restore=None,exact=True):
    shutil.copytree(template/'datin',target/'datin',copy_function=shutil.copyfile)
    (target/'outdat/new').mkdir(parents=True)
    path=target/'datin/input.run'
    path.write_text(assignment(path.read_text(),'max_steps',str(steps)))
    path=target/'datin/input.output';text=path.read_text()
    text=assignment(text,'restore_directory',repr(str(restore or '')))
    text=assignment(text,'exact_restart','.true.' if exact else '.false.')
    if not writing:text=text.replace('enabled = .true.','enabled = .false.')
    path.write_text(text)
    catalog=(template/'input.products.jsonl').read_text() if writing else '{"catalog_schema":"astr-insitu-products-v1"}\n'
    (target/'input.products.jsonl').write_text(catalog)


def components(build,validation,root,mpiexec):
    rows=[('sampling',1,['1,1,1','pinned'],'pinned'),
          ('sampling',2,['2,1,1','device-aware','sixth'],'device-aware'),
          ('sampling',4,['2,2,1','pinned','legacy'],'pinned'),
          ('vectors',2,['2,1,1','pinned'],'pinned'),
          ('vectors',2,['1,2,1','device-aware'],'device-aware'),
          ('vectors',2,['1,1,2','pinned'],'pinned'),
          ('owned',1,['1,1,1','pinned'],'pinned'),
          ('owned',2,['1,1,2','device-aware'],'device-aware'),
          ('owned',4,['2,2,1','pinned'],'pinned'),
          ('wall',1,['1,1,1','pinned','legacy'],'pinned'),
          ('wall',2,['1,2,1','device-aware','sixth'],'device-aware'),
          ('wall',4,['2,2,1','pinned','curve'],'pinned'),
          ('sbli',2,['pinned'],'pinned'),('sbli',2,['device-aware'],'device-aware'),
          ('profile',1,[],'pinned'),('profile',2,[],'pinned'),('probes',2,[],'pinned')]
    targets=dict(sampling='astr_backend_insitu_sampling',vectors='astr_backend_insitu_vectors',
        owned='astr_backend_owned_vector',wall='astr_backend_insitu_wall',sbli='astr_backend_sbli_wall',
        profile='astr_backend_profile_wall',probes='astr_backend_sparse_probe')
    records=[]
    def run(precision,name,ranks,args,transport):
        log=root/('%02d_'%len(records)+precision+'_'+name+'.log')
        print('COMPONENT',log.stem,flush=True)
        command([mpiexec,'--bind-to','none','-n',ranks,build/('build_'+precision)/'bin'/name,*args],
                root,environment(transport),log)
        records.append(dict(log=log.name,ranks=ranks,precision=precision,transport=transport))
    for name,ranks,args,transport in rows:run('fp64',targets[name],ranks,args,transport)
    for precision in ('fp64','fp32'):
        packet=validation/'mp'/precision
        run(precision,'astr_backend_mp_positivity_math',1,[packet/'input.bin',packet/'reference.bin'],'pinned')
        for key,ranks,case,transport in [('np1_x',1,'0','pinned'),('np2_x',2,'1','device'),
                                        ('np2_y',2,'1','pinned'),('np2_z',2,'0','device')]:
            run(precision,'astr_backend_mp_positivity_stage',ranks,[packet/key,case,transport],transport)
    for name in ('astr_backend_profile_wall','astr_backend_sparse_probe'):run('fp32',name,2,[],'pinned')
    return records


def execute(build,validation,root,mpiexec):
    root.mkdir(parents=True,exist_ok=False);(root/'components').mkdir()
    manifest=json.loads((validation/'manifest.json').read_text())
    summary=dict(scope='numerical/communication short cases; no renderer or timing',components=[],families={})
    summary['components']=components(build,validation,root/'components',mpiexec)
    for name,family in manifest['families'].items():
        print('ORDINARY',name,flush=True)
        template=validation/'families'/name;paths={};dims=family['topology'];ranks=dims[0]*dims[1]*dims[2]
        binary=build/('build_'+family['precision'])/'bin/astr'
        for phase,steps,writing in [('full',4,True),('quiet',4,False),('prefix',2,True),('tail',4,True)]:
            target=root/name/phase
            restore=paths['prefix']/'outdat/new/checkpoints/step000000000002' if phase=='tail' else None
            prepare_case(template,target,steps,writing,restore)
            command([mpiexec,'--bind-to','none','-n',ranks,binary,'run','datin/input.case'],
                    target,environment(family['transport'],family['topology']),target/'run.log')
            paths[phase]=target
        if name=='fp32_filter':
            for phase,steps,restore in [('lean_prefix',2,None),
                ('nonexact',4,root/name/'lean_prefix/outdat/new/checkpoints/step000000000002')]:
                target=root/name/phase;prepare_case(template,target,steps,True,restore,exact=False)
                command([mpiexec,'--bind-to','none','-n',ranks,binary,'run','datin/input.case'],
                        target,environment(family['transport'],family['topology']),target/'run.log')
                paths[phase]=target
        comparator=build/('build_'+family['precision'])/'bin/astr_backend_compare'
        summary['families'][name]=inspect_family(template,paths,comparator,'gpufusion',root/name/'analysis',
            validation/'original_fp64_incoming.h5')
        (root/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('build-root','validation','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    execute(args.build_root.resolve(),args.validation.resolve(),args.output.resolve(),args.mpiexec.resolve())


if __name__=='__main__':main()
