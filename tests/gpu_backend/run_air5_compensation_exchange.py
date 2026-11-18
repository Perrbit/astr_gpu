"""Explicit AIR5 compensation phases, not the legacy all-periodic fast controller."""
import argparse, hashlib, json, os, re, shutil, subprocess
from pathlib import Path
import numpy as np
from run_air5_properties_components import read
ROOT=Path(__file__).resolve().parents[2]
FILES=['src/chemistry_compensation.F90','src/chemistry_air5_data.F90','src/chemistry_core.F90',
       'src/chemistry_properties.F90','src/chemistry_pure_vt_body.inc','src/chemistry_vibrational_difference_body.inc',
       'src_gpu/halo_transport.cu','src_gpu/halo_transport.h','src_gpu/halo_transport.F90',
       'src_gpu/device_halo.cu','src_gpu/device_state.h','src_gpu/device_state_internal.cuh',
       'src_gpu/chemistry_compensation_math.cuh','src_gpu/chemistry_transport.h','src_gpu/chemistry_transport.F90',
       'src_gpu/chemistry_properties_math.cuh','src_gpu/chemistry_properties.cu','src_gpu/chemistry_properties.h',
       'src_gpu/chemistry_properties.F90','src_gpu/chemistry_air5_data.h',
       'src_gpu/CMakeLists.txt','tests/gpu_backend/CMakeLists.txt','tests/gpu_backend/air5_compensation_oracle.F90',
       'tests/gpu_backend/air5_compensation_exchange_probe.cu','tests/gpu_backend/run_air5_compensation_exchange.py']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--properties-reference',type=Path,required=True);p.add_argument('--mpirun',type=Path,required=True)
    p.add_argument('--memcheck',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);hashes={}
    for name in FILES:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    binary=out/'candidate';shutil.copy2(a.candidate.resolve(),binary)
    status,values=read(a.properties_reference.resolve()/'cpu_all.bin');assert np.all(status[120]==0)
    values[120,:11].astype('<f8').tofile(out/'input.bin')
    receipt=dict(source_sha256=hashes,candidate_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (out/'source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    pattern=re.compile(r'AIR5_COMPENSATION_EXCHANGE_PASS rank=(\d+) mode=(\d+) math=(\d+) phases=(\d+) packets=(\d+) payloads=(\d+) cache_ratio=(\S+)')
    commands=[];runs=[]
    def execute(cmd,env,log):
        commands.append(dict(command=list(map(str,cmd)),environment={k:v for k,v in env.items() if k.startswith('OMPI_MCA_')},log=log))
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        with (out/log).open('w') as f:subprocess.run(list(map(str,cmd)),cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        return (out/log).read_text()
    for mode in ['pinned','device']:
        env=dict(os.environ,OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc')
        env.update(dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_opal_cuda_support='0') if mode=='pinned'
                   else dict(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1'))
        for topo in [(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1)]:
            ranks=int(np.prod(topo));tag=mode+'_'+''.join(map(str,topo))
            text=execute([a.mpirun.resolve(),'--bind-to','none','-np',ranks,binary,out/'input.bin',*topo,min(2,ranks),mode],env,tag+'.log')
            rows=[dict(rank=int(r),mode=int(m),math=int(n),phases=int(ph),packets=int(pk),payloads=int(pl),cache_budget_ratio=float(ratio))
                  for r,m,n,ph,pk,pl,ratio in pattern.findall(text)]
            assert len(rows)==ranks and {r['rank'] for r in rows}==set(range(ranks))
            assert all(r['math']==9 and r['phases']==40 and r['cache_budget_ratio']<=1 for r in rows)
            runs.append(dict(mode=mode,topology=topo,ranks=ranks,rank_contexts=4*ranks,observations=rows))
            (out/'completed_runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    env=dict(os.environ,OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    text=execute([a.mpirun.resolve(),'--bind-to','none','-np',2,a.memcheck.resolve(),'--tool','memcheck','--leak-check','full','--error-exitcode',99,
                  binary,out/'input.bin',2,1,1,2,'pinned'],env,'memcheck_pinned.log')
    assert text.count('ERROR SUMMARY: 0 errors')==2 and text.count('LEAK SUMMARY: 0 bytes leaked')==2
    report=dict(**receipt,runs=runs,rank_contexts=sum(r['rank_contexts'] for r in runs),
                phase_observations=sum(x['phases'] for r in runs for x in r['observations']),
                send_signatures_checked=sum(x['packets'] for r in runs for x in r['observations']),
                live_payloads_checked=sum(x['payloads'] for r in runs for x in r['observations']),
                max_cache_budget_ratio=max(x['cache_budget_ratio'] for r in runs for x in r['observations']),
                memcheck='pinned two ranks, zero errors/leaks; device MPI tool limitations are not waived',
                scope='per-axis math/real transport, general sequence and final halo/cache; legacy all-periodic NP1 fast path not invoked or repaired')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['rank_contexts','phase_observations','send_signatures_checked','live_payloads_checked','max_cache_budget_ratio','memcheck','scope']},indent=2))


if __name__=='__main__':main()
