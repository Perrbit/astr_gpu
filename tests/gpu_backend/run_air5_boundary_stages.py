"""MPI state-boundary scheduling with frozen original GPU stages, no CFD update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from run_air5_boundary_components import BASE, ROOT, Air5RadauReference
from run_air5_flux_components import extract
from run_air5_properties_components import read as read_properties

FILES = BASE + ['src/chemistry_pure_vt_body.inc', 'src/chemistry_vibrational_difference_body.inc',
    'src_gpu/chemistry_core_gpu.cuf', 'src_gpu/chemistry_flow_state_gpu.cuf', 'src_gpu/chemistry_boundary_gpu.cuf',
    'src_gpu/chemistry_boundary.F90', 'src_gpu/chemistry_boundary.cu', 'src_gpu/chemistry_boundary.h',
    'src_gpu/chemistry_boundary_math.cuh', 'src_gpu/chemistry_properties_math.cuh', 'src_gpu/chemistry_air5_data.h',
    'src_gpu/chemistry_characteristic_math.cuh', 'src_gpu/CMakeLists.txt', 'tests/gpu_backend/CMakeLists.txt',
    'tests/gpu_backend/air5_boundary_stage_probe.F90', 'tests/gpu_backend/run_air5_boundary_stages.py',
    'chemMech/air5_kimjo12.json']
CASES = np.array([[0,1,1,0], [0,0,0,1], [1,1,0,0], [1,0,1,1], [2,1,1,0], [2,0,0,1],
                  [2,1,1,1], [0,1,1,1], [3,1,1,0], [4,0,0,1], [5,1,1,0]], dtype='<i4')
ENV = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
           OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')


def run(command, work, log):
    with (work/log).open('w') as f:
        subprocess.run(list(map(str,command)), cwd=work, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)


def build_reference(out, compiler):
    work = out/'old_gpu_build';work.mkdir();source=out/'source'
    text=(source/'src_gpu/chemistry_boundary_gpu.cuf').read_text()
    names=['record_hbl_status','build_air5_hbl_outflow_state_gpu','air5_hbl_outflow_boundary_kernel',
           'air5_hbl_wall_boundary_kernel','air5_hbl_farfield_boundary_kernel','air5_hbl_inflow_boundary_kernel',
           'check_hbl_status','apply_air5_hbl_boundary_gpu']
    (work/'frozen_hbl_functions.inc').write_text('\n'.join(extract(text,name) for name in names))
    names=['build_air5_pressure_outlet_state_gpu','air5_postshock_boundary_kernel','apply_air5_postshock_boundary_gpu']
    (work/'frozen_postshock_functions.inc').write_text('\n'.join(extract(text,name) for name in names))
    flags=[compiler,'-O2','-Mpreprocess','-Mextend','-cuda','-gpu=rdc,maxregcount:128','-D_CUDA',
           '-DASTR_AIR5_STAGE_REFERENCE','-I'+str(work),'-I'+str(source/'src')]
    objects=[];commands=[]
    for i,name in enumerate(BASE+['src_gpu/chemistry_core_gpu.cuf','src_gpu/chemistry_flow_state_gpu.cuf',
                                'tests/gpu_backend/air5_boundary_stage_probe.F90']):
        path=source/name
        if name.endswith('stage_probe.F90'):
            path=work/'reference.cuf';shutil.copy2(source/name,path)
        obj=work/f'unit{i}.o';cmd=flags+['-c',path,'-o',obj]
        run(cmd,work,f'build{i}.log');objects.append(obj);commands.append(list(map(str,cmd)))
    cmd=flags+objects+['-o',work/'probe'];run(cmd,work,'link.log');commands.append(list(map(str,cmd)))
    return work/'probe',commands


def fixture(directory, topology, properties):
    directory.mkdir();s,p=read_properties(properties/'cpu_all.bin');index=120
    assert np.all(s[index]==0)
    base=p[index,:11];assert np.all(base[5:10]>0)
    model=Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    y=base[5:10]/base[0];tw=1000.;pressure=p[index,22];rho=pressure/(np.dot(y,model.gas_constant)*tw)
    wall=np.zeros(11);wall[0]=rho;wall[5:10]=rho*y;wall[10]=model.ev_from_tv(wall[5:10],tw)
    wall[4]=model.q5_from_state(rho,wall[1:4],wall[5:10],wall[10],tw)
    im,jm,km,h=8,7,6,5;shape=(im+2*h+1,jm+2*h+1,km+2*h+1,11)
    physical=(im+1,jm+1,km+1);incident=np.column_stack([base,base*.875])
    info=[]
    for rank in range(int(np.prod(topology))):
        coord=np.unravel_index(rank,topology)
        gi=coord[0]*im+np.arange(im+1);gj=coord[1]*jm+np.arange(jm+1);gk=coord[2]*km+np.arange(km+1)
        mk=gk%(topology[2]*km)
        scale=1.+(gi[:,None,None]+2*gj[None,:,None]+3*mk[None,None,:])/1024.
        q=(-777.-np.arange(np.prod(shape))/2**20).reshape(shape,order='F')
        q[h:h+im+1,h:h+jm+1,h:h+km+1,:]=scale[:,:,:,None]*base
        coords=np.empty(physical+(3,));coords[:,:,:,0]=.125*gi[:,None,None]
        coords[:,:,:,1]=.125*gj[None,:,None];coords[:,:,:,2]=.125*gk[None,None,:]
        fraction=gj/(topology[1]*jm)
        profile=(1.-fraction[:,None])*wall+fraction[:,None]*base
        carry=np.empty(physical+(11,))
        carry[:]=np.ldexp(1.+np.arange(carry.size).reshape(carry.shape,order='F')/65536.,-90)
        file=directory/f'input.rank{rank:04d}.bin'
        with file.open('wb') as f:
            np.array([im,jm,km,h,len(CASES)],'<i4').tofile(f)
            for a in [q,coords,profile,base,incident,np.array([tw,1.,.125*im*topology[0]/2.,pressure]),carry]:
                np.asarray(a,dtype='<f8').ravel(order='F').tofile(f)
            CASES.T.ravel(order='F').tofile(f)
        info.append(dict(rank=rank,coordinate=list(map(int,coord)),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    return dict(topology=topology,property_row=index,temperature=float(p[index,15]),tv=float(p[index,21]),
                cases=CASES.tolist(),inputs=info,scope='manufactured states; incident selector targets are not a physical shock')


def read(path):
    with path.open('rb') as f:
        header=np.fromfile(f,'<i4',11);im,jm,km,h,n=map(int,header[:5])
        qshape=(im+2*h+1,jm+2*h+1,km+2*h+1,11);physical=(im+1,jm+1,km+1)
        cases=[]
        for _ in range(n):
            cfg=np.fromfile(f,'<i4',4);q=np.fromfile(f,'<f8',int(np.prod(qshape))).reshape(qshape,order='F')
            carry=np.fromfile(f,'<f8',int(np.prod(physical))*11).reshape(physical+(11,),order='F') if cfg[2] else None
            cache=np.fromfile(f,'<f8',int(np.prod(physical))*12).reshape(physical+(12,),order='F')
            cases.append((cfg,q,carry,cache))
        assert f.read()==b''
    return header,cases


def budget(new, ref):
    tolerance=1e-13*np.maximum(1.,np.max(abs(ref),axis=-1))[...,None]+5e-12*abs(ref)
    ratio=float(np.max(abs(new-ref)/tolerance));assert np.isfinite(new).all() and ratio<=1.,ratio
    return ratio


def compare(new,ref,initial_path):
    nh,nc=read(new);rh,rc=read(ref);assert np.array_equal(nh,rh)
    im,jm,km,h=map(int,nh[:4]);lower,upper=nh[5:8],nh[8:11]
    with initial_path.open('rb') as f:
        np.fromfile(f,'<i4',5)
        initial=np.fromfile(f,'<f8',(im+2*h+1)*(jm+2*h+1)*(km+2*h+1)*11).reshape(nc[0][1].shape,order='F')
        coords=np.fromfile(f,'<f8',(im+1)*(jm+1)*(km+1)*3).reshape((im+1,jm+1,km+1,3),order='F')
        profile=np.fromfile(f,'<f8',(jm+1)*11).reshape((jm+1,11),order='F')
        farfield=np.fromfile(f,'<f8',11);incident=np.fromfile(f,'<f8',22).reshape((11,2),order='F')
        wall_temperature,origin,incident_x,pressure_target=np.fromfile(f,'<f8',4)
    results=[]
    for (cfg,q,carry,cache),(rf,rq,rcarry,rcache) in zip(nc,rc):
        assert np.array_equal(cfg,rf);mode=int(cfg[0]);qp=q[h:h+im+1,h:h+jm+1,h:h+km+1]
        assert np.all(qp[:,:,:,0]>0) and np.all(qp[:,:,:,5:10]>=0)
        assert np.max(abs(np.sum(qp[:,:,:,5:10],axis=-1)-qp[:,:,:,0]))<=128*np.finfo(float).eps*max(1.,float(np.max(qp[:,:,:,0])))
        qratio=budget(q,rq);cache_ratios={}
        for name,a,b in [('rho',0,1),('velocity',1,4),('temperature',4,5),('tv',5,6),('pressure',6,7),('species',7,12)]:
            cache_ratios[name]=budget(cache[:,:,:,a:b],rcache[:,:,:,a:b])
        untouched=np.ones(q.shape[:3],bool)
        if upper[0]:untouched[h+im:,h:h+jm+1,h:h+km+1]=False
        if lower[0]:untouched[:h+1,h:h+jm+1,h:h+km+1]=False
        if mode<=2:
            if lower[1]:untouched[h:h+im+1,:h+1,h:h+km+1]=False
            if upper[1]:untouched[h:h+im+1,h+jm+(1 if mode==2 else 0):,h:h+km+1]=False
        assert np.array_equal(q[untouched].view('u8'),initial[untouched].view('u8'))
        assert np.array_equal(rq[untouched].view('u8'),initial[untouched].view('u8'))
        if carry is not None:
            assert np.array_equal(carry.view('u8'),rcarry.view('u8'))
        for state in [q,rq]:
            if lower[0]:
                target=state[:h+1,h:h+jm+1,h:h+km+1]
                expected=np.broadcast_to(profile[None,:,None,:] if mode<=2 else farfield,target.shape)
                assert np.array_equal(target.view('u8'),expected.view('u8'))
            if mode==3 and upper[0]:
                target=state[h+im:,h:h+jm+1,h:h+km+1]
                assert np.array_equal(target.view('u8'),np.broadcast_to(incident[:,1],target.shape).view('u8'))
            if mode==1 and upper[1]:
                selected=np.where((coords[:,jm,:,0]<incident_x)[:,:,None],incident[:,0],incident[:,1])
                target=state[h:h+im+1,h+jm:,h:h+km+1]
                expected=np.broadcast_to(selected[:,None,:,:],target.shape).copy()
                if lower[0]:expected[0,0,:,:]=profile[-1]
                assert np.array_equal(target.view('u8'),expected.view('u8'))
            if mode==2 and upper[1]:
                target=state[h:h+im+1,h+jm+1:,h:h+km+1]
                expected=np.broadcast_to(initial[h:h+im+1,h+jm:h+jm+1,h:h+km+1],target.shape)
                assert np.array_equal(target.view('u8'),expected.view('u8'))
        if mode<=2 and lower[1]:
            for field in [4,5]:
                temperature=cache[:,0,:,field]
                assert np.max(abs(temperature-wall_temperature))<=1e-13*max(1.,wall_temperature)+5e-12*wall_temperature
        if mode==5 and upper[0]:
            assert np.max(abs(cache[im,:,:,6]-pressure_target))<=1e-13*max(1.,pressure_target)+5e-12*pressure_target
        results.append(dict(mode=mode,preserve=int(cfg[1]),compensated=int(cfg[2]),debug=int(cfg[3]),
                            q_budget_ratio=qratio,cache_budget_ratios=cache_ratios,untouched_words_exact=True,
                            carry_words_exact=True,copy_targets_exact=True,wall_and_pressure_constraints=True))
    return results


def check_aliases(directory,topology):
    records=[read(directory/f'rank{rank:04d}.bin') for rank in range(int(np.prod(topology)))];pairs=0
    for rank,(header,cases) in enumerate(records):
        im,jm,km,h=map(int,header[:4]);coord=list(np.unravel_index(rank,topology));last=[im,jm,km]
        for axis in range(3):
            peer=coord.copy();peer[axis]+=1
            if peer[axis]==topology[axis]:
                if axis!=2:continue
                peer[axis]=0
            neighbor=records[np.ravel_multi_index(tuple(peer),topology)][1]
            for (_,q,_,cache),(_,other,_,other_cache) in zip(cases,neighbor):
                a=q[h:h+im+1,h:h+jm+1,h:h+km+1];b=other[h:h+im+1,h:h+jm+1,h:h+km+1]
                assert np.array_equal(np.take(a,last[axis],axis=axis).view('u8'),np.take(b,0,axis=axis).view('u8'))
                assert np.array_equal(np.take(cache,last[axis],axis=axis).view('u8'),np.take(other_cache,0,axis=axis).view('u8'))
                pairs+=1
    return dict(q_and_physical_cache_pairs_exact=pairs,carry='not asserted: initial interior carry intentionally differs between ranks')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True);p.add_argument('--mpirun',type=Path,required=True)
    p.add_argument('--properties-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--memcheck',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in FILES:
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    candidate=out/'candidate';shutil.copy2(args.candidate.resolve(),candidate)
    (out/'source_receipt.json').write_text(json.dumps(dict(source_sha256=hashes,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest()),indent=2)+'\n')
    original,build_commands=build_reference(out,args.compiler.resolve());commands=[];runs=[]
    (out/'build_commands.json').write_text(json.dumps(build_commands,indent=2)+'\n')
    for topology in [(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1)]:
        tag=''.join(map(str,topology));work=out/tag;work.mkdir();info=fixture(work/'inputs',topology,args.properties_reference.resolve())
        (work/'fixture.json').write_text(json.dumps(info,indent=2)+'\n');ranks=int(np.prod(topology))
        for backend,binary in [('old_gpu',original),('cpp',candidate)]:
            directory=work/backend;directory.mkdir()
            cmd=[args.mpirun.resolve(),'--bind-to','none','-np',ranks,binary,work/'inputs',directory,*topology,min(2,ranks)]
            commands.append(list(map(str,cmd)));run(cmd,work,backend+'.log')
        result=[compare(work/'cpp'/f'rank{rank:04d}.bin',work/'old_gpu'/f'rank{rank:04d}.bin',
                        work/'inputs'/f'input.rank{rank:04d}.bin') for rank in range(ranks)]
        text=(work/'cpp.log').read_text();assert text.count('AIR5_BOUNDARY_STAGE_FAILURE_PASS')==4*ranks
        aliases={backend:check_aliases(work/backend,topology) for backend in ['old_gpu','cpp']}
        report=dict(topology=topology,ranks=ranks,rank_cases=ranks*len(CASES),checks=result,global_failure_rank_checks=4*ranks,aliases=aliases)
        (work/'report.json').write_text(json.dumps(report,indent=2)+'\n');runs.append(report)
    work=out/'211';directory=work/'memcheck';directory.mkdir()
    cmd=[args.mpirun.resolve(),'--bind-to','none','-np',2,args.memcheck.resolve(),'--tool','memcheck','--leak-check','full','--error-exitcode',99,
         candidate,work/'inputs',directory,2,1,1,2];commands.append(list(map(str,cmd)));run(cmd,work,'memcheck.log')
    text=(work/'memcheck.log').read_text();assert text.count('ERROR SUMMARY: 0 errors')==2 and text.count('LEAK SUMMARY: 0 bytes leaked')==2
    for rank in range(2):compare(directory/f'rank{rank:04d}.bin',work/'old_gpu'/f'rank{rank:04d}.bin',work/'inputs'/f'input.rank{rank:04d}.bin')
    report=dict(source_sha256=hashes,commands=commands,runs=runs,total_rank_cases=sum(r['rank_cases'] for r in runs),
                global_failure_rank_checks=sum(r['global_failure_rank_checks'] for r in runs),memcheck='two ranks, zero errors/leaks',
                scope='MPI state boundary sequence and explicit physical primitive refresh; no characteristic RHS/halo/CFD')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(total_rank_cases=report['total_rank_cases'],global_failure_rank_checks=report['global_failure_rank_checks'],
                         max_q_budget_ratio=max(c['q_budget_ratio'] for r in runs for rank in r['checks'] for c in rank),memcheck=report['memcheck']),indent=2))


if __name__=='__main__':main()
