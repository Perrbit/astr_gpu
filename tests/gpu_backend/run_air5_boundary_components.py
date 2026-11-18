"""Original pressure-state outlet and isothermal-wall operators, not NSCBC RHS."""
import argparse,hashlib,json,math
from pathlib import Path
import shutil,subprocess,sys
import numpy as np
from run_air5_flux_components import extract
from run_air5_properties_components import read as read_properties
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from air5_radau_reference import Air5RadauReference

BASE=['src/chemistry_air5_data.F90','src/chemistry_core.F90','src/chemistry_properties.F90']
FILES=BASE+['src/chemistry_pure_vt_body.inc','src/chemistry_vibrational_difference_body.inc','src/chemistry_boundary_state.F90',
 'src_gpu/chemistry_core_gpu.cuf','src_gpu/chemistry_flow_state_gpu.cuf','src_gpu/chemistry_boundary_gpu.cuf',
 'src_gpu/chemistry_boundary_math.cuh','src_gpu/chemistry_boundary.cu','src_gpu/chemistry_boundary.F90','src_gpu/chemistry_boundary.h',
 'src_gpu/chemistry_properties_math.cuh','src_gpu/chemistry_air5_data.h','src_gpu/CMakeLists.txt','tests/gpu_backend/CMakeLists.txt',
 'tests/gpu_backend/air5_boundary_probe.cu','tests/gpu_backend/air5_boundary_reference.F90',
 'tests/gpu_backend/run_air5_boundary_components.py','chemMech/air5_kimjo12.json']
def run(cmd,out,log):
 with (out/log).open('w') as f:subprocess.run(list(map(str,cmd)),cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True)
def build(out,compiler):
 source=out/'source';commands={}
 for backend in ['cpu','old_gpu']:
  gpu=backend=='old_gpu';work=out/backend;work.mkdir();objects=[];commands[backend]=[]
  if gpu:
   text=(source/'src_gpu/chemistry_boundary_gpu.cuf').read_text();names=['build_air5_pressure_outlet_state_gpu','record_hbl_status','air5_hbl_wall_boundary_kernel','build_air5_hbl_outflow_state_gpu','get_air5_top_target_gpu']
  else:
   text=(source/'src/chemistry_boundary_state.F90').read_text();names=['build_air5_pressure_outlet_state','build_air5_hbl_wall_state','build_air5_hbl_outflow_state','build_air5_hbl_similarity_farfield_state']
  (work/'frozen_boundary_functions.inc').write_text('\n'.join(extract(text,name) for name in names))
  flags=[str(compiler),'-O2','-Mpreprocess','-Mextend','-I'+str(work),'-I'+str(source/'src')]
  if gpu:flags += ['-cuda','-gpu=rdc,maxregcount:128','-D_CUDA','-DASTR_AIR5_REFERENCE_GPU']
  files=BASE+(['src_gpu/chemistry_core_gpu.cuf','src_gpu/chemistry_flow_state_gpu.cuf'] if gpu else [])+['tests/gpu_backend/air5_boundary_reference.F90']
  for name in files:
   path=source/name
   if gpu and name.endswith('air5_boundary_reference.F90'):path=work/'reference.cuf';shutil.copy2(source/name,path)
   obj=work/(Path(name).stem+'.o');objects.append(obj);cmd=flags+['-c',path,'-o',obj];commands[backend].append(list(map(str,cmd)));run(cmd,work,obj.stem+'.log')
  cmd=flags+objects+['-o',work/'probe'];commands[backend].append(list(map(str,cmd)));run(cmd,work,'link.log')
 return commands
def read(path):
 with path.open('rb') as f:
  n=int(np.fromfile(f,'<i4',1)[0]);s=np.fromfile(f,'<i4',n);v=np.fromfile(f,'<f8').reshape(n,11)
 return s,v
def inputs(reference):
 statuses,p=read_properties(reference/'cpu_all.bin');valid=np.flatnonzero(np.all(statuses==0,axis=1));valid=valid[np.linspace(0,len(valid)-1,24,dtype=int)]
 model=Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json');rows=[];labels=[];tags=[]
 for index in valid:
  q=p[index,:11];rho=q[0];y=q[5:10]/rho;gamma=1.+(y@model.gas_constant)/(y@model.cv_tr);c=math.sqrt(gamma*p[index,22]/rho)
  for preserve in [0,1]:
   for wall_t in [600.,300.,8000.,299.,8001.]:
    x=np.zeros(32);x[:11]=q;x[11:22]=1.05*q;x[22]=wall_t;x[23]=preserve;x[31]=1;rows.append(x);labels.append(f'wall-{index}-{preserve}-{wall_t}');tags.append(dict(kind='wall',mach=0))
   for mach in [.2,.8,1.2,2.,0.,-.2]:
    a=q.copy();old=(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/(2*rho);a[1]=rho*mach*c
    a[4]+=((a[1]*a[1]+a[2]*a[2]+a[3]*a[3])/(2*rho))-old
    for factor in [1.,.95,1.05]:
     x=np.zeros(32);x[:11]=a;x[22]=p[index,22]*factor;x[23]=preserve;rows.append(x);labels.append(f'outlet-{index}-{preserve}-{mach}-{factor}');tags.append(dict(kind='outlet',mach=mach))
 first=valid[0];one=p[first,:11];y_one=one[5:10]/one[0]
 second=next(i for i in valid if np.max(abs(p[i,5:10]/p[i,0]-y_one))>1e-3)
 two=p[second,:11]*(1.05*p[first,22]/p[second,22])
 for preserve in [0,1]:
  x=np.zeros(32);x[:11]=one;x[11:22]=two;x[22]=600.;x[23]=preserve;x[31]=1
  rows.append(x);labels.append(f'wall-mixed-composition-{preserve}');tags.append(dict(kind='wall',mach=0))
 return np.array(rows),labels,tags,model
def compare(ref,new,labels):
 rs,rv=ref;ns,nv=new
 if not np.array_equal(rs,ns):
  i=int(np.flatnonzero(rs!=ns)[0]);raise AssertionError((labels[i],int(rs[i]),int(ns[i])))
 assert np.isfinite(rv).all() and np.isfinite(nv).all()
 maximum=0.
 for i,(a,b) in enumerate(zip(nv,rv)):
  tolerance=1e-13*max(1.,float(max(abs(b))))+5e-12*abs(b);ratio=float(max(abs(a-b)/tolerance))
  assert ratio<=1,(labels[i],ratio);maximum=max(maximum,ratio)
 return dict(maximum_original_component_budget_ratio=maximum,statuses_exact=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--compiler',type=Path,required=True)
 parser.add_argument('--properties-reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--memcheck',type=Path)
 args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);hashes={}
 for name in FILES:
  dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst);hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
 candidate=out/'candidate';shutil.copy2(args.candidate.resolve(),candidate);rows,labels,tags,model=inputs(args.properties_reference)
 with (out/'input.bin').open('wb') as f:np.array([len(rows)],'<i4').tofile(f);rows.astype('<f8').tofile(f)
 report=dict(source_sha256=hashes,inputs=len(rows),commands=build(out,args.compiler),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
 products={}
 for backend,binary in [('cpu',out/'cpu/probe'),('old_gpu',out/'old_gpu/probe'),('cpp',candidate)]:
  run([binary,out/'input.bin',out/(backend+'.bin')],out,backend+'.log');products[backend]=read(out/(backend+'.bin'))
 (out/'labels.json').write_text(json.dumps(labels,indent=2)+'\n')
 report['comparisons']={b:compare(products[b],products['cpp'],labels) for b in ['cpu','old_gpu']}
 s,v=products['cpp'];success=0;copies=0
 for i,(x,q,tag) in enumerate(zip(rows,v,tags)):
  if s[i]:continue
  success+=1;rho=q[0];z=q[5:10];kin=(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/(2*rho)
  t=(q[4]-q[10]-z@model.formation_energy-kin)/(z@model.cv_tr);p=t*(z@model.gas_constant)
  if tag['kind']=='wall':
   assert np.all(q[1:4]==0),(labels[i],'wall velocity')
   assert np.max(abs(z/rho-x[5:10]/x[0]))<=64*np.finfo(float).eps,(labels[i],'wall composition')
   assert abs(t-x[22])<=1e-13*max(1.,x[22])+5e-12*abs(x[22]),(labels[i],'wall T',t)
   ev=sum(z[k]*model.species_vibrational_energy(k,x[22]) for k in range(5))
   assert abs(q[10]-ev)<=1e-13*max(1.,abs(ev))+5e-12*abs(ev),(labels[i],'wall Tv energy')
  elif tag['mach']>=1.:
   assert np.array_equal(q.view('u8'),x[:11].view('u8')),(labels[i],'supersonic copy');copies+=1
  else:
   assert tag['mach']>0
   assert abs(p-x[22])<=1e-13*max(1.,abs(x[22]))+5e-12*abs(x[22]),(labels[i],'outlet pressure',p)
 report['independent']=dict(successful_states=success,supersonic_copy_words=copies)
 run([candidate,out/'input.bin',out/'field_scalar.bin',out/'cpu.bin'],out,'fields.log');report['fields']='successful ghost fill, failed word retention and unchanged carry'
 if args.memcheck:
  run([args.memcheck,'--tool','memcheck','--leak-check','full','--error-exitcode','99',candidate,out/'input.bin',out/'memcheck.bin',out/'cpu.bin'],out,'memcheck.log')
  text=(out/'memcheck.log').read_text();assert 'ERROR SUMMARY: 0 errors' in text and 'LEAK SUMMARY: 0 bytes leaked' in text;report['memcheck']='zero errors/leaks'
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','commands']},indent=2))
if __name__=='__main__':main()
