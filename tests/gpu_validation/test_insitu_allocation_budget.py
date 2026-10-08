"""Overflow-safe admission arithmetic, without CUDA or a physical run."""
from pathlib import Path
import subprocess
import os
import re
import pytest


def test_admission_arithmetic(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / 'probe.cpp'
    source.write_text(r'''
#include "insitu_allocation_budget.h"
#include <cassert>
#include <limits>
int main() {
  using astr_insitu::admit_device_allocation;
  assert(admit_device_allocation(200, 900, 1000, 300, 200));
  assert(!admit_device_allocation(201, 900, 1000, 300, 200));
  assert(!admit_device_allocation(1, 699, 1000, 300, 200));
  assert(!admit_device_allocation(1, 199, 1000, 2000, 200));
  assert(admit_device_allocation(300, 1200, 1000, 300, 200));
  auto max=std::numeric_limits<std::uint64_t>::max();
  assert(!admit_device_allocation(max, 900, 1000, max, 200));
  assert(admit_device_allocation(max-100, max, max, max, 100));
  assert(!admit_device_allocation(max-99, max, max, max, 100));
}''')
    binary = tmp_path / 'probe'
    subprocess.run(['g++', '-std=c++17', '-I', str(root/'src_gpu'), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_shared_atomic_reservation_and_release(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path/'shared_probe.cpp'
    source.write_text(r'''
#include "insitu_shared_allocation_budget.h"
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cassert>
#include <limits>
int main() {
  using namespace astr_insitu;
  auto* budget=static_cast<SharedAllocationBudget*>(mmap(nullptr,sizeof(SharedAllocationBudget),
    PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
  assert(budget!=MAP_FAILED);
  initialize_shared_allocation_budget(*budget);
  for(int round=0;round<32;++round) {
    int done[2],go[2];assert(pipe(done)==0 && pipe(go)==0);
    pid_t children[2];
    for(int i=0;i<2;++i) {
      children[i]=fork();assert(children[i]>=0);
      if(children[i]==0) {
        close(done[0]);close(go[1]);
        bool admitted;
        { SharedAllocationLock lock(*budget);admitted=reserve_shared_allocation(*budget,600,1000); }
        char result=admitted?1:0;assert(write(done[1],&result,1)==1);
        char release;assert(read(go[0],&release,1)==1);
        if(admitted) { SharedAllocationLock lock(*budget);release_shared_allocation(*budget,600); }
        _exit(0);
      }
    }
    close(done[1]);close(go[0]);
    char a,b;assert(read(done[0],&a,1)==1 && read(done[0],&b,1)==1);
    assert(a+b==1);
    { SharedAllocationLock lock(*budget);assert(budget->held==600 && budget->peak==600); }
    assert(write(go[1],"xx",2)==2);
    for(auto child:children) { int status;assert(waitpid(child,&status,0)==child && status==0); }
    close(done[0]);close(go[1]);
    { SharedAllocationLock lock(*budget);assert(budget->held==0); }
  }
  { SharedAllocationLock lock(*budget);
    const auto max=std::numeric_limits<std::uint64_t>::max();
    assert(reserve_shared_allocation(*budget,max,max));
    assert(!reserve_shared_allocation(*budget,1,max));
    release_shared_allocation(*budget,max);
    assert(budget->held==0 && budget->admissions==33 && budget->refusals==33);
    bool caught=false;
    try { release_shared_allocation(*budget,1); } catch(const std::exception&) { caught=true; }
    assert(caught);
  }
  destroy_shared_allocation_budget(*budget);
  assert(munmap(budget,sizeof(SharedAllocationBudget))==0);
}''')
    binary=tmp_path/'shared_probe'
    subprocess.run(['g++','-std=c++11','-pthread','-I',str(root/'src'),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=15)


@pytest.mark.skipif('ASTR_INSITU_DEVICE_PROBE_BIN' not in os.environ, reason='Set the root-built CUDA probe directory')
def test_device_array_and_sort_preflight():
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set the matching MPI prefix; do not launch this MPI-enabled Viskores probe directly'
    command = [str(Path(prefix)/'bin/mpirun'), '--prefix', prefix,
               '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
               '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
               '--mca', 'opal_cuda_support', '0', '-np', '1',
               str(Path(os.environ['ASTR_INSITU_DEVICE_PROBE_BIN'])/'insitu_allocation_guard_probe')]
    result = subprocess.run(command,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'array_rejected_before_malloc=1' in result.stdout
    assert 'scratch_rejection=1' in result.stdout


@pytest.mark.skipif('ASTR_INSITU_SHARED_PROBE_EXE' not in os.environ, reason='Set the root-built shared allocation probe')
@pytest.mark.parametrize('mode', ('success','refuse'))
def test_shared_native_lifecycle(tmp_path, mode):
    prefix=os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Select the matching MPI prefix'
    output=tmp_path/'render'
    output.mkdir()
    command=[str(Path(prefix)/'bin/mpirun'),'--prefix',prefix,
        '--mca','pml','ob1','--mca','btl','self,tcp','--mca','osc','sm',
        '--mca','coll_hcoll_enable','0','--mca','coll_ucc_enable','0',
        '--mca','opal_cuda_support','0','--oversubscribe','-np','4',
        os.environ['ASTR_INSITU_SHARED_PROBE_EXE'],str(output),mode]
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    log=result.stdout+result.stderr
    (tmp_path/'probe.log').write_text(log)
    if mode=='refuse':
        assert result.returncode!=0,log
        assert 'ASTR INSITU SHARED ALLOCATION REFUSED' in log,log
        assert 'preflight_only=1 no_device_allocation=1' in log,log
        assert 'concurrent over-budget reservation was admitted' not in log,log
    else:
        assert result.returncode==0,log
        reports=re.findall(r'ASTR_INSITU_SHARED_ALLOCATION_END held=(\d+) peak=(\d+) '
            r'admissions=(\d+) releases=(\d+) refusals=(\d+)',log)
        assert len(reports)==4,log
        assert all(int(held)==0 and int(peak)>0 and int(admissions)==6 and
                   int(releases)==6 and int(refusals)==0
                   for held,peak,admissions,releases,refusals in reports),log
        assert log.count('immediate_and_deferred_release=1')==4,log


@pytest.mark.skipif('ASTR_OUTPUT_INSITU_EXE' not in os.environ,
                    reason='Select the root-built strict renderer for the native refusal gate')
def test_native_frame_preflight_refusal(tmp_path):
    from test_insitu_curve_demo import scale_configuration
    from test_insitu_device_curve_wall_fields import current_arguments
    from run_output_restart_validation import run_case

    args = current_arguments(tmp_path, samples=False)
    config = scale_configuration('standard-device', 'pinned').replace(
        'device_budget_bytes=6442450944', 'device_budget_bytes=314572800')
    case, _ = run_case(args, Path(__file__).resolve().parents[2], 'gpu', 2,
        'allocation_refusal', 2, enabled=False, checkpoint_enabled=False,
        grid='32,32,32', tgv_mapping='periodic', insitu_config=config,
        postprocess_transport='pinned', reject='ASTR INSITU ALLOCATION REFUSED',
        failure_after_start=True)
    log = (case/'run.log').read_text()
    refused = re.findall(r'ASTR INSITU ALLOCATION REFUSED source=(.*?) requested=(\d+)', log)
    assert refused and all(source in ('Viskores array', 'Viskores unmanaged/Thrust temporary',
                                      'OpenGL display buffer') and int(size) > 0
                           for source, size in refused), log
    assert 'device allocation would exceed budget/reserve' in log, log
    assert not list((case/'outdat/render').glob('*.jpeg'))
    assert not list((case/'outdat/new').rglob('COMPLETE'))
