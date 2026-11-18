"""Explicit old-GPU -> C++ nonexact continuation of existing wall-time state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments
from run_output_restart_validation import run_case
from run_native_precision_restart_gate import check_state
from prepare_legacy_wall_reference import definitions
from run_cpp_statistics_gate import configuration
from run_cpp_tgv_gate import checkpoint,compare,logged_steps
from run_legacy_wall_runtime import curves,compare_curves,wall_history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mpiexec',type=Path,required=True)
    p.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    p.add_argument('--memcheck',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'cpp_astr';shutil.copy2(a.binary,binary);binary.chmod(0o555)
    source=a.reference.resolve()/('gpu_np1_'+a.precision+'_csv')
    args=arguments(out);args.executable=binary;args.mpiexec=str(a.mpiexec.resolve());args.unrestricted=True
    args.native_precision=a.precision;args.insitu_write_eps='0';args.scale_timestep=2e-5;args.maximum_cfl=.5
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    config=configuration(True).replace('statistics_window=.0005,.0115','statistics_window=.00001,.0002')
    config=config.replace('statistics=.true.,render=f',"statistics=.true.,render=f,products='channel_walls'")
    case,_=run_case(args,ROOT,'gpu',1,'cpp-nonexact-old-wall',4,
        restore=source/'outdat/new/checkpoints/step000000000002',exact_restart=False,
        grid='32,32,32',tgv_mapping='y-wavy',checkpoint_interval=2,insitu_config=config,
        product_catalog_records=definitions(),gpu_precision_mode='' if a.precision=='fp32' else 'fp64',
        numerics=dict(filter_scheme='explicit10'),memcheck=a.memcheck)
    final=case/'outdat/new/checkpoints/step000000000004'
    original=source/'outdat/new/checkpoints/step000000000004'
    error=compare(checkpoint(final/'state.h5'),checkpoint(original/'state.h5'),None if a.precision=='fp32' else 2e-10)
    values=curves(case,True);differences=compare_curves(values,curves(source),a.precision)
    history=wall_history(final/'statistics.h5',original/'statistics.h5',a.precision)
    with h5py.File(final/'statistics.h5') as f,h5py.File(original/'statistics.h5') as old:
        if not np.array_equal(f['wall_statistics/metadata'][...],old['wall_statistics/metadata'][...]):
            raise AssertionError('migration lost wall endpoint count/window/coverage history')
    report=dict(precision=a.precision,mode='explicit nonexact; not cross-implementation exact',source=str(source),
        memcheck=a.memcheck,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),flow_scaled=error,
        curves=differences,wall_history=history,clock=logged_steps(case,count=2,first_step=3).tolist(),
        physical=check_state(final/'state.h5',a.precision))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('explicit nonexact old GPU wall-time continuation preserves its approved history')


if __name__=='__main__':main()
