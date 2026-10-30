"""Numerical slice storage from root-built objects, not a CFD calculation."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get('ASTR_SLICE_BUILD', ROOT / 'build_gpu_probe')).resolve()
MPIEXEC = os.environ.get('ASTR_CATALOG_MPIEXEC', shutil.which('mpiexec') or 'mpiexec')
GPU = os.environ.get('ASTR_SLICE_GPU') == '1'


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    return compile_probe('index_slice_fields_probe.F90', tmp_path_factory)


def compile_probe(source, tmp_path_factory):
    link = BUILD / 'src/CMakeFiles/astr.dir/link.txt'
    if not link.exists():
        pytest.skip('Build astr through root CMake first')
    tokens = shlex.split(link.read_text())
    command, skip = [], False
    for token in tokens:
        if skip:
            skip = False
        elif token == '-o':
            skip = True
        elif not (token.endswith('/astr.F90.o') or token.startswith('-Wl,--dependency-file=')):
            command.append(token)
    includes = next(line.split('=', 1)[1] for line in (link.parent / 'flags.make').read_text().splitlines()
                    if line.startswith('Fortran_INCLUDES ='))
    executable = tmp_path_factory.mktemp('index_slice_fields') / 'probe'
    command += shlex.split(includes) + ['-I', str(BUILD / 'src'),
        str(ROOT / 'tests/gpu_validation' / source), '-o', str(executable)]
    if GPU:
        command.append('-DTEST_INDEX_FIELDS_GPU')
    result = subprocess.run(command, cwd=BUILD / 'src', capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.fixture(scope='module')
def diagnostic_probe(tmp_path_factory):
    return compile_probe('index_slice_diagnostics_probe.F90', tmp_path_factory)


@pytest.mark.parametrize('ranks,axis', [(1, 1), (2, 1), (2, 2), (2, 3), (4, 1)])
def test_curved_affine_field_and_sixth_order_wall_stencil(diagnostic_probe, ranks, axis):
    wrapper = []
    mpi_options = ['--mca', 'coll_hcoll_enable', '0']
    if os.environ.get('ASTR_SLICE_MEMCHECK') == '1':
        assert GPU
        wrapper = ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99']
        # This private slice exchange sends host buffers. Do not initialize UCX/UCC CUDA contexts
        # before the probe binds its GPUs; those external context probes trip sanitizer API checks.
        mpi_options += ['--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp', '--mca', 'osc', 'pt2pt',
                        '--mca', 'coll', '^cuda,hcoll,ucc', '--mca', 'opal_cuda_support', '0']
    result = subprocess.run([MPIEXEC, '--oversubscribe', *mpi_options, '-n', str(ranks),
                             *wrapper, str(diagnostic_probe), str(axis)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert float(result.stdout.split('SLICE_DIAGNOSTIC_ERROR=')[1].split()[0]) <= 2e-10
    if wrapper:
        assert (result.stdout + result.stderr).count('ERROR SUMMARY: 0 errors') == ranks


def test_curved_polynomial_sixth_order_convergence(diagnostic_probe):
    errors = []
    for intervals in (16, 32, 64):
        result = subprocess.run([MPIEXEC, '--oversubscribe', '--mca', 'coll_hcoll_enable', '0', '-n', '2',
            str(diagnostic_probe), '1', 'convergence', str(intervals)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        errors.append(float(result.stdout.split('SLICE_DIAGNOSTIC_ERROR=')[1].split()[0]))
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(orders > 5.8), (errors, orders.tolist())


@pytest.mark.parametrize('ranks,split,plane,index', [(1,1,3,8),(2,1,3,8),(2,2,3,8),
                                                   (2,3,3,8),(2,1,1,8),(4,2,1,0)])
def test_image_mesh_keeps_all_cells_across_partitions(diagnostic_probe,ranks,split,plane,index):
    result=subprocess.run([MPIEXEC,'--oversubscribe','--mca','coll_hcoll_enable','0','-n',str(ranks),
        str(diagnostic_probe),str(split),'mesh',str(plane),str(index)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'IMAGE_MESH_CELLS=256' in result.stdout
    assert float(result.stdout.split('SLICE_DIAGNOSTIC_ERROR=')[1].split()[0])<=2e-10


@pytest.mark.skipif(not GPU,reason='Retained device image workspace')
@pytest.mark.parametrize('split,plane,index',[(1,3,8),(2,3,8),(3,3,8)])
def test_image_mesh_can_be_consumed_without_field_download(diagnostic_probe,split,plane,index):
    mpi_options=['--mca','coll_hcoll_enable','0']
    wrapper=[]
    if os.environ.get('ASTR_SLICE_MEMCHECK')=='1':
        mpi_options+=['--mca','pml','ob1','--mca','btl','self,vader,tcp','--mca','osc','pt2pt',
                      '--mca','coll','^cuda,hcoll,ucc','--mca','opal_cuda_support','0']
        wrapper=['compute-sanitizer','--tool','memcheck','--error-exitcode','99']
    result=subprocess.run([MPIEXEC,'--oversubscribe',*mpi_options,'-n','2',*wrapper,
        str(diagnostic_probe),str(split),'retained-mesh',str(plane),str(index)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'IMAGE_MESH_CELLS=256' in result.stdout
    assert 'field_d2h_bytes=0 geometry_host_bytes=0 oracle_readback_separate=1' in result.stdout
    assert float(result.stdout.split('SLICE_DIAGNOSTIC_ERROR=')[1].split()[0])<=2e-10
    if wrapper:
        assert (result.stdout+result.stderr).count('ERROR SUMMARY: 0 errors')==2


@pytest.mark.skipif(not GPU,reason='Private image MPI device workspace')
@pytest.mark.parametrize('transport',['pinned','device-aware'])
def test_image_private_transport_matches_native_affine_source(diagnostic_probe,transport):
    env=dict(os.environ)
    options=['--mca','coll','^cuda,hcoll,ucc','--mca','osc','pt2pt']
    if transport=='device-aware':
        env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='true',UCX_MEMTYPE_CACHE='n',
            UCX_CUDA_COPY_ENABLE_FABRIC='no',UCX_CUDA_COPY_DMABUF='no',UCX_CUDA_IPC_ENABLE_MNNVL='no',
            UCX_TLS='self,sm,cuda_copy,cuda_ipc')
    else:
        options+=['--mca','pml','ob1','--mca','btl','self,vader,tcp','--mca','opal_cuda_support','0']
    wrapper=[]
    if os.environ.get('ASTR_SLICE_MEMCHECK')=='1':
        wrapper=['compute-sanitizer','--tool','memcheck','--error-exitcode','99']
        if transport=='device-aware':
            wrapper+=['--suppressions',str(ROOT/'tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml')]
    result=subprocess.run([MPIEXEC,'--oversubscribe',*options,'-n','2',*wrapper,
        str(diagnostic_probe),'3','retained-mesh','3','8',transport],env=env,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    assert float(result.stdout.split('SLICE_DIAGNOSTIC_ERROR=')[1].split()[0])<=2e-10
    rows=[line for line in result.stdout.splitlines() if line.startswith('PRIVATE_MPI ')]
    assert len(rows)==2
    counts=[int(line.split('download=')[1].split()[0])+int(line.split('upload=')[1].split()[0]) for line in rows]
    assert (sum(counts)>0) if transport=='pinned' else sum(counts)==0
    if wrapper:
        assert (result.stdout+result.stderr).count('ERROR SUMMARY: 0 errors')==2


@pytest.mark.skipif(not GPU,reason='Private streamline device input')
@pytest.mark.parametrize('axis',[1,2,3])
def test_frozen_streamline_inputs_and_private_coordinate_halos(diagnostic_probe,axis):
    wrapper=[]
    if os.environ.get('ASTR_SLICE_MEMCHECK')=='1':
        wrapper=['compute-sanitizer','--tool','memcheck','--error-exitcode','99']
    result=subprocess.run([MPIEXEC,'--oversubscribe','--mca','pml','ob1','--mca','btl','self,vader,tcp',
        '--mca','osc','pt2pt','--mca','coll','^cuda,hcoll,ucc','--mca','opal_cuda_support','0','-n','2',
        *wrapper,str(diagnostic_probe),str(axis),'trace-input'],capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    assert float(result.stdout.split('TRACE_INPUT_MAXABS=')[1].split()[0])<=2e-10
    assert result.stdout.count('ASTR_RUNTIME_LINE_HALO ')==2
    if wrapper:
        assert (result.stdout+result.stderr).count('ERROR SUMMARY: 0 errors')==2


@pytest.mark.skipif(not GPU,reason='Explicit GPU allocation-recovery check')
def test_recoverable_device_allocation_allows_next_product_kernel(diagnostic_probe):
    result=subprocess.run([MPIEXEC,'--oversubscribe','--mca','coll_hcoll_enable','0','-n','2',
        str(diagnostic_probe),'1','allocation-recovery'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert float(result.stdout.split('SLICE_DIAGNOSTIC_ERROR=')[1].split()[0])<=2e-10


@pytest.mark.skipif(not GPU,reason='Native CPU/GPU diagnostic comparison')
def test_native_diagnostic_difference_report(diagnostic_probe,tmp_path):
    result=subprocess.run([MPIEXEC,'--oversubscribe','--mca','coll_hcoll_enable','0','-n','2',
        str(diagnostic_probe),'1','native-reference'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    width=int(result.stdout.split('SLICE_NATIVE_BYTES=')[1].split()[0])
    difference=float(result.stdout.split('BACKEND_DIFFERENCE=')[1].split()[0])
    analytic=float(result.stdout.split('SLICE_DIAGNOSTIC_ERROR=')[1].split()[0])
    assert np.isfinite([difference,analytic]).all()
    if width==8:
        assert difference<=2e-10 and analytic<=2e-10
    (tmp_path/'native_diagnostic_report.json').write_text(json.dumps(dict(source_bytes=width,
        cpu_gpu_maxabs=difference,analytic_maxabs=analytic,
        fp32_bias_threshold=None,definition='same frozen source; explicit sixth-order diagnostics'),indent=2)+'\n')


def archive_definition(name='changing', fields=None, interval=4):
    return dict(id=name, kind='slice', enable_request=1, source='instantaneous',
        fields=fields or ['velocity_x'],
        sampling=dict(mode='grid_index', axis='k', index=8, index_ranges=dict(i=[0,16], j=[0,16])),
        output=dict(channel='data', format='hdf5-xdmf'), schedule=dict(mode='steps', interval=interval))


def write_catalog(path, products):
    path.write_text('\n'.join(json.dumps(record) for record in
        [{'catalog_schema':'astr-insitu-products-v1'}, *products])+'\n')


def test_complete_step_archive_hot_update(diagnostic_probe, tmp_path):
    write_catalog(tmp_path/'products.jsonl', [archive_definition()])
    write_catalog(tmp_path/'candidate3.jsonl', [archive_definition(fields=['pressure'], interval=3)])
    write_catalog(tmp_path/'candidate7.jsonl', [archive_definition(fields=['pressure'], interval=2)])
    write_catalog(tmp_path/'candidate9.jsonl', [archive_definition('new-plane', ['density'], interval=2)])
    result = subprocess.run([MPIEXEC, '--oversubscribe', '--mca', 'coll_hcoll_enable', '0', '-n', '2',
        str(diagnostic_probe), '1', 'archive', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    root = tmp_path/'output/slice_products'
    def steps(name, version):
        return [int(path.stem[4:]) for path in sorted((root/name/version).glob('segment*/step*.h5'))]
    assert steps('changing', 'version000000000001') == [0]
    assert steps('changing', 'version000000000002') == [4,7,9]
    assert steps('new-plane', 'version000000000001') == [10,12]
    status = json.loads((tmp_path/'output/products.status.json').read_text())
    assert [row['state'] for row in status['products']] == ['not-in-catalog','active']


@pytest.fixture(scope='module')
def slice_write_fault(tmp_path_factory):
    flags = BUILD/'src/CMakeFiles/astr_checkpoint_filesystem.dir/flags.make'
    includes = next(line.split('=',1)[1] for line in flags.read_text().splitlines() if line.startswith('C_INCLUDES ='))
    output = tmp_path_factory.mktemp('index_slice_fault')/'fault.so'
    result = subprocess.run(['cc','-std=c99','-Wall','-Wextra','-Werror','-shared','-fPIC',
        *shlex.split(includes),str(ROOT/'tests/gpu_validation/index_slice_fault_preload.c'),'-ldl','-o',str(output)],
        capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr
    return output


def test_collective_hdf_write_failure_isolates_one_product(diagnostic_probe, slice_write_fault, tmp_path):
    definitions = [archive_definition('faulty'), archive_definition('good')]
    for name in ('products.jsonl','candidate3.jsonl','candidate7.jsonl','candidate9.jsonl'):
        write_catalog(tmp_path/name, definitions)
    environment = dict(os.environ, LD_PRELOAD=str(slice_write_fault), ASTR_TEST_SLICE_WRITE_FAILURE='1')
    result = subprocess.run([MPIEXEC,'--oversubscribe','--mca','coll_hcoll_enable','0','-n','2',
        str(diagnostic_probe),'1','archive',str(tmp_path)],capture_output=True,text=True,env=environment)
    assert result.returncode == 0,result.stdout+result.stderr
    root = tmp_path/'output/slice_products'
    failed = next((root/'faulty').glob('*/segment*'))
    good = next((root/'good').glob('*/segment*'))
    assert [p.name for p in failed.glob('step*.h5')] == ['step000000000000.h5']
    assert len(ET.parse(failed/'series.xdmf').findall('.//Grid[@GridType="Uniform"]')) == 1
    assert len(ET.parse(good/'series.xdmf').findall('.//Grid[@GridType="Uniform"]')) == 4
    status = json.loads((tmp_path/'output/products.status.json').read_text())
    assert [row['state'] for row in status['products']] == ['resource-disabled','active']


@pytest.mark.parametrize('topology,axis,index', [((1,1,1),0,5),((1,2,1),1,0),((1,1,2),2,8),
    ((2,2,1),2,0),((2,1,2),1,12),((1,2,2),0,0)])
def test_production_archive_asymmetric_axes_regions_and_empty_ranks(diagnostic_probe,tmp_path,topology,axis,index):
    intervals=np.array([16,12,8]);bounds=np.array([[2,14],[1,11],[0,8]])
    bounds[axis]=index
    definition=archive_definition(fields=['density','velocity_x','pressure','velocity_gradient_xy',
                                          'density_gradient_magnitude'],interval=100)
    definition['sampling']=dict(mode='grid_index',axis='ijk'[axis],index=index,
        index_ranges={letter:bounds[d].tolist() for d,letter in enumerate('ijk') if d!=axis})
    for name in ('products.jsonl','candidate3.jsonl','candidate7.jsonl','candidate9.jsonl'):
        write_catalog(tmp_path/name,[definition])
    result=subprocess.run([MPIEXEC,'--oversubscribe','--mca','coll_hcoll_enable','0','-n',str(np.prod(topology)),
        str(diagnostic_probe),'1','archive-ownership',str(tmp_path),','.join(map(str,topology))],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    paths=list((tmp_path/'output/slice_products').glob('*/*/segment*/step*.h5'))
    assert len(paths)==1
    frame=paths[0]
    varying=[d for d in range(3) if d!=axis]
    shape=(bounds[varying[1],1]-bounds[varying[1],0]+1,bounds[varying[0],1]-bounds[varying[0],0]+1)
    grid=np.indices(shape)
    global_index=np.full((3,*shape),index)
    for dimension,array_dimension in zip(varying,(1,0)):
        global_index[dimension]=grid[array_dimension]+bounds[dimension,0]
    cell=intervals//topology
    owners=np.minimum(global_index//cell[:,None,None],np.array(topology)[:,None,None]-1)
    rank=owners[0]+topology[0]*(owners[1]+topology[1]*owners[2])
    xi,eta,zeta=global_index/16
    coordinates=np.stack([xi+eta/8,eta,zeta+xi*eta/32],axis=-1)
    with h5py.File(frame) as data,h5py.File(frame.parent/'geometry.h5') as geometry:
        np.testing.assert_array_equal(data['pressure'],2+rank/1024)
        np.testing.assert_array_equal(geometry['coordinates'],coordinates)
        np.testing.assert_array_equal(data['index_ranges'],bounds.reshape(-1))
        assert np.max(np.abs(data['velocity_gradient_xy'][...]-.5))<=2e-10
        assert np.max(np.abs(data['density_gradient_magnitude'][...]-np.sqrt(.01**2+.02**2+.03**2)))<=2e-10


@pytest.mark.skipif(not GPU,reason='GPU transfer sharing check')
def test_matching_products_reuse_one_native_2d_buffer(diagnostic_probe,tmp_path):
    definitions=[archive_definition('first',['velocity_x','pressure']),archive_definition('second',['velocity_x','pressure'])]
    for name in ('products.jsonl','candidate3.jsonl','candidate7.jsonl','candidate9.jsonl'):
        write_catalog(tmp_path/name,definitions)
    result=subprocess.run([MPIEXEC,'--oversubscribe','--mca','coll_hcoll_enable','0','-n','2',
        str(diagnostic_probe),'1','archive',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    lines=[line for line in result.stdout.splitlines() if line.startswith('ASTR_SLICE ') and 'id=second ' in line]
    assert len(lines)==8 and all('field_d2h_bytes=0 ' in line and 'reused_2d=T' in line for line in lines)
    root=tmp_path/'output/slice_products'
    first=sorted((root/'first').glob('*/segment*/step*.h5'))
    second=sorted((root/'second').glob('*/segment*/step*.h5'))
    for a,b in zip(first,second,strict=True):
        with h5py.File(a) as x,h5py.File(b) as y:
            for field in ('velocity_x','pressure'):
                np.testing.assert_array_equal(x[field],y[field])


@pytest.mark.parametrize('ranks,decomposition', [(1, 1), (2, 1), (2, 2), (2, 3), (4, 1)])
def test_field_subset_and_index_region(probe, tmp_path, ranks, decomposition):
    result = subprocess.run([MPIEXEC, '--oversubscribe', '--mca', 'coll_hcoll_enable', '0', '-n', str(ranks),
                             str(probe), str(tmp_path), str(decomposition)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / 'data.h5') as data:
        fields = set(data) - {'metadata', 'coordinates', 'index_ranges', 'field_ids'}
        assert fields == {'velocity_x', 'pressure'}
        k, i = np.meshgrid(np.arange(1, 4), np.arange(1, 8), indexing='ij')
        j = np.full_like(i, 2)
        index = (i, j, k)[decomposition - 1]
        cells = (8, 6, 4)[decomposition - 1] // ranks
        owner = np.minimum(index // cells, ranks - 1)
        encoded = 100*i + 10*j + k + owner/1024
        np.testing.assert_array_equal(data['velocity_x'], encoded + 2/16)
        np.testing.assert_array_equal(data['pressure'], encoded + 5/16)
        np.testing.assert_array_equal(data['field_ids'], [2, 5])
        np.testing.assert_array_equal(data['index_ranges'], [1, 7, 2, 2, 1, 3])
    assert 'SLICE_FIELD_BYTES=336' in result.stdout
    if GPU:
        assert 'SLICE_DOWNLOAD_BYTES=336' in result.stdout


def test_separate_static_geometry_and_frame_filename(probe, tmp_path):
    result = subprocess.run([MPIEXEC, '--oversubscribe', '--mca', 'coll_hcoll_enable', '0', '-n', '2',
                             str(probe), str(tmp_path), '2', 'shared-geometry'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / 'geometry.h5') as geometry, h5py.File(tmp_path / 'step12.h5') as frame:
        assert 'coordinates' in geometry and 'coordinates' not in frame
        assert 'velocity' not in frame and 'pressure' not in geometry
        assert geometry['coordinates'].shape == (3, 7, 3)
        assert frame['velocity_x'].shape == (3, 7)
    xml = ET.parse(tmp_path / 'step12.h5.xdmf')
    refs = [node.text.strip() for node in xml.findall('.//DataItem')]
    assert 'geometry.h5:/coordinates' in refs
    assert 'step12.h5:/velocity_x' in refs and not any(ref.endswith(':/velocity') for ref in refs)


@pytest.mark.parametrize('ranks', [1, 2])
def test_legacy_volume_default_is_unchanged(probe, tmp_path, ranks):
    result = subprocess.run([MPIEXEC, '--oversubscribe', '--mca', 'coll_hcoll_enable', '0', '-n', str(ranks),
                             str(probe), str(tmp_path), '2', 'legacy-volume'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with h5py.File(tmp_path / 'data.h5') as data:
        assert data['density'].shape == (5, 7, 9)
        assert data['velocity'].shape == (5, 7, 9, 3)
        np.testing.assert_array_equal(data['velocity'][..., 0], data['velocity_x'])
        assert 'field_ids' not in data
    assert 'SLICE_FIELD_BYTES=15120' in result.stdout
