"""Extract actual NVHPC CPU/CUDA filter matrices; no CFD advancement."""
import json
import os
import subprocess

import numpy as np
import pytest

from review_high_order_p6_filter_compatibility import current_normal, sample
from scipy.linalg import block_diag
from test_high_order_boundary_closures import derivative_matrix
from test_high_order_boundary_geometry import ROOT
from test_high_order_explicit_wall_candidates import state_projection
from test_high_order_filter_kernel import filter_probe_modules

pytestmark = pytest.mark.skipif(os.environ.get('ASTR_HB6_FILTER_KERNEL_PROBE') != '1',
                               reason='Explicit native CUDA filter matrix probe')


def test_actual_cuda_matrices_keep_joint_gate(tmp_path):
    source = tmp_path/'matrix_probe.cuf'
    source.write_text(filter_probe_modules()+'''program matrix_probe
  use astr_precision
  use commvar
  use commarray_gpu
  use host_filter_probe
  use device_filter_probe
  implicit none
  integer,parameter :: sizes(5)=[8,9,32,64,128],kinds(3)=[41,42,421]
  integer :: ni,ki,n,kind,axis,layout,column,status,unit,dims(3)
  real(real_kind),allocatable :: q(:,:,:,:),f(:),matrix(:,:)
  character(128) :: filename
  type(dim3) :: block,grid
  call filter_coefficient_explicit()
  high_order_wall_enabled_d=.true.
  do ki=1,size(kinds)
    kind=kinds(ki)
    high_order_adiabatic_wall_enabled_d=kind/=41
    do ni=1,size(sizes)
      n=sizes(ni)
      allocate(f(-hm:n+hm),matrix(0:n,0:n))
      do column=0,n
        f=0._real_kind;f(column)=1._real_kind
        matrix(:,column)=spafilter10exp(f,4,n,kind)
      enddo
      write(filename,'("cpu_",i0,"_",i0,".bin")') kind,n
      open(newunit=unit,file=filename,status='new',access='stream',form='unformatted')
      write(unit) matrix
      close(unit)
      do axis=1,3
        dims=0;dims(axis)=n
        high_order_wall_axis_d=axis
        allocate(q(-hm:dims(1)+hm,-hm:dims(2)+hm,-hm:dims(3)+hm,5))
        allocate(q_d,mold=q);allocate(qwork_d,mold=q)
        allocate(filter_work_d(-hm:dims(1)+hm,-hm:dims(2)+hm,-hm:dims(3)+hm))
        select case(axis)
        case(1)
          block=dim3(512,1,1);grid=dim3((n+512)/512,1,1)
        case(2)
          block=dim3(32,16,1);grid=dim3((n+512)/512,1,1)
        case(3)
          block=dim3(64,1,8);grid=dim3((n+512)/512,1,1)
        end select
        do layout=0,1
          do column=0,n
            q=0._real_kind
            select case(axis)
            case(1);q(column,0,0,1)=1._real_kind
            case(2);q(0,column,0,1)=1._real_kind
            case(3);q(0,0,column,1)=1._real_kind
            end select
            q_d=q;qwork_d=q;filter_work_d=q(:,:,:,1)
            if(layout==0) then
              select case(axis)
              case(1);call filter_x_global_kernel<<<grid,block>>>(dims(1),dims(2),dims(3),4,5,.true.)
              case(2);call filter_y_pong_to_q_halo_global_kernel<<<grid,block>>>(dims(1),dims(2),dims(3),4,5,.true.)
              case(3);call filter_z_halo_global_kernel<<<grid,block>>>(dims(1),dims(2),dims(3),4,5,.true.)
              end select
            else
              select case(axis)
              case(1);call filter_x_scalar_global_kernel<<<grid,block>>>(dims(1),dims(2),dims(3),4,1,.true.)
              case(2);call filter_y_scalar_to_q_halo_global_kernel<<<grid,block>>>(dims(1),dims(2),dims(3),4,1,.true.)
              case(3);call filter_z_scalar_global_kernel<<<grid,block>>>(dims(1),dims(2),dims(3),4,1,.true.)
              end select
            endif
            status=cudaDeviceSynchronize()
            if(status/=cudaSuccess) error stop 'matrix filter kernel'
            if(layout==0) then
              select case(axis)
              case(1);matrix(:,column)=qwork_d(0:n,0,0,1)
              case(2);matrix(:,column)=q_d(0,0:n,0,1)
              case(3);matrix(:,column)=qwork_d(0,0,0:n,1)
              end select
            else
              select case(axis)
              case(1);matrix(:,column)=filter_work_d(0:n,0,0)
              case(2);matrix(:,column)=q_d(0,0:n,0,1)
              case(3);matrix(:,column)=filter_work_d(0,0,0:n)
              end select
            endif
          enddo
          write(filename,'("gpu_",i0,"_",i0,"_",i0,"_",i0,".bin")') kind,n,axis,layout
          open(newunit=unit,file=filename,status='new',access='stream',form='unformatted')
          write(unit) matrix
          close(unit)
        enddo
        deallocate(q,q_d,qwork_d,filter_work_d)
      enddo
      deallocate(f,matrix)
    enddo
  enddo
  print *, 'FILTER_MATRIX_PROBE_DONE'
end program
''')
    compiler = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran'
    build = subprocess.run([compiler, '-Mpreprocess', '-Mextend', '-Mnostack_arrays', '-O2', '-cuda',
                            '-gpu=rdc,maxregcount:128', str(ROOT/'src/astr_precision.F90'), str(ROOT/'src/constdef.F90'),
                            str(ROOT/'src/boundary_high_order.F90'), str(ROOT/'src_gpu/boundary_high_order_gpu.cuf'),
                            str(source), '-o', 'matrix_probe'], cwd=tmp_path, text=True, capture_output=True, timeout=120)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode == 0, build.stdout+build.stderr
    run = subprocess.run([str(tmp_path/'matrix_probe')], cwd=tmp_path, text=True, capture_output=True, timeout=240)
    (tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0 and 'FILTER_MATRIX_PROBE_DONE' in run.stdout, run.stdout+run.stderr
    records = []
    for kind in (41,42,421):
        for n in (8,9,32,64,128):
            cpu = np.fromfile(tmp_path/f'cpu_{kind}_{n}.bin','<f8').reshape(n+1,n+1,order='F')
            unique = [cpu]
            for axis in (1,2,3):
                for layout in (0,1):
                    gpu = np.fromfile(tmp_path/f'gpu_{kind}_{n}_{axis}_{layout}.bin','<f8').reshape(n+1,n+1,order='F')
                    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=2e-10)
                    if not any(np.array_equal(gpu, old) for old in unique):
                        unique.append(gpu)
            for version, filtering in enumerate(unique):
                pd = state_projection(n,'dirichlet')
                pt = state_projection(n,'dirichlet' if kind==41 else 'neumann')
                a = current_normal(n,0.,kind);diffusion = current_normal(n,1.,kind)-a
                d = n*derivative_matrix(n,True);tangent = pt if kind==421 else pd
                tf = filtering[1:-1]@tangent
                blocks = [('normal',block_diag(filtering,filtering[1:-1]@pd,filtering[1:-1]@pt),a,diffusion),
                          ('tangent',tf,np.zeros_like(tf),(d@d@tangent)[1:-1])]
                for nu in (0.,1e-6,1e-5,1e-4,1/1600):
                    for cfl in (None,0.,.0001,.001,.005,.01,.025,.05,.1,.15,.2):
                        report = sample(blocks,nu,cfl,n)
                        records.append(dict(bctype=kind,intervals=n,arithmetic_version=version,**report))
                        assert report['passes'], records[-1]
    (tmp_path/'matrix_spectra.json').write_text(json.dumps(dict(
        scope='actual compiled matrices, frozen normal/tangent gate; not nonlinear stability proof',
        samples=records), indent=2, allow_nan=False)+'\n')
