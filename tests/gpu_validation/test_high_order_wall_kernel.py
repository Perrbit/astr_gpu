"""Isolate the actual wall RHS kernel's two-sided device stencil addressing."""
import os
from pathlib import Path
import subprocess

import pytest
from test_high_order_boundary_geometry import procedure,ROOT


@pytest.mark.skipif(os.environ.get('ASTR_HB1D_KERNEL_PROBE')!='1',reason='Explicit native CUDA kernel probe')
def test_production_wall_mass_stencil(tmp_path):
    kernel=procedure(ROOT/'src_gpu/boundary_gpu.cuf','subroutine','high_order_wall_rhs_kernel')
    source=tmp_path/'probe.cuf'
    source.write_text('''module commarray_gpu
  use astr_precision
  use cudafor
  real(real_kind),device,allocatable :: q_d(:,:,:,:),qrhs_d(:,:,:,:),jacob_d(:,:,:),dxi_d(:,:,:,:,:)
end module
module wall_probe
  use astr_precision
  use cudafor
  use boundary_high_order_gpu
  implicit none
contains
'''+kernel+'''end module
program probe
  use wall_probe
  use commarray_gpu
  implicit none
  integer,parameter :: n=8,h=5
  integer :: i,j,k,m,status
  real(real_kind) :: q(-h:n+h,-h:n+h,-h:n+h,5),result(-h:n+h,-h:n+h,-h:n+h,5)
  real(real_kind) :: expected,error
  allocate(q_d(-h:n+h,-h:n+h,-h:n+h,5),qrhs_d(-h:n+h,-h:n+h,-h:n+h,5), &
    jacob_d(-h:n+h,-h:n+h,-h:n+h),dxi_d(-h:n+h,-h:n+h,-h:n+h,3,3))
  do k=-h,n+h
  do j=-h,n+h
  do i=-h,n+h
    q(i,j,k,:)=[1._real_kind,.1_real_kind+.005_real_kind*i**3,0._real_kind, &
      .2_real_kind+.003_real_kind*k**3,0._real_kind]
    q(i,j,k,5)=2.5_real_kind/.126_real_kind+.5_real_kind*sum(q(i,j,k,2:4)**2)
  enddo
  enddo
  enddo
  q_d=q;qrhs_d=0._real_kind;jacob_d=1._real_kind;dxi_d=0._real_kind
  do m=1,3
    dxi_d(:,:,:,m,m)=1._real_kind
  enddo
  high_order_adiabatic_wall_enabled_d=.true.
  high_order_slip_wall_enabled_d=.true.
  call high_order_wall_rhs_kernel<<<1,256>>>(n,n,n,.126_real_kind,2.5_real_kind,0._real_kind,0._real_kind,1,1)
  status=cudaDeviceSynchronize()
  if(status/=cudaSuccess) error stop 'wall kernel launch'
  result=qrhs_d
  error=0._real_kind
  do k=0,n
  do i=0,n
    expected=-.015_real_kind*i*i-.009_real_kind*k*k
    error=max(error,abs(result(i,0,k,1)-expected),abs(result(i,n,k,1)-expected))
  enddo
  enddo
  print *, 'WALL_KERNEL_MASS_MAXABS',error
  if(error>2.e-12_real_kind) error stop 'wall two-sided mass stencil'
end program
''')
    compiler='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran'
    command=[compiler,'-Mpreprocess','-Mextend','-O2','-cuda','-gpu=rdc,maxregcount:128',
        '-I'+str(ROOT/'src'),str(ROOT/'src/astr_precision.F90'),str(ROOT/'src/constdef.F90'),
        str(ROOT/'src_gpu/boundary_high_order_gpu.cuf'),str(source),'-o','probe']
    build=subprocess.run(command,cwd=tmp_path,text=True,capture_output=True,timeout=120)
    assert build.returncode==0,build.stdout+build.stderr
    result=subprocess.run([str(tmp_path/'probe')],cwd=tmp_path,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'WALL_KERNEL_MASS_MAXABS' in result.stdout
