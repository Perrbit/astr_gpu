"""Actual CPU and CUDA filter procedures, isolated from flow advancement."""
import os
import subprocess

import pytest
from test_high_order_boundary_geometry import procedure,ROOT


def filter_probe_source():
    host='\n'.join(procedure(ROOT/'src/filter.F90',kind,name) for kind,name in (
        ('subroutine','filter_coefficient_explicit'),('function','spafilter10exp')))
    device='\n'.join(procedure(ROOT/'src_gpu/solver_gpu.cuf','function',name) for name in (
        'filter10_value','filter6_value','filter8_value','filter6_boundary_point1','filter6_boundary_point2'))
    kernels='\n'.join(procedure(ROOT/'src_gpu/solver_gpu.cuf','subroutine',name) for name in (
        'filter_x_global_kernel','filter_y_pong_to_q_halo_global_kernel','filter_z_halo_global_kernel',
        'filter_x_scalar_global_kernel','filter_y_scalar_to_q_halo_global_kernel','filter_z_scalar_global_kernel'))
    return '''module commvar
  integer,parameter :: hm=5
end module
module commarray_gpu
  use astr_precision
  use cudafor
  real(real_kind),device,allocatable :: q_d(:,:,:,:),qwork_d(:,:,:,:),filter_work_d(:,:,:)
end module
module host_filter_probe
  use astr_precision
  implicit none
  real(real_kind),allocatable :: coef2e(:),coef4e(:),coef6e(:),coef8e(:),coef10e(:)
  real(real_kind),allocatable :: coef3be(:,:),coef4be(:,:),coef6be(:,:)
contains
'''+host+'''end module
module device_filter_probe
  use astr_precision
  use cudafor
  use constdef
  implicit none
contains
'''+device+kernels+'''end module
program probe
  use astr_precision
  use commvar
  use commarray_gpu
  use host_filter_probe
  use device_filter_probe
  implicit none
  integer :: n,ni,i,j,k,m,nt,axis,field,layout,status,checks
  integer,parameter :: sizes(4)=[8,12,16,32]
  logical :: flat
  real(real_kind),allocatable :: q(:,:,:,:),reference(:,:,:,:),full(:,:,:,:),scalar(:,:,:,:)
  real(real_kind) :: error,max_error,storage_error,max_storage_error
  type(dim3) :: block,grid
  call filter_coefficient_explicit()
  max_error=0._real_kind;max_storage_error=0._real_kind;checks=0
  do ni=1,size(sizes)
    n=sizes(ni)
    allocate(q(-hm:n+hm,-hm:n+hm,-hm:n+hm,5),reference,mold=q)
    allocate(full,mold=q);allocate(scalar,mold=q)
    allocate(q_d,mold=q);allocate(qwork_d,mold=q)
    allocate(filter_work_d(-hm:n+hm,-hm:n+hm,-hm:n+hm))
    do field=1,2
      do m=1,5
      do k=-hm,n+hm
      do j=-hm,n+hm
      do i=-hm,n+hm
        q(i,j,k,m)=real(m,real_kind)
        if(field==2) q(i,j,k,m)=q(i,j,k,m)+ &
          .125_real_kind*(real(i,real_kind)/n)**2+.0625_real_kind*(real(j,real_kind)/n)**3+ &
          .03125_real_kind*(real(k,real_kind)/n)**4+.01_real_kind*sin(.31_real_kind*i+.21_real_kind*j+.13_real_kind*k+m)
      enddo
      enddo
      enddo
      enddo
      do nt=1,4
      do axis=1,3
        reference=q
        do m=1,5
          select case(axis)
          case(1)
            do k=0,n
            do j=0,n
              reference(0:n,j,k,m)=spafilter10exp(q(:,j,k,m),nt,n)
            enddo
            enddo
          case(2)
            do k=0,n
            do i=0,n
              reference(i,0:n,k,m)=spafilter10exp(q(i,:,k,m),nt,n)
            enddo
            enddo
          case(3)
            do j=0,n
            do i=0,n
              reference(i,j,0:n,m)=spafilter10exp(q(i,j,:,m),nt,n)
            enddo
            enddo
          end select
        enddo
        if(field==1.and.any(reference(0:n,0:n,0:n,:)/=q(0:n,0:n,0:n,:))) &
          error stop 'CPU constant preservation'
        do layout=0,1
          flat=layout==1
          select case(axis)
          case(1)
            block=dim3(512,1,1);grid=dim3((n+512)/512,n+1,n+1)
            if(flat) grid=dim3(((n+1)**3+511)/512,1,1)
          case(2)
            block=dim3(32,16,1);grid=dim3((n+32)/32,(n+16)/16,n+1)
            if(flat) grid=dim3(((n+1)**2+511)/512,1,n+1)
          case(3)
            block=dim3(64,1,8);grid=dim3((n+64)/64,n+1,(n+8)/8)
            if(flat) grid=dim3(((n+1)**2+511)/512,n+1,1)
          end select
          q_d=q;qwork_d=q
          select case(axis)
          case(1)
            call filter_x_global_kernel<<<grid,block>>>(n,n,n,nt,5,flat)
          case(2)
            call filter_y_pong_to_q_halo_global_kernel<<<grid,block>>>(n,n,n,nt,5,flat)
          case(3)
            call filter_z_halo_global_kernel<<<grid,block>>>(n,n,n,nt,5,flat)
          end select
          status=cudaDeviceSynchronize()
          if(status/=cudaSuccess) error stop 'full filter kernel'
          if(axis==2) then
            full=q_d
          else
            full=qwork_d
          endif
          q_d=q;qwork_d=q;scalar=q
          do m=1,5
            filter_work_d=q(:,:,:,m)
            select case(axis)
            case(1)
              call filter_x_scalar_global_kernel<<<grid,block>>>(n,n,n,nt,m,flat)
            case(2)
              call filter_y_scalar_to_q_halo_global_kernel<<<grid,block>>>(n,n,n,nt,m,flat)
            case(3)
              call filter_z_scalar_global_kernel<<<grid,block>>>(n,n,n,nt,m,flat)
            end select
            status=cudaDeviceSynchronize()
            if(status/=cudaSuccess) error stop 'scalar filter kernel'
            if(axis==2) then
              scalar=q_d
            else
              scalar(:,:,:,m)=filter_work_d
            endif
          enddo
          error=max(maxval(abs(full(0:n,0:n,0:n,:)-reference(0:n,0:n,0:n,:))), &
            maxval(abs(scalar(0:n,0:n,0:n,:)-reference(0:n,0:n,0:n,:))))
          storage_error=maxval(abs(full(0:n,0:n,0:n,:)-scalar(0:n,0:n,0:n,:)))
          max_error=max(max_error,error);max_storage_error=max(max_storage_error,storage_error)
          if(error>2.e-10_real_kind.or.storage_error>2.e-10_real_kind) then
            print *, 'FILTER_FAILURE',n,nt,axis,field,layout,error,storage_error
            error stop 'CPU/GPU or storage mismatch'
          endif
          checks=checks+1
        enddo
      enddo
      enddo
    enddo
    deallocate(q,reference,full,scalar,q_d,qwork_d,filter_work_d)
  enddo
  print *, 'FILTER_KERNEL_CHECKS',checks
  print *, 'FILTER_KERNEL_MAXABS',max_error
  print *, 'FILTER_STORAGE_MAXABS',max_storage_error
end program
'''


@pytest.mark.skipif(os.environ.get('ASTR_HB6_FILTER_KERNEL_PROBE')!='1',reason='Explicit native CUDA filter probe')
def test_cpu_gpu_physical_filter_and_both_workspaces(tmp_path,record_property):
    source=tmp_path/'filter_probe.cuf';source.write_text(filter_probe_source())
    compiler='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran'
    result=subprocess.run([compiler,'-Mpreprocess','-Mextend','-Mnostack_arrays','-O2','-cuda',
        '-gpu=rdc,maxregcount:128',str(ROOT/'src/astr_precision.F90'),str(ROOT/'src/constdef.F90'),
        str(source),'-o','filter_probe'],cwd=tmp_path,text=True,capture_output=True,timeout=120)
    (tmp_path/'build.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(tmp_path/'filter_probe')],cwd=tmp_path,text=True,capture_output=True,timeout=120)
    (tmp_path/'run.log').write_text(result.stdout+result.stderr)
    record_property('probe_directory',str(tmp_path))
    assert result.returncode==0,result.stdout+result.stderr
    assert 'FILTER_KERNEL_MAXABS' in result.stdout
