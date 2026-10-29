"""HB0/D17: exercise production geombc and its real CUDA upload procedure."""
from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]


def procedure(path,kind,name):
    source=path.read_text()
    match=re.search(rf'^[ \t]+(?:attributes\([^\n)]*\)\s+)?(?:real\([^\n)]*\)\s+)?{kind} {name}\b.*?^[ \t]+end {kind} {name}\b',source,re.M|re.S|re.I)
    if not match:
        raise ValueError(f'Cannot isolate production {name}')
    return match.group(0)+'\n'


def probe_source(gpu):
    extrapolation=procedure(ROOT/'src/commfunc.F90','function','gradextrp')
    geometry=procedure(ROOT/'src/bc.F90','subroutine','geombc')
    upload=procedure(ROOT/'src_gpu/commarray_gpu.cuf','subroutine','copy_flow_to_gpu') if gpu else ''
    declaration='''
  real(real_kind),allocatable :: q(:,:,:,:),rho(:,:,:),vel(:,:,:,:),prs(:,:,:),tmp(:,:,:)
  real(real_kind),allocatable :: spc(:,:,:,:),tve(:,:,:,:),jacob(:,:,:),x(:,:,:,:),dxi(:,:,:,:,:)
  real(real_kind),allocatable :: bnorm_i0(:,:,:),bnorm_im(:,:,:),bnorm_j0(:,:,:), &
    bnorm_jm(:,:,:),bnorm_k0(:,:,:),bnorm_km(:,:,:)
'''
    device_declaration=declaration.replace(',allocatable',',device,allocatable')
    for name in ('bnorm_i0','bnorm_im','bnorm_j0','bnorm_jm','bnorm_k0','bnorm_km',
                 'jacob','spc','tve','rho','vel','prs','tmp','dxi','q','x'):
        device_declaration=re.sub(rf'\b{name}\b',name+'_d',device_declaration)
    device_module=('module device_probe\n use cudafor\n use astr_precision\n'+device_declaration+
                   'contains\n'+upload+'end module\n') if gpu else ''
    return '''module astr_precision
  use iso_fortran_env,only: real64
  integer,parameter :: real_kind=real64
end module
module commarray
  use astr_precision
'''+declaration+'''end module
module commfunc
  use astr_precision
  real(real_kind),parameter :: num65d3=65._real_kind/3, num20d3=20._real_kind/3
contains
'''+extrapolation+'''end module
module boundary_probe
  use astr_precision
  use commarray
  integer,parameter :: im=6,jm=8,km=3,hm=5
  integer :: jrk,jrkm,bctype(6)
contains
'''+geometry+'''end module
'''+device_module+'''program probe
  use boundary_probe
  use commfunc,only: gradextrp
'''+('  use device_probe\n' if gpu else '')+'''
  implicit none
  integer :: i,j,k,m,n,l,side,failures
  real(real_kind) :: s,eta,expected(4),error,actual_jacob
  real(real_kind),allocatable :: original(:,:,:),metrics(:,:,:,:,:),copied(:,:,:),copied_metrics(:,:,:,:,:)
  logical :: selected
  character(20) :: ownership
  call get_command_argument(1,ownership)
  jrkm=2
  select case(trim(ownership))
  case('both'); jrk=0; jrkm=0
  case('lower'); jrk=0
  case('upper'); jrk=2
  case('interior'); jrk=1
  case default; error stop 'unknown ownership'
  end select
  bctype=[1,1,41,41,1,1]
  allocate(jacob(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
  jacob=-37._real_kind; dxi=-53._real_kind
  ! Mapping x=xi, y=(1+.01xi+.02zeta)(eta+.005eta^2), z=zeta.
  do k=0,km
    do j=0,jm
      do i=0,im
        s=1+.01_real_kind*i+.02_real_kind*k;eta=real(j,real_kind)
        jacob(i,j,k)=s*(1+.01_real_kind*eta)
        dxi(i,j,k,:,:)=0
        dxi(i,j,k,1,1)=1;dxi(i,j,k,3,3)=1
        dxi(i,j,k,2,1)=-.01_real_kind*(eta+.005_real_kind*eta**2)/jacob(i,j,k)
        dxi(i,j,k,2,2)=1/jacob(i,j,k)
        dxi(i,j,k,2,3)=-.02_real_kind*(eta+.005_real_kind*eta**2)/jacob(i,j,k)
      enddo
    enddo
  enddo
  original=jacob;metrics=dxi
  call geombc()
  error=0
  do side=0,1
    selected=(side==0.and.jrk==0).or.(side==1.and.jrk==jrkm)
    do k=0,km
      do i=0,im
        do l=1,5
          j=-l
          if(side==1) j=jm+l
          actual_jacob=original(i,j,k)
          if(selected.and.l<=4) then
            actual_jacob=(1+.01_real_kind*i+.02_real_kind*k)*(1+.01_real_kind*j)
          endif
          error=max(error,abs(jacob(i,j,k)-actual_jacob))
          do n=1,3
            do m=1,3
              if(selected.and.l<=4) then
                if(side==0) expected=gradextrp(metrics(i,0,k,m,n),metrics(i,1,k,m,n))
                if(side==1) expected=gradextrp(metrics(i,jm,k,m,n),metrics(i,jm-1,k,m,n))
                error=max(error,abs(dxi(i,j,k,m,n)-expected(l)))
              else
                error=max(error,abs(dxi(i,j,k,m,n)-metrics(i,j,k,m,n)))
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo
  error=max(error,maxval(abs(jacob(0:im,0:jm,0:km)-original(0:im,0:jm,0:km))))
  error=max(error,maxval(abs(dxi(0:im,0:jm,0:km,:,:)-metrics(0:im,0:jm,0:km,:,:))))
'''+('''
  allocate(q(1,1,1,5),rho(1,1,1),vel(1,1,1,3),prs(1,1,1),tmp(1,1,1),x(1,1,1,3))
  q=1;rho=1;vel=0;prs=1;tmp=1;x=0
  allocate(q_d(1,1,1,5),rho_d(1,1,1),vel_d(1,1,1,3),prs_d(1,1,1),tmp_d(1,1,1),x_d(1,1,1,3))
  allocate(jacob_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),dxi_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
  allocate(bnorm_i0_d(1,1,1),bnorm_im_d(1,1,1),bnorm_j0_d(1,1,1), &
    bnorm_jm_d(1,1,1),bnorm_k0_d(1,1,1),bnorm_km_d(1,1,1))
  call copy_flow_to_gpu()
  allocate(copied(-hm:im+hm,-hm:jm+hm,-hm:km+hm),copied_metrics(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
  copied=jacob_d;copied_metrics=dxi_d
  error=max(error,maxval(abs(copied-jacob)),maxval(abs(copied_metrics-dxi)))
''' if gpu else '')+'''
  write(*,'(a,es24.16)') 'maximum_geometry_error=',error
  if(error>5.e-13_real_kind) error stop 'geombc column ownership or CUDA upload mismatch'
end program
'''


@pytest.fixture(scope='module',params=['cpu','gpu'])
def executable(request,tmp_path_factory):
    gpu=request.param=='gpu'
    compiler=shutil.which('nvfortran' if gpu else 'gfortran')
    if not compiler or (gpu and os.environ.get('ASTR_HB0_GPU_PROBE')!='1'):
        pytest.skip('GPU probe requires nvfortran and ASTR_HB0_GPU_PROBE=1')
    build=tmp_path_factory.mktemp('hb0_geometry_'+request.param)
    source=build/'geombc.F90';source.write_text(probe_source(gpu))
    command=[compiler,'-O0']
    command+=['-cuda'] if gpu else ['-ffree-line-length-none','-fcheck=bounds']
    result=subprocess.run(command+[str(source),'-o',str(build/'probe')],cwd=build,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return build/'probe'


@pytest.mark.parametrize('ownership',['both','lower','upper','interior'])
def test_geombc_column_ownership_and_geometry_upload(executable,ownership):
    result=subprocess.run([str(executable),ownership],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
