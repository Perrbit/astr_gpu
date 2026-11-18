! Frozen original face routines and their host/device observation driver.
module air5_faces_reference_math
  use iso_fortran_env,only:real64
  use constdef,only:num1d60,num1d12,num2d3
  use chemistry_air5_data
  use chemistry_model
  use chemistry_state_layout
  use chemistry_flow_runtime,only:air5_symmetric_species_convection
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
  use chemistry_thermo_gpu
  implicit none
  integer,device :: air5_convection_symmetric_d=0,air5_flow_status_d=0
#else
  use chemistry_thermo
  use commarray,only:momentum_flux=>sigma,energy_flux=>qflux,species_flux=>jsflux, &
    vibrational_flux=>evflux,species_energy_flux=>esflux
  implicit none
#endif
contains
#include "frozen_faces_functions.inc"
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(global) subroutine batch(axis,ntype,kind,capturing,im,jm,km,out)
    integer,value :: axis,ntype,kind,capturing,im,jm,km
    real(real64),device :: out(11,*)
    integer :: id,p(3),n(3),last(3),face
    real(real64) :: flux(11)
    id=(blockidx%x-1)*blockdim%x+threadidx%x-1
    n=[im+1,jm+1,km+1];n(axis)=n(axis)+1;last=[im,jm,km]
    if(id>=product(n))return
    p=[mod(id,n(1)),mod(id/n(1),n(2)),id/(n(1)*n(2))];p(axis)=p(axis)-1;face=p(axis)
    select case(kind)
    case(0)
      call air5_selective_convective_face_flux_gpu(p(1),p(2),p(3),axis,face,last(axis),ntype,im,jm,km,capturing,flux)
    case(1)
      call air5_low_order_convective_face_flux(p(1),p(2),p(3),axis,face,flux)
    case default
      call air5_diffusive_face_flux(p(1),p(2),p(3),axis,face,last(axis),ntype,flux,kind-2)
    end select
    out(:,id+1)=flux
  end subroutine
#endif
end module
program air5_faces_reference
  use commarray
  use chemistry_flow_runtime
  use air5_faces_reference_math
#ifdef ASTR_AIR5_REFERENCE_GPU
  use commarray_gpu
#endif
  implicit none
  integer,parameter :: h=5
  integer :: im,jm,km,ncases,header(4),i,j,k,u,v,job,axis,kind,n,packet(6),first(3),last(3),nt(3),id,point(3)
  integer,allocatable :: jobs(:,:),mask(:,:,:)
  real(real64),allocatable :: table(:,:,:,:),out(:,:)
  real(real64) :: flux(11)
  character(1024) :: input,output
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device,allocatable :: out_d(:,:)
#endif
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)header;im=header(1);jm=header(2);km=header(3);ncases=header(4)
  allocate(jobs(5,ncases),table(64,-h:im+h,-h:jm+h,-h:km+h),mask(0:im,0:jm,0:km))
  read(u)jobs,table,mask;close(u)
  allocate(q(-h:im+h,-h:jm+h,-h:km+h,11),rho(-h:im+h,-h:jm+h,-h:km+h), &
    vel(-h:im+h,-h:jm+h,-h:km+h,3),prs(-h:im+h,-h:jm+h,-h:km+h),spc(-h:im+h,-h:jm+h,-h:km+h,5), &
    dxi(-h:im+h,-h:jm+h,-h:km+h,3,3),jacob(-h:im+h,-h:jm+h,-h:km+h), &
    sigma(-h:im+h,-h:jm+h,-h:km+h,6),qflux(-h:im+h,-h:jm+h,-h:km+h,3), &
    jsflux(-h:im+h,-h:jm+h,-h:km+h,5,3),evflux(-h:im+h,-h:jm+h,-h:km+h,3), &
    esflux(-h:im+h,-h:jm+h,-h:km+h,6),lshock(0:im,0:jm,0:km))
  do k=-h,km+h;do j=-h,jm+h;do i=-h,im+h
    q(i,j,k,:)=table(1:11,i,j,k);rho(i,j,k)=table(12,i,j,k);vel(i,j,k,:)=table(13:15,i,j,k)
    prs(i,j,k)=table(16,i,j,k);spc(i,j,k,:)=table(17:21,i,j,k);dxi(i,j,k,:,:)=reshape(table(22:30,i,j,k),[3,3])
    jacob(i,j,k)=table(31,i,j,k);sigma(i,j,k,:)=table(32:37,i,j,k);qflux(i,j,k,:)=table(38:40,i,j,k)
    jsflux(i,j,k,:,:)=reshape(table(41:55,i,j,k),[5,3]);evflux(i,j,k,:)=table(56:58,i,j,k);esflux(i,j,k,:)=table(59:64,i,j,k)
  enddo;enddo;enddo
  lshock=mask/=0
#ifdef ASTR_AIR5_REFERENCE_GPU
  allocate(q_d(-h:im+h,-h:jm+h,-h:km+h,11),rho_d(-h:im+h,-h:jm+h,-h:km+h), &
    vel_d(-h:im+h,-h:jm+h,-h:km+h,3),prs_d(-h:im+h,-h:jm+h,-h:km+h),spc_d(-h:im+h,-h:jm+h,-h:km+h,5), &
    dxi_d(-h:im+h,-h:jm+h,-h:km+h,3,3),jacob_d(-h:im+h,-h:jm+h,-h:km+h), &
    sigma_d(-h:im+h,-h:jm+h,-h:km+h,6),qflux_d(-h:im+h,-h:jm+h,-h:km+h,3), &
    species_flux_d(-h:im+h,-h:jm+h,-h:km+h,5,3),ev_flux_d(-h:im+h,-h:jm+h,-h:km+h,3), &
    species_energy_flux_d(-h:im+h,-h:jm+h,-h:km+h,6),shock_mask_d(0:im,0:jm,0:km))
  q_d=q;rho_d=rho;vel_d=vel;prs_d=prs;spc_d=spc;dxi_d=dxi;jacob_d=jacob
  sigma_d=sigma;qflux_d=qflux;species_flux_d=jsflux;ev_flux_d=evflux;species_energy_flux_d=esflux;shock_mask_d=int(mask,1)
#endif
  open(newunit=v,file=trim(output),access='stream',form='unformatted',status='new');write(v)ncases
  do job=1,ncases
    axis=jobs(1,job);kind=jobs(3,job);first=0;first(axis)=-1;last=[im,jm,km];nt=last+1;nt(axis)=nt(axis)+1;n=product(nt)
    allocate(out(11,n))
    test_symmetric=jobs(5,job)/=0;test_capturing=jobs(4,job)/=0
#ifdef ASTR_AIR5_REFERENCE_GPU
    air5_convection_symmetric_d=jobs(5,job);air5_flow_status_d=0
    allocate(out_d(11,n))
    call batch<<<(n+127)/128,128>>>(axis,jobs(2,job),kind,jobs(4,job),im,jm,km,out_d)
    out=out_d;i=air5_flow_status_d
    if(i/=0)error stop 'original GPU face component failed'
    deallocate(out_d)
#else
    id=0
    do k=first(3),last(3);do j=first(2),last(2);do i=first(1),last(1)
      id=id+1;point=[i,j,k]
      select case(kind)
      case(0)
        call air5_selective_convective_face_flux(i,j,k,axis,point(axis),last(axis),jobs(2,job),flux)
      case(1)
        call air5_low_order_convective_face_flux(i,j,k,axis,point(axis),flux)
      case default
        call projected_air5_face_flux(i,j,k,axis,point(axis),last(axis),jobs(2,job),flux,kind-2)
      end select
      out(:,id)=flux
    enddo;enddo;enddo
#endif
    packet=[jobs(:,job),n];write(v)packet,out;deallocate(out)
  enddo
  close(v)
end program
