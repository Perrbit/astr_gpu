module production_statistics_cpp
  use iso_c_binding, only: c_ptr,c_int,c_size_t,c_loc,c_null_ptr
  use device_state, only: c_device_real_kind
  implicit none
  private
  integer(c_int),parameter,public :: production_measure=0,production_sum=1,production_endpoint=2, &
    production_wall_measure=3,production_wall_sum=4,production_wall_endpoint=5
  public :: production_prepare_cpp,production_reset_cpp,production_plane_cpp,production_wall_cpp,production_transfer_cpp
  public :: production_geometry_cpp
  interface
    integer(c_int) function production_prepare_cpp(parent,segments,frame) bind(C,name='astr_device_production_prepare')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent,frame
      real(c_device_real_kind),intent(in) :: segments(*)
    end function
    integer(c_int) function production_reset_cpp(parent) bind(C,name='astr_device_production_reset')
      import c_ptr,c_int
      type(c_ptr),value :: parent
    end function
    integer(c_int) function production_plane_cpp(parent,interval,previous) bind(C,name='astr_device_production_plane')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: interval
      integer(c_int),value :: previous
    end function
    integer(c_int) function production_wall_cpp(parent,interval,previous,nondimensional,constants) &
        bind(C,name='astr_device_production_wall')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      real(c_device_real_kind),value :: interval
      integer(c_int),value :: previous,nondimensional
      real(c_device_real_kind),intent(in) :: constants(6)
    end function
    integer(c_int) function production_transfer_cpp(parent,buffer,upload,host,bytes) &
        bind(C,name='astr_device_production_transfer')
      import c_ptr,c_int,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      integer(c_int),value :: buffer,upload
      real(c_device_real_kind),intent(inout) :: host(*)
      integer(c_size_t),value :: bytes
    end function
  end interface
contains
  integer(c_int) function production_geometry_cpp(parent,x,bnorm,im,jm,km,hm,ia,ig0,lower_wall) result(status)
    type(c_ptr),value :: parent
    integer,intent(in) :: im,jm,km,hm,ia,ig0
    logical,intent(in) :: lower_wall
    real(c_device_real_kind),intent(in) :: x(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),bnorm(0:im,0:km,3)
    real(c_device_real_kind),allocatable :: segments(:,:,:)
    real(c_device_real_kind),allocatable,target :: frame(:,:,:)
    real(c_device_real_kind) :: di(3),normal(3),tangent(3),norm
    integer :: i,j,k
    type(c_ptr) :: frame_pointer
    allocate(segments(0:im,0:jm,km))
    do k=1,km
      do j=0,jm
        do i=0,im
          di=x(i,j,k,:)-x(i,j,k-1,:)
          segments(i,j,k)=sqrt(di(1)*di(1)+di(2)*di(2)+di(3)*di(3))
        enddo
      enddo
    enddo
    frame_pointer=c_null_ptr
    if(lower_wall) then
      allocate(frame(0:im,0:km,6))
      do k=0,km
        do i=0,im
          normal=bnorm(i,k,:)
          if(ig0+i==0) then
            di=x(i+1,0,k,:)-x(i,0,k,:)
          elseif(ig0+i==ia) then
            di=x(i,0,k,:)-x(i-1,0,k,:)
          else
            di=x(i+1,0,k,:)-x(i-1,0,k,:)
          endif
          tangent=di-dot_product(di,normal)*normal
          norm=sqrt(dot_product(tangent,tangent))
          if(norm>1.e-300_c_device_real_kind) then
            tangent=tangent/norm
          else
            tangent=0.e0_c_device_real_kind
          endif
          frame(i,k,1:3)=normal;frame(i,k,4:6)=tangent
        enddo
      enddo
      frame_pointer=c_loc(frame)
    endif
    status=production_prepare_cpp(parent,segments,frame_pointer)
  end function
end module
