program insitu_geometry_probe
  use iso_fortran_env, only: real64,int32
  use insitu_geometry, only: hex_volume,quad_area,wall_frame
  implicit none
  real(real64),allocatable :: coordinates(:,:,:,:),volumes(:,:,:),areas(:,:,:)
  real(real64) :: hex(3,0:1,0:1,0:1),quad(3,0:1,0:1)
  real(real64) :: vectors(3,3),normal(3),tangent(3)
  integer(int32) :: dims(3)
  integer :: unit,status,i,j,k,c,a,b,d,w
  logical :: ok
  character(1024) :: input,output,mode
  call get_command_argument(1,input)
  call get_command_argument(2,output)
  call get_command_argument(3,mode)
  open(newunit=unit,file=trim(input),access='stream',form='unformatted', &
    status='old',convert='little_endian',iostat=status)
  if(status/=0) stop 1
  if(trim(mode)=='frame') then
    read(unit,iostat=status) vectors
    close(unit)
    if(status/=0) stop 8
    call wall_frame(vectors(:,1),vectors(:,2),vectors(:,3),normal,tangent,ok)
    if(.not.ok) stop 9
    open(newunit=unit,file=trim(output),access='stream',form='unformatted', &
      status='new',convert='little_endian',iostat=status)
    if(status/=0) stop 6
    write(unit,iostat=status) normal,tangent
    close(unit)
    if(status/=0) stop 7
    stop
  endif
  read(unit,iostat=status) dims
  if(status/=0.or.any(dims<1).or.any(dims>64)) stop 2
  allocate(coordinates(3,0:dims(1),0:dims(2),0:dims(3)), &
    volumes(dims(1),dims(2),dims(3)),areas(dims(1),dims(3),2))
  read(unit,iostat=status) coordinates
  close(unit)
  if(status/=0) stop 3
  do k=0,dims(3)-1
  do j=0,dims(2)-1
  do i=0,dims(1)-1
    do d=0,1
    do b=0,1
    do a=0,1
      hex(:,a,b,d)=coordinates(:,i+a,j+b,k+d)
    enddo
    enddo
    enddo
    call hex_volume(hex,volumes(i+1,j+1,k+1),ok)
    if(.not.ok) then
      print *, 'invalid positive cell Jacobian',i,j,k
      stop 4
    endif
  enddo
  enddo
  enddo
  do w=1,2
    j=(w-1)*dims(2)
    do k=0,dims(3)-1
    do i=0,dims(1)-1
      do b=0,1
      do a=0,1
        quad(:,a,b)=coordinates(:,i+a,j,k+b)
      enddo
      enddo
      call quad_area(quad,areas(i+1,k+1,w),ok)
      if(.not.ok) stop 5
    enddo
    enddo
  enddo
  open(newunit=unit,file=trim(output),access='stream',form='unformatted', &
    status='new',convert='little_endian',iostat=status)
  if(status/=0) stop 6
  write(unit,iostat=status) volumes,areas
  close(unit)
  if(status/=0) stop 7
end program
