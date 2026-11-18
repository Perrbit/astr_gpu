program sponge_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use halo_transport
  use sponge_operator_cpp
  implicit none
  integer :: rank,ranks,ierr,local_rank,status,mode,unit,h(5),first(3),last(3),edge(4),stored_mode
  integer :: topology(3),coordinate(3),low(3),high(3),axis,d,c,debug,counts(2),lo(3),hi(3),cases=0
  integer(c_int) :: begin(2),ending(2),transport
  logical :: selected(2),periodic(3)
  real(real_kind) :: error,maximum=0,constant_error=0
  real(real_kind),allocatable :: initial(:,:,:,:),reference(:,:,:,:),previous(:,:,:,:),actual(:,:,:,:),xcoef(:),ycoef(:),sync_result(:,:,:,:)
  type(c_ptr) :: state
  character(1024) :: root,path
  character(80) :: text
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'sponge probe needs Open MPI local rank'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,root);call get_command_argument(2,text);read(text,*) topology
  if(product(topology)/=ranks) error stop 'sponge topology mismatch'
  call get_command_argument(3,text)
  select case(trim(text))
  case('pinned');transport=halo_pinned
  case('device');transport=halo_device
  case default;error stop 'sponge transport must be pinned/device'
  end select
  coordinate=[mod(rank,topology(1)),mod(rank/topology(1),topology(2)),rank/(topology(1)*topology(2))]
  low=MPI_PROC_NULL;high=MPI_PROC_NULL;periodic=[.false.,.false.,.true.]
  do d=1,2
    if(coordinate(d)>0) low(d)=rank-product(topology(:d-1))
    if(coordinate(d)<topology(d)-1) high(d)=rank+product(topology(:d-1))
  enddo
  low(3)=modulo(coordinate(3)-1,topology(3))*topology(1)*topology(2)+coordinate(2)*topology(1)+coordinate(1)
  high(3)=modulo(coordinate(3)+1,topology(3))*topology(1)*topology(2)+coordinate(2)*topology(1)+coordinate(1)
  do mode=1,3
    write(path,'(a,a,i1,a,i8.8,a)') trim(root),'/reference.mode',mode,'.rank',rank,'.bin'
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
    read(unit) h,first,last,edge,stored_mode
    if(h(5)/=5.or.stored_mode/=mode) error stop 'sponge record identity'
    allocate(initial(-h(4):h(1)+h(4),-h(4):h(2)+h(4),-h(4):h(3)+h(4),5))
    allocate(reference,mold=initial);allocate(previous,mold=initial);allocate(actual,mold=initial);allocate(sync_result,mold=initial)
    read(unit) initial,reference,previous;close(unit)
    if(.not.all(ieee_is_finite(reference)).or..not.all(ieee_is_finite(previous))) error stop 'nonfinite sponge reference'
    if(real_bytes==8.and.maxval(abs(reference-previous))>2.e-10_real_kind) error stop 'original CPU/old GPU sponge mismatch'
    do c=1,5
      previous(:,:,:,c)=real(c,real_kind)
    enddo
    selected=[mode/=2,mode/=1];begin=edge([1,3]);ending=edge([2,4])
    where(.not.selected) begin=-1
    counts=0
    do axis=1,2
      if(begin(axis)<0) cycle
      lo=first;hi=last;lo(axis)=max(lo(axis),begin(axis));hi(axis)=min(hi(axis),ending(axis))
      counts(axis)=product(max(0,hi-lo+1))
    enddo
    allocate(xcoef(counts(1)),ycoef(counts(2)));xcoef=.04e0_real_kind;ycoef=.03e0_real_kind
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),int(h(1),c_int),int(h(2),c_int),int(h(3),c_int), &
        int(h(4),c_int),int(debug,c_int),state))
      call check(device_state_upload(state,device_q,initial,bytes(size(initial))))
      call check(device_transport_prepare(state,int(MPI_COMM_WORLD,c_int),int(low,c_int),int(high,c_int), &
        int(topology,c_int),5_c_int,transport,0_c_int,merge(1_c_int,0_c_int,periodic)))
      call check(sponge_prepare_cpp(state,int(first,c_int),int(last,c_int),begin,ending,xcoef,ycoef))
      do axis=1,2
        if(.not.selected(axis)) cycle
        do d=0,2
          call check(device_exchange_axis(state,device_q,int(d,c_int),0_c_int))
        enddo
        call check(sponge_apply_cpp(state,int(axis-1,c_int)))
      enddo
      call check(device_state_download(state,device_q,actual,bytes(size(actual))))
      if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite new sponge result'
      error=maxval(abs(actual-reference))/max(1.e0_real_kind,maxval(abs(reference)))
      if(debug==0) then
        sync_result=actual
      elseif(any(real_to_word(actual)/=real_to_word(sync_result))) then
        error stop 'sponge synchronization words differ'
      endif
      if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) then
        print *, 'SPONGE_DIFFERENCE ',rank,mode,debug,error
        error stop 'sponge physical/shared/halo mismatch'
      endif
      maximum=max(maximum,error);cases=cases+1
      do c=1,5
        write(*,'(a,4(1x,i0),1x,es24.16e3)') 'CPP_SPONGE_NATIVE_DIFFERENCE',rank,mode,debug,c, &
          maxval(abs(actual(:,:,:,c)-reference(:,:,:,c)))/max(1._real_kind,maxval(abs(reference(:,:,:,c))))
      enddo
      call check(device_state_upload(state,device_q,previous,bytes(size(previous))))
      do axis=1,2
        if(.not.selected(axis)) cycle
        do d=0,2
          call check(device_exchange_axis(state,device_q,int(d,c_int),0_c_int))
        enddo
        call check(sponge_apply_cpp(state,int(axis-1,c_int)))
      enddo
      call check(device_state_download(state,device_q,actual,bytes(size(actual))))
      error=maxval(abs(actual-previous))
      if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite sponge constant result'
      if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'sponge constant identity'
      constant_error=max(constant_error,error)
      call check(device_state_destroy(state))
    enddo
    deallocate(initial,reference,previous,actual,xcoef,ycoef,sync_result)
  enddo
  print *, 'CPP_SPONGE_PASS ',rank,cases,maximum,constant_error
  print *, 'CPP_SPONGE_REAL_BYTES ',rank,real_bytes
  call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*int(real_bytes,c_size_t)
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0) then
      print *, 'SPONGE_STATUS ',rank,code
      error stop 'sponge API failed'
    endif
  end subroutine
end program
