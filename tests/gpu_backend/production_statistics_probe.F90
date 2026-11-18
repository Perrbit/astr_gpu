program production_statistics_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare
  use production_statistics_cpp
  implicit none
  type(c_ptr) :: state
  integer :: rank,ranks,ierr,status,local_rank,kind,sample,debug,pass,unit,h(8),im,jm,km,hm,selector
  integer :: checked=0
  real(real_kind) :: constants(8),error,maximum=0
  real(real_kind),allocatable :: x(:,:,:,:),fields(:,:,:,:),gradient(:,:,:,:,:),thermal(:,:,:,:),bnorm(:,:,:)
  real(real_kind),allocatable :: measure(:,:),plane(:,:,:),endpoint(:,:,:),wall(:,:),wall_endpoint(:,:)
  real(real_kind),allocatable :: saved_plane(:,:,:),saved_endpoint(:,:,:),saved_wall(:,:),saved_wall_endpoint(:,:)
  real(real_kind),allocatable :: final_plane(:,:,:),final_endpoint(:,:,:),final_wall(:,:),final_wall_endpoint(:,:)
  real(real_kind),allocatable :: received(:),guard(:),restored(:)
  type(device_view) :: view
  character(1024) :: root,path
  character(80) :: text
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'production statistics probe needs Open MPI local rank'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,root)
  do kind=1,2
    do debug=0,1
      do pass=1,3
        do sample=merge(3,1,pass==3),3
          write(path,'(a,a,i1,a,i1,a)') trim(root),'.kind',kind,'.sample',sample,'.bin'
          open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
          read(unit) h,constants
          im=h(1);jm=h(2);km=h(3);hm=h(4)
          if(h(5)/=kind.or.h(6)/=sample) error stop 'production statistics capture identity'
          if(.not.allocated(x)) then
            allocate(x(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),fields(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6))
            allocate(gradient(0:im,0:jm,0:km,3,3),thermal(0:im,0:jm,0:km,3),bnorm(0:im,0:km,3))
            allocate(measure(0:im,0:jm),plane(0:im,0:jm,12),endpoint(0:im,0:jm,12))
            allocate(wall(0:im,3),wall_endpoint(0:im,3))
            allocate(saved_plane,mold=plane);allocate(saved_endpoint,mold=endpoint)
            allocate(saved_wall,mold=wall);allocate(saved_wall_endpoint,mold=wall_endpoint)
            allocate(final_plane,mold=plane);allocate(final_endpoint,mold=endpoint)
            allocate(final_wall,mold=wall);allocate(final_wall_endpoint,mold=wall_endpoint)
          endif
          read(unit) x,fields,gradient,thermal,bnorm,measure,plane,endpoint,wall,wall_endpoint;close(unit)
          if(sample==1.or.pass==3) then
            if(pass/=2) then
              call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
                int(hm,c_int),int(debug,c_int),state))
              call check(fluid_prepare(state,1_c_int))
              call check(production_geometry_cpp(state,x,bnorm,im,jm,km,hm,h(8),0,kind==2))
            else
              call check(production_reset_cpp(state))
            endif
            if(pass==3) then
              call restore(production_sum,reshape(saved_plane,[size(saved_plane)]))
              call restore(production_endpoint,reshape(saved_endpoint,[size(saved_endpoint)]))
              if(kind==2) then
                call restore(production_wall_sum,reshape(saved_wall,[size(saved_wall)]))
                call restore(production_wall_endpoint,reshape(saved_wall_endpoint,[size(saved_wall_endpoint)]))
              endif
            endif
          endif
          call check(device_state_upload(state,device_rho,fields(:,:,:,1),bytes(size(fields)/6)))
          call check(device_state_upload(state,device_velocity,fields(:,:,:,2:4),bytes(size(fields)/2)))
          call check(device_state_upload(state,device_pressure,fields(:,:,:,5),bytes(size(fields)/6)))
          call check(device_state_upload(state,device_temperature,fields(:,:,:,6),bytes(size(fields)/6)))
          call check(device_state_upload(state,device_dvelocity,gradient,bytes(size(gradient))))
          call check(device_state_upload(state,device_dtemperature,thermal,bytes(size(thermal))))
          do selector=device_q,device_save
            call check(device_state_view(state,int(selector,c_int),view))
            allocate(guard(product(view%extent)));guard=77.e0_real_kind+selector
            call check(device_state_upload(state,int(selector,c_int),guard,bytes(size(guard))))
            deallocate(guard)
          enddo
          if(kind==1) then
            call check(production_plane_cpp(state,constants(1),int(h(7),c_int)))
          else
            call check(production_wall_cpp(state,constants(1),int(h(7),c_int),1_c_int,constants(3:8)))
          endif
          do selector=0,merge(5,2,kind==2)
            select case(selector)
            case(0);call compare(selector,reshape(measure,[size(measure)]))
            case(1);call compare(selector,reshape(plane,[size(plane)]))
            case(2);call compare(selector,reshape(endpoint,[size(endpoint)]))
            case(3);call compare(selector,measure(:,0))
            case(4);call compare(selector,reshape(wall,[size(wall)]))
            case(5);call compare(selector,reshape(wall_endpoint,[size(wall_endpoint)]))
            end select
          enddo
          if(sample==2.and.pass==1) then
            call retrieve(production_sum,saved_plane)
            call retrieve(production_endpoint,saved_endpoint)
            if(kind==2) then
              call retrieve_wall(production_wall_sum,saved_wall)
              call retrieve_wall(production_wall_endpoint,saved_wall_endpoint)
            endif
          endif
          if(sample==3) then
            if(pass==1) then
              call retrieve(production_sum,final_plane);call retrieve(production_endpoint,final_endpoint)
              if(kind==2) then
                call retrieve_wall(production_wall_sum,final_wall)
                call retrieve_wall(production_wall_endpoint,final_wall_endpoint)
              endif
            else
              call exact(production_sum,reshape(final_plane,[size(final_plane)]))
              call exact(production_endpoint,reshape(final_endpoint,[size(final_endpoint)]))
              if(kind==2) then
                call exact(production_wall_sum,reshape(final_wall,[size(final_wall)]))
                call exact(production_wall_endpoint,reshape(final_wall_endpoint,[size(final_wall_endpoint)]))
              endif
            endif
          endif
          do selector=device_q,device_save
            call check(device_state_view(state,int(selector,c_int),view))
            allocate(guard(product(view%extent)))
            call check(device_state_download(state,int(selector,c_int),guard,bytes(size(guard))))
            if(any(guard/=77.e0_real_kind+selector)) error stop 'production statistics modified solver q/RHS/save'
            deallocate(guard)
          enddo
          call field_unchanged(device_rho,reshape(fields(:,:,:,1),[size(fields)/6]))
          call field_unchanged(device_velocity,reshape(fields(:,:,:,2:4),[size(fields)/2]))
          call field_unchanged(device_pressure,reshape(fields(:,:,:,5),[size(fields)/6]))
          call field_unchanged(device_temperature,reshape(fields(:,:,:,6),[size(fields)/6]))
          call field_unchanged(device_dvelocity,reshape(gradient,[size(gradient)]))
          call field_unchanged(device_dtemperature,reshape(thermal,[size(thermal)]))
          checked=checked+1
        enddo
        if(pass/=1) call check(device_state_destroy(state))
      enddo
    enddo
  enddo
  print *, 'CPP_PRODUCTION_STATISTICS_PASS ',rank,checked,maximum,' reset_repeat_and_restore_bits_equal'
  print *, 'CPP_PRODUCTION_STATISTICS_REAL_BYTES ',rank,real_bytes
  call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*int(real_bytes,c_size_t)
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0) then
      print *, 'PRODUCTION_STATISTICS_STATUS ',rank,code
      error stop 'production statistics API failed'
    endif
  end subroutine
  subroutine compare(buffer,reference)
    integer,intent(in) :: buffer
    real(real_kind),intent(in) :: reference(:)
    integer :: channels,c,first,last,nodes
    allocate(received(size(reference)))
    call check(production_transfer_cpp(state,int(buffer,c_int),0_c_int,received,bytes(size(received))))
    if(.not.all(ieee_is_finite(received))) error stop 'nonfinite production statistics'
    error=maxval(abs(received-reference))/max(1.e0_real_kind,maxval(abs(reference)))
    block
      real(real_kind) :: channel_error
      channels=1
      if(buffer==production_sum.or.buffer==production_endpoint) channels=12
      if(buffer==production_wall_sum.or.buffer==production_wall_endpoint) channels=3
      nodes=size(reference)/channels
      do c=1,channels
        first=(c-1)*nodes+1;last=c*nodes
        channel_error=maxval(abs(received(first:last)-reference(first:last)))/max(1._real_kind,maxval(abs(reference(first:last))))
        write(*,'(a,7(1x,i0),2(1x,es24.16e3))') 'CPP_PRODUCTION_STATISTICS_NATIVE_DIFFERENCE', &
          rank,kind,debug,pass,sample,buffer,c,maxval(abs(received(first:last)-reference(first:last))), &
          channel_error
        if(real_bytes==8.and.channel_error>2.e-10_real_kind) error stop 'production statistics per-channel mismatch'
      enddo
    end block
    if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) then
      print *, 'PRODUCTION_STATISTICS_DIFFERENCE ',rank,kind,debug,pass,sample,buffer,error
      error stop 'production statistics frozen GPU mismatch'
    endif
    maximum=max(maximum,error);deallocate(received)
  end subroutine
  subroutine restore(buffer,reference)
    integer(c_int),intent(in) :: buffer
    real(real_kind),intent(in) :: reference(:)
    allocate(restored,source=reference)
    call check(production_transfer_cpp(state,buffer,1_c_int,restored,bytes(size(restored))))
    deallocate(restored)
  end subroutine
  subroutine retrieve(buffer,values)
    integer(c_int),intent(in) :: buffer
    real(real_kind),intent(out) :: values(:,:,:)
    call check(production_transfer_cpp(state,buffer,0_c_int,values,bytes(size(values))))
  end subroutine
  subroutine retrieve_wall(buffer,values)
    integer(c_int),intent(in) :: buffer
    real(real_kind),intent(out) :: values(:,:)
    call check(production_transfer_cpp(state,buffer,0_c_int,values,bytes(size(values))))
  end subroutine
  subroutine exact(buffer,reference)
    integer(c_int),intent(in) :: buffer
    real(real_kind),intent(in) :: reference(:)
    allocate(received(size(reference)))
    call check(production_transfer_cpp(state,buffer,0_c_int,received,bytes(size(received))))
    if(any(real_to_word(received)/=real_to_word(reference))) error stop 'production statistics resume/repeat bits'
    deallocate(received)
  end subroutine
  subroutine field_unchanged(buffer,reference)
    integer(c_int),intent(in) :: buffer
    real(real_kind),intent(in) :: reference(:)
    allocate(received(size(reference)))
    call check(device_state_download(state,buffer,received,bytes(size(received))))
    if(any(real_to_word(received)/=real_to_word(reference))) error stop 'production statistics modified cache/gradient'
    deallocate(received)
  end subroutine
end program
