program upwind_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use mpi
  use device_state
  use halo_transport
  use fluid_explicit
  use shock_sensor_cpp
  use upwind_operator_cpp
  use fluid_step, only: fluid_advance_prepared_stage
  implicit none
  type(c_ptr) :: state
  type(device_view) :: before,after
  integer :: rank,ranks,local_rank,ierr,cart,split_axis=1,h(9),g(3),upper(3),origin(3),hm, &
    family,backend,stage,debug,unit,record_unit,a,c,i,j,k,gradient_mode,cases=0,step_cases=0,step_header(4)
  integer(c_int) :: dims(3),lower(3),higher(3),periodic(3),low(3),high(3),first(3),last(3),mode=halo_pinned
  logical :: periods(3)
  logical :: expanded_mask=.false.,mixed_flux_probe=.false.,single_flux=.false.,characteristic_storage_probe=.false.
  logical :: high_open_probe=.false.
  real(c_float),allocatable :: interface_values(:,:,:,:)
  real(c_float) :: padding
  real(real_kind),allocatable :: native_work(:,:,:,:),native_interfaces(:,:,:,:,:)
  integer(c_int64_t) :: storage_cases=0,storage_values=0,storage_marked=0,storage_unmarked=0,storage_bytes=0
  real(real_kind),allocatable :: gq(:,:,:,:),gp(:,:,:,:),gg(:,:,:,:,:),gj(:,:,:),gd(:,:,:,:,:), &
    gr(:,:,:,:),sensor_ref(:,:,:),q(:,:,:,:),p(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:), &
    gradient(:,:,:,:,:),observed_gradient(:,:,:,:,:),rhs(:,:,:,:),expected(:,:,:,:),scratch(:,:,:,:),raw(:,:,:), &
    global_base(:,:,:,:),global_full(:,:,:,:),global_post(:,:,:,:),base(:,:,:,:),full(:,:,:,:),post(:,:,:,:)
  integer(c_int8_t),allocatable :: gm(:,:,:),mask(:,:,:),expected_mask(:,:,:)
  real(real_kind),allocatable :: gt(:,:,:,:),expected_temperature(:,:,:,:),observed_temperature(:,:,:,:)
  real(real_kind),allocatable :: gold_rhs(:,:,:,:,:)
  real(real_kind) :: temperature_maximum=0
  integer :: temperature_cases=0
  real(real_kind) :: constants(4),error,maximum=0,gradient_maximum=0,nan,step_rhs_max=0,step_q_max=0,dt
  character(1024) :: root,path,steps_root
  character(80) :: text,suffix
  character(3) :: backend_name
  character(10),parameter :: families(3)=[character(10)::'mp7','mp7_roe','mp7ld_flat']
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'upwind probe needs matching Open MPI launcher'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=1.and.ranks/=2) error stop 'upwind probe covers NP=1/2'
  call get_command_argument(1,root)
  if(ranks==2) then
    call get_command_argument(2,text);read(text,*) split_axis
  endif
  if(split_axis<1.or.split_axis>3) error stop 'upwind split axis must be 1/2/3'
  call get_command_argument(3,text)
  if(trim(text)=='device') then
    mode=halo_device
  elseif(len_trim(text)>0.and.trim(text)/='pinned') then
    error stop 'upwind transport must be pinned/device'
  endif
  call get_command_argument(4,steps_root)
  call get_command_argument(5,text);expanded_mask=trim(text)=='compact-expanded'
  high_open_probe=trim(text)=='high-open'
  call get_command_argument(6,text);mixed_flux_probe=trim(text)=='mixed-flux'
  characteristic_storage_probe=trim(text)=='characteristic-storage'
  if(expanded_mask.and.len_trim(steps_root)>0) error stop 'compact convection probe does not run prepared RK stages'
  write(path,'("observations.rank",i8.8,".bin")') rank
  open(newunit=record_unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(record_unit) real_bytes,rank,split_axis
  nan=ieee_value(0.e0_real_kind,ieee_quiet_nan)
  do family=1,3
  if(expanded_mask.and.family/=3) cycle
  if(high_open_probe.and.family/=3) cycle
  if(mixed_flux_probe.and.family/=1) cycle
  if(characteristic_storage_probe.and.family/=2) cycle
  do backend=1,2
    single_flux=mixed_flux_probe.and.backend==2
    backend_name='cpu'
    if(backend==2) backend_name='gpu'
    do stage=1,3
      write(suffix,'(a,i2.2,a)') '.rk',stage,'.bin'
      path=trim(root)//'/'//trim(families(family))//'_'//backend_name//trim(suffix)
      open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
      read(unit) h,constants;g=h(:3);hm=h(4);periodic=h(7:9);upper=g;origin=0
      if(ranks==2) then
        upper(split_axis)=g(split_axis)/2;origin(split_axis)=rank*upper(split_axis)
      endif
      allocate(gq(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,5),gp(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,6), &
        gg(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,3,3),gj(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm), &
        gd(0:g(1),0:g(2),0:g(3),3,3),sensor_ref(0:g(1),0:g(2),0:g(3)),gm(0:g(1),0:g(2),0:g(3)), &
        gr(0:g(1),0:g(2),0:g(3),5))
      read(unit) gq,gp,gg,gj,gd
      if(high_open_probe) then
        allocate(gt(0:g(1),0:g(2),0:g(3),3));read(unit) gt
      endif
      read(unit) sensor_ref,gm,gr;close(unit)
      allocate(q(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
        p(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
        scratch(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
        metric(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,3,3), &
        jac(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
        raw(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
        gradient(0:upper(1),0:upper(2),0:upper(3),3,3),observed_gradient(0:upper(1),0:upper(2),0:upper(3),3,3), &
        rhs(0:upper(1),0:upper(2),0:upper(3),5),expected(0:upper(1),0:upper(2),0:upper(3),5), &
        mask(0:upper(1),0:upper(2),0:upper(3)),expected_mask(0:upper(1),0:upper(2),0:upper(3)))
      allocate(gold_rhs(0:upper(1),0:upper(2),0:upper(3),5,0:1))
      q=gq(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm,:)
      p=gp(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm,:)
      metric=gg(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm,:,:)
      jac=gj(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm)
      gradient=gd(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:,:)
      expected=gr(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:)
      expected_mask=gm(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3))
      if(high_open_probe) then
        allocate(expected_temperature(0:upper(1),0:upper(2),0:upper(3),3))
        expected_temperature=gt(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:)
        allocate(observed_temperature,mold=expected_temperature)
      endif
      if(len_trim(steps_root)>0) then
        path=trim(steps_root)//'/'//trim(families(family))//'_'//backend_name//trim(suffix)
        allocate(global_base(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,5), &
          global_full(0:g(1),0:g(2),0:g(3),5),global_post(0:g(1),0:g(2),0:g(3),5), &
          base(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
          full(0:upper(1),0:upper(2),0:upper(3),5),post(0:upper(1),0:upper(2),0:upper(3),5))
        open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
        read(unit) step_header,global_base,global_full,global_post;close(unit)
        if(any(step_header/=h(:4))) error stop 'prepared RK reference layout changed'
        base=global_base(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm, &
          origin(3)-hm:origin(3)+upper(3)+hm,:)
        full=global_full(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:)
        post=global_post(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:)
      endif
      low=merge(1,0,periodic==0.and.origin==0);high=merge(1,0,periodic==0.and.origin+upper==g)
      first=low;last=upper-high;dims=1;dims(split_axis)=ranks;periods=periodic/=0
      if(high_open_probe) last=upper
      call MPI_Cart_create(MPI_COMM_WORLD,3,dims,periods,.false.,cart,ierr);call check(ierr)
      do a=1,3
        call MPI_Cart_shift(cart,a-1,1,lower(a),higher(a),ierr);call check(ierr)
      enddo
      do debug=0,1
        call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),state))
        call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,int(merge(1,0,family==3),c_int)))
        if(device_state_view(state,device_interface_flux,before)/=device_not_allocated) error stop 'unrequested interface workspace'
        if(single_flux) then
          call check(upwind_prepare_mixed_flux_cpp(state))
          call check(device_state_view(state,device_interface_flux,before))
          call check(upwind_prepare_mixed_flux_cpp(state))
        else
          call check(upwind_prepare_cpp(state));call check(device_state_view(state,device_interface_flux,before))
          call check(upwind_prepare_cpp(state))
        endif
        call check(device_state_view(state,device_interface_flux,after))
        if(before%element_bytes/=merge(4,real_bytes,single_flux)) error stop 'upwind interface storage kind differs'
        if(.not.c_associated(before%data,after%data)) error stop 'upwind prepare changed workspace'
        if(single_flux.or.characteristic_storage_probe) then
          allocate(interface_values(-1:upper(1)+1,-1:upper(2)+1,-1:upper(3)+1,5))
          padding=transfer(int(z'7fc13579',c_int32_t),padding)
          if(single_flux.and.debug==0.and.stage==1) write(*,'(a,3(1x,i0))') 'CPP_MIXED_FLUX_STORAGE',rank, &
            size(interface_values,kind=c_size_t)*4,size(interface_values,kind=c_size_t)*8
        endif
        if(characteristic_storage_probe) allocate(native_work(-1:upper(1)+1,-1:upper(2)+1,-1:upper(3)+1,5), &
          native_interfaces(-1:upper(1)+1,-1:upper(2)+1,-1:upper(3)+1,5,0:2))
        call check(shock_prepare_cpp(state))
        call check(device_transport_prepare(state,int(cart,c_int),lower,higher,dims,6_c_int,mode,0_c_int,periodic))
        call check(device_state_upload(state,device_q,q,bytes(size(q))))
        call check(device_state_upload(state,device_rho,p(:,:,:,1),bytes(size(jac))))
        call check(device_state_upload(state,device_velocity,p(:,:,:,2:4),bytes(3*size(jac))))
        call check(device_state_upload(state,device_pressure,p(:,:,:,5),bytes(size(jac))))
        call check(device_state_upload(state,device_temperature,p(:,:,:,6),bytes(size(jac))))
        call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
        call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))))
        do a=0,2
          call check(device_exchange_axis(state,device_q,int(a,c_int),0_c_int))
          call check(device_exchange_axis(state,device_rho,int(a,c_int),0_c_int))
          call check(device_exchange_axis(state,device_velocity,int(a,c_int),0_c_int))
          call check(device_exchange_axis(state,device_pressure,int(a,c_int),0_c_int))
          call check(device_exchange_axis(state,device_temperature,int(a,c_int),0_c_int))
        enddo
        do gradient_mode=0,merge(0,1,expanded_mask)
          if(gradient_mode==0) then
            call check(device_state_upload(state,device_dvelocity,gradient,bytes(size(gradient))))
          else
            if(high_open_probe) then
              call check(fluid_gradient_high_open(state,low,high))
            else
              call check(fluid_gradient_basic_boundary(state,low,high))
            endif
            call check(device_state_download(state,device_dvelocity,observed_gradient,bytes(size(gradient))))
            do a=1,3
              do c=1,3
                error=maxval(abs(observed_gradient(:,:,:,c,a)-gradient(:,:,:,c,a)))/max(1.e0_real_kind,maxval(abs(gradient(:,:,:,c,a))))
                if(.not.ieee_is_finite(error)) error stop 'nonfinite coupled upwind gradient difference'
                if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'coupled upwind gradient differs'
                gradient_maximum=max(gradient_maximum,error)
              enddo
            enddo
            if(high_open_probe) then
              call check(device_state_download(state,device_dtemperature,observed_temperature,bytes(size(observed_temperature))))
              do c=1,3
                error=maxval(abs(observed_temperature(:,:,:,c)-expected_temperature(:,:,:,c)))/ &
                  max(1.e0_real_kind,maxval(abs(expected_temperature(:,:,:,c))))
                if(.not.ieee_is_finite(error)) error stop 'nonfinite high-open temperature gradient difference'
                if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'high-open temperature gradient differs'
                temperature_maximum=max(temperature_maximum,error)
              enddo
              temperature_cases=temperature_cases+1
            endif
          endif
          raw=nan;call check(device_state_upload(state,device_shock_sensor,raw,bytes(size(raw))))
          call check(shock_raw_cpp(state,low,high))
          do a=0,2
            call check(device_exchange_axis(state,device_shock_sensor,int(a,c_int),0_c_int))
          enddo
          call check(shock_expand_cpp(state,low,high,constants(4)))
          if(expanded_mask) call check(shock_mask_exchange_cpp(state))
          call check(shock_mask_download_cpp(state,mask,int(size(mask),c_size_t)))
          if(any(mask/=expected_mask)) error stop 'coupled upwind mask differs'
          call check(fluid_rhs_clear(state))
          do a=0,2
            if(single_flux) then
              interface_values=padding
              call check(device_state_upload_f32(state,device_interface_flux,interface_values, &
                int(size(interface_values),c_size_t)*4))
            endif
            if(characteristic_storage_probe) then
              native_work=nan
              call check(device_state_upload(state,device_interface_flux,native_work,bytes(size(native_work))))
            endif
            if(high_open_probe) then
              call check(upwind_high_order_axis_cpp(state,int(a,c_int),constants(1),constants(2),constants(3),low,high,first,last))
            else
              call check(upwind_axis_cpp(state,int(a,c_int),h(5),h(6),constants(1),constants(2),constants(3), &
                0_c_int,low,high,first,last))
            endif
            if(single_flux) call check_interface_padding(a+1)
            if(characteristic_storage_probe) then
              call check(device_state_download(state,device_interface_flux,native_work,bytes(size(native_work))))
              native_interfaces(:,:,:,:,a)=native_work
            endif
          enddo
          call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
          if(.not.all(ieee_is_finite(rhs))) error stop 'nonfinite upwind RHS'
          if(debug==0) then
            gold_rhs(:,:,:,:,gradient_mode)=rhs
          elseif(any(real_to_word(rhs)/=real_to_word(gold_rhs(:,:,:,:,gradient_mode)))) then
            error stop 'upwind synchronization changed RHS words'
          endif
          write(record_unit) family,backend,stage,debug,gradient_mode,upper,origin,rhs,mask
          do c=1,5
            error=maxval(abs(rhs(:,:,:,c)-expected(:,:,:,c)))/max(1.e0_real_kind,maxval(abs(expected(:,:,:,c))))
            if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) then
              write(*,*) 'UPWIND_RHS_ERROR',rank,family,backend,stage,debug,gradient_mode,c,error
              error stop 'upwind RHS differs from frozen CPU/old GPU'
            endif
            maximum=max(maximum,error)
          enddo
          if(characteristic_storage_probe) call check_characteristic_storage()
          cases=cases+1
        enddo
        call check(device_state_download(state,device_q,scratch,bytes(size(q))))
        if(any(real_to_word(q(0:upper(1),0:upper(2),0:upper(3),:))/= &
          real_to_word(scratch(0:upper(1),0:upper(2),0:upper(3),:)))) error stop 'upwind changed physical q'
        call check(device_state_view(state,device_interface_flux,after))
        if(.not.c_associated(before%data,after%data)) error stop 'upwind workspace was not reused'
        if(len_trim(steps_root)>0) call prepared_rk_check()
        if(allocated(interface_values)) deallocate(interface_values)
        if(allocated(native_work)) deallocate(native_work,native_interfaces)
        call check(device_state_destroy(state))
      enddo
      call MPI_Comm_free(cart,ierr);call check(ierr)
      if(allocated(base)) deallocate(global_base,global_full,global_post,base,full,post)
      if(allocated(gt)) deallocate(gt,expected_temperature,observed_temperature)
      deallocate(gq,gp,gg,gj,gd,sensor_ref,gm,gr,q,p,scratch,metric,jac,raw,gradient,observed_gradient,rhs,expected,mask,expected_mask,gold_rhs)
    enddo
  enddo
  enddo
  write(*,'(a,2(1x,i0),2(1x,es24.16e3))') 'CPP_UPWIND_PASS',rank,cases,maximum,gradient_maximum
  write(*,'(a,2(1x,i0))') 'CPP_UPWIND_ELEMENT_BYTES',rank,real_bytes
  if(high_open_probe) write(*,'(a,2(1x,i0),1x,es24.16e3)') &
    'CPP_HIGH_OPEN_TEMPERATURE_PASS',rank,temperature_cases,temperature_maximum
  if(len_trim(steps_root)>0) write(*,'(a,2(1x,i0),2(1x,es24.16e3))') &
    'CPP_UPWIND_STEP_PASS',rank,step_cases,step_rhs_max,step_q_max
  if(characteristic_storage_probe) write(*,'(a,7(1x,i0))') 'CPP_CHARACTERISTIC_STORAGE_PASS',rank, &
    storage_cases,storage_values,storage_marked,storage_unmarked,storage_bytes*2,storage_bytes
  close(record_unit)
  call MPI_Finalize(ierr)
contains
  subroutine check_characteristic_storage()
    type(c_ptr) :: candidate
    type(device_view) :: view,repeat_view
    real(real_kind),allocatable :: q_before(:,:,:,:),cache(:,:,:,:),source_raw(:,:,:),candidate_raw(:,:,:), &
      predicted(:,:,:,:),actual_rhs(:,:,:,:)
    integer(c_int8_t),allocatable :: candidate_mask(:,:,:)
    real(c_float) :: expected_value
    integer :: axis,ii,jj,kk,component,point(3),left(3),right(3)
    logical :: active,marked
    allocate(q_before,source=q)
    allocate(cache(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
      source_raw(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
      candidate_raw(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
      predicted(0:upper(1),0:upper(2),0:upper(3),5),actual_rhs(0:upper(1),0:upper(2),0:upper(3),5), &
      candidate_mask(0:upper(1),0:upper(2),0:upper(3)))
    call check(device_state_download(state,device_q,q_before,bytes(size(q_before))))
    call check(device_state_download(state,device_rho,cache(:,:,:,1),bytes(size(jac))))
    call check(device_state_download(state,device_velocity,cache(:,:,:,2:4),bytes(3*size(jac))))
    call check(device_state_download(state,device_pressure,cache(:,:,:,5),bytes(size(jac))))
    call check(device_state_download(state,device_temperature,cache(:,:,:,6),bytes(size(jac))))
    call check(device_state_download(state,device_dvelocity,observed_gradient,bytes(size(observed_gradient))))
    call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),candidate))
    call check(fluid_prepare(candidate,1_c_int));call check(fluid_rhs_prepare(candidate,0_c_int))
    call check(upwind_prepare_mixed_characteristic_cpp(candidate))
    call check(device_state_view(candidate,device_interface_flux,view))
    call check(upwind_prepare_mixed_characteristic_cpp(candidate))
    call check(device_state_view(candidate,device_interface_flux,repeat_view))
    if(view%element_bytes/=4.or..not.c_associated(view%data,repeat_view%data)) error stop 'MP3 storage identity'
    storage_bytes=size(interface_values,kind=c_int64_t)*4
    call check(shock_prepare_cpp(candidate))
    call check(device_transport_prepare(candidate,int(cart,c_int),lower,higher,dims,6_c_int,mode,0_c_int,periodic))
    call check(device_state_upload(candidate,device_q,q_before,bytes(size(q_before))))
    call check(device_state_upload(candidate,device_rho,cache(:,:,:,1),bytes(size(jac))))
    call check(device_state_upload(candidate,device_velocity,cache(:,:,:,2:4),bytes(3*size(jac))))
    call check(device_state_upload(candidate,device_pressure,cache(:,:,:,5),bytes(size(jac))))
    call check(device_state_upload(candidate,device_temperature,cache(:,:,:,6),bytes(size(jac))))
    call check(device_state_upload(candidate,device_metric,metric,bytes(size(metric))))
    call check(device_state_upload(candidate,device_jacobian,jac,bytes(size(jac))))
    call check(device_state_upload(candidate,device_dvelocity,observed_gradient,bytes(size(observed_gradient))))
    candidate_raw=nan
    call check(device_state_upload(candidate,device_shock_sensor,candidate_raw,bytes(size(candidate_raw))))
    call check(shock_raw_cpp(candidate,low,high))
    do axis=0,2
      call check(device_exchange_axis(candidate,device_shock_sensor,int(axis,c_int),0_c_int))
    enddo
    call check(shock_expand_cpp(candidate,low,high,constants(4)))
    call check(shock_mask_download_cpp(candidate,candidate_mask,int(size(candidate_mask),c_size_t)))
    call check(device_state_download(state,device_shock_sensor,source_raw,bytes(size(source_raw))))
    call check(device_state_download(candidate,device_shock_sensor,candidate_raw,bytes(size(candidate_raw))))
    if(any(candidate_mask/=mask)) error stop 'MP3 changed integer sensor masks'
    if(any(real_to_word(source_raw)/=real_to_word(candidate_raw))) error stop 'MP3 changed raw sensor bits'
    call check(fluid_rhs_clear(candidate));predicted=0.e0_real_kind
    do axis=0,2
      interface_values=padding
      call check(device_state_upload_f32(candidate,device_interface_flux,interface_values,int(size(interface_values),c_size_t)*4))
      call check(upwind_axis_cpp(candidate,int(axis,c_int),h(5),h(6),constants(1),constants(2),constants(3), &
        0_c_int,low,high,first,last))
      call check(device_state_download_f32(candidate,device_interface_flux,interface_values,int(size(interface_values),c_size_t)*4))
      do kk=-1,upper(3)+1
      do jj=-1,upper(2)+1
      do ii=-1,upper(1)+1
        point=[ii,jj,kk]
        active=all(point>=merge(-1,0,[0,1,2]==axis)).and.all(point<=upper)
        if(active) then
          left=point;right=point;left(axis+1)=max(0,left(axis+1));right(axis+1)=min(upper(axis+1),right(axis+1)+1)
          marked=mask(left(1),left(2),left(3))/=0.or.mask(right(1),right(2),right(3))/=0
          if(marked) then
            storage_marked=storage_marked+1
          else
            storage_unmarked=storage_unmarked+1
          endif
        endif
        do component=1,5
          if(active) then
            expected_value=real(native_interfaces(ii,jj,kk,component,axis),c_float)
            if(.not.ieee_is_finite(interface_values(ii,jj,kk,component))) error stop 'MP3 nonfinite stored flux'
            if(transfer(interface_values(ii,jj,kk,component),0_c_int32_t)/=transfer(expected_value,0_c_int32_t)) &
              error stop 'MP3 is not one final FP32 store of the FP64 physical flux'
            storage_values=storage_values+1
          elseif(transfer(interface_values(ii,jj,kk,component),0_c_int32_t)/=transfer(padding,0_c_int32_t)) then
            error stop 'MP3 changed unused interface padding'
          endif
        enddo
      enddo
      enddo
      enddo
      do component=1,5
      do kk=0,upper(3)
      do jj=0,upper(2)
      do ii=0,upper(1)
        right=[ii,jj,kk];left=right;left(axis+1)=left(axis+1)-1
        predicted(ii,jj,kk,component)=predicted(ii,jj,kk,component)+( &
          real(real(native_interfaces(left(1),left(2),left(3),component,axis),c_float),real_kind)- &
          real(real(native_interfaces(right(1),right(2),right(3),component,axis),c_float),real_kind))
      enddo
      enddo
      enddo
      enddo
    enddo
    call check(device_state_download(candidate,device_rhs,actual_rhs,bytes(size(actual_rhs))))
    if(.not.all(ieee_is_finite(actual_rhs)).or.any(actual_rhs/=predicted)) error stop 'MP3 FP64 difference of stored interfaces'
    call check(device_state_download(candidate,device_q,scratch,bytes(size(scratch))))
    if(any(real_to_word(q_before)/=real_to_word(scratch))) &
      error stop 'MP3 operator changed input q bytes'
    call check(device_state_destroy(candidate))
    storage_cases=storage_cases+1
    deallocate(q_before,cache,source_raw,candidate_raw,predicted,actual_rhs,candidate_mask)
  end subroutine
  subroutine check_interface_padding(axis)
    integer,intent(in) :: axis
    integer :: point(3),component
    logical :: active
    call check(device_state_download_f32(state,device_interface_flux,interface_values, &
      int(size(interface_values),c_size_t)*4))
    do k=-1,upper(3)+1
    do j=-1,upper(2)+1
    do i=-1,upper(1)+1
      point=[i,j,k]
      active=all(point>=merge(-1,0,[1,2,3]==axis)).and.all(point<=upper)
      do component=1,5
        if(active) then
          if(.not.ieee_is_finite(interface_values(i,j,k,component))) error stop 'nonfinite mixed interface'
        elseif(transfer(interface_values(i,j,k,component),0_c_int32_t)/=transfer(padding,0_c_int32_t)) then
          error stop 'mixed interface changed unused padding'
        endif
      enddo
    enddo
    enddo
    enddo
  end subroutine
  subroutine prepared_rk_check()
    integer :: component
    real(real_kind) :: reynolds,tc,tc1,c5
    dt=1.e-5_real_kind;reynolds=1600.e0_real_kind
    if(mixed_flux_probe) dt=1.e-4_real_kind
    if(family==3) then
      dt=1.e-6_real_kind;reynolds=1000.e0_real_kind
    endif
    tc=110.3e0_real_kind/273.15e0_real_kind;tc1=1.e0_real_kind+tc;c5=(constants(1)-1.e0_real_kind)*constants(2)**2
    call check(device_state_upload(state,device_q,base,bytes(size(base))))
    call check(fluid_rk_save(state))
    call check(device_state_upload(state,device_q,q,bytes(size(q))))
    call check(fluid_advance_prepared_stage(state,stage,dt,constants(1)*constants(2)**2,1.e0_real_kind/(constants(1)-1.e0_real_kind), &
      reynolds,.72e0_real_kind,c5,tc,tc1,family==3,device_exchange_axis,device_exchange_begin,device_exchange_end, &
      gradient_stage,convection_stage,viscous_stage,sensor_operator=sensor_stage))
    call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
    call check(device_state_download(state,device_q,scratch,bytes(size(scratch))))
    if(.not.all(ieee_is_finite(scratch(0:upper(1),0:upper(2),0:upper(3),:)))) error stop 'nonfinite prepared RK state'
    do component=1,5
      error=maxval(abs(rhs(:,:,:,component)-full(:,:,:,component)))/max(1.e0_real_kind,maxval(abs(full(:,:,:,component))))
      if(.not.all(ieee_is_finite(rhs(:,:,:,component))).or.(real_bytes==8.and..not.(error<=2.e-10_real_kind))) then
        write(*,*) 'UPWIND_STEP_RHS_ERROR',rank,family,backend,stage,debug,component,error
        error stop 'prepared upwind/viscous RHS differs'
      endif
      step_rhs_max=max(step_rhs_max,error)
      error=maxval(abs(scratch(0:upper(1),0:upper(2),0:upper(3),component)-post(:,:,:,component)))/ &
        max(1.e0_real_kind,maxval(abs(post(:,:,:,component))))
      if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) then
        write(*,*) 'UPWIND_STEP_Q_ERROR',rank,family,backend,stage,debug,component,error
        error stop 'prepared upwind RK result differs'
      endif
      step_q_max=max(step_q_max,error)
    enddo
    step_cases=step_cases+1
  end subroutine
  integer(c_int) function gradient_stage(handle) result(status)
    type(c_ptr),value :: handle
    status=fluid_gradient_basic_boundary(handle,low,high)
  end function
  integer(c_int) function sensor_stage(handle) result(status)
    type(c_ptr),value :: handle
    integer(c_int) :: axis
    status=0
    if(family/=3) status=gradient_stage(handle)
    if(status/=0) return
    status=shock_raw_cpp(handle,low,high)
    if(status/=0) return
    do axis=0,2
      status=device_exchange_axis(handle,device_shock_sensor,axis,0_c_int)
      if(status/=0) return
    enddo
    status=shock_expand_cpp(handle,low,high,constants(4))
    if(status==0.and.expanded_mask) status=shock_mask_exchange_cpp(handle)
  end function
  integer(c_int) function convection_stage(handle) result(status)
    type(c_ptr),value :: handle
    integer(c_int) :: axis
    status=fluid_rhs_clear(handle)
    do axis=0,2
      if(status/=0) return
      status=upwind_axis_cpp(handle,axis,h(5),h(6),constants(1),constants(2),constants(3),0_c_int,low,high,first,last)
    enddo
  end function
  integer(c_int) function viscous_stage(handle) result(status)
    type(c_ptr),value :: handle
    status=fluid_flux_divergence_basic(handle,1_c_int,low,high,first,last)
  end function
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*real_bytes
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    write(*,*) 'UPWIND_STATUS',rank,family,backend,stage,debug,status
    error stop 'upwind operation failed'
  end subroutine
end program
