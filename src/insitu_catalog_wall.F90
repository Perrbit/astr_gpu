module insitu_catalog_wall
  use mpi
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_int,c_char,c_null_char,c_int64_t,c_double
  use astr_precision, only: real_kind,real_bytes,c_real_kind
  use astr_precision_mpi, only: mpi_real_type
  use ieee_arithmetic, only: ieee_is_finite
  use commvar, only: use_gpu,numq,im,jm,km,ia,ja,ka,nondimen,roinf,uinf,pinf
  use parallel, only: ig0,jg0,kg0
  use insitu_run_config, only: insitu_options,catalog_product
  use insitu_session, only: catalog_product_options,catalog_wall_statistics_info,begin_catalog_renderer
  use insitu_fields, only: catalog_wall_fields_supported,capture_channel_walls,capture_air5_walls,admit_optional_fields
  use insitu_wall_statistics, only: wall_statistics_render_values
  use insitu_product_schedule, only: runtime_product_state,runtime_catalog_count,runtime_catalog_state, &
    runtime_catalog_due,runtime_catalog_success,runtime_catalog_wait,runtime_catalog_disable,runtime_catalog_reject, &
    runtime_active,runtime_waiting
  use index_slice_archive, only: runtime_product_segment,refresh_runtime_product_status
  use checkpoint_state_io, only: checkpoint_state_identity
  use benchmark_runtime, only: benchmark_field_io_disabled,insitu_clock,report_insitu_timing
#ifdef _CUDA
  use insitu_products_gpu, only: catalog_wall_gpu_supported,begin_catalog_wall_gpu,end_catalog_wall_gpu,catalog_wall_line_gpu
  use insitu_products_gpu, only: catalog_sbli_wall_gpu_supported,capture_catalog_sbli_wall_gpu
  use index_slice_gpu, only: validate_index_slice_transport
#endif
  implicit none
  private
  public :: observe_catalog_walls
  type :: wall_frame
    type(runtime_product_state) :: state
    real(real_kind) :: window(2)=0,duration=0,factor=1
    integer(int64) :: count=0
    integer :: source=0,supplier=0,column=0
    logical :: due=.false.
  end type
  type :: wall_line
    integer :: index=-1
    character(8) :: side=''
    real(real_kind),allocatable :: values(:,:)
  end type
contains
  subroutine mpi_ok(status)
    integer,intent(in) :: status
    integer :: ignored
    if(status/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,status,ignored)
  end subroutine

  subroutine agree_error(message)
    character(*),intent(inout) :: message
    integer :: rank,candidate,owner,err
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    candidate=huge(candidate)
    if(len_trim(message)>0) candidate=rank
    call MPI_Allreduce(candidate,owner,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,err);call mpi_ok(err)
    if(owner==huge(owner)) return
    call MPI_Bcast(message,len(message),MPI_CHARACTER,owner,MPI_COMM_WORLD,err);call mpi_ok(err)
  end subroutine

  subroutine wall_source(definition,identity,frame,ready,ok,message,host)
    type(catalog_product),intent(in) :: definition
    type(checkpoint_state_identity),intent(in) :: identity
    type(wall_frame),intent(inout) :: frame
    logical,intent(out) :: ready,ok
    logical,intent(in) :: host
    character(*),intent(out) :: message
    logical :: active
    ready=.false.;ok=.false.;message=''
    if((definition%view%sampling/='wall'.and.definition%view%sampling/='wall_midspan').or.definition%axis/=2) then
      message='existing wall fields support the j-normal lower/upper walls';return
    endif
    if(definition%view%sampling=='wall_midspan') then
      if(host) then
        message='existing SBLI physical-midspan producer requires explicit device processing';return
      endif
      frame%source=0;frame%supplier=2;frame%factor=1;frame%window=0;frame%duration=0;frame%count=0
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_PRODUCTS)
      if(.not.use_gpu.or..not.catalog_sbli_wall_gpu_supported()) then
        message='SBLI midspan producer requires an admitted rectilinear MP7-LD x-slab with a stationary impermeable wall';return
      endif
#else
      message='existing SBLI physical-midspan wall producer requires the GPU device-product build';return
#endif
      if(definition%view%side/='lower') then
        message='existing SBLI physical-midspan curves contain only the lower wall';return
      endif
      if(definition%source/='instantaneous') then
        message='existing SBLI physical-midspan curves do not have a time-mean accumulator';return
      endif
      select case(trim(definition%view%field))
      case('p_pinf');frame%column=4
      case('skin_friction');frame%column=5
      case('wall_heat_into_wall_normalized');frame%column=6
      case default
        message='SBLI midspan curves supply p_pinf, skin_friction and wall_heat_into_wall_normalized';return
      end select
      ready=.true.;ok=.true.;return
    endif
    if(.not.catalog_wall_fields_supported()) then
      message='existing wall field producer is not admitted for this flow/layout';return
    endif
#ifdef _CUDA
    if(use_gpu.and..not.host.and..not.catalog_wall_gpu_supported()) then
      message='existing GPU wall field producer is not admitted for this flow/layout';return
    endif
#endif
    if(numq==11.and.definition%view%side=='upper') then
      message='existing AIR5 wall fields contain only the lower noncatalytic wall';return
    endif
    frame%factor=1
    select case(trim(definition%view%field))
    case('wall_pressure');frame%column=4
    case('wall_shear_x');frame%column=5
    case('wall_heat_into_gas');frame%column=6
    case('skin_friction')
      frame%column=5
      if(.not.all(ieee_is_finite([roinf,uinf])).or.roinf<=0.or.uinf==0) then
        message='skin_friction needs finite positive reference density and nonzero reference speed';return
      endif
      frame%factor=2/roinf/uinf/uinf
    case('p_pinf')
      frame%column=4
      if(.not.ieee_is_finite(pinf).or.pinf<=0) then
        message='p_pinf needs a finite positive reference pressure';return
      endif
      frame%factor=1/pinf
    case default
      message='existing wall curve producer supports pressure, signed shear, heat into gas, Cf and p_pinf';return
    end select
    if(.not.ieee_is_finite(frame%factor)) then
      message='wall quantity normalization is not representable';return
    endif
    frame%source=0;frame%supplier=0;frame%window=0;frame%duration=0;frame%count=0
    if(definition%source=='instantaneous') then
      ready=.true.;ok=.true.;return
    endif
    if(definition%source/='reynolds'.or.definition%statistics_source/='wall-time') then
      message='existing wall means require reynolds with statistics_source=wall-time; no Favre wall accumulator';return
    endif
    call catalog_wall_statistics_info(int(identity%step),identity%time,active,ready,frame%window,frame%duration,frame%count)
    if(.not.active) then
      message='the existing wall-time accumulation was not enabled at startup';return
    endif
    frame%source=1;frame%supplier=1;ok=.true.
  end subroutine

  subroutine observe_catalog_walls(identity)
    type(checkpoint_state_identity),intent(in) :: identity
    type(insitu_options) :: options
    type(runtime_product_state) :: state
    type(wall_frame),allocatable :: frames(:)
    type(wall_line),allocatable :: lines(:)
    real(real_kind),allocatable :: coordinates(:,:,:,:),fields(:,:,:,:),means(:,:,:,:),local(:,:),send(:,:)
    logical,allocatable :: owned(:,:,:)
    logical :: enabled,ready,ok,dirty,selected,captured,host
    integer :: n,p,q,s,rank,err,w,k,i,old,nlines,status
    integer(int64) :: download,total_download,mean_download
    real(8) :: started
    character(512) :: message
    character(16) :: mode
    character(:),allocatable :: directory
    if(benchmark_field_io_disabled()) return
    n=runtime_catalog_count()
    if(n==0) return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call mpi_ok(err)
    call catalog_product_options(options,enabled)
    host=options%render.and.options%processing_backend=='host'
    allocate(frames(n),lines(n));dirty=.false.
    do p=1,n
      call runtime_catalog_state(p,state)
      if(state%definition%view%kind/='wall_curve') cycle
      if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
      message=''
      if(.not.enabled) then
        message='wall extraction needs enabled base in-situ options and explicit resource budgets'
      elseif(state%definition%view%channel=='image') then
        if(.not.options%render) then
          message='wall charts require rendering enabled in the base configuration'
        elseif(.not.use_gpu) then
          message='runtime wall charts require the existing GPU/EGL renderer binding'
        endif
#if !defined(_CUDA) || !defined(ASTR_INSITU_DEVICE_RENDERING)
        message='runtime wall charts require the Catalyst device rendering build'
#endif
      endif
      if(len_trim(message)==0) call wall_source(state%definition,identity,frames(p),ready,ok,message,host)
      call agree_error(message)
      if(len_trim(message)>0) then
        call runtime_catalog_reject(p,trim(message));dirty=.true.;cycle
      endif
      old=state%status
      call runtime_catalog_wait(p,.not.ready)
      if((old==runtime_waiting).eqv.ready) dirty=.true.
      if(.not.ready) cycle
      if(.not.runtime_catalog_due(p,identity%step,real(identity%time,real64))) cycle
      frames(p)%state=state;frames(p)%due=.true.
    enddo
    mode=options%postprocess_transport
    if(mode=='') mode='pinned'
    do s=0,2
      selected=.false.
      do p=1,n
        selected=selected.or.(frames(p)%due.and.frames(p)%supplier==s)
      enddo
      if(.not.selected) cycle
      captured=.false.;nlines=0;total_download=0
      do p=1,n
        if(.not.frames(p)%due.or.frames(p)%supplier/=s) cycle
        state=frames(p)%state;ok=.true.;message=''
        if(.not.captured) then
          started=insitu_clock()
#ifdef _CUDA
          if(use_gpu.and..not.host) then
            call validate_index_slice_transport(mode,ok,message)
            call agree_error(message)
            ok=len_trim(message)==0
            if(ok) then
              if(s==2) then
                call capture_catalog_sbli_wall_gpu(mode,int(identity%step),identity%time,options%host_budget_bytes, &
                  options%device_budget_bytes,options%device_reserve_bytes,lines(1)%values,download,ok)
                if(ok) then
                  lines(1)%index=0;lines(1)%side='lower';nlines=1;total_download=download
                else
                  message='optional SBLI wall source allocation would exceed its budget or available memory'
                  if(allocated(lines(1)%values)) deallocate(lines(1)%values)
                endif
              else
                call begin_catalog_wall_gpu(mode,s==1,frames(p)%duration,options%host_budget_bytes, &
                  options%device_budget_bytes,options%device_reserve_bytes,ok)
                if(.not.ok) message='optional wall source allocation would exceed its budget or available memory'
              endif
            endif
          else
#endif
            if(numq==11) then
              call capture_air5_walls(coordinates,fields,owned,options%host_budget_bytes, &
                options%device_budget_bytes,options%device_reserve_bytes,download,ok)
            else
              call capture_channel_walls(coordinates,fields,owned,options%host_budget_bytes, &
                options%device_budget_bytes,options%device_reserve_bytes,download,ok)
            endif
            if(.not.ok) message='optional host wall source allocation would exceed its budget or available memory'
            total_download=total_download+download
            if(ok.and.s==1) then
              call wall_statistics_render_values(int(identity%step),identity%time,means,mean_download,ok)
              total_download=total_download+mean_download
              if(ok) then
                fields(:,:,:,1:min(3,size(fields,4)))=means(:,:,:,1:min(3,size(fields,4)))
                if(numq==11) fields(:,:,:,1:18)=means(:,:,:,1:18)
              else
                message='optional host wall mean allocation failed'
              endif
              if(allocated(means)) deallocate(means)
            endif
#ifdef _CUDA
          endif
#endif
          captured=ok
          call report_insitu_timing('runtime_wall_capture',started,int(identity%step))
        endif
        q=1
        do while(q<=nlines)
          if(lines(q)%index==int(state%definition%index).and.lines(q)%side==state%definition%view%side) exit
          q=q+1
        enddo
        if(ok.and.q>nlines) then
          started=insitu_clock()
          allocate(local(6,im+1),send(6,ia+1),lines(q)%values(6,ia+1),stat=status)
          if(.not.admit_optional_fields(status==0,'wall line host allocation',ok)) then
            message='optional wall line host allocation failed'
            if(allocated(local)) deallocate(local)
            if(allocated(send)) deallocate(send)
            if(allocated(lines(q)%values)) deallocate(lines(q)%values)
          endif
        endif
        if(ok.and.q>nlines) then
          local=0;send=0;lines(q)%values=0;download=0
          w=0;k=int(state%definition%index)-kg0
          if(state%definition%view%side=='lower'.and.jg0==0) w=1
          if(state%definition%view%side=='upper'.and.jg0+jm==ja) w=1+merge(1,0,jg0==0)
          if(k<0.or.k>km.or.(k==km.and.kg0+km<ka)) k=-1
#ifdef _CUDA
          if(use_gpu.and..not.host) then
            call catalog_wall_line_gpu(k,w,local,download,ok)
            if(.not.ok) message='optional wall line device allocation failed'
          else
#endif
            if(w>0.and.k>=0) then
              do i=0,im
                local(1:3,i+1)=coordinates(i,k,w,:)
                if(numq==11) then
                  local(4:6,i+1)=fields(i,k,w,[7,13,17])
                else
                  local(4:6,i+1)=fields(i,k,w,1:3)
                endif
              enddo
            endif
#ifdef _CUDA
          endif
#endif
          call agree_error(message)
          ok=len_trim(message)==0
          if(ok.and.w>0.and.k>=0) then
            do i=0,im
              if(i==im.and.ig0+im<ia) cycle
              send(:,ig0+i+1)=local(:,i+1)
            enddo
          endif
          if(ok) then
            call MPI_Reduce(send,lines(q)%values,6*(ia+1),mpi_real_type,MPI_SUM,0,MPI_COMM_WORLD,err);call mpi_ok(err)
            lines(q)%index=int(state%definition%index);lines(q)%side=state%definition%view%side
            nlines=q;total_download=total_download+download
          else
            deallocate(lines(q)%values)
          endif
          deallocate(local,send)
          call report_insitu_timing('runtime_wall_select_reduce',started,int(identity%step))
        endif
        started=insitu_clock()
        if(ok) call runtime_product_segment(p,state,directory,ok,message)
        if(ok) then
          if(state%definition%view%channel=='image') then
            call render_curve(directory,state,identity,frames(p),lines(q)%values,options,ok,message)
          else
            call write_curve(directory,state,identity,frames(p),lines(q)%values,rank,ok,message)
          endif
        endif
        call report_insitu_timing('runtime_wall_publication_inclusive',started,int(identity%step))
        if(ok) then
          call runtime_catalog_success(p,identity%step,real(identity%time,real64))
        else
          call runtime_catalog_disable(p,identity%step,real(identity%time,real64),trim(message))
          if(rank==0) print '(a)','ASTR_RUNTIME_WALL_DISABLED '//state%definition%id//': '//trim(message)
        endif
        dirty=.true.
      enddo
#ifdef _CUDA
      if(use_gpu.and.captured.and..not.host) call end_catalog_wall_gpu()
#endif
      if(allocated(coordinates)) deallocate(coordinates)
      if(allocated(fields)) deallocate(fields)
      if(allocated(owned)) deallocate(owned)
      do q=1,nlines
        deallocate(lines(q)%values)
      enddo
      write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0)') 'ASTR_RUNTIME_WALL_GROUP rank=',rank,' step=',identity%step, &
        ' source=',s,' selected_lines=',nlines,' selected_line_d2h_bytes=',total_download
    enddo
    if(dirty) call refresh_runtime_product_status(identity)
  end subroutine

  subroutine render_curve(directory,state,identity,frame,values,options,ok,message)
    use insitu_run_config, only: pack_catalog_render
    character(*),intent(in) :: directory
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    type(wall_frame),intent(in) :: frame
    real(real_kind),intent(in) :: values(:,:)
    type(insitu_options),intent(in) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
#if defined(_CUDA) && defined(ASTR_INSITU_DEVICE_RENDERING)
    character(32) :: labels(11)
    integer(c_int) :: words(7)
    real(real64) :: settings(31)
    real(real_kind) :: statistics(3)
    character(c_char) :: buffer(512)
    integer :: status,d
    interface
      integer(c_int) function render(pipeline,backend,script,comm,step,t,id,source,definition,directory,version, &
          labels,words,settings,values,nodes,column,factor,nondimensional,provider,statistics,samples,host_budget,host) &
          bind(C,name='astr_insitu_render_wall_product')
        import c_int,c_char,c_int64_t,c_double,c_real_kind
        character(c_char),intent(in) :: pipeline(*),backend(*),script(*),id(*),source(*),definition(*),directory(*),labels(*)
        integer(c_int),value :: comm,step,nodes,column,nondimensional,provider,host
        integer(c_int64_t),value :: version,samples,host_budget
        integer(c_int),intent(in) :: words(7)
        real(c_double),intent(in) :: settings(31)
        real(c_real_kind),value :: t,factor
        real(c_real_kind),intent(in) :: values(*),statistics(3)
      end function
      subroutine get_error(buffer,capacity) bind(C,name='astr_insitu_runtime_error')
        import c_char,c_int
        character(c_char),intent(out) :: buffer(*)
        integer(c_int),value :: capacity
      end subroutine
    end interface
    call begin_catalog_renderer(ok,message)
    if(.not.ok) return
    call pack_catalog_render(state%definition%view,labels,words,settings)
    statistics=[frame%window,frame%duration]
    status=render(trim(options%rendering_pipeline)//c_null_char,trim(options%implementation_path)//c_null_char, &
      trim(options%pipeline_file)//c_null_char,int(MPI_COMM_WORLD,c_int),int(identity%step,c_int),identity%time, &
      state%definition%id//c_null_char,state%definition%source//c_null_char,state%definition%record//c_null_char, &
      directory//c_null_char,int(state%version,c_int64_t),labels,words,settings,values,size(values,2), &
      frame%column,frame%factor,merge(1,0,nondimen),frame%source,statistics,frame%count,options%host_budget_bytes, &
      merge(1_c_int,0_c_int,options%processing_backend=='host'))
    ok=status==0
    if(.not.ok) then
      call get_error(buffer,int(size(buffer),c_int));message=''
      do d=1,size(buffer)
        if(buffer(d)==c_null_char) exit
        message(d:d)=buffer(d)
      enddo
    endif
#else
    ok=.false.;message='runtime wall charts require the Catalyst device rendering build'
#endif
  end subroutine

  subroutine write_curve(directory,state,identity,frame,values,rank,ok,message)
    character(*),intent(in) :: directory
    type(runtime_product_state),intent(in) :: state
    type(checkpoint_state_identity),intent(in) :: identity
    type(wall_frame),intent(in) :: frame
    real(real_kind),intent(in) :: values(:,:)
    integer,intent(in) :: rank
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    character(64) :: filename
    character(:),allocatable :: path,temporary
    integer :: unit,status,closed,err,i,ignored
    real(real64) :: bounds(6)
    interface
      integer(c_int) function rename_file(old,new) bind(C,name='rename')
        import c_int,c_char
        character(c_char),intent(in) :: old(*),new(*)
      end function
      integer(c_int) function remove_file(path) bind(C,name='remove')
        import c_int,c_char
        character(c_char),intent(in) :: path(*)
      end function
    end interface
    status=0;message=''
    if(rank==0) then
      write(filename,'("frame.step",i12.12,".csv")') identity%step
      path=directory//'/'//trim(filename);temporary=path//'.partial'
      open(newunit=unit,file=temporary,status='new',action='write',iostat=status)
      if(status==0) then
        write(unit,'(a,i0,a,l1)',iostat=status) '# phase=completed_step source_bytes=',real_bytes,' nondimensional=',nondimen
        if(status==0) then
          if(state%definition%view%sampling=='wall_midspan') then
            write(unit,'(5a)',iostat=status) '# field=',trim(state%definition%view%field), &
              ' source=',state%definition%source,' sampling=physical_midspan_interpolation; no spanwise average'
            if(status==0) write(unit,'(a)',iostat=status) &
              '# legacy SBLI lower wall: p/p_inf, Cf, heat positive into wall divided by rho_inf U_inf^3'
          else
            write(unit,'(5a)',iostat=status) '# field=',trim(state%definition%view%field), &
              ' source=',state%definition%source,' sampling=global_k_index; no spanwise average'
            if(status==0) write(unit,'(a)',iostat=status) &
              '# inward wall normal; shear along projected global +x tangent; heat positive into gas'
          endif
        endif
        if(status==0) write(unit,'(a,3(es26.17,1x))',iostat=status) &
          '# rho_inf,U_inf,p_inf=',real([roinf,uinf,pinf],real64)
        if(status==0) write(unit,'(a,3(es26.17,1x),a,i0)',iostat=status) '# requested_window,covered_duration=', &
          real(frame%window,real64),real(frame%duration,real64),' samples=',frame%count
        if(status==0) write(unit,'(a)',iostat=status) 'step,time,global_i,x,y,z,value'
        bounds=state%definition%view%bounds
        do i=1,size(values,2)
          if(status/=0) exit
          if(any(values(1:3,i)<bounds(1:6:2)).or.any(values(1:3,i)>bounds(2:6:2))) cycle
          if(.not.all(ieee_is_finite(values(:,i))).or. &
              .not.ieee_is_finite(values(frame%column,i)*frame%factor)) then
            status=1;exit
          endif
          write(unit,'(i0,",",es26.17,",",i0,4(",",es26.17))',iostat=status) identity%step, &
            real(identity%time,real64),i-1,real(values(1:3,i),real64),real(values(frame%column,i)*frame%factor,real64)
        enddo
        close(unit,iostat=closed)
        if(status==0) status=closed
      endif
      if(status==0) status=rename_file(temporary//c_null_char,path//c_null_char)
      if(status/=0) ignored=remove_file(temporary//c_null_char)
    endif
    call MPI_Bcast(status,1,MPI_INTEGER,0,MPI_COMM_WORLD,err);call mpi_ok(err)
    ok=status==0
    if(.not.ok) message='cannot publish the selected wall CSV frame'
  end subroutine
end module
