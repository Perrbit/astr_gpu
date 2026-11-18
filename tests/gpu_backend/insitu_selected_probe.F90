program insitu_selected_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare
  use insitu_moments_cpp
  use insitu_index_api
  use insitu_sample_moments
  use insitu_time_integral, only: clipped_trapezoid
  implicit none
  integer,parameter :: hm=5,global=16,ny=4,nk=3,nodes=ny*nk,nt=6
  real(c_double),parameter :: times(nt)=[0.d0,.001d0,.0017d0,.003d0,.0049d0,.006d0]
  real(c_double),parameter :: constant_primitives(6)=[0.d0,.25d0,-.5d0,.125d0,3.d0,3.d0]
  type(c_ptr) :: parent,plan,resumed,old_parent,empty
  type(device_view) :: view,input
  type(index_region) :: region
  real(c_double),allocatable :: q(:,:,:,:),saved_q(:,:,:,:),cpu(:,:),actual(:,:),again(:,:),profile(:,:),expected(:,:), &
    rho_field(:,:,:),velocity_field(:,:,:,:),xyz(:,:,:,:)
  real(c_double) :: window(2),weights(2),duration,const2,const6,primitive(5),density,terms(24),span(nk), &
    u(3),pressure,label,constant(5),state_error=0,profile_error=0,external_error=0
  integer(c_int) :: ids(5),indices(3,nodes),slots(3),positions(nodes),nfields,components,covariance
  integer(c_int64_t) :: usage(9),before(9),base
  integer :: ierr,rank,ranks,local_rank,status,cart,dims(3),coords(3),extent(3),origin(3), &
    debug,case_id,fixture,frame,n,i,j,k,c,term_count,checks=0,restores=0,external_checks=0,targets
  logical :: previous,ok,thermal
  character(64) :: argument
#ifdef ASTR_SELECTED_ALLOCATION_TEST
  interface
    subroutine fail_allocation(after) bind(C,name='astr_selected_test_fail_allocation')
      import c_int
      integer(c_int),value :: after
    end subroutine
    integer(c_int) function allocation_failures() bind(C,name='astr_selected_test_allocation_failures')
      import c_int
    end function
  end interface
#endif
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'Open MPI local rank required'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind before MPI')
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,argument);read(argument,*) dims
  if(any(dims<1).or.product(dims)/=ranks.or.any(mod(global,dims)/=0)) error stop 'selected topology mismatch'
  call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.false.,.false.,.false.],.false.,cart,ierr)
  call MPI_Cart_coords(cart,rank,3,coords,ierr)
  extent=global/dims;origin=coords*extent
  span=[.125d0,.25d0,.375d0]
  allocate(q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5), &
    saved_q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5))
  n=0
  do k=0,nk-1
  do j=0,ny-1
    n=n+1;indices(:,n)=[merge(0,extent(1),mod(j+k,2)==0),2*j,2*k]
  enddo
  enddo
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),int(extent(1),c_int),int(extent(2),c_int), &
      int(extent(3),c_int),hm,int(debug,c_int),parent),'create parent')
    do case_id=0,8
      covariance=merge(1_c_int,0_c_int,case_id>=6);nfields=case_id
      if(case_id>=6) nfields=case_id-3
      ids=0
      select case(nfields)
      case(1);ids(1)=6
      case(2);ids(:2)=[5,6]
      case(3);ids(:3)=[4,2,3]
      case(4);ids(:4)=[5,4,2,3]
      case(5);ids=[6,4,2,5,3]
      end select
      slots=0
      if(covariance==1) then
        do c=1,nfields
          if(ids(c)<=4) slots(ids(c)-1)=c
        enddo
      endif
      components=sample_moment_size(int(nfields),covariance==1)
      term_count=sample_profile_size(int(nfields),covariance==1)
      allocate(cpu(nodes,components),actual(nodes,components),again(nodes,components), &
        profile(term_count,ny),expected(term_count,ny))
      do fixture=1,4
        if(fixture>1.and.case_id/=8) cycle
        window=[.0004d0,.0042d0]
        if(fixture==2) window=[1.d0,2.d0]
        cpu=0
        call check(selected_create(parent,nodes,indices,components,plan),'allocate sparse group')
        base=int(nodes,c_int64_t)*(12+8*components)+4
        call check(selected_metrics(plan,usage),'initial sparse resources')
        if(usage(1)/=base.or.usage(6)/=12*nodes.or.any(usage([3,4,5,7,8,9])/=0)) error stop 'eager sparse workspace/transfer'
        old_parent=parent;status=device_state_destroy(parent)
        if(status/=device_invalid_argument.or..not.c_associated(parent,old_parent)) error stop 'live sparse parent destroyed'
        resumed=c_null_ptr
        const2=ieee_value(0.d0,ieee_quiet_nan);const6=const2
        if(any(ids(:nfields)==6)) const2=2
        thermal=any(ids(:nfields)>=5)
        if(thermal) const6=2.5d0
        do frame=1,nt
          q=ieee_value(0.d0,ieee_quiet_nan)
          weights=0;previous=frame>1
          if(fixture==3.and.frame==4) previous=.false.
          if(previous) then
            call clipped_trapezoid(times(frame-1),times(frame),[1.d0,0.d0],[0.d0,1.d0], &
              window(1),window(2),weights,duration,ok)
            if(.not.ok) error stop 'selected clipped quadrature'
          endif
          do n=1,nodes
            i=indices(1,n);j=indices(2,n);k=indices(3,n)
            label=real(origin(1)+i+global*(origin(2)+j+global*(origin(3)+k)),c_double)
            density=1+label/8192+times(frame)/64
            u=[label/4096+times(frame),-label/8192+2*times(frame),label/16384-3*times(frame)]
            pressure=2+label/8192+times(frame)*3
            if(fixture==4) then
              density=2;u=[.25d0,-.5d0,.125d0];pressure=3
            endif
            q(i,j,k,1)=density
            if(thermal) then
              q(i,j,k,2:4)=density*u
              q(i,j,k,5)=pressure*const6+.5d0*density*sum(u**2)
            else
              do c=1,nfields
                q(i,j,k,ids(c))=density*u(ids(c)-1)
              enddo
            endif
            call decode_sample_primitives(q(i,j,k,:),int(nfields),int(ids(:nfields)),const2,const6, &
              primitive(:nfields),density,ok)
            if(.not.ok) error stop 'CPU primitive oracle'
            call push_sample_moments(cpu(n,:),primitive(:nfields),density,weights,previous,int(slots),ok)
            if(.not.ok) error stop 'CPU selected update oracle'
          enddo
          call check(device_state_upload(parent,device_q,q,nbytes(size(q))),'explicit sparse fixture q')
          call check(selected_metrics(plan,before),'pre-sample sparse counters')
          call check(selected_push(plan,nfields,ids,const2,const6,weights,merge(1_c_int,0_c_int,previous),covariance), &
            'sparse q sampling')
          call check(selected_metrics(plan,usage),'post-sample sparse counters')
          if(usage(1)/=base.or.usage(3)/=4*frame.or.usage(4)/=before(4).or.usage(5)/=0.or. &
            usage(6)/=12*nodes.or.usage(7)/=frame.or.any(usage(8:9)/=0)) error stop 'sample copied a point field/allocated output'
          call check(selected_download(plan,actual,nbytes(size(actual))),'explicit sparse observation')
          call compare(actual,cpu,state_error)
          call check(selected_view(plan,0_c_int,view),'borrow sparse state')
          if(view%extent(1)/=nodes.or.view%extent(2)/=components.or.view%stride(2)/=nodes) error stop 'sparse state layout'
          if(frame==3) then
            call check(selected_create(parent,nodes,indices,components,resumed),'allocate restore group')
            call check(selected_upload(resumed,actual,nbytes(size(actual))),'explicit sparse restore')
            restores=restores+1
          elseif(frame>3) then
            call check(selected_push(resumed,nfields,ids,const2,const6,weights,merge(1_c_int,0_c_int,previous),covariance), &
              'continued sparse sampling')
            call check(selected_download(resumed,again,nbytes(size(again))),'continued sparse observation')
            if(any(transfer(actual,[0_c_int64_t],size(actual))/=transfer(again,[0_c_int64_t],size(again)))) &
              error stop 'sparse continuation state bits differ'
          endif
          call check(device_state_download(parent,device_q,saved_q,nbytes(size(q))),'solver isolation observation')
          if(any(transfer(q,[0_c_int64_t],size(q))/=transfer(saved_q,[0_c_int64_t],size(q)))) error stop 'sparse update changed q'
          checks=checks+1
        enddo
        expected=0
        if(fixture/=2) then
          do k=1,nk
          do j=1,ny
            n=(k-1)*ny+j
            call sample_profile_terms(int(components),int(nfields),cpu(n,:),span(k),covariance==1,terms,ok)
            if(.not.ok) error stop 'CPU profile terms oracle'
            expected(:,j)=expected(:,j)+terms(:term_count)
          enddo
          enddo
#ifdef ASTR_SELECTED_ALLOCATION_TEST
          if(case_id==8.and.fixture==1.and.debug==0) then
            call check(selected_metrics(plan,before),'pre-fault profile resources')
            call fail_allocation(2_c_int)
            status=selected_profile(plan,ny,nk,span,nfields,covariance,profile,nbytes(size(profile)))
            if(status/=2.or.allocation_failures()/=1) error stop 'profile second allocation fault was not injected'
            call check(selected_metrics(plan,usage),'post-fault profile resources')
            if(usage(1)/=before(1)) error stop 'failed profile retained a partially initialized workspace'
            call check(selected_download(plan,again,nbytes(size(again))),'post-fault moment observation')
            if(any(transfer(again,[0_c_int64_t],size(again))/=transfer(actual,[0_c_int64_t],size(actual)))) &
              error stop 'profile allocation failure modified statistics'
            write(*,'(a,1x,i0)') 'CPP_INSITU_SELECTED_ALLOCATION_FAULT_PASS',rank
          endif
#endif
          call check(selected_profile(plan,ny,nk,span,nfields,covariance,profile,nbytes(size(profile))),'reduced profile')
          call compare(profile,expected,profile_error)
          expected=profile
          call check(selected_profile(plan,ny,nk,span,nfields,covariance,profile,nbytes(size(profile))),'repeated profile')
          if(any(transfer(profile,[0_c_int64_t],size(profile))/=transfer(expected,[0_c_int64_t],size(expected)))) &
            error stop 'profile repeat bits differ'
          call check(selected_metrics(plan,usage),'profile resource counters')
          if(usage(1)/=base+8*nk+nbytes(size(profile)).or.usage(2)/=usage(1).or.usage(9)/=8*nk.or. &
            usage(8)/=2*(nbytes(size(profile))+4)) error stop 'profile downloaded points or reuploaded weights'
        else
          status=selected_profile(plan,ny,nk,span,nfields,covariance,profile,nbytes(size(profile)))
          if(status/=device_nonphysical) error stop 'uncovered sparse profile accepted'
        endif
        if(fixture==4) then
          if(any(actual(:,3+3*nfields+1:)/=0)) error stop 'constant selected covariance is nonzero'
          do c=1,nfields
            constant(c)=constant_primitives(ids(c))
          enddo
          do c=1,nfields
            if(any(actual(:,3+nfields+c)/=constant(c)).or.any(actual(:,3+2*nfields+c)/=constant(c))) &
              error stop 'constant selected means differ'
          enddo
        endif
        call check(selected_destroy(resumed),'release restore group')
        call check(selected_destroy(plan),'release sparse group')
      enddo
      deallocate(cpu,actual,again,profile,expected)
    enddo
    call check(selected_create(parent,0_c_int,indices,3_c_int,empty),'empty rank group')
    call check(selected_push(empty,0_c_int,ids,0.d0,0.d0,[0.d0,0.d0],0_c_int,0_c_int),'empty rank sample')
    call check(selected_download(empty,constant,0_c_size_t),'empty checkpoint observation')
    call check(selected_profile(empty,0_c_int,0_c_int,span,0_c_int,0_c_int,constant,0_c_size_t),'empty profile')
    call check(selected_metrics(empty,usage),'empty resources')
    if(usage(1)/=40.or.any(usage(3:9)/=0)) error stop 'empty group transferred data or used a full field'
    call check(selected_destroy(empty),'release empty group')
    call check(selected_create(parent,nodes,indices,6_c_int,plan),'invalid source group')
    status=selected_push(plan,1_c_int,[7_c_int],1.d0,1.d0,[0.d0,0.d0],0_c_int,0_c_int)
    if(status/=device_invalid_argument) error stop 'invalid selected descriptor accepted'
    q=77;q(indices(1,1),indices(2,1),indices(3,1),1)=0
    call check(device_state_upload(parent,device_q,q,nbytes(size(q))),'nonpositive selected density fixture')
    status=selected_push(plan,1_c_int,[2_c_int],0.d0,0.d0,[0.d0,0.d0],0_c_int,0_c_int)
    if(status/=device_nonphysical) error stop 'nonpositive selected density accepted'
    ! Failed samples are discarded; they are not retried or used as a new seed.
    call check(selected_destroy(plan),'release failed selected group')
    call external_packets()
    call check(device_state_destroy(parent),'release parent')
  enddo
  write(*,'(a,4(1x,i0),3(1x,es24.16e3))') 'CPP_INSITU_SELECTED_PASS',rank,checks,restores,external_checks, &
    state_error,profile_error,external_error
  call MPI_Comm_free(cart,ierr);call MPI_Finalize(ierr)
contains
  subroutine external_packets()
    real(c_double) :: rho,values(2)
    integer :: variant,packet,node,position,point(3),first_target
    call check(fluid_prepare(parent,0_c_int),'prepare external packet source')
    allocate(rho_field(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm), &
      velocity_field(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,3), &
      xyz(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,3))
    region%first=origin;region%extent=[1,ny,nk];region%lower=0;region%shape=1;region%origin=origin
    region%intervals=global;region%periodic=0;region%cartesian_support=0;region%nondimen=1
    region%tempconst=.4d0;region%tempconst1=1.4d0;region%reynolds=1600;region%prandtl=.72d0
    region%const5=1;region%cp=3.5d0
    q=77;rho_field=2;velocity_field=0
    call check(device_state_upload(parent,device_q,q,nbytes(size(q))),'independent parent q')
    do k=-hm,extent(3)+hm
    do j=-hm,extent(2)+hm
    do i=-hm,extent(1)+hm
      xyz(i,j,k,:)=[2.d0+real(i,c_double)/32,real(j,c_double)/16,-real(k,c_double)/32]
    enddo
    enddo
    enddo
    do variant=1,2
      components=9;nfields=2;slots=0
      allocate(cpu(nodes,components),actual(nodes,components))
      cpu=0
      call check(selected_create(parent,nodes,indices,components,plan),'external selected group')
      do frame=1,3
        do k=0,extent(3)
        do j=0,extent(2)
        do i=0,extent(1)
          rho_field(i,j,k)=2+real(j+k,c_double)/64
          velocity_field(i,j,k,1)=real(j,c_double)/16+real(frame,c_double)/64
          velocity_field(i,j,k,2)=-real(k,c_double)/32+real(frame,c_double)/128
        enddo
        enddo
        enddo
        call check(device_state_upload(parent,device_rho,rho_field,nbytes(size(rho_field))),'external rho fixture')
        call check(device_state_upload(parent,device_velocity,velocity_field,nbytes(size(velocity_field))),'external field fixture')
        call check(index_begin(parent,0_c_int,nodes,0_c_int,[integer(c_int)::],3_c_int,[1_c_int,2_c_int,3_c_int], &
          spread(0.d0,1,49),merge(1_c_int,0_c_int,variant==1),xyz,nbytes(size(xyz))),'external producer group')
        call check(index_evaluate(parent,region),'external producer evaluation')
        call check(index_source(parent,merge(1_c_int,0_c_int,variant==1),input),'borrow external packet')
        weights=0
        if(frame>1) weights=[.0005d0,.0005d0]
        targets=merge(nodes,nodes/2,variant==1)
        input%extent(1)=targets
        do packet=1,variant
          first_target=(packet-1)*targets
          do node=1,targets
            positions(node)=nodes-(first_target+node)+1
            point=[0,mod(node-1,ny),(node-1)/ny]
            if(variant==1) then
              rho=xyz(point(1),point(2),point(3),1);values=xyz(point(1),point(2),point(3),2:3)
            else
              rho=rho_field(point(1),point(2),point(3));values=velocity_field(point(1),point(2),point(3),1:2)
            endif
            position=positions(node)
            call push_sample_moments(cpu(position,:),values,rho,weights,frame>1,[0,0,0],ok)
            if(.not.ok) error stop 'CPU external moment oracle'
          enddo
          call check(selected_scatter(plan,int(targets,c_int),positions,input,nfields,weights,merge(1_c_int,0_c_int,frame>1)), &
            'external selected scatter')
          external_checks=external_checks+1
        enddo
        call check(selected_download(plan,actual,nbytes(size(actual))),'external result observation')
        call compare(actual,cpu,external_error)
        call check(index_end(parent),'release external source group')
      enddo
      call check(device_state_download(parent,device_q,saved_q,nbytes(size(q))),'external solver isolation')
      if(any(saved_q/=77)) error stop 'external source consulted/changed solver q'
      call check(selected_metrics(plan,usage),'external resources')
      if(usage(1)/=nodes*(12+8*components+4)+4.or.usage(5)/=0.or. &
        usage(6)/=12*nodes+3*nodes*4.or.usage(3)/=4*3*variant.or.usage(7)/=3*variant) &
        error stop 'external scatter workspace or transfer count differs'
      call check(selected_destroy(plan),'release external group')
      deallocate(cpu,actual)
    enddo
    deallocate(rho_field,velocity_field,xyz)
  end subroutine
  integer(c_size_t) function nbytes(n)
    integer,intent(in) :: n
    nbytes=int(n,c_size_t)*8
  end function
  subroutine compare(a,b,worst)
    real(c_double),intent(in) :: a(:,:),b(:,:)
    real(c_double),intent(inout) :: worst
    integer :: c
    if(any(shape(a)/=shape(b)).or..not.all(ieee_is_finite(a)).or..not.all(ieee_is_finite(b))) error stop 'selected state finite/layout'
    do c=1,size(a,2)
      worst=max(worst,maxval(abs(a(:,c)-b(:,c)))/max(1.d0,maxval(abs(b(:,c)))))
    enddo
    if(worst>2.d-10) error stop 'selected mathematical comparison exceeds 2e-10'
  end subroutine
  subroutine check(code,where)
    integer(c_int),intent(in) :: code
    character(*),intent(in) :: where
    integer :: ignored
    if(code==0) return
    print *, 'selected failure: ',where,code
    call MPI_Abort(MPI_COMM_WORLD,int(code),ignored)
  end subroutine
end program
