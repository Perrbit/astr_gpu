program keep_probe_cpp
  use iso_c_binding, only: c_ptr,c_int,c_double,c_size_t,c_sizeof,c_int64_t
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use mpi, only: MPI_Init,MPI_Finalize,MPI_COMM_SELF
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use keep_operator, only: keep_explicit_volume,keep_split_plan,build_keep_split_plan,keep_split_variables, &
    keep_monomial,keep_split_accumulate
  use compact_operator, only: compact_line_partition,partition_compact_line,compact_derivative_core,compact_derivative_rhs
  use compact_line_solver, only: compact_matrix,compact_reference_factor,factor_compact_reference,solve_compact_reference
  use compact_line_cpp
  use device_state
  use fluid_explicit
  implicit none
  real(real_kind),parameter :: pi=acos(-1.e0_real_kind),gamma=1.4e0_real_kind
  real(real_kind),parameter :: coefficients(3)=[.75e0_real_kind,-.15e0_real_kind,1.e0_real_kind/60]
  real(real_kind),allocatable :: q(:,:,:,:),rho(:,:,:),vel(:,:,:,:),prs(:,:,:),jac(:,:,:),metric(:,:,:,:,:), &
    result(:,:,:,:),reference(:,:,:,:),gold(:,:,:,:),compact_gold(:,:,:,:,:)
  real(real_kind) :: errors(5,3),cpu_errors(5,3),orders(5),h,x,density,pressure,velocity(3),dr,dp,dv(3), &
    exact(5),enthalpy,dh,balance,scale,kinetic,dp_discrete,difference,maximum=0,net_error
  type(c_ptr) :: state
  integer :: axis,level,n,upper(3),halo=3,rotation(3),node(3),neighbor(3),i,j,k,c,debug,l,status,ierr
  call check(device_bind(0_c_int),'pre-MPI device binding')
  call MPI_Init(ierr);call check(ierr,'MPI initialization')
  do axis=1,3
    rotation=[axis,mod(axis,3)+1,mod(axis+1,3)+1]
    do level=1,3
      n=16*2**level;upper=5;upper(axis)=n-1;h=2*pi/n
      allocate(q(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
        rho(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo),vel(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3), &
        prs(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo),jac(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
        metric(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3,3), &
        result(0:upper(1),0:upper(2),0:upper(3),5),reference(0:upper(1),0:upper(2),0:upper(3),5), &
        gold(0:upper(1),0:upper(2),0:upper(3),5),compact_gold(0:upper(1),0:upper(2),0:upper(3),5,2))
      metric=0;jac=h
      do c=1,3
        metric(:,:,:,c,c)=1
      enddo
      metric(:,:,:,axis,axis)=1/h
      do k=-halo,upper(3)+halo
      do j=-halo,upper(2)+halo
      do i=-halo,upper(1)+halo
        node=[i,j,k];x=modulo(node(axis),n)*h
        rho(i,j,k)=1+.1e0_real_kind*sin(x);prs(i,j,k)=1+.1e0_real_kind*cos(2*x)
        vel(i,j,k,rotation)=[.2e0_real_kind+.1e0_real_kind*sin(2*x),.1e0_real_kind*cos(x),.05e0_real_kind*sin(3*x)]
        q(i,j,k,1)=rho(i,j,k);q(i,j,k,2:4)=rho(i,j,k)*vel(i,j,k,:)
        q(i,j,k,5)=prs(i,j,k)/(gamma-1)+.5e0_real_kind*rho(i,j,k)*sum(vel(i,j,k,:)**2)
      enddo
      enddo
      enddo
      reference=0
      call keep_explicit_volume(upper(1),upper(2),upper(3),halo,[3,3,3],[0,0,0],upper, &
        q,rho,vel,prs,jac,metric,gamma,reference)
      do debug=0,1
        status=device_state_create(0_c_int,int(upper(1),c_int),int(upper(2),c_int),int(upper(3),c_int), &
          int(halo,c_int),int(debug,c_int),state)
        call check(status,'create')
        call check(fluid_prepare(state,0_c_int),'primitive buffers')
        call check(fluid_rhs_prepare(state,0_c_int),'RHS buffers')
        call check(device_state_upload(state,device_q,q,bytes(size(q))),'q upload')
        call check(device_state_upload(state,device_rho,rho,bytes(size(rho))),'rho upload')
        call check(device_state_upload(state,device_velocity,vel,bytes(size(vel))),'velocity upload')
        call check(device_state_upload(state,device_pressure,prs,bytes(size(prs))),'pressure upload')
        call check(device_state_upload(state,device_metric,metric,bytes(size(metric))),'metric upload')
        call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))),'Jacobian upload')
        call check(fluid_keep_convection(state,gamma),'KEEP convection')
        call check(device_state_download(state,device_rhs,result,bytes(size(result))),'RHS observation')
        if(.not.all(ieee_is_finite(result))) error stop 'KEEP nonfinite output'
        difference=0
        do c=1,5
          difference=max(difference,maxval(abs(result(:,:,:,c)+reference(:,:,:,c)))/max(1.e0_real_kind,maxval(abs(reference(:,:,:,c)))))
        enddo
        maximum=max(maximum,difference)
        if(real_bytes==8.and.difference>2.e-10_real_kind) error stop 'KEEP CPU volume parity'
        if(debug==0) then
          gold=result
        elseif(any(real_to_word(result)/=real_to_word(gold))) then
          error stop 'KEEP sync changed words'
        endif
        call compact_case()
        result=gold
        call check(device_state_destroy(state),'destroy')
      enddo
      errors(:,level)=0;cpu_errors(:,level)=0;balance=0;scale=1
      do i=0,n-1
        node=0;node(axis)=i;x=i*h
        density=rho(node(1),node(2),node(3));pressure=prs(node(1),node(2),node(3));velocity=vel(node(1),node(2),node(3),:)
        dr=.1e0_real_kind*cos(x);dp=-.2e0_real_kind*sin(2*x)
        dv(rotation)=[.2e0_real_kind*cos(2*x),-.1e0_real_kind*sin(x),.15e0_real_kind*cos(3*x)]
        exact(1)=dr*velocity(axis)+density*dv(axis)
        exact(2:4)=exact(1)*velocity+density*velocity(axis)*dv;exact(axis+1)=exact(axis+1)+dp
        enthalpy=gamma*pressure/(gamma-1)+.5e0_real_kind*density*sum(velocity**2)
        dh=gamma*dp/(gamma-1)+.5e0_real_kind*dr*sum(velocity**2)+density*sum(velocity*dv)
        exact(5)=dh*velocity(axis)+enthalpy*dv(axis)
        errors(:,level)=max(errors(:,level),abs(-result(node(1),node(2),node(3),:)/h-exact))
        cpu_errors(:,level)=max(cpu_errors(:,level),abs(reference(node(1),node(2),node(3),:)/h-exact))
        dp_discrete=0
        do l=1,3
          neighbor=node;neighbor(axis)=i+l
          dp_discrete=dp_discrete+coefficients(l)*prs(neighbor(1),neighbor(2),neighbor(3))/h
          neighbor(axis)=i-l
          dp_discrete=dp_discrete-coefficients(l)*prs(neighbor(1),neighbor(2),neighbor(3))/h
        enddo
        kinetic=-sum(velocity*result(node(1),node(2),node(3),2:4))/h &
          +.5e0_real_kind*sum(velocity**2)*result(node(1),node(2),node(3),1)/h
        balance=balance+h*(kinetic-velocity(axis)*dp_discrete)
        scale=scale+h*(abs(kinetic)+abs(velocity(axis)*dp_discrete))
      enddo
      if(.not.ieee_is_finite(balance).or..not.ieee_is_finite(scale)) error stop 'nonfinite KEEP exchange'
      if(real_bytes==8.and.abs(balance)>2.e-12_real_kind*scale) error stop 'KEEP kinetic pressure exchange'
      if(real_bytes==8.and.maxval(abs(sum(sum(sum(result,dim=1),dim=1),dim=1)))> &
        2.e-12_real_kind*max(1.e0_real_kind,sum(abs(result)))) error stop 'KEEP conservative net sum'
      net_error=maxval(abs(sum(sum(sum(result,dim=1),dim=1),dim=1)))/max(1._real_kind,sum(abs(result)))
      write(*,'(a,2(1x,i0),2(1x,es24.16e3))') 'CPP_KEEP_INVARIANTS',axis,n,abs(balance)/scale,net_error
      write(*,'(a,2(1x,i0),7(1x,es24.16e3))') 'CPP_KEEP',axis,n,difference,balance,errors(:,level)
      write(*,'(a,2(1x,i0),5(1x,es24.16e3))') 'CPP_KEEP_CPU_ANALYTIC',axis,n,cpu_errors(:,level)
      deallocate(q,rho,vel,prs,jac,metric,result,reference,gold,compact_gold)
    enddo
    do level=1,2
      orders=log(errors(:,level)/errors(:,level+1))/log(2.e0_real_kind)
      if(real_bytes==8.and.any(orders<5.5e0_real_kind)) error stop 'KEEP sixth-order convergence'
      write(*,'(a,1x,i0,5(1x,f12.6))') 'CPP_KEEP_ORDERS',axis,orders
      orders=log(cpu_errors(:,level)/cpu_errors(:,level+1))/log(2._real_kind)
      write(*,'(a,1x,i0,5(1x,f12.6))') 'CPP_KEEP_CPU_ORDERS',axis,orders
    enddo
  enddo
  call basic_boundaries()
  write(*,'(a,1x,es24.16e3)') 'CPP_KEEP_PASS',maximum
  write(*,'(a,1x,i0)') 'CPP_KEEP_ELEMENT_BYTES',real_bytes
  call MPI_Finalize(ierr);call check(ierr,'MPI finalization')
contains
  subroutine compact_case()
    type(keep_split_plan) :: split
    type(compact_line_partition) :: part
    type(compact_matrix) :: core,full
    type(compact_reference_factor) :: factor
    type(compact_distributed_cpp_plan) :: plan
    type(compact_stencil_cpp) :: stencil
    type(device_view) :: input_view,solution_view,nodes_view
    real(real_kind),allocatable :: basis(:,:),right_hand(:,:),derivatives(:,:),variables(:,:),expected(:,:)
    real(real_kind) :: v(9),edge(2),local_error,normalizer
    integer :: p,b,mode,capacity,total,lines,start,columns,ii,jj,kk,point_index(3),component
    call build_keep_split_plan(split)
    if(split%terms/=294.or.split%bases/=85) error stop 'KEEP split table identity'
    call partition_compact_line(n,0,1,.true.,part)
    call compact_derivative_core(part,core,edge)
    full=core;full%cyclic=.true.;full%lower(1)=edge(1);full%upper(n)=edge(2)
    call factor_compact_reference(full,factor)
    allocate(basis(-halo:n-1+halo,split%bases),right_hand(n,split%bases),derivatives(n,split%bases), &
      variables(n,9),expected(n,5))
    do p=-halo,n-1+halo
      point_index=0;point_index(axis)=p
      ii=point_index(1);jj=point_index(2);kk=point_index(3)
      call keep_split_variables(rho(ii,jj,kk),vel(ii,jj,kk,:),prs(ii,jj,kk), &
        jac(ii,jj,kk)*metric(ii,jj,kk,axis,:),gamma,v)
      if(p>=0.and.p<n) variables(p+1,:)=v
      do b=1,split%bases
        basis(p,b)=keep_monomial(v,split%right(:,b))
      enddo
    enddo
    call compact_derivative_rhs(part,n-1,halo,basis,right_hand)
    call solve_compact_reference(factor,right_hand,derivatives)
    do p=1,n
      call keep_split_accumulate(split,variables(p,:),derivatives(p,:),expected(p,:))
    enddo
    if(.not.all(ieee_is_finite(expected))) error stop 'nonfinite independent compact KEEP reference'
    call check(compact_keep_prepare_cpp(state,int(split%terms,c_int),int(split%bases,c_int), &
      split%left,split%right,split%component,split%derivative,split%weight),'native KEEP split table')
    lines=product(upper+1)/(upper(axis)+1);total=split%bases*lines
    stencil%axis=axis-1;stencil%first=0;stencil%total=n;stencil%periodic=1;stencil%kind=2
    stencil%component_offset=1;stencil%active_first=0;stencil%active_last=upper
    do mode=1,2
      capacity=merge(min(127,total),total,mode==1)
      call prepare_distributed_compact_cpp(state,core,edge,part,MPI_COMM_SELF,capacity,0_c_int,plan)
      call compact_distributed_views_cpp(plan,input_view,solution_view,nodes_view)
      if(input_view%element_bytes/=real_bytes.or.nodes_view%element_bytes/=real_bytes) &
        error stop 'compact KEEP native element width'
      result=7
      call check(device_state_upload(state,device_rhs,result,bytes(size(result))),'nonzero initial compact RHS')
      start=0
      do while(start<total)
        columns=min(capacity,total-start);stencil%start=start
        call check(compact_keep_rhs_cpp(state,plan%handle,stencil,int(columns,c_int),gamma),'compact KEEP basis RHS')
        call solve_distributed_compact_cpp(plan,columns)
        call restore_distributed_compact_cpp(plan,columns)
        call check(compact_keep_commit_cpp(state,plan%handle,stencil,int(columns,c_int),gamma),'compact KEEP accumulation')
        start=start+columns
      enddo
      call check(device_state_download(state,device_rhs,result,bytes(size(result))),'compact KEEP result')
      if(.not.all(ieee_is_finite(result))) error stop 'nonfinite compact KEEP result'
      local_error=0
      do component=1,5
        normalizer=max(1._real_kind,maxval(abs(7-expected(:,component))))
        do kk=0,upper(3)
        do jj=0,upper(2)
        do ii=0,upper(1)
          point_index=[ii,jj,kk];p=point_index(axis)+1
          local_error=max(local_error,abs(result(ii,jj,kk,component)-(7-expected(p,component)))/normalizer)
        enddo
        enddo
        enddo
      enddo
      if(real_bytes==8.and.local_error>2.e-10_real_kind) error stop 'compact KEEP LU reference mismatch'
      if(debug==0) then
        compact_gold(:,:,:,:,mode)=result
      elseif(any(real_to_word(result)/=real_to_word(compact_gold(:,:,:,:,mode)))) then
        error stop 'compact KEEP same-batch synchronization words'
      endif
      write(*,'(a,4(1x,i0),1x,es24.16e3)') 'CPP_KEEP_COMPACT',axis,n,debug,capacity,local_error
      if(mode==2) then
        ! Constant operator inputs are a manufactured property, not repaired flow data.
        call check(device_state_upload(state,device_rho,0*rho+1,bytes(size(rho))),'constant density')
        call check(device_state_upload(state,device_velocity,0*vel,bytes(size(vel))),'constant velocity')
        call check(device_state_upload(state,device_pressure,0*prs+1,bytes(size(prs))),'constant pressure')
        call check(fluid_keep_convection(state,gamma),'constant explicit KEEP')
        call check(device_state_download(state,device_rhs,result,bytes(size(result))),'constant explicit observation')
        if(any(result/=0)) error stop 'constant explicit KEEP invariant'
        stencil%start=0
        call check(compact_keep_rhs_cpp(state,plan%handle,stencil,int(total,c_int),gamma),'constant compact basis')
        call solve_distributed_compact_cpp(plan,total)
        call restore_distributed_compact_cpp(plan,total)
        call check(compact_keep_commit_cpp(state,plan%handle,stencil,int(total,c_int),gamma),'constant compact KEEP')
        call check(device_state_download(state,device_rhs,result,bytes(size(result))),'constant compact observation')
        if(any(result/=0)) error stop 'constant compact KEEP invariant'
        write(*,'(a,3(1x,i0))') 'CPP_KEEP_CONSTANT_EXACT',axis,n,debug
      endif
      call release_distributed_compact_cpp(plan)
    enddo
  end subroutine

  subroutine basic_boundaries()
    integer(c_int) :: types(3),first(3),last(3)
    integer :: mask,exclude,a,indices(3),cases
    logical :: low(3),high(3)
    real(real_kind) :: nan,local_max
    nan=ieee_value(0.e0_real_kind,ieee_quiet_nan)
    upper=[8,7,6];cases=0;local_max=0
    allocate(q(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
      rho(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
      vel(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3), &
      prs(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
      jac(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
      metric(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3,3), &
      result(0:upper(1),0:upper(2),0:upper(3),5),reference(0:upper(1),0:upper(2),0:upper(3),5), &
      gold(0:upper(1),0:upper(2),0:upper(3),5))
    do debug=0,1
      call check(device_state_create(0_c_int,int(upper(1),c_int),int(upper(2),c_int),int(upper(3),c_int), &
        int(halo,c_int),int(debug,c_int),state),'physical create')
      call check(fluid_prepare(state,0_c_int),'physical primitives')
      call check(fluid_rhs_prepare(state,0_c_int),'physical RHS')
      do mask=0,63
        do a=1,3
          low(a)=btest(mask,2*a-2);high(a)=btest(mask,2*a-1)
          types(a)=3
          if(low(a)) types(a)=1
          if(high(a)) types(a)=2
          if(low(a).and.high(a)) types(a)=4
        enddo
        do k=-halo,upper(3)+halo
        do j=-halo,upper(2)+halo
        do i=-halo,upper(1)+halo
          indices=[i,j,k];x=.13e0_real_kind*i+.07e0_real_kind*j-.11e0_real_kind*k
          rho(i,j,k)=1+.03e0_real_kind*sin(x);prs(i,j,k)=1+.02e0_real_kind*cos(2*x)
          vel(i,j,k,:)=[.2e0_real_kind+.03e0_real_kind*cos(x),.01e0_real_kind*sin(2*x),-.02e0_real_kind*cos(3*x)]
          jac(i,j,k)=1+.01e0_real_kind*sin(x)
          do a=1,3
          do c=1,3
            metric(i,j,k,a,c)=merge(1.e0_real_kind,.05e0_real_kind,a==c)+.01e0_real_kind*cos(x+a+c)
          enddo
          enddo
          q(i,j,k,1)=rho(i,j,k);q(i,j,k,2:4)=rho(i,j,k)*vel(i,j,k,:)
          q(i,j,k,5)=prs(i,j,k)/(gamma-1)+.5e0_real_kind*rho(i,j,k)*sum(vel(i,j,k,:)**2)
          if(any(low.and.indices<0).or.any(high.and.indices>upper)) then
            rho(i,j,k)=nan;prs(i,j,k)=nan;vel(i,j,k,:)=nan;jac(i,j,k)=nan;metric(i,j,k,:,:)=nan;q(i,j,k,:)=nan
          endif
        enddo
        enddo
        enddo
        call check(device_state_upload(state,device_q,q,bytes(size(q))),'physical q')
        call check(device_state_upload(state,device_rho,rho,bytes(size(rho))),'physical rho')
        call check(device_state_upload(state,device_velocity,vel,bytes(size(vel))),'physical velocity')
        call check(device_state_upload(state,device_pressure,prs,bytes(size(prs))),'physical pressure')
        call check(device_state_upload(state,device_metric,metric,bytes(size(metric))),'physical metric')
        call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))),'physical Jacobian')
        do exclude=0,1
          first=0;last=upper
          if(exclude==1) then
            where(low) first=1
            where(high) last=upper-1
          endif
          do k=0,upper(3)
          do j=0,upper(2)
          do i=0,upper(1)
          do c=1,5
            gold(i,j,k,c)=7+.01e0_real_kind*(i+3*j+7*k+11*c)
          enddo
          enddo
          enddo
          enddo
          reference=0
          call keep_explicit_volume(upper(1),upper(2),upper(3),halo,int(types),int(first),int(last), &
            q,rho,vel,prs,jac,metric,gamma,reference)
          reference=gold-reference
          call check(device_state_upload(state,device_rhs,gold,bytes(size(gold))),'initial boundary RHS')
          call check(fluid_keep_accumulate_basic(state,gamma,types,first,last),'physical accumulate')
          call check(device_state_download(state,device_rhs,result,bytes(size(result))),'physical result')
          if(.not.all(ieee_is_finite(result)).or..not.all(ieee_is_finite(reference))) error stop 'physical KEEP ghost read'
          difference=0
          do c=1,5
            difference=max(difference,maxval(abs(result(:,:,:,c)-reference(:,:,:,c)))/ &
              max(1.e0_real_kind,maxval(abs(reference(:,:,:,c)))))
          enddo
          if(real_bytes==8.and..not.(difference<=2.e-10_real_kind)) error stop 'physical KEEP CPU volume parity'
          do k=0,upper(3)
          do j=0,upper(2)
          do i=0,upper(1)
            if(all([i,j,k]>=first).and.all([i,j,k]<=last)) cycle
            if(any(real_to_word(result(i,j,k,:))/=real_to_word(gold(i,j,k,:)))) &
              error stop 'physical KEEP changed boundary-owned RHS'
          enddo
          enddo
          enddo
          local_max=max(local_max,difference);cases=cases+1
        enddo
      enddo
      call check(device_state_destroy(state),'physical destroy')
    enddo
    deallocate(q,rho,vel,prs,jac,metric,result,reference,gold)
    maximum=max(maximum,local_max)
    write(*,'(a,1x,i0,1x,es24.16e3)') 'CPP_KEEP_BASIC_BOUNDARY_PASS',cases,local_max
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*real_bytes
  end function
  subroutine check(code,label)
    integer,intent(in) :: code
    character(*),intent(in) :: label
    if(code==0) return
    write(*,'(a,1x,a,1x,i0)') 'CPP_KEEP_FAILURE',label,code
    error stop 1
  end subroutine
end program
