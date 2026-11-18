program compact_line_probe
  use iso_c_binding, only: c_ptr,c_int,c_int64_t,c_size_t,c_associated
  use ieee_arithmetic, only: ieee_is_finite
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use compact_line_solver, only: compact_matrix,compact_reference_factor, &
    factor_compact_reference,solve_compact_reference,compact_matvec,compact_residual
  use compact_line_cpp
  use device_state
  implicit none
  integer,parameter :: lengths(7)=[2,7,127,128,129,257,513],counts(3)=[1,5,11],tiles(2)=[4,128],capacity=11
  type(compact_matrix) :: matrix
  type(compact_reference_factor) :: reference
  type(compact_cpp_plan) :: plan
  type(device_view) :: input_view,output_view,again_input,again_output
  type(c_ptr) :: state
  real(real_kind),allocatable :: exact(:,:),rhs(:,:),cpu(:,:),actual(:,:),gold(:,:),unchanged(:,:)
  real(real_kind) :: scaled,residual,manufactured,maximum_scaled=0,maximum_residual=0
  integer :: tile,n,kind,debug,ir,nrhs,i,j,cases=0
  integer(c_int) :: status
  integer(c_size_t) :: maximum_bytes=0
  do tile=1,size(tiles)
    do ir=1,size(lengths)
      n=lengths(ir)
      allocate(exact(n,capacity),rhs(n,capacity),cpu(n,capacity),actual(n,capacity),gold(n,capacity),unchanged(n,capacity))
      do j=1,capacity
        do i=1,n
          exact(i,j)=sin(0.013e0_real_kind*i*j)+0.15e0_real_kind*cos(0.021e0_real_kind*i*(j+1))
        enddo
      enddo
      do kind=1,3
        matrix%n=n;matrix%cyclic=kind/=3
        allocate(matrix%lower(n),matrix%diagonal(n),matrix%upper(n))
        matrix%diagonal=1.e0_real_kind
        matrix%lower=1.e0_real_kind/3.e0_real_kind;matrix%upper=matrix%lower
        if(kind==2) then
          matrix%lower=0.49e0_real_kind;matrix%upper=0.49e0_real_kind
        elseif(kind==3) then
          do i=1,n
            matrix%diagonal(i)=2.e0_real_kind+0.01e0_real_kind*i/n
            matrix%lower(i)=-0.2e0_real_kind+0.03e0_real_kind*i/n
            matrix%upper(i)=0.15e0_real_kind+0.02e0_real_kind*i/n
          enddo
          matrix%lower(1)=0;matrix%upper(n)=0
        endif
        call compact_matvec(matrix,exact,rhs)
        call factor_compact_reference(matrix,reference)
        call solve_compact_reference(reference,rhs,cpu)
        residual=compact_residual(matrix,cpu,rhs)
        if(.not.all(ieee_is_finite(cpu)).or..not.ieee_is_finite(residual)) error stop 'nonfinite LU reference'
        if(real_bytes==8.and.residual>1.e-12_real_kind) error stop 'independent LU residual'
        do debug=0,1
          status=device_state_create(0_c_int,3_c_int,3_c_int,3_c_int,1_c_int,int(debug,c_int),state)
          if(status/=0) error stop 'compact component context creation'
          call prepare_compact_cpp(state,matrix,tiles(tile),capacity,plan)
          maximum_bytes=max(maximum_bytes,plan%device_bytes)
          call compact_views_cpp(plan,input_view,output_view)
          if(input_view%element_bytes/=real_bytes.or.output_view%element_bytes/=real_bytes.or. &
            input_view%rank/=2.or.any(input_view%extent(:2)/=[n,capacity]).or. &
            input_view%stride(1)/=1.or.input_view%stride(2)/=n) error stop 'borrowed matrix view layout'
          status=device_state_destroy(state)
          if(status/=device_invalid_argument.or..not.c_associated(state)) error stop 'parent lifetime guard'
          call upload_compact_rhs_cpp(plan,rhs,capacity)
          do nrhs=1,size(counts)
            j=counts(nrhs)
            call solve_compact_cpp(plan,j)
            call download_compact_solution_cpp(plan,actual,j)
            if(.not.all(ieee_is_finite(actual(:,:j)))) error stop 'nonfinite compact C++ result'
            scaled=0;manufactured=0
            do i=1,j
              scaled=max(scaled,maxval(abs(actual(:,i)-cpu(:,i)))/max(1.e0_real_kind,maxval(abs(cpu(:,i)))))
              manufactured=max(manufactured,maxval(abs(actual(:,i)-exact(:,i)))/ &
                max(1.e0_real_kind,maxval(abs(exact(:,i)))))
            enddo
            residual=compact_residual(matrix,actual(:,:j),rhs(:,:j))
            if(.not.all(ieee_is_finite([scaled,manufactured,residual]))) error stop 'nonfinite compact diagnostics'
            if(real_bytes==8.and.(scaled>2.e-10_real_kind.or.manufactured>2.e-10_real_kind.or.residual>1.e-12_real_kind)) &
              error stop 'compact C++ manufactured/LU/residual gate'
            if(debug==1) then
              if(any(real_to_word(actual(:,:j))/=real_to_word(gold(:,:j)))) error stop 'compact sync isolation'
            elseif(j==capacity) then
              gold=actual
            endif
            maximum_scaled=max(maximum_scaled,scaled);maximum_residual=max(maximum_residual,residual)
            cases=cases+1
            write(*,'(a,5(1x,i0),2(1x,es24.16e3))') 'CPP_COMPACT',n,kind,tiles(tile),debug,j,scaled,residual
          enddo
          call download_compact_rhs_cpp(plan,unchanged,capacity)
          if(any(real_to_word(unchanged)/=real_to_word(rhs))) error stop 'compact C++ modified RHS bits'
          unchanged=0
          call upload_compact_rhs_cpp(plan,unchanged,capacity)
          call solve_compact_cpp(plan,capacity)
          call download_compact_solution_cpp(plan,actual,capacity)
          if(any(actual/=0)) error stop 'compact zero RHS invariant'
          call compact_views_cpp(plan,again_input,again_output)
          if(.not.c_associated(again_input%data,input_view%data).or. &
            .not.c_associated(again_output%data,output_view%data)) error stop 'compact buffer changed during solve'
          call release_compact_cpp(plan)
          status=device_state_destroy(state)
          if(status/=0.or.c_associated(state)) error stop 'compact component context release'
        enddo
        deallocate(matrix%lower,matrix%diagonal,matrix%upper)
      enddo
      deallocate(exact,rhs,cpu,actual,gold,unchanged)
    enddo
  enddo
  write(*,'(a,1x,i0,2(1x,es24.16e3))') 'CPP_COMPACT_PASS',cases,maximum_scaled,maximum_residual
  write(*,'(a,1x,i0)') 'CPP_COMPACT_OWNED_DEVICE_BYTES',maximum_bytes
  write(*,'(a,1x,i0)') 'CPP_COMPACT_ELEMENT_BYTES',real_bytes
end program
