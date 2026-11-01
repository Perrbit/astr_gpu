  attributes(global) subroutine h1_common_capture_kernel(im,jm,km,left,down,before_diffusion)
    use commarray_gpu, only: qrhs_d
    integer,value :: im,jm,km,left,down
    logical,value :: before_diffusion
    integer :: id,i,j,k
    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(id>=(im+1)*(jm+1)*(km+1)) return
    i=mod(id,im+1); j=mod(id/(im+1),jm+1); k=id/((im+1)*(jm+1))
    if((left/=0.and.i==0).or.(down/=0.and.j==0)) then
      h1_common_d(i,j,k,:)=0.e0_real_kind
    elseif(before_diffusion) then
      h1_common_d(i,j,k,:)=-qrhs_d(i,j,k,:)
    else
      h1_common_d(i,j,k,:)=h1_common_d(i,j,k,:)+qrhs_d(i,j,k,:)
    endif
  end subroutine h1_common_capture_kernel

  subroutine capture_h1_common_gpu(before_diffusion)
    use commvar, only: im,jm,km,diffterm
    use boundary_high_order, only: h1_stage_budget_candidate
    use boundary_gpu, only: gpu_physical_boundary_apply_mask
    use gpu_check, only: sync_after_kernel
    logical,intent(in) :: before_diffusion
    integer :: left,right,down,up
    if(.not.(h1_stage_budget_candidate.and.diffterm)) return
    if(.not.allocated(h1_common_d)) allocate(h1_common_d(0:im,0:jm,0:km,5))
    call gpu_physical_boundary_apply_mask(1,left,right)
    call gpu_physical_boundary_apply_mask(2,down,up)
    call h1_common_capture_kernel<<<((im+1)*(jm+1)*(km+1)+127)/128,128>>>( &
      im,jm,km,left,down,before_diffusion)
    call sync_after_kernel('h1_common_capture_kernel',.true.)
  end subroutine capture_h1_common_gpu

  attributes(global) subroutine h1_add_common_kernel(im,jm,km)
    integer,value :: im,jm,km
    integer :: id,i,j,k
    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(id>=(im+1)*(jm+1)*(km+1)) return
    i=mod(id,im+1); j=mod(id/(im+1),jm+1); k=id/((im+1)*(jm+1))
    h1_low_d(i,j,k,:)=h1_low_d(i,j,k,:)+h1_common_d(i,j,k,:)
  end subroutine h1_add_common_kernel

  attributes(global) subroutine h1_budget_caps_kernel(dt,a,b,c,stage,im,jm,km,hm,left,right,down,up,roundoff_safe)
    use commarray_gpu, only: q_d,qsave_d,qrhs_d,jacob_d
    use boundary_high_order_gpu, only: boundary_budget_fraction5,boundary_budget_margin5,boundary_budget_roundoff_fraction5
    real(real_kind),value :: dt,a,b,c
    integer,value :: stage,im,jm,km,hm,left,right,down,up
    logical,value :: roundoff_safe
    real(real_kind) :: base(5),changes(5,6),work(5),scale,full(5)
    integer :: id,i,j,k,axis,prev(3),m,nparts
    logical :: open_face,valid
    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(id>=(im+1)*(jm+1)*(km+1)) return
    i=mod(id,im+1); j=mod(id/(im+1),jm+1); k=id/((im+1)*(jm+1))
    h1_residual_d(i,j,k,:)=0.e0_real_kind
    if((left/=0.and.i==0).or.(down/=0.and.j==0)) return
    scale=c*dt/jacob_d(i,j,k)
    full=h1_low_d(i,j,k,:)
    if(stage==1) then
      base=(q_d(i,j,k,:)*jacob_d(i,j,k)+dt*full)/jacob_d(i,j,k)
    else
      base=(a*qsave_d(i,j,k,:)+b*q_d(i,j,k,:)*jacob_d(i,j,k)+c*dt*full)/jacob_d(i,j,k)
    endif
    if(roundoff_safe) then
      valid=boundary_budget_margin5(base,64)
    else
      valid=admissible_mp_state(base)
    endif
    if(.not.valid) then
      call record_mp_failure(2,i,j,k,base)
      return
    endif
    do axis=1,3
      prev=[i,j,k]; prev(axis)=prev(axis)-1
      changes(:,2*axis-1)=delta_d(prev(1),prev(2),prev(3),:,axis)
      changes(:,2*axis)=-delta_d(i,j,k,:,axis)
    enddo
    open_face=(right/=0.and.i==im).or.(up/=0.and.j==jm)
    h1_residual_d(i,j,k,:)=qrhs_d(i,j,k,:)-full-sum(changes,dim=2)
    nparts=6
    if(open_face) then
      nparts=7
    endif
    do m=1,6
      work=real(nparts,real_kind)*scale*changes(:,m)
      if(roundoff_safe) then
        h1_caps_d(i,j,k,m)=boundary_budget_roundoff_fraction5(base,work)
      else
        h1_caps_d(i,j,k,m)=boundary_budget_fraction5(base,work)
      endif
    enddo
    if(open_face) then
      work=7.e0_real_kind*scale*h1_residual_d(i,j,k,:)
      if(roundoff_safe) then
        h1_caps_d(i,j,k,7)=boundary_budget_roundoff_fraction5(base,work)
      else
        h1_caps_d(i,j,k,7)=boundary_budget_fraction5(base,work)
      endif
    endif
  end subroutine h1_budget_caps_kernel

  attributes(global) subroutine h1_budget_cap_plane_kernel(axis,im,jm,km,nplane,unpack,has_lo,has_hi)
    integer,value :: axis,im,jm,km,nplane,unpack,has_lo,has_hi
    integer :: id,s,m,lo(3),hi(3),dims(3)
    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(id>=7*nplane) return
    s=mod(id,nplane); m=id/nplane+1; dims=[im,jm,km]
    select case(axis)
    case(1); lo=[0,mod(s,jm+1),s/(jm+1)]
    case(2); lo=[mod(s,im+1),0,s/(im+1)]
    case(3); lo=[mod(s,im+1),s/(im+1),0]
    end select
    hi=lo; hi(axis)=dims(axis)
    if(unpack==0) then
      send_lo_d(id+1,1,1,1)=h1_caps_d(lo(1),lo(2),lo(3),m)
      send_hi_d(id+1,1,1,1)=h1_caps_d(hi(1),hi(2),hi(3),m)
    else
      if(has_lo/=0) h1_caps_d(lo(1),lo(2),lo(3),m)= &
        min(h1_caps_d(lo(1),lo(2),lo(3),m),recv_lo_d(id+1,1,1,1))
      if(has_hi/=0) h1_caps_d(hi(1),hi(2),hi(3),m)= &
        min(h1_caps_d(hi(1),hi(2),hi(3),m),recv_hi_d(id+1,1,1,1))
    endif
  end subroutine h1_budget_cap_plane_kernel

  attributes(global) subroutine h1_budget_correct_kernel(im,jm,km,left,down)
    use commarray_gpu, only: qrhs_d
    integer,value :: im,jm,km,left,down
    integer :: id,i,j,k,axis,prev(3),next(3),m,ignored
    real(real_kind) :: theta(7),changes(5,6)
    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(id>=(im+1)*(jm+1)*(km+1)) return
    i=mod(id,im+1); j=mod(id/(im+1),jm+1); k=id/((im+1)*(jm+1))
    do axis=1,3
      prev=[i,j,k]; next=prev; prev(axis)=prev(axis)-1; next(axis)=next(axis)+1
      changes(:,2*axis-1)=delta_d(prev(1),prev(2),prev(3),:,axis)
      changes(:,2*axis)=-delta_d(i,j,k,:,axis)
      theta(2*axis-1)=min(h1_caps_d(i,j,k,2*axis-1),h1_caps_d(prev(1),prev(2),prev(3),2*axis))
      theta(2*axis)=min(h1_caps_d(i,j,k,2*axis),h1_caps_d(next(1),next(2),next(3),2*axis-1))
    enddo
    theta(7)=h1_caps_d(i,j,k,7); h1_theta_d(i,j,k,:)=theta
    if((left/=0.and.i==0).or.(down/=0.and.j==0)) return
    if(any(theta<1.e0_real_kind)) ignored=atomicadd(limited_d,1)
    do m=1,6
      if(theta(m)<1.e0_real_kind) qrhs_d(i,j,k,:)=qrhs_d(i,j,k,:)+(theta(m)-1.e0_real_kind)*changes(:,m)
    enddo
    if(theta(7)<1.e0_real_kind) qrhs_d(i,j,k,:)=qrhs_d(i,j,k,:)+ &
      (theta(7)-1.e0_real_kind)*h1_residual_d(i,j,k,:)
  end subroutine h1_budget_correct_kernel

  attributes(global) subroutine h1_budget_roundoff_check_kernel(dt,a,b,c,stage,im,jm,km)
    use commarray_gpu, only: q_d,qsave_d,qrhs_d,jacob_d
    use boundary_high_order_gpu, only: boundary_budget_margin5
    real(real_kind),value :: dt,a,b,c
    integer,value :: stage,im,jm,km
    real(real_kind) :: trial(5)
    integer :: id,i,j,k
    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(id>=(im+1)*(jm+1)*(km+1)) return
    i=mod(id,im+1); j=mod(id/(im+1),jm+1); k=id/((im+1)*(jm+1))
    if(stage==1) then
      trial=(q_d(i,j,k,:)*jacob_d(i,j,k)+qrhs_d(i,j,k,:)*dt)/jacob_d(i,j,k)
    else
      trial=(a*qsave_d(i,j,k,:)+b*q_d(i,j,k,:)*jacob_d(i,j,k)+c*qrhs_d(i,j,k,:)*dt)/jacob_d(i,j,k)
    endif
    if(.not.boundary_budget_margin5(trial,32)) call record_mp_failure(3,i,j,k,trial)
  end subroutine h1_budget_roundoff_check_kernel

  subroutine limit_h1_stage_gpu(dt,a,b,c,stage)
    use commvar, only: im,jm,km,hm,npdci,npdcj,npdck,gamma,bfacmpld,nstep,diffterm
    use boundary_high_order, only: h1_stage_budget_candidate,h1_outlet_filter_candidate
    use boundary_gpu, only: audit_h1_low_boundary_gpu,gpu_physical_boundary_apply_mask
    use parallel, only: mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,lio
    use halo_transport_gpu, only: exchange_host_pair
    use halo_exchange_gpu, only: exchange_field_halo_gpu
    use gpu_check, only: sync_after_kernel
    use validation_io, only: rhs_validation_requested,write_rhs_validation_snapshot
    use commarray_gpu, only: copy_rhs_from_gpu
    use mpi, only: MPI_PROC_NULL
    real(real_kind),intent(in) :: dt,a,b,c
    integer,intent(in) :: stage
    integer :: axis,types(3),dims(3),counts(3),lower(3),upper(3),n,left,right,down,up,count
    integer,device :: types_d(3)
    real(real_kind),allocatable :: sample(:,:,:,:)
    type(dim3) :: block,grid
    if(.not.h1_stage_budget_candidate) return
    if(mp_positivity_enabled) error stop 'H1 test budget cannot be combined with the old MP budget'
    call allocate_mp_positivity_gpu()
    if(.not.allocated(h1_low_d)) allocate(h1_low_d(0:im,0:jm,0:km,5),h1_residual_d(0:im,0:jm,0:km,5), &
      h1_caps_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,7),h1_theta_d(0:im,0:jm,0:km,7))
    types=[npdci,npdcj,npdck]; types_d=types; dims=[im,jm,km]
    counts=[(jm+1)*(km+1),(im+1)*(km+1),(im+1)*(jm+1)]
    lower=[mpileft,mpidown,mpiback]; upper=[mpiright,mpiup,mpifront]
    block=dim3(128,1,1); failure_d=0; limited_d=0
    do axis=1,3
      n=(dims(axis)+2)*counts(axis); grid=dim3((n+127)/128,1,1)
      call mp_faces_kernel<<<grid,block>>>(axis,im,jm,km,hm,types(axis),gamma,bfacmpld)
      call sync_after_kernel('h1_budget_faces',.true.)
      call exchange_mp_planes(axis,1)
    enddo
    call check_mp_failure('H1 budget face input/flux')
    n=(im+1)*(jm+1)*(km+1); grid=dim3((n+127)/128,1,1)
    call h1_low_rhs_audit_kernel<<<grid,block>>>(h1_low_d,im,jm,km,hm,types_d,gamma)
    call sync_after_kernel('h1_budget_low_rhs',.true.)
    call check_mp_failure('H1 budget low RHS')
    call audit_h1_low_boundary_gpu(h1_low_d)
    if(diffterm) then
      if(rhs_validation_requested()) then
        allocate(sample(0:im,0:jm,0:km,5)); sample=h1_low_d
        call write_rhs_validation_snapshot('budget_low_conv',stage_index=stage,values=sample)
        sample=h1_common_d
        call write_rhs_validation_snapshot('budget_common',stage_index=stage,values=sample)
        deallocate(sample)
      endif
      call h1_add_common_kernel<<<grid,block>>>(im,jm,km)
      call sync_after_kernel('h1_add_common_kernel',.true.)
    endif
    call gpu_physical_boundary_apply_mask(1,left,right)
    call gpu_physical_boundary_apply_mask(2,down,up)
    h1_caps_d=1.e0_real_kind
    call h1_budget_caps_kernel<<<grid,block>>>(dt,a,b,c,stage,im,jm,km,hm,left,right,down,up,h1_outlet_filter_candidate)
    call sync_after_kernel('h1_budget_caps',.true.)
    call check_mp_failure('H1 complete low-order RK state')
    do axis=1,3
      count=7*counts(axis)
      call h1_budget_cap_plane_kernel<<<(count+127)/128,128>>>(axis,im,jm,km,counts(axis),0,0,0)
      call sync_after_kernel('h1_budget_caps_pack',.true.)
      send_lo_h(1:count,1,1,1)=send_lo_d(1:count,1,1,1)
      send_hi_h(1:count,1,1,1)=send_hi_d(1:count,1,1,1)
      call exchange_host_pair(send_lo_h(1:count,:,:,:),send_hi_h(1:count,:,:,:), &
        recv_hi_h(1:count,:,:,:),recv_lo_h(1:count,:,:,:),count,lower(axis),upper(axis),24511,24512)
      if(lower(axis)/=MPI_PROC_NULL) recv_lo_d(1:count,1,1,1)=recv_lo_h(1:count,1,1,1)
      if(upper(axis)/=MPI_PROC_NULL) recv_hi_d(1:count,1,1,1)=recv_hi_h(1:count,1,1,1)
      call h1_budget_cap_plane_kernel<<<(count+127)/128,128>>>(axis,im,jm,km,counts(axis),1, &
        merge(1,0,lower(axis)/=MPI_PROC_NULL),merge(1,0,upper(axis)/=MPI_PROC_NULL))
      call sync_after_kernel('h1_budget_caps_min')
    enddo
    call exchange_field_halo_gpu(h1_caps_d(:,:,:,1:6),6)
    call exchange_field_halo_gpu(h1_caps_d(:,:,:,7:7),1)
    call h1_budget_correct_kernel<<<grid,block>>>(im,jm,km,left,down)
    call sync_after_kernel('h1_budget_correct',.true.)
    if(h1_outlet_filter_candidate) then
      call h1_budget_roundoff_check_kernel<<<grid,block>>>(dt,a,b,c,stage,im,jm,km)
    else
      call mp_check_update_kernel<<<grid,block>>>(dt,a,b,c,stage,im,jm,km)
    endif
    call sync_after_kernel('h1_budget_check',.true.)
    call check_mp_failure('H1 complete limited RK state')
    if(rhs_validation_requested()) then
      allocate(sample(0:im,0:jm,0:km,5)); sample=h1_low_d
      call write_rhs_validation_snapshot('budget_low',stage_index=stage,values=sample)
      sample=h1_residual_d
      call write_rhs_validation_snapshot('budget_residual',stage_index=stage,values=sample)
      deallocate(sample); allocate(sample(0:im,0:jm,0:km,7)); sample=h1_theta_d
      call write_rhs_validation_snapshot('budget_theta',stage_index=stage,values=sample)
      call copy_rhs_from_gpu()
      call write_rhs_validation_snapshot('limited',stage_index=stage)
    endif
    count=limited_d
    if(lio) write(*,'(a,3(1x,i0))') 'H1_BUDGET_GPU',nstep,stage,count
  end subroutine limit_h1_stage_gpu
