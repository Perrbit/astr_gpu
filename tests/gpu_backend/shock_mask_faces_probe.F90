program shock_mask_faces_probe
  use mpi
  implicit none
  integer,parameter :: width=7
  integer :: ierr,rank,nrank,axis,mode,i,lower,upper,local_wrong,global_wrong
  integer :: mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,isize,jsize,ksize
  logical :: lihomo,ljhomo,lkhomo
  integer(1) :: sendbuf(width,2),recvbuf(width,2),expected(width,2)
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nrank,ierr)
  local_wrong=0
  do axis=1,3
    do mode=0,3
      mpileft=MPI_PROC_NULL;mpiright=MPI_PROC_NULL
      mpidown=MPI_PROC_NULL;mpiup=MPI_PROC_NULL
      mpiback=MPI_PROC_NULL;mpifront=MPI_PROC_NULL
      isize=1;jsize=1;ksize=1
      lihomo=.false.;ljhomo=.false.;lkhomo=.false.
      lower=MPI_PROC_NULL;upper=MPI_PROC_NULL
      if(mode==2) then
        if(rank>0) lower=rank-1
        if(rank<nrank-1) upper=rank+1
      else if(mode==3.and.nrank>1) then
        lower=modulo(rank-1,nrank);upper=modulo(rank+1,nrank)
      endif
      select case(axis)
      case(1)
        mpileft=lower;mpiright=upper
        lihomo=mode==1.or.mode==3
        if(mode>=2) isize=nrank
      case(2)
        mpidown=lower;mpiup=upper
        ljhomo=mode==1.or.mode==3
        if(mode>=2) jsize=nrank
      case(3)
        mpiback=lower;mpifront=upper
        lkhomo=mode==1.or.mode==3
        if(mode>=2) ksize=nrank
      end select
      do i=1,width
        sendbuf(i,1)=int(rank*20+i,1)
        sendbuf(i,2)=int(rank*20+width+i,1)
      enddo
      expected=0_1
      if(mode==1.or.(mode==3.and.nrank==1)) then
        expected(:,1)=sendbuf(:,2)
        expected(:,2)=sendbuf(:,1)
      else
        do i=1,width
          if(lower/=MPI_PROC_NULL) expected(i,1)=int(lower*20+width+i,1)
          if(upper/=MPI_PROC_NULL) expected(i,2)=int(upper*20+i,1)
        enddo
      endif
      recvbuf=-99_1
      call exchange_shock_mask_faces(sendbuf,recvbuf,width,axis)
      local_wrong=local_wrong+count(recvbuf/=expected)
    enddo
  enddo
  call MPI_Allreduce(local_wrong,global_wrong,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
  if(rank==0) write(*,'(a,2(i0,1x))') 'SHOCK_MASK_FACES ranks/wrong=',nrank,global_wrong
  call MPI_Finalize(ierr)
  if(global_wrong/=0) stop 1
contains
  include 'exchange_shock_mask_faces.inc'
end program
