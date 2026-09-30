module insitu_runtime
  use mpi
  use iso_c_binding, only: c_int,c_char,c_null_char
  use iso_fortran_env, only: iostat_end,iostat_eor
  implicit none
  private
  public :: insitu_check
#ifdef ASTR_WITH_CATALYST
  interface
    integer(c_int) function backend_check(path,comm) bind(C,name='astr_catalyst_check')
      import c_int,c_char
      character(kind=c_char),intent(in) :: path(*)
      integer(c_int),value :: comm
    end function
  end interface
#endif
contains
  subroutine require_all(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    integer :: local_bad,any_bad,ierr,ignored,rank
    local_bad=0
    if(.not.ok) local_bad=1
    call MPI_Allreduce(local_bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,1,ignored)
    if(any_bad/=0) then
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
      if(rank==0) write(*,'(A)') 'ASTR INSITU ERROR: '//message
      call MPI_Abort(MPI_COMM_WORLD,1,ignored)
    endif
  end subroutine

  subroutine insitu_check()
    integer :: rank,ierr
#ifdef ASTR_WITH_CATALYST
    character(1024) :: filename,root_filename
    character(4096) :: implementation_path,root_path
    character(4096) :: line
    character(256) :: message
    integer :: narg,status,length,source,scratch,j,close_status
    logical :: exists,valid
    namelist /insitu/ implementation_path
#endif
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
    call require_all(ierr==MPI_SUCCESS,'cannot query MPI rank')
#ifndef ASTR_WITH_CATALYST
    call require_all(.false.,'insitu-check requires ASTR_WITH_CATALYST=ON')
#else
    narg=command_argument_count()
    call require_all(narg==2,'usage: astr insitu-check <configuration file>')
    filename=''
    call get_command_argument(2,filename,length,status)
    call require_all(status==0.and.length>0.and.length<=len(filename),'invalid configuration path')
    root_filename=filename
    call MPI_Bcast(root_filename,len(root_filename),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call require_all(ierr==MPI_SUCCESS,'cannot broadcast configuration path')
    call require_all(filename==root_filename,'configuration paths differ between ranks')

    ! Each rank parses its own file, then compares canonical values, not raw bytes.
    implementation_path=''
    open(newunit=source,file=trim(filename),status='old',action='read',iostat=status)
    call require_all(status==0,'cannot open configuration on every rank')
    open(newunit=scratch,status='scratch',form='formatted',action='readwrite',iostat=status)
    call require_all(status==0,'cannot create normalized configuration buffer')
    valid=.true.
    do
      read(source,'(A)',advance='no',iostat=status) line
      if(status==iostat_end) exit
      if(status/=iostat_eor) then
        valid=.false.
        exit
      endif
      do j=1,len_trim(line)
        if(iachar(line(j:j))==13) line(j:j)=' '
      enddo
      write(scratch,'(A)',iostat=status) trim(line)
      if(status/=0) then
        valid=.false.
        exit
      endif
    enddo
    close(source,iostat=close_status)
    call require_all(valid.and.close_status==0,'cannot normalize configuration (line limit 4095 characters)')
    rewind(scratch,iostat=status)
    call require_all(status==0,'cannot rewind configuration')
    read(scratch,nml=insitu,iostat=status,iomsg=message)
    if(status/=0) write(*,'(A,I0,2A)') 'ASTR INSITU rank ',rank,': ',trim(message)
    close(scratch,iostat=close_status)
    call require_all(status==0.and.close_status==0,'invalid &insitu namelist')
    call require_all(len_trim(implementation_path)>0,'implementation_path is required')
    call require_all(len_trim(implementation_path)<=1024,'implementation_path exceeds 1024 characters')
    root_path=implementation_path
    call MPI_Bcast(root_path,len(root_path),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call require_all(ierr==MPI_SUCCESS,'cannot broadcast backend configuration')
    call require_all(implementation_path==root_path,'backend configuration differs between ranks')
    exists=.false.
    inquire(file=trim(implementation_path)//'/libcatalyst-paraview.so',exist=exists,iostat=status)
    call require_all(status==0.and.exists,'ParaView Catalyst implementation library is missing')
    if(rank==0) write(*,'(A)') 'ASTR INSITU: backend admission only; no mesh, sampling, or rendering'
    status=backend_check(trim(implementation_path)//c_null_char,int(MPI_COMM_WORLD,c_int))
    call require_all(status==0,'Catalyst backend admission failed')
    if(rank==0) write(*,'(A)') 'ASTR INSITU: ParaView backend lifecycle PASS; flow was not advanced'
#endif
  end subroutine
end module
