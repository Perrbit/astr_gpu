program timestep_counter_probe
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind
  use iso_c_binding, only: c_int,c_char,c_int64_t,c_null_char
  use mpi
  use commvar, only: nstep,nsrpt,maxstep,feqlist,feqchkpt,feqavg
  use statistic, only: nsamples,nstep_sbeg
  use mainloop, only: loop_counter
  use parallel, only: bcast,mpirank,lio
  use hdf5io, only: h5io_init,h5io_end,h5write,h5read,h5srite,h5sread
  use run_control_config, only: run_control_options,run_finished
  use run_control_runtime, only: active_run
  use readwrite, only: apply_run_input
  implicit none
  integer :: ierr,unit
  integer(int64) :: expected,restored
  character(1024) :: path
  character(64) :: name
  type(run_control_options) :: options
  integer(c_int) :: count,status
  integer(c_int64_t) :: steps(3)
  interface
    integer(c_int) function catalog(path,steps,capacity,count) bind(C,name='astr_output_frame_catalog')
      import c_int,c_char,c_int64_t
      character(c_char),intent(in) :: path(*)
      integer(c_int64_t),intent(out) :: steps(*)
      integer(c_int),value :: capacity
      integer(c_int),intent(out) :: count
    end function
  end interface
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  lio=mpirank==0
  call get_command_argument(1,path)
  if(storage_size(nstep)/=64.or.storage_size(nsrpt)/=64.or.storage_size(maxstep)/=64.or. &
    storage_size(nsamples)/=64.or.storage_size(nstep_sbeg)/=64.or.storage_size(loop_counter)/=64) &
    error stop 'solver counter kind is not 64 bits'
  expected=int(huge(0),int64)+11_int64
  active_run%max_steps=expected+1
  active_run%list_steps=expected
  active_run%statistics%sample_steps=expected+3
  call apply_run_input()
  if(maxstep/=expected+1.or.feqlist/=expected.or.feqchkpt/=expected.or.feqavg/=expected+3) &
    error stop 'named runtime counters were clamped in solver aliases'
  loop_counter=expected
  loop_counter=loop_counter+1
  if(loop_counter/=expected+1) error stop 'complete-step loop counter narrowed'
  nstep=0
  if(lio) nstep=expected
  call bcast(nstep)
  if(nstep/=expected) error stop 'MPI step counter narrowed'
  options%max_steps=expected+1
  if(run_finished(options,nstep,0.0_real_kind)) error stop '64-bit stop fired early'
  nstep=nstep+1
  if(.not.run_finished(options,nstep,0.0_real_kind)) error stop '64-bit stop failed at target'
  call h5io_init(trim(path)//'_collective.h5','write')
  call h5write('nstep',nstep)
  call h5io_end()
  call h5io_init(trim(path)//'_collective.h5','read')
  call h5read('nstep',restored)
  call h5io_end()
  if(restored/=nstep) error stop 'collective HDF5 counter narrowed'
  expected=1000000000000007_int64
  if(lio) call h5srite(expected,'nstep',trim(path)//'_serial.h5',newfile=.true.)
  call MPI_Barrier(MPI_COMM_WORLD,ierr)
  call h5sread(restored,'nstep',trim(path)//'_serial.h5')
  if(restored/=expected) error stop 'serial HDF5 counter narrowed'
  write(name,'("step",i0.12)') expected
  if(index(name,'*')/=0) error stop 'step filename overflow'
  read(name(5:),*) restored
  if(restored/=expected) error stop 'step filename lost digits'
  status=catalog(trim(path)//'_frames'//c_null_char,steps,3_c_int,count)
  if(status/=0.or.count/=3) error stop '64-bit frame catalog rejected valid names'
  if(any(steps/=[5_int64,expected,huge(0_int64)])) error stop 'frame catalog narrowed or reordered steps'
  print *, 'TIMESTEP_COUNTER_PASS'
  call MPI_Finalize(ierr)
end program
