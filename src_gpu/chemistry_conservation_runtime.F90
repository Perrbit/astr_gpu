module chemistry_conservation_cpp
  use iso_c_binding, only: c_ptr,c_int64_t,c_double
  use iso_fortran_env, only: int64
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use parallel, only: mpirank
  use chemistry_air5_data, only: air5_num_species,air5_atom_count,air5_molar_mass
  use chemistry_state_layout, only: air5_num_conservative,air5_idx_density,air5_idx_total_energy, &
    air5_idx_species_first,air5_idx_species_last
  use chemistry_device_statistics, only: air5_conservation_prepare,air5_conservation_collect
  implicit none
  private
  public :: configure_air5_conservation_cpp,begin_air5_conservation_file_cpp,close_air5_conservation_file_cpp
  public :: air5_conservation_state_cpp,restore_air5_conservation_state_cpp,sample_air5_conservation_cpp
  logical,save :: enabled=.false.,baseline_set=.false.,echo_baseline=.false.
  integer(int64),save :: first_step=0,last_step=0,samples=0
  real(c_double),save :: baseline(air5_num_conservative)=0
  integer,save :: unit
contains
  subroutine configure_air5_conservation_cpp(handle,active,budget)
    type(c_ptr),intent(in) :: handle
    logical,intent(in) :: active
    integer(int64),intent(in) :: budget
    integer :: status
    integer(c_int64_t) :: bytes
    enabled=active
    if(.not.enabled) return
    status=air5_conservation_prepare(handle,int(budget,c_int64_t),bytes)
    call require(status==0,'C++ AIR5 conservation workspace')
    if(mpirank==0) write(*,'(a,i0,a)') 'ASTR_OUTPUT_AIR5_CONSERVATION device_bytes=',bytes, &
      ' sample_download_bytes=88 fixed_order=yes'
  end subroutine

  subroutine begin_air5_conservation_file_cpp(path,ok)
    character(*),intent(in) :: path
    logical,intent(out) :: ok
    integer :: status
    ok=.true.
    if(.not.enabled.or.mpirank/=0)return
    open(newunit=unit,file=path,status='new',action='write',iostat=status)
    if(status==0) write(unit,'(a)',iostat=status) &
      '# step phase q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 element_n element_o '// &
      'species_mass_closure rel_mass rel_energy rel_element_n rel_element_o max_species_abs_drift'
    if(status==0.and.echo_baseline) write(unit,'(a)',iostat=status) &
      '# continuation: phase=0 repeats the persisted baseline, not a new sample'
    ok=status==0
  end subroutine

  subroutine close_air5_conservation_file_cpp()
    integer :: status
    if(.not.enabled)return
    status=0
    if(mpirank==0)close(unit,iostat=status)
    call require(status==0,'close C++ AIR5 conservation diagnostic')
  end subroutine

  subroutine air5_conservation_state_cpp(metadata,totals)
    integer(int64),intent(out) :: metadata(5)
    real(c_double),intent(out) :: totals(air5_num_conservative)
    metadata=[int(merge(1,0,enabled),int64),int(merge(1,0,baseline_set),int64),first_step,last_step,samples]
    totals=baseline
  end subroutine

  subroutine restore_air5_conservation_state_cpp(metadata,totals)
    integer(int64),intent(in) :: metadata(5)
    real(c_double),intent(in) :: totals(air5_num_conservative)
    baseline=totals
    baseline_set=metadata(2)==1
    echo_baseline=baseline_set
    first_step=metadata(3);last_step=metadata(4);samples=metadata(5)
  end subroutine

  subroutine sample_air5_conservation_cpp(handle,step,phase)
    type(c_ptr),intent(in) :: handle
    integer(int64),intent(in) :: step
    integer,intent(in) :: phase
    real(c_double) :: local(air5_num_conservative),total(air5_num_conservative)
    integer :: status
    if(.not.enabled.or.(phase==0.and.baseline_set)) return
    call require(phase==0.or.(baseline_set.and.step==last_step+1),'C++ AIR5 conservation sample sequence')
    status=air5_conservation_collect(handle,local)
    call require(status==0,'collect C++ AIR5 conservation integrals')
    call MPI_Allreduce(local,total,air5_num_conservative,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,status)
    call require(status==MPI_SUCCESS.and.all(ieee_is_finite(total)),'C++ AIR5 global conservation integrals')
    if(phase==0) then
      baseline=total;first_step=step;baseline_set=.true.
    else
      if(echo_baseline)call write_record(first_step,0,baseline)
      echo_baseline=.false.;samples=samples+1
    endif
    call write_record(step,phase,total)
    last_step=step
  end subroutine

  subroutine write_record(step,phase,total)
    integer(int64),intent(in) :: step
    integer,intent(in) :: phase
    real(c_double),intent(in) :: total(air5_num_conservative)
    real(c_double) :: elements(2),reference_elements(2),relative_elements(2),closure,relative_mass,relative_energy,drift
    integer :: e,s,c,status
    status=0
    if(mpirank==0)then
      elements=0;reference_elements=0
      do e=1,2
        do s=1,air5_num_species
          c=air5_idx_species_first+s-1
          elements(e)=elements(e)+total(c)*real(air5_atom_count(e,s),c_double)/air5_molar_mass(s)
          reference_elements(e)=reference_elements(e)+baseline(c)*real(air5_atom_count(e,s),c_double)/air5_molar_mass(s)
        enddo
      enddo
      closure=sum(total(air5_idx_species_first:air5_idx_species_last))-total(air5_idx_density)
      relative_mass=(total(air5_idx_density)-baseline(air5_idx_density))/ &
        max(abs(baseline(air5_idx_density)),tiny(1.0_c_double))
      relative_energy=(total(air5_idx_total_energy)-baseline(air5_idx_total_energy))/ &
        max(abs(baseline(air5_idx_total_energy)),tiny(1.0_c_double))
      relative_elements=(elements-reference_elements)/max(abs(reference_elements),tiny(1.0_c_double))
      drift=maxval(abs(total(air5_idx_species_first:air5_idx_species_last)- &
        baseline(air5_idx_species_first:air5_idx_species_last)))
      write(unit,'(I10,1X,I1,1X,19(ES24.16E3,1X))',iostat=status) step,phase,total,elements,closure, &
        relative_mass,relative_energy,relative_elements,drift
      if(status==0)flush(unit,iostat=status)
    endif
    call require(status==0,'write C++ AIR5 conservation diagnostic')
  end subroutine

  subroutine require(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    integer :: ierr
    if(ok)return
    write(*,'(a,i0,2a)') 'ASTR_GPU_CPP AIR5 rank=',mpirank,' ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end module
