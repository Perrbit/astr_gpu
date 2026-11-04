module case_model_options
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: reset_model_options,read_model_group,publish_model_options,get_case_option
  public :: write_model_option_receipt
  integer,parameter :: count=32,token_length=1024
  real(real64),parameter :: unset=-huge(0.0_real64)
  character(40),parameter :: keys(count)=[character(40) :: &
    'profile_inflow_mode','wall_blowing_mode','nscbc_farfield_mode','high_order_wall_mode', &
    'conservative_boundary_file','nscbc_farfield_rho','nscbc_farfield_u','nscbc_farfield_v', &
    'nscbc_farfield_w','nscbc_farfield_t','nscbc_farfield_shock_x','nscbc_farfield_shock_angle_deg', &
    'perfect_gas_prandtl','sutherland_temperature_k','channel_force_mode','channel_force_fixed', &
    'shuosher_shock_x','gpu_precision_mode','gpu_mixed_candidate','gpu_mp_positivity', &
    'source_mode','chemistry_tolerance','filter_mode','convection_limiter','diffusion_limiter', &
    'top_mode','top_tau','compensation','compensation_restart','step_history', &
    'low_activity_threshold','source_evaluation']
  character(48),parameter :: retired(count)=[character(48) :: &
    'ASTR_PROFILE_INFLOW_MODE','ASTR_WALL_BLOWING_MODE','ASTR_NSCBC_FARFIELD_MODE','ASTR_HIGH_ORDER_WALL_MODE', &
    'ASTR_CONSERVATIVE_BOUNDARY_FILE','ASTR_NSCBC_FARFIELD_RHO','ASTR_NSCBC_FARFIELD_U','ASTR_NSCBC_FARFIELD_V', &
    'ASTR_NSCBC_FARFIELD_W','ASTR_NSCBC_FARFIELD_T','ASTR_NSCBC_FARFIELD_SHOCK_X', &
    'ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG','ASTR_PERFECT_GAS_PRANDTL','ASTR_SUTHERLAND_TEMPERATURE_K', &
    'ASTR_CHANNEL_FORCE_MODE','ASTR_CHANNEL_FORCE_FIXED','ASTR_SHUOSHER_SHOCK_X', &
    'ASTR_GPU_PRECISION_MODE','ASTR_GPU_MIXED_CANDIDATE','ASTR_GPU_MP_POSITIVITY', &
    'ASTR_AIR5_SOURCE_MODE','ASTR_AIR5_CHEMISTRY_TOLERANCE','ASTR_AIR5_FILTER_MODE', &
    'ASTR_AIR5_CONVECTION_LIMITER','ASTR_AIR5_DIFFUSION_LIMITER','ASTR_AIR5_TOP_MODE','ASTR_AIR5_TOP_TAU', &
    'ASTR_AIR5_COMPENSATION','ASTR_AIR5_COMPENSATION_RESTART','ASTR_AIR5_STEP_HISTORY', &
    'ASTR_AIR5_LOW_ACTIVITY','ASTR_AIR5_SOURCE_EVALUATION']
  character(token_length),save :: tokens(count)=''
  character(64) :: profile_inflow_mode='',wall_blowing_mode='',nscbc_farfield_mode='',high_order_wall_mode=''
  character(token_length) :: conservative_boundary_file=''
  character(64) :: channel_force_mode='',gpu_precision_mode='',gpu_mixed_candidate='',gpu_mp_positivity=''
  character(64) :: source_mode='',chemistry_tolerance='',filter_mode='',convection_limiter='',diffusion_limiter=''
  character(64) :: top_mode='',compensation='',compensation_restart='',step_history='',source_evaluation=''
  real(real64) :: nscbc_farfield_rho=unset,nscbc_farfield_u=unset,nscbc_farfield_v=unset,nscbc_farfield_w=unset
  real(real64) :: nscbc_farfield_t=unset,nscbc_farfield_shock_x=unset,nscbc_farfield_shock_angle_deg=unset
  real(real64) :: perfect_gas_prandtl=unset,sutherland_temperature_k=unset,channel_force_fixed=unset
  real(real64) :: shuosher_shock_x=unset,top_tau=unset,low_activity_threshold=unset
contains
  subroutine reset_model_options()
    tokens=''
    profile_inflow_mode='';wall_blowing_mode='';nscbc_farfield_mode='';high_order_wall_mode=''
    conservative_boundary_file='';channel_force_mode='';gpu_precision_mode='';gpu_mixed_candidate=''
    gpu_mp_positivity='';source_mode='';chemistry_tolerance='';filter_mode='';convection_limiter=''
    diffusion_limiter='';top_mode='';compensation='';compensation_restart='';step_history='';source_evaluation=''
    nscbc_farfield_rho=unset;nscbc_farfield_u=unset;nscbc_farfield_v=unset;nscbc_farfield_w=unset
    nscbc_farfield_t=unset;nscbc_farfield_shock_x=unset;nscbc_farfield_shock_angle_deg=unset
    perfect_gas_prandtl=unset;sutherland_temperature_k=unset;channel_force_fixed=unset
    shuosher_shock_x=unset;top_tau=unset;low_activity_threshold=unset
  end subroutine

  subroutine read_model_group(unit,group,status,message)
    integer,intent(in) :: unit
    character(*),intent(in) :: group
    integer,intent(out) :: status
    character(*),intent(out) :: message
    namelist /boundary_options/ profile_inflow_mode,wall_blowing_mode,nscbc_farfield_mode, &
      high_order_wall_mode,conservative_boundary_file,nscbc_farfield_rho,nscbc_farfield_u, &
      nscbc_farfield_v,nscbc_farfield_w,nscbc_farfield_t,nscbc_farfield_shock_x,nscbc_farfield_shock_angle_deg
    namelist /transport_options/ perfect_gas_prandtl,sutherland_temperature_k,channel_force_mode,channel_force_fixed
    namelist /initial_conditions/ shuosher_shock_x
    namelist /precision_options/ gpu_precision_mode,gpu_mixed_candidate,gpu_mp_positivity
    namelist /chemistry/ source_mode,chemistry_tolerance,filter_mode,convection_limiter,diffusion_limiter, &
      top_mode,top_tau,compensation,compensation_restart,step_history,low_activity_threshold,source_evaluation
    message=''
    select case(group)
    case('boundary_options');read(unit,nml=boundary_options,iostat=status,iomsg=message)
    case('transport_options');read(unit,nml=transport_options,iostat=status,iomsg=message)
    case('initial_conditions');read(unit,nml=initial_conditions,iostat=status,iomsg=message)
    case('precision_options');read(unit,nml=precision_options,iostat=status,iomsg=message)
    case('chemistry');read(unit,nml=chemistry,iostat=status,iomsg=message)
    case default
      error stop 'Unknown model-option namelist group'
    end select
  end subroutine

  subroutine publish_model_options(communicator)
    use mpi
    integer,intent(in) :: communicator
    integer :: rank,status,ignored,i,length,env_status
    call MPI_Comm_rank(communicator,rank,status)
    if(status/=MPI_SUCCESS) call MPI_Abort(communicator,status,ignored)
    if(rank==0) then
      tokens=''
      tokens(1:5)=[character(token_length) :: profile_inflow_mode,wall_blowing_mode,nscbc_farfield_mode, &
        high_order_wall_mode,conservative_boundary_file]
      call number(6,nscbc_farfield_rho);call number(7,nscbc_farfield_u);call number(8,nscbc_farfield_v)
      call number(9,nscbc_farfield_w);call number(10,nscbc_farfield_t);call number(11,nscbc_farfield_shock_x)
      call number(12,nscbc_farfield_shock_angle_deg);call number(13,perfect_gas_prandtl)
      call number(14,sutherland_temperature_k);tokens(15)=channel_force_mode;call number(16,channel_force_fixed)
      call number(17,shuosher_shock_x)
      tokens(18:26)=[character(token_length) :: gpu_precision_mode,gpu_mixed_candidate,gpu_mp_positivity, &
        source_mode,chemistry_tolerance,filter_mode,convection_limiter,diffusion_limiter,top_mode]
      call number(27,top_tau)
      tokens(28:30)=[character(token_length) :: compensation,compensation_restart,step_history]
      call number(31,low_activity_threshold);tokens(32)=source_evaluation
      ! No physical value is ever fetched from the process environment.
      do i=1,count
        if(len_trim(retired(i))==0) cycle
        call get_environment_variable(trim(retired(i)),length=length,status=env_status)
        if(env_status==0.or.env_status==-1) write(*,'(4A)') &
          'ASTR_CASE ignored retired environment ',trim(retired(i)),'; input.case field: ',trim(keys(i))
      enddo
    endif
    call MPI_Bcast(tokens,token_length*count,MPI_CHARACTER,0,communicator,status)
    if(status/=MPI_SUCCESS) call MPI_Abort(communicator,status,ignored)
  contains
    subroutine number(index,value)
      integer,intent(in) :: index
      real(real64),intent(in) :: value
      if(.not.ieee_is_finite(value)) then
        write(*,'(2A)') 'ASTR_CASE nonfinite model option: ',trim(keys(index))
        call MPI_Abort(communicator,90,ignored)
      endif
      if(value/=unset) write(tokens(index),'(ES26.17E3)') value
      tokens(index)=adjustl(tokens(index))
    end subroutine
  end subroutine

  subroutine get_case_option(name,value,length,status)
    character(*),intent(in) :: name
    character(*),optional,intent(out) :: value
    integer,optional,intent(out) :: length,status
    integer :: i,n,result
    do i=1,count
      if(name==keys(i)) exit
    enddo
    if(i>count) error stop 'Unknown physical/numerical input key'
    n=len_trim(tokens(i));result=0
    if(n==0) result=1
    if(present(value)) then
      value=tokens(i)
      if(n>len(value)) result=-1
    endif
    if(present(length)) length=n
    if(present(status)) status=result
  end subroutine

  subroutine write_model_option_receipt(unit)
    integer,intent(in) :: unit
    character(40) :: option_names(count)
    character(token_length) :: option_values(count)
    namelist /model_option_receipt/ option_names,option_values
    option_names=keys;option_values=tokens
    write(unit,nml=model_option_receipt)
  end subroutine
end module case_model_options
