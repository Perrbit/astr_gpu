module output_input_resources
  use iso_c_binding, only: c_int,c_char,c_null_char
  implicit none
  private
  public :: set_inflow_resource_root,inflow_source_path,inflow_source_name,discover_inflow_sources
  public :: set_initial_resource_root,initial_source_path,initial_source_name
  character(1200),save :: inflow_root=''
  character(1200),save :: initial_root=''
  integer,save :: frozen_count=0
  interface
    function inflow_count(path,count) bind(C,name='astr_checkpoint_inflow_count') result(status)
      import c_int,c_char
      character(c_char),intent(in) :: path(*)
      integer(c_int),intent(out) :: count
      integer(c_int) :: status
    end function
  end interface
contains
  subroutine set_initial_resource_root(path)
    character(*),intent(in) :: path
    if(len_trim(path)>len(initial_root)) error stop 'initial resource root is too long'
    initial_root=path
  end subroutine

  function initial_source_name(dimension) result(name)
    integer,intent(in) :: dimension
    character(12) :: name
    if(dimension<1.or.dimension>3) error stop 'initial resource dimension must be 1:3'
    write(name,'("flowini",i1,"d.h5")') dimension
  end function

  function initial_source_path(dimension) result(path)
    integer,intent(in) :: dimension
    character(1400) :: path
    path='datin/'//initial_source_name(dimension)
    if(len_trim(initial_root)>0) path=trim(initial_root)//'/'//initial_source_name(dimension)
  end function

  subroutine discover_inflow_sources(path,count,ok)
    character(*),intent(in) :: path
    integer,intent(out) :: count
    logical,intent(out) :: ok
    integer(c_int) :: found,status
    status=inflow_count(trim(path)//c_null_char,found)
    count=int(found)
    ok=status==0
  end subroutine

  subroutine set_inflow_resource_root(path,count)
    character(*),intent(in) :: path
    integer,intent(in),optional :: count
    if(len_trim(path)>len(inflow_root)) error stop 'inflow resource root is too long'
    inflow_root=path
    frozen_count=0
    if(present(count)) frozen_count=count
  end subroutine

  function inflow_source_name(number) result(name)
    integer,intent(in) :: number
    character(14) :: name
    if(number<0.or.number>99999) error stop 'inflow resource index is outside 00000:99999'
    write(name,'("islice",i5.5,".h5")') number
  end function

  function inflow_source_path(number) result(path)
    integer,intent(in) :: number
    character(1400) :: path
    if(frozen_count>0.and.number>=frozen_count) error stop 'frozen inflow resource range exhausted'
    path='inflow/'//inflow_source_name(number)
    if(len_trim(inflow_root)>0) path=trim(inflow_root)//'/'//inflow_source_name(number)
  end function
end module
