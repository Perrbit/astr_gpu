module astr_precision_io
  use iso_c_binding, only: c_float,c_double
  use iso_fortran_env, only: int32
  use astr_precision, only: real_kind,real_bytes
  implicit none
  private
  public :: read_source_reals,source_real_matches
  public :: pack_logical32,unpack_logical32
  interface read_source_reals
    module procedure read_source_scalar,read_source_vector,read_source_matrix
  end interface
contains
  elemental integer(int32) function pack_logical32(value) result(word)
    logical,intent(in) :: value
    word=merge(1_int32,0_int32,value)
  end function

  elemental logical function unpack_logical32(word) result(value)
    integer(int32),intent(in) :: word
    ! Existing NVHPC batches use -1 for true; never load that as GNU LOGICAL.
    value=word/=0_int32
  end function

  elemental logical function source_real_matches(saved,configured,source_bytes) result(matches)
    real(real_kind),intent(in) :: saved,configured
    integer,optional,intent(in) :: source_bytes
    matches=saved==configured
    if(present(source_bytes)) then
      if(source_bytes==4.and.real_bytes==8) &
        matches=real(saved,c_float)==real(configured,c_float)
    endif
  end function

  subroutine read_source_scalar(unit,value,source_bytes,status)
    integer,intent(in) :: unit
    integer,optional,intent(in) :: source_bytes
    real(real_kind),intent(out) :: value
    integer,intent(out) :: status
    real(real_kind) :: values(1)
    call read_source_vector(unit,values,source_bytes,status)
    if(status==0) value=values(1)
  end subroutine

  subroutine read_source_vector(unit,values,source_bytes,status)
    integer,intent(in) :: unit
    integer,optional,intent(in) :: source_bytes
    real(real_kind),intent(out) :: values(:)
    integer,intent(out) :: status
    integer :: bytes
    bytes=real_bytes
    if(present(source_bytes)) bytes=source_bytes
    if(bytes==real_bytes) then
      read(unit,iostat=status) values
    else
      ! Only small file-control records use these conversion buffers, never fields.
      select case(bytes)
      case(4)
        block
          real(c_float) :: saved(size(values))
          read(unit,iostat=status) saved
          if(status==0) values=real(saved,real_kind)
        end block
      case(8)
        block
          real(c_double) :: saved(size(values))
          read(unit,iostat=status) saved
          if(status==0) values=real(saved,real_kind)
        end block
      case default
        status=1
      end select
    endif
  end subroutine

  subroutine read_source_matrix(unit,values,source_bytes,status)
    integer,intent(in) :: unit
    integer,optional,intent(in) :: source_bytes
    real(real_kind),intent(out) :: values(:,:)
    integer,intent(out) :: status
    real(real_kind) :: flat(size(values))
    call read_source_vector(unit,flat,source_bytes,status)
    if(status==0) values=reshape(flat,shape(values))
  end subroutine
end module astr_precision_io
