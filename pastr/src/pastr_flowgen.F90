module pastr_flowgen

    use iso_fortran_env, only: wp => real64
    use pastr_constdef

    implicit none

contains
    
    subroutine flowgen

      use pastr_io,     only: parse_command_line

      character(len=32) :: gentype

      call parse_command_line(  string=gentype )

      if(trim(gentype)=='int2d') then
        print*,' ** to interpolate between 2 2D grids'
        call flowinterp2d()
      else
        stop '  !! gentype not defined !! @ flowgen'
      endif

    end subroutine flowgen

    subroutine flowinterp2d

      use pastr_commvar, only :  gridfile
      use pastr_h5io
      use pastr_input,only : read_astr_input
      use pastr_io,     only: parse_command_line
      use pastr_tecio
      use structured_grid_interp

      integer :: im,jm,km,in,jn,kn,dims(3)
      real(wp),allocatable :: x(:,:),y(:,:),ro(:,:),u1(:,:),u2(:,:),t(:,:)
      character(len=32) :: gridin
      real(wp),allocatable :: xo(:,:),yo(:,:),roo(:,:),u1o(:,:),u2o(:,:),to(:,:)

      call read_astr_input()

      dims=h5_getdimensio('x',trim(gridfile))
      im=dims(1)-1
      jm=dims(2)-1
      km=dims(3)-1
      print*,' ** dimension of input data:',im,jm

      allocate(x(0:im,0:jm),y(0:im,0:jm))
      call H5ReadSubset(x,im,jm,km,'x',trim(gridfile),kslice=0)
      call H5ReadSubset(y,im,jm,km,'y',trim(gridfile),kslice=0)

      allocate(ro(0:im,0:jm))
      call H5ReadSubset(ro,im,jm,km,'ro','outdat/flowfield.h5',kslice=0)
      allocate(u1(0:im,0:jm))
      call H5ReadSubset(u1,im,jm,km,'u1','outdat/flowfield.h5',kslice=0)
      allocate(u2(0:im,0:jm))
      call H5ReadSubset(u2,im,jm,km,'u2','outdat/flowfield.h5',kslice=0)
      allocate( t(0:im,0:jm))
      call H5ReadSubset(t, im,jm,km, 't','outdat/flowfield.h5',kslice=0)

      call parse_command_line(  string=gridin )

      dims=h5_getdimensio('x',trim(gridin))
      in=dims(1)-1
      jn=dims(2)-1
      kn=dims(3)-1
      print*,' ** dimension of interpolate data:',in,jn

      allocate(xo(0:in,0:jn),yo(0:in,0:jn))
      call H5ReadSubset(xo,in,jn,kn,'x',trim(gridin),kslice=0)
      call H5ReadSubset(yo,in,jn,kn,'y',trim(gridin),kslice=0)

      allocate(roo(0:in,0:jn))
      allocate(u1o(0:in,0:jn))
      allocate(u2o(0:in,0:jn))
      allocate( to(0:in,0:jn))

      call interpolate_field_2d(xs=x ,ys=y, fs=ro, fs2=u1, fs3=u2, fs4=t, &
                                xd=xo,yd=yo,fd=roo,fd2=u1o,fd3=u2o,fd4=to )

      call tecbin('tecini_2d.plt',xo,'x',yo,'y',roo,'ro',u1o,'u',u2o,'v',to,'T')

      call H5WriteArray(roo,in,jn,'ro','flowini2d.h5')
      call H5WriteArray(u1o,in,jn,'u1','flowini2d.h5')
      call H5WriteArray(u2o,in,jn,'u2','flowini2d.h5')
      call H5WriteArray( to,in,jn, 't','flowini2d.h5')

    end subroutine flowinterp2d

end module pastr_flowgen