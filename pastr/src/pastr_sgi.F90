module structured_grid_interp

  use pastr_utility, only : progress_bar

  implicit none

  private

  integer, parameter :: dp = kind(1.0d0)
  real(dp), parameter :: tol_newton = 1.0d-10
  integer, parameter :: max_newton_iter = 30

  public :: interpolate_field_2d
  public :: interpolate_field_3d

contains

  subroutine interpolate_field_2d(xs, ys, fs, fs2, fs3, fs4, fs5, &
                                  xd, yd, fd, fd2, fd3, fd4, fd5, mask_ok)
    implicit none
    real(dp), intent(in) :: xs(:,:), ys(:,:), fs(:,:)
    real(dp), intent(in),dimension(:,:),optional :: fs2,fs3,fs4,fs5
    real(dp), intent(in)  :: xd(:,:), yd(:,:)
    real(dp), intent(out) :: fd(size(xd,1), size(xd,2))
    real(dp), intent(out),dimension(size(xd,1), size(xd,2)),optional :: fd2,fd3,fd4,fd5
    logical,  intent(out), optional :: mask_ok(size(xd,1), size(xd,2))

    integer :: i, j
    logical :: found

    do j = lbound(xd,2), ubound(xd,2)
      do i = lbound(xd,1), ubound(xd,1)
        call locate_and_interp_2d(xs, ys, fs, fs2, fs3, fs4, fs5,  &
                                  xd(i,j), yd(i,j),  fd(i,j),      &
                                                    fd2(i,j),      &
                                                    fd3(i,j),      &
                                                    fd4(i,j),      &
                                                    fd5(i,j), found)
        if (present(mask_ok)) mask_ok(i,j) = found
        if (.not. found) fd(i,j) = 0.0_dp

      end do

      call progress_bar(iteration=j,maximum=ubound(xd,2),info2show=' interpolating',barlength=50)
    end do
  end subroutine interpolate_field_2d

  subroutine interpolate_field_3d(xs, ys, zs, fs, xd, yd, zd, fd, mask_ok)
    implicit none
    real(dp), intent(in)  :: xs(:,:,:), ys(:,:,:), zs(:,:,:), fs(:,:,:)
    real(dp), intent(in)  :: xd(:,:,:), yd(:,:,:), zd(:,:,:)
    real(dp), intent(out) :: fd(size(xd,1), size(xd,2), size(xd,3))
    logical,  intent(out), optional :: mask_ok(size(xd,1), size(xd,2), size(xd,3))

    integer :: i, j, k
    logical :: found

    do k = lbound(xd,3), ubound(xd,3)
      do j = lbound(xd,2), ubound(xd,2)
        do i = lbound(xd,1), ubound(xd,1)
          call locate_and_interp_3d(xs, ys, zs, fs, xd(i,j,k), yd(i,j,k), zd(i,j,k), &
                                    fd(i,j,k), found)
          if (present(mask_ok)) mask_ok(i,j,k) = found
          if (.not. found) fd(i,j,k) = 0.0_dp
        end do
      end do
    end do
  end subroutine interpolate_field_3d

  subroutine locate_and_interp_2d(x, y, f, f2, f3, f4, f5, xp, yp, fp, fp2, fp3, fp4, fp5, found)
    implicit none
    real(dp), intent(in)  :: x(:,:), y(:,:), f(:,:)
    real(dp), intent(in), optional :: f2(:,:), f3(:,:), f4(:,:), f5(:,:)
    real(dp), intent(in)  :: xp, yp
    real(dp), intent(out) :: fp
    real(dp), intent(out), optional :: fp2, fp3, fp4, fp5
    logical,  intent(out) :: found
  
    integer :: i, j
    integer :: ilo, ihi, jlo, jhi
    integer :: ibest, jbest
    real(dp) :: xi, eta
    logical :: ok
    real(dp) :: dist2, best_dist2, xc, yc
  
    ilo = lbound(x,1)
    ihi = ubound(x,1) - 1
    jlo = lbound(x,2)
    jhi = ubound(x,2) - 1
  
    found = .false.
    fp  = 0.0_dp
    if(present(f2)) fp2 = 0.0_dp
    if(present(f3)) fp3 = 0.0_dp
    if(present(f4)) fp4 = 0.0_dp
    if(present(f5)) fp5 = 0.0_dp
  
    ! First: try to find a true containing cell
    do j = jlo, jhi
      do i = ilo, ihi
        if (.not. point_in_cell_bbox_2d(x, y, i, j, xp, yp)) cycle
  
        call invert_bilinear_cell(x, y, i, j, xp, yp, xi, eta, ok)
        if (.not. ok) cycle
  
        if (xi >= -1.0d-8 .and. xi <= 1.0d0 + 1.0d-8 .and. &
            eta >= -1.0d-8 .and. eta <= 1.0d0 + 1.0d-8) then
          fp = bilinear_scalar(f, i, j, xi, eta)
          if(present(f2)) fp2 = bilinear_scalar(f2, i, j, xi, eta)
          if(present(f3)) fp3 = bilinear_scalar(f3, i, j, xi, eta)
          if(present(f4)) fp4 = bilinear_scalar(f4, i, j, xi, eta)
          if(present(f5)) fp5 = bilinear_scalar(f5, i, j, xi, eta)
          found = .true.
          return
        end if
      end do
    end do
  
    ! If not found: choose the nearest cell center and extrapolate from it
    best_dist2 = huge(1.0_dp)
    ibest = ilo
    jbest = jlo
  
    do j = jlo, jhi
      do i = ilo, ihi
        xc = 0.25_dp * (x(i,j) + x(i+1,j) + x(i,j+1) + x(i+1,j+1))
        yc = 0.25_dp * (y(i,j) + y(i+1,j) + y(i,j+1) + y(i+1,j+1))
        dist2 = (xc - xp)**2 + (yc - yp)**2
        if (dist2 < best_dist2) then
          best_dist2 = dist2
          ibest = i
          jbest = j
        end if
      end do
    end do
  
    call invert_bilinear_cell(x, y, ibest, jbest, xp, yp, xi, eta, ok)
    if (ok) then
      fp = bilinear_scalar(f, ibest, jbest, xi, eta)
      if(present(f2)) fp2 = bilinear_scalar(f2, ibest, jbest, xi, eta)
      if(present(f3)) fp3 = bilinear_scalar(f3, ibest, jbest, xi, eta)
      if(present(f4)) fp4 = bilinear_scalar(f4, ibest, jbest, xi, eta)
      if(present(f5)) fp5 = bilinear_scalar(f5, ibest, jbest, xi, eta)
      found = .true.
    else
      ! final fallback: nearest node value
      call nearest_node_value_2d(x, y, f, xp, yp, fp)
      if(present(f2)) call nearest_node_value_2d(x, y, f2, xp, yp, fp2)
      if(present(f3)) call nearest_node_value_2d(x, y, f3, xp, yp, fp3)
      if(present(f4)) call nearest_node_value_2d(x, y, f4, xp, yp, fp4)
      if(present(f5)) call nearest_node_value_2d(x, y, f5, xp, yp, fp5)
      found = .true.
    end if
  
  end subroutine locate_and_interp_2d

  subroutine locate_and_interp_3d(x, y, z, f, xp, yp, zp, fp, found)
    implicit none
    real(dp), intent(in)  :: x(:,:,:), y(:,:,:), z(:,:,:), f(:,:,:)
    real(dp), intent(in)  :: xp, yp, zp
    real(dp), intent(out) :: fp
    logical,  intent(out) :: found
  
    integer :: i, j, k
    integer :: ilo, ihi, jlo, jhi, klo, khi
    integer :: ibest, jbest, kbest
    real(dp) :: xi, eta, zeta
    logical :: ok
    real(dp) :: dist2, best_dist2, xc, yc, zc
  
    ilo = lbound(x,1)
    ihi = ubound(x,1) - 1
    jlo = lbound(x,2)
    jhi = ubound(x,2) - 1
    klo = lbound(x,3)
    khi = ubound(x,3) - 1
  
    found = .false.
    fp = 0.0_dp
  
    ! First: try to find a true containing cell
    do k = klo, khi
      do j = jlo, jhi
        do i = ilo, ihi
          if (.not. point_in_cell_bbox_3d(x, y, z, i, j, k, xp, yp, zp)) cycle
  
          call invert_trilinear_cell(x, y, z, i, j, k, xp, yp, zp, xi, eta, zeta, ok)
          if (.not. ok) cycle
  
          if (xi   >= -1.0d-8 .and. xi   <= 1.0d0 + 1.0d-8 .and. &
              eta  >= -1.0d-8 .and. eta  <= 1.0d0 + 1.0d-8 .and. &
              zeta >= -1.0d-8 .and. zeta <= 1.0d0 + 1.0d-8) then
            fp = trilinear_scalar(f, i, j, k, xi, eta, zeta)
            found = .true.
            return
          end if
        end do
      end do
    end do
  
    ! If not found: choose nearest cell center and extrapolate from it
    best_dist2 = huge(1.0_dp)
    ibest = ilo
    jbest = jlo
    kbest = klo
  
    do k = klo, khi
      do j = jlo, jhi
        do i = ilo, ihi
          xc = 0.125_dp * (x(i,j,k) + x(i+1,j,k) + x(i,j+1,k) + x(i+1,j+1,k) + &
                           x(i,j,k+1) + x(i+1,j,k+1) + x(i,j+1,k+1) + x(i+1,j+1,k+1))
          yc = 0.125_dp * (y(i,j,k) + y(i+1,j,k) + y(i,j+1,k) + y(i+1,j+1,k) + &
                           y(i,j,k+1) + y(i+1,j,k+1) + y(i,j+1,k+1) + y(i+1,j+1,k+1))
          zc = 0.125_dp * (z(i,j,k) + z(i+1,j,k) + z(i,j+1,k) + z(i+1,j+1,k) + &
                           z(i,j,k+1) + z(i+1,j,k+1) + z(i,j+1,k+1) + z(i+1,j+1,k+1))
  
          dist2 = (xc - xp)**2 + (yc - yp)**2 + (zc - zp)**2
          if (dist2 < best_dist2) then
            best_dist2 = dist2
            ibest = i
            jbest = j
            kbest = k
          end if
        end do
      end do
    end do
  
    call invert_trilinear_cell(x, y, z, ibest, jbest, kbest, xp, yp, zp, xi, eta, zeta, ok)
    if (ok) then
      fp = trilinear_scalar(f, ibest, jbest, kbest, xi, eta, zeta)
      found = .true.
    else
      call nearest_node_value_3d(x, y, z, f, xp, yp, zp, fp)
      found = .true.
    end if
  
  end subroutine locate_and_interp_3d

  subroutine nearest_node_value_2d(x, y, f, xp, yp, fp)
    implicit none
    real(dp), intent(in)  :: x(:,:), y(:,:), f(:,:)
    real(dp), intent(in)  :: xp, yp
    real(dp), intent(out) :: fp
  
    integer :: i, j, ibest, jbest
    real(dp) :: dist2, best_dist2
  
    best_dist2 = huge(1.0_dp)
    ibest = lbound(x,1)
    jbest = lbound(x,2)
  
    do j = lbound(x,2), ubound(x,2)
      do i = lbound(x,1), ubound(x,1)
        dist2 = (x(i,j)-xp)**2 + (y(i,j)-yp)**2
        if (dist2 < best_dist2) then
          best_dist2 = dist2
          ibest = i
          jbest = j
        end if
      end do
    end do
  
    fp = f(ibest,jbest)
  end subroutine nearest_node_value_2d

  subroutine nearest_node_value_3d(x, y, z, f, xp, yp, zp, fp)
    implicit none
    real(dp), intent(in)  :: x(:,:,:), y(:,:,:), z(:,:,:), f(:,:,:)
    real(dp), intent(in)  :: xp, yp, zp
    real(dp), intent(out) :: fp
  
    integer :: i, j, k, ibest, jbest, kbest
    real(dp) :: dist2, best_dist2
  
    best_dist2 = huge(1.0_dp)
    ibest = lbound(x,1)
    jbest = lbound(x,2)
    kbest = lbound(x,3)
  
    do k = lbound(x,3), ubound(x,3)
      do j = lbound(x,2), ubound(x,2)
        do i = lbound(x,1), ubound(x,1)
          dist2 = (x(i,j,k)-xp)**2 + (y(i,j,k)-yp)**2 + (z(i,j,k)-zp)**2
          if (dist2 < best_dist2) then
            best_dist2 = dist2
            ibest = i
            jbest = j
            kbest = k
          end if
        end do
      end do
    end do
  
    fp = f(ibest,jbest,kbest)
  end subroutine nearest_node_value_3d

  logical function point_in_cell_bbox_2d(x, y, i, j, xp, yp)
    implicit none
    real(dp), intent(in) :: x(:,:), y(:,:), xp, yp
    integer, intent(in) :: i, j
    real(dp) :: xmin, xmax, ymin, ymax, eps

    eps = 1.0d-12

    xmin = min( min(x(i,j), x(i+1,j)), min(x(i,j+1), x(i+1,j+1)) )
    xmax = max( max(x(i,j), x(i+1,j)), max(x(i,j+1), x(i+1,j+1)) )
    ymin = min( min(y(i,j), y(i+1,j)), min(y(i,j+1), y(i+1,j+1)) )
    ymax = max( max(y(i,j), y(i+1,j)), max(y(i,j+1), y(i+1,j+1)) )

    point_in_cell_bbox_2d = (xp >= xmin-eps .and. xp <= xmax+eps .and. &
                             yp >= ymin-eps .and. yp <= ymax+eps)
  end function point_in_cell_bbox_2d

  logical function point_in_cell_bbox_3d(x, y, z, i, j, k, xp, yp, zp)
    implicit none
    real(dp), intent(in) :: x(:,:,:), y(:,:,:), z(:,:,:), xp, yp, zp
    integer, intent(in) :: i, j, k
    real(dp) :: xmin, xmax, ymin, ymax, zmin, zmax, eps
    real(dp) :: xx(8), yy(8), zz(8)

    eps = 1.0d-12

    xx = [ x(i,j,k), x(i+1,j,k), x(i,j+1,k), x(i+1,j+1,k), &
           x(i,j,k+1), x(i+1,j,k+1), x(i,j+1,k+1), x(i+1,j+1,k+1) ]
    yy = [ y(i,j,k), y(i+1,j,k), y(i,j+1,k), y(i+1,j+1,k), &
           y(i,j,k+1), y(i+1,j,k+1), y(i,j+1,k+1), y(i+1,j+1,k+1) ]
    zz = [ z(i,j,k), z(i+1,j,k), z(i,j+1,k), z(i+1,j+1,k), &
           z(i,j,k+1), z(i+1,j,k+1), z(i,j+1,k+1), z(i+1,j+1,k+1) ]

    xmin = minval(xx); xmax = maxval(xx)
    ymin = minval(yy); ymax = maxval(yy)
    zmin = minval(zz); zmax = maxval(zz)

    point_in_cell_bbox_3d = (xp >= xmin-eps .and. xp <= xmax+eps .and. &
                             yp >= ymin-eps .and. yp <= ymax+eps .and. &
                             zp >= zmin-eps .and. zp <= zmax+eps)
  end function point_in_cell_bbox_3d

  subroutine invert_bilinear_cell(x, y, i, j, xp, yp, xi, eta, ok)
    implicit none
    real(dp), intent(in)  :: x(:,:), y(:,:), xp, yp
    integer,  intent(in)  :: i, j
    real(dp), intent(out) :: xi, eta
    logical,  intent(out) :: ok

    integer :: iter
    real(dp) :: xx, yy, rx, ry, dx_dxi, dx_deta, dy_dxi, dy_deta
    real(dp) :: detj, dxi, deta

    xi  = 0.5_dp
    eta = 0.5_dp
    ok = .false.

    do iter = 1, max_newton_iter
      call bilinear_map_and_jac(x, y, i, j, xi, eta, xx, yy, dx_dxi, dx_deta, dy_dxi, dy_deta)

      rx = xx - xp
      ry = yy - yp

      if (abs(rx) + abs(ry) < tol_newton) then
        ok = .true.
        return
      end if

      detj = dx_dxi*dy_deta - dx_deta*dy_dxi
      if (abs(detj) < 1.0d-14) return

      dxi  = (-dy_deta*rx + dx_deta*ry) / detj
      deta = ( dy_dxi*rx  - dx_dxi*ry ) / detj

      xi  = xi  + dxi
      eta = eta + deta

      if (abs(dxi) + abs(deta) < tol_newton) then
        ok = .true.
        return
      end if
    end do
  end subroutine invert_bilinear_cell

  subroutine bilinear_map_and_jac(x, y, i, j, xi, eta, xx, yy, dx_dxi, dx_deta, dy_dxi, dy_deta)
    implicit none
    real(dp), intent(in)  :: x(:,:), y(:,:), xi, eta
    integer,  intent(in)  :: i, j
    real(dp), intent(out) :: xx, yy, dx_dxi, dx_deta, dy_dxi, dy_deta

    real(dp) :: n00, n10, n01, n11
    real(dp) :: dn00_dxi, dn10_dxi, dn01_dxi, dn11_dxi
    real(dp) :: dn00_deta, dn10_deta, dn01_deta, dn11_deta

    n00 = (1.0_dp-xi)*(1.0_dp-eta)
    n10 = xi*(1.0_dp-eta)
    n01 = (1.0_dp-xi)*eta
    n11 = xi*eta

    dn00_dxi  = -(1.0_dp-eta)
    dn10_dxi  =  (1.0_dp-eta)
    dn01_dxi  = -eta
    dn11_dxi  =  eta

    dn00_deta = -(1.0_dp-xi)
    dn10_deta = -xi
    dn01_deta =  (1.0_dp-xi)
    dn11_deta =  xi

    xx = n00*x(i,j) + n10*x(i+1,j) + n01*x(i,j+1) + n11*x(i+1,j+1)
    yy = n00*y(i,j) + n10*y(i+1,j) + n01*y(i,j+1) + n11*y(i+1,j+1)

    dx_dxi  = dn00_dxi *x(i,j) + dn10_dxi *x(i+1,j) + dn01_dxi *x(i,j+1) + dn11_dxi *x(i+1,j+1)
    dx_deta = dn00_deta*x(i,j) + dn10_deta*x(i+1,j) + dn01_deta*x(i,j+1) + dn11_deta*x(i+1,j+1)

    dy_dxi  = dn00_dxi *y(i,j) + dn10_dxi *y(i+1,j) + dn01_dxi *y(i,j+1) + dn11_dxi *y(i+1,j+1)
    dy_deta = dn00_deta*y(i,j) + dn10_deta*y(i+1,j) + dn01_deta*y(i,j+1) + dn11_deta*y(i+1,j+1)
  end subroutine bilinear_map_and_jac

  real(dp) function bilinear_scalar(f, i, j, xi, eta)
    implicit none
    real(dp), intent(in) :: f(:,:), xi, eta
    integer,  intent(in) :: i, j

    bilinear_scalar = (1.0_dp-xi)*(1.0_dp-eta)*f(i,j)   + &
                      xi*(1.0_dp-eta)*f(i+1,j)         + &
                      (1.0_dp-xi)*eta*f(i,j+1)         + &
                      xi*eta*f(i+1,j+1)
  end function bilinear_scalar

  subroutine invert_trilinear_cell(x, y, z, i, j, k, xp, yp, zp, xi, eta, zeta, ok)
    implicit none
    real(dp), intent(in)  :: x(:,:,:), y(:,:,:), z(:,:,:), xp, yp, zp
    integer,  intent(in)  :: i, j, k
    real(dp), intent(out) :: xi, eta, zeta
    logical,  intent(out) :: ok

    integer :: iter
    real(dp) :: xx, yy, zz, rx, ry, rz
    real(dp) :: jmat(3,3), rhs(3), du(3), detj

    xi = 0.5_dp
    eta = 0.5_dp
    zeta = 0.5_dp
    ok = .false.

    do iter = 1, max_newton_iter
      call trilinear_map_and_jac(x, y, z, i, j, k, xi, eta, zeta, xx, yy, zz, jmat)

      rx = xx - xp
      ry = yy - yp
      rz = zz - zp

      if (abs(rx) + abs(ry) + abs(rz) < tol_newton) then
        ok = .true.
        return
      end if

      rhs = -[rx, ry, rz]
      call solve_3x3(jmat, rhs, du, detj)
      if (abs(detj) < 1.0d-14) return

      xi   = xi   + du(1)
      eta  = eta  + du(2)
      zeta = zeta + du(3)

      if (abs(du(1)) + abs(du(2)) + abs(du(3)) < tol_newton) then
        ok = .true.
        return
      end if
    end do
  end subroutine invert_trilinear_cell

  subroutine trilinear_map_and_jac(x, y, z, i, j, k, xi, eta, zeta, xx, yy, zz, jmat)
    implicit none
    real(dp), intent(in)  :: x(:,:,:), y(:,:,:), z(:,:,:), xi, eta, zeta
    integer,  intent(in)  :: i, j, k
    real(dp), intent(out) :: xx, yy, zz, jmat(3,3)

    real(dp) :: n(8), dxi(8), deta(8), dzeta(8)
    real(dp) :: xv(8), yv(8), zv(8)

    xv = [ x(i,j,k), x(i+1,j,k), x(i,j+1,k), x(i+1,j+1,k), &
           x(i,j,k+1), x(i+1,j,k+1), x(i,j+1,k+1), x(i+1,j+1,k+1) ]

    yv = [ y(i,j,k), y(i+1,j,k), y(i,j+1,k), y(i+1,j+1,k), &
           y(i,j,k+1), y(i+1,j,k+1), y(i,j+1,k+1), y(i+1,j+1,k+1) ]

    zv = [ z(i,j,k), z(i+1,j,k), z(i,j+1,k), z(i+1,j+1,k), &
           z(i,j,k+1), z(i+1,j,k+1), z(i,j+1,k+1), z(i+1,j+1,k+1) ]

    n(1) = (1-xi)*(1-eta)*(1-zeta)
    n(2) = xi*(1-eta)*(1-zeta)
    n(3) = (1-xi)*eta*(1-zeta)
    n(4) = xi*eta*(1-zeta)
    n(5) = (1-xi)*(1-eta)*zeta
    n(6) = xi*(1-eta)*zeta
    n(7) = (1-xi)*eta*zeta
    n(8) = xi*eta*zeta

    dxi(1) = -(1-eta)*(1-zeta)
    dxi(2) =  (1-eta)*(1-zeta)
    dxi(3) = -eta*(1-zeta)
    dxi(4) =  eta*(1-zeta)
    dxi(5) = -(1-eta)*zeta
    dxi(6) =  (1-eta)*zeta
    dxi(7) = -eta*zeta
    dxi(8) =  eta*zeta

    deta(1) = -(1-xi)*(1-zeta)
    deta(2) = -xi*(1-zeta)
    deta(3) =  (1-xi)*(1-zeta)
    deta(4) =  xi*(1-zeta)
    deta(5) = -(1-xi)*zeta
    deta(6) = -xi*zeta
    deta(7) =  (1-xi)*zeta
    deta(8) =  xi*zeta

    dzeta(1) = -(1-xi)*(1-eta)
    dzeta(2) = -xi*(1-eta)
    dzeta(3) = -(1-xi)*eta
    dzeta(4) = -xi*eta
    dzeta(5) =  (1-xi)*(1-eta)
    dzeta(6) =  xi*(1-eta)
    dzeta(7) =  (1-xi)*eta
    dzeta(8) =  xi*eta

    xx = sum(n*xv)
    yy = sum(n*yv)
    zz = sum(n*zv)

    jmat(1,1) = sum(dxi*xv)
    jmat(1,2) = sum(deta*xv)
    jmat(1,3) = sum(dzeta*xv)

    jmat(2,1) = sum(dxi*yv)
    jmat(2,2) = sum(deta*yv)
    jmat(2,3) = sum(dzeta*yv)

    jmat(3,1) = sum(dxi*zv)
    jmat(3,2) = sum(deta*zv)
    jmat(3,3) = sum(dzeta*zv)
  end subroutine trilinear_map_and_jac

  real(dp) function trilinear_scalar(f, i, j, k, xi, eta, zeta)
    implicit none
    real(dp), intent(in) :: f(:,:,:), xi, eta, zeta
    integer,  intent(in) :: i, j, k

    trilinear_scalar = &
      (1-xi)*(1-eta)*(1-zeta)*f(i,j,k)         + &
      xi*(1-eta)*(1-zeta)*f(i+1,j,k)           + &
      (1-xi)*eta*(1-zeta)*f(i,j+1,k)           + &
      xi*eta*(1-zeta)*f(i+1,j+1,k)             + &
      (1-xi)*(1-eta)*zeta*f(i,j,k+1)           + &
      xi*(1-eta)*zeta*f(i+1,j,k+1)             + &
      (1-xi)*eta*zeta*f(i,j+1,k+1)             + &
      xi*eta*zeta*f(i+1,j+1,k+1)
  end function trilinear_scalar

  subroutine solve_3x3(a, b, x, deta)
    implicit none
    real(dp), intent(in)  :: a(3,3), b(3)
    real(dp), intent(out) :: x(3), deta
    real(dp) :: aa(3,3), bb(3), factor
    integer :: i, j, k, p
    real(dp) :: tmp, rowtmp(3)

    aa = a
    bb = b

    deta = determinant_3x3(a)
    if (abs(deta) < 1.0d-14) then
      x = 0.0_dp
      return
    end if

    do k = 1, 2
      p = k
      do i = k+1, 3
        if (abs(aa(i,k)) > abs(aa(p,k))) p = i
      end do

      if (p /= k) then
        rowtmp = aa(k,:)
        aa(k,:) = aa(p,:)
        aa(p,:) = rowtmp
        tmp = bb(k)
        bb(k) = bb(p)
        bb(p) = tmp
      end if

      do i = k+1, 3
        factor = aa(i,k) / aa(k,k)
        aa(i,k:3) = aa(i,k:3) - factor * aa(k,k:3)
        bb(i) = bb(i) - factor * bb(k)
      end do
    end do

    x(3) = bb(3) / aa(3,3)
    x(2) = (bb(2) - aa(2,3)*x(3)) / aa(2,2)
    x(1) = (bb(1) - aa(1,2)*x(2) - aa(1,3)*x(3)) / aa(1,1)
  end subroutine solve_3x3

  real(dp) function determinant_3x3(a)
    implicit none
    real(dp), intent(in) :: a(3,3)

    determinant_3x3 = a(1,1)*(a(2,2)*a(3,3) - a(2,3)*a(3,2)) - &
                      a(1,2)*(a(2,1)*a(3,3) - a(2,3)*a(3,1)) + &
                      a(1,3)*(a(2,1)*a(3,2) - a(2,2)*a(3,1))
  end function determinant_3x3

end module structured_grid_interp