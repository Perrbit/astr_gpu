module index_slice_stencil
  use astr_precision, only: real_kind
  implicit none
  private
  public :: slice_stencil_start,slice_stencil_weights,slice_diagnostics
contains
  pure integer function slice_stencil_start(node,intervals,periodic) result(first)
    integer,intent(in) :: node,intervals
    logical,intent(in) :: periodic
    first=node-3
    if(.not.periodic) first=min(max(first,0),intervals-6)
  end function

  pure subroutine slice_stencil_weights(weights)
    real(real_kind),intent(out) :: weights(7,0:6)
    real(real_kind) :: numerator,denominator
    integer :: node,k,j,m
    ! Derivatives of the seven Lagrange basis polynomials at each integer node.
    do node=0,6
      do k=0,6
        denominator=1
        do j=0,6
          if(j/=k) denominator=denominator*real(k-j,real_kind)
        enddo
        weights(k+1,node)=0
        do m=0,6
          if(m==k) cycle
          numerator=1
          do j=0,6
            if(j/=k.and.j/=m) numerator=numerator*real(node-j,real_kind)
          enddo
          weights(k+1,node)=weights(k+1,node)+numerator/denominator
        enddo
      enddo
    enddo
  end subroutine

  pure subroutine slice_diagnostics(gradient,density_gradient,values)
    real(real_kind),intent(in) :: gradient(3,3),density_gradient(3)
    real(real_kind),intent(out) :: values(15)
    values(:9)=reshape(gradient,[9])
    values(10)=-0.5_real_kind*sum(gradient*transpose(gradient))
    values(11)=gradient(1,1)+gradient(2,2)+gradient(3,3)
    values(12)=gradient(3,2)-gradient(2,3)
    values(13)=gradient(1,3)-gradient(3,1)
    values(14)=gradient(2,1)-gradient(1,2)
    values(15)=sqrt(sum(density_gradient*density_gradient))
  end subroutine
end module
