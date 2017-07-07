module precond_psuedoinv_mod
  use iso_c_binding
  use types_as_c
  use constants

  implicit none

contains

  subroutine compsep_assemble_U(nobs, ncomp, lmax, lmax_per_comp, mixing_scalars, bl, sqrt_invCl, alpha, U) bind(c)
    integer(i4b), value :: nobs, ncomp, lmax
    integer(i4b), dimension(ncomp) :: lmax_per_comp
    real(sp), dimension(nobs + ncomp, ncomp, 0:lmax) :: U
    real(dp), dimension(nobs, ncomp) :: mixing_scalars
    real(dp), dimension(nobs, 0:lmax) :: bl
    real(dp), dimension(ncomp, 0:lmax) :: sqrt_invCl
    real(dp), dimension(nobs) :: alpha
    !--
    integer(i4b) :: l, k, nu

    do l = 0, lmax
       do k = 1, ncomp
          do nu = 1, nobs
             if (l .le. lmax_per_comp(k)) then
                U(nu, k, l) = real(mixing_scalars(nu, k) * bl(nu, l) * alpha(nu), kind=sp)
             else
                ! We need to specifically truncate components as there is no function of
                ! (k, l) above. For bands this is handled implicitly by bl(nu, l) being
                ! zero.
                U(nu, k, l) = 0_sp
             end if
          end do
          U(nobs + k, :, l) = 0_sp
          if (l .le. lmax_per_comp(k)) then
             U(nobs + k, k, l) = real(sqrt_invCl(k, l), kind=sp)
          end if
       end do

    end do

  end subroutine compsep_assemble_U




  ! Apply a matrix in the block-diagonal form of U and U+ in Seljebotn et al. (2017)
  ! to a set of spherical harmonic vectors. Each block has size (nobs + ncomp) x (ncomp),
  ! but the input and output vectors are stacked after one another with different sizes.
  !
  ! For simplicity we assume a common lmax for everything here, and the input/output vector
  ! has shape (*, ncomp + nobs) -- only a portion of the the second dimension is used on input
  ! (transpose=0) or output (transpose=1)
  subroutine compsep_apply_U_block_diagonal(nobs, ncomp, lmax, transpose, blocks, x) bind(c)
    integer(i4b), value :: nobs, ncomp, lmax, transpose
    real(sp), dimension(1:(nobs + ncomp), 1:ncomp, 0:lmax) :: blocks
    real(sp), dimension(1:(lmax + 1)**2, ncomp + nobs) :: x
    !--

    real(sp), dimension(ncomp) :: col_buf
    real(sp), dimension(nobs + ncomp) :: row_buf
    integer(i4b) :: idx, l, m, neg

    if (transpose == 0) then
       idx = 1
       do m = 0, lmax
          do l = m, lmax
             do neg = 0, 1
                if (m == 0 .and. neg == 1) cycle

                col_buf(:) = x(idx, 1:ncomp)
                row_buf(:) = matmul(blocks(:, :, l), col_buf)
                x(idx, 1:ncomp + nobs) = row_buf(:)

                idx = idx + 1
             end do
          end do
       end do
    else
       idx = 1
       do m = 0, lmax
          do l = m, lmax
             do neg = 0, 1
                if (m == 0 .and. neg == 1) cycle

                row_buf(:) = x(idx, 1:ncomp + nobs)
                col_buf(:) = matmul(row_buf, blocks(:, :, l))
                x(idx, 1:ncomp) = col_buf(1:ncomp)

                idx = idx + 1
             end do
          end do
       end do

    end if
  end subroutine compsep_apply_U_block_diagonal


end module precond_psuedoinv_mod
