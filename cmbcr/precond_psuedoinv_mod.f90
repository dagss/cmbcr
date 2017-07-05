module precond_psuedoinv_mod
  use iso_c_binding
  use types_as_c
  use constants

  implicit none

contains

  ! Apply a matrix in the block-diagonal form of U and U+ in Seljebotn et al. (2017)
  ! to a set of spherical harmonic vectors. Each block has size (nobs + ncomp) x (ncomp),
  ! but the input and output vectors are stacked after one another with different sizes.
  !
  ! For simplicity we assume a common lmax for everything here, and the input/output/blocks
  ! needs to be padded.
  !
  ! x_comp is modified in-place; x_obs is overwritten.
  subroutine compsep_apply_U_block_diagonal(nobs, ncomp, lmax, blocks, x_comp, x_obs, transpose) bind(c)
    integer(i4b), value :: nobs, ncomp
    integer(i4b), value :: lmax
    real(c_float), dimension(1:(nobs + ncomp), 1:ncomp, 0:lmax) :: blocks
    real(c_float), dimension(1:(lmax + 1)**2, ncomp) :: x_comp
    real(c_float), dimension(1:(lmax + 1)**2, nobs) :: x_obs
    character(c_char), value :: transpose
    !--

    real(c_float), dimension(ncomp) :: col_buf
    real(c_float), dimension(nobs + ncomp) :: row_buf
    integer(i4b) :: idx, l, m, neg, nsh

    nsh = (lmax + 1)**2

    if (transpose == 'N' .or. transpose == 'n') then
       idx = 1
       do m = 0, lmax
          do l = m, lmax
             do neg = 0, 1
                if (m == 0 .and. neg == 1) cycle

                col_buf(:) = x_comp(idx, :)
                row_buf(:) = matmul(blocks(:, :, l), col_buf)
                x_obs(idx, :) = row_buf(1:nobs)
                x_comp(idx, :) = row_buf(nobs + 1:nobs + ncomp)
                
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

                row_buf(1:nobs) = x_obs(idx, :)
                row_buf(nobs + 1:nobs + ncomp) = x_comp(idx, :)
                
                col_buf(:) = matmul(row_buf, blocks(:, :, l))
                
                x_comp(idx, :) = col_buf(:)
                
                idx = idx + 1
             end do
          end do
       end do
       
    end if

       


    
  end subroutine compsep_apply_U_block_diagonal
  

end module precond_psuedoinv_mod
