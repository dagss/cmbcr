module harmonic_preconditioner
  use iso_c_binding
  use types_as_c
  use constants

  implicit none

  interface
     SUBROUTINE SPBTRF( UPLO, N, KD, AB, LDAB, INFO )
       CHARACTER          UPLO
       INTEGER            INFO, KD, LDAB, N
       REAL               AB( LDAB, * )
     END SUBROUTINE SPBTRF

     SUBROUTINE SPBTRS( UPLO, N, KD, NRHS, AB, LDAB, B, LDB, INFO )
       CHARACTER          UPLO
       INTEGER            INFO, KD, NRHS, LDAB, LDB
       REAL               AB( LDAB, * ), B(LDB, *)
     END SUBROUTINE SPBTRS

     SUBROUTINE DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
       DOUBLE PRECISION ALPHA, BETA
       INTEGER K, LDA, LDB, LDC, M, N
       CHARACTER TRANSA, TRANSB
       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
     END SUBROUTINE DGEMM

     subroutine sharp_normalized_associated_legendre_table( &
           m, spin, lmax, ntheta, theta, theta_stride, l_stride, spin_stride, out) bind(c)
       use iso_c_binding
       integer(c_intptr_t), value :: m, lmax, ntheta, theta_stride, l_stride, spin_stride
       integer(c_int), value :: spin
       real(c_double), dimension(1:ntheta) :: theta
       !-- in this wrapper hardcode in some strides...
       real(c_double), dimension(0:(lmax - m), 1:ntheta) :: out
     end subroutine sharp_normalized_associated_legendre_table

     !function make_complex_plan(length) result(c_result) bind(c)
     !  use iso_c_binding
     !  type(c_ptr), intent(out) :: c_result
     !  integer(c_size_t), value :: length
     !end function make_complex_plan

     !subroutine kill_complex_plan(p) bind(c)
     !  use iso_c_binding
     !  type(c_ptr), value :: p
     !end subroutine kill_complex_plan

     !subroutine complex_plan_forward(plan, data) bind(c)
     !  use iso_c_binding
     !  type(c_ptr), value :: plan
     !  complex(dpc), dimension(*) :: data
     !end subroutine complex_plan_forward

     !subroutine complex_plan_backward(plan, data) bind(c)
     !  use iso_c_binding
     !  type(c_ptr), value :: plan
     !  complex(dpc), dimension(*) :: data
     !end subroutine complex_plan_backward

  end interface

contains


  subroutine compute_complex_Yh_D_Y_block(m, mp, lmax, ntheta, thetas, phase_map, out_real, out_imag) bind(c)
    integer(i4b) :: m, mp, lmax, ntheta
    real(dp), dimension(1:ntheta) :: thetas
    complex(dpc), dimension(0:2 * lmax, 1:ntheta) :: phase_map
    real(dp), dimension(0:(lmax - m), 0:(lmax - mp)) :: out_real, out_imag
    !--
    real(dp), allocatable, dimension(:, :) :: lambda, lambda_p, buf
    integer(i4b) :: l, itheta

    ! NOTE: ONLY TESTED FOR m == mp AND m == -mp

    allocate(lambda(0:(lmax - abs(m)), ntheta))
    allocate(lambda_p(0:(lmax - abs(mp)), ntheta))
    allocate(buf(0:(lmax - abs(mp)), ntheta))

    call sharp_normalized_associated_legendre_table( &
         int(abs(m), kind=c_intptr_t), 0, int(lmax, kind=c_intptr_t), int(ntheta, kind=c_intptr_t), &
         thetas, int(lmax - abs(m) + 1, kind=c_intptr_t), int(1, kind=c_intptr_t), int(1, kind=c_intptr_t), lambda)
    if (m == mp) then
       lambda_p(:, :) = lambda
    else
       call sharp_normalized_associated_legendre_table( &
            int(abs(mp), kind=c_intptr_t), 0, int(lmax, kind=c_intptr_t), int(ntheta, kind=c_intptr_t), &
            thetas, int(lmax - abs(mp) + 1, kind=c_intptr_t), int(1, kind=c_intptr_t), int(1, kind=c_intptr_t), lambda_p)
    end if

    ! real part
    do l = abs(mp), lmax
       do itheta = 1, ntheta
          buf(l - abs(mp), itheta) = real(phase_map(abs(m - mp), itheta)) * lambda_p(l - abs(mp) , itheta)
       end do
    end do

    call dgemm('N', 'T', lmax - abs(m) + 1, lmax - abs(mp) + 1, ntheta, 1.0_dp, &
         lambda, lmax - abs(m) + 1, &
         buf, lmax - abs(mp) + 1, &
         0.0_dp, out_real, lmax - abs(m) + 1)

    ! imag part
    do l = abs(m), lmax
       do itheta = 1, ntheta
          buf(l - abs(m), itheta) = aimag(phase_map(abs(m - mp), itheta)) * lambda_p(l - abs(m), itheta)
       end do
    end do
    call dgemm('N', 'T', lmax - abs(m) + 1, lmax - abs(mp) + 1, ntheta, 1.0_dp, &
         lambda, lmax - abs(m) + 1, &
         buf, lmax - abs(mp) + 1, &
         0.0_dp,&
         out_imag, lmax - abs(m) + 1)

  end subroutine compute_complex_Yh_D_Y_block

  subroutine matrix_m_block_complex_to_real(lmax, m, mp, ar, ai, br, bi, cr, ci, dr, di, out) bind(c)
    !
    !         [ a b ]    [ (+m,+mp)  (+m,-mp) ]
    !         [ c d ] == [ (-m,+mp)  (-m,-mp) ]

    integer(i4b) :: m, mp, lmax
    real(dp), dimension(m:lmax, m:lmax) :: ar, ai
    real(dp), dimension(m:lmax, mp:lmax) :: br, bi
    real(dp), dimension(mp:lmax, m:lmax) :: cr, ci
    real(dp), dimension(mp:lmax, mp:lmax) :: dr, di
    real(dp), dimension(0:merge(lmax - m, 2 * (lmax - m) + 1, m == 0), 0:merge(lmax - mp, 2 * (lmax - mp) + 1, mp == 0)) :: out
    !--
    integer(i4b) :: l, lp

    ! NOTE: ONLY TESTED FOR m == mp AND m == -mp

    if (m == 0 .and. mp == 0) then
       out = ar(:, :)
    else if (m > 0 .and. mp == 0) then
       do l = m, lmax
          do lp = mp, lmax + 1
             out(2 * (l - m), lp - mp) = (+ar(l, lp) +cr(l, lp)) * sqrt(0.5_dp)
             out(2 * (l - m), lp - mp) = (+ai(l, lp) -ci(l, lp)) * sqrt(0.5_dp)
          end do
       end do
    else if (m == 0 .and. mp > 0) then
       do l = m, lmax
          do lp = mp, lmax + 1
             out(l - m, 2 * (lp - mp)) = (+ar(l, lp) +br(l, lp)) * sqrt(0.5_dp)
             out(l - m, 2 * (lp - mp) + 1) = (-ai(l, lp) +bi(l, lp)) * sqrt(0.5_dp)
          end do
       end do
    else
       do l = m, lmax
          do lp = mp, lmax
             out(2 * (l - m), 2 * (lp - mp))         = (+ar(l,lp) +br(l,lp) +cr(l,lp) +dr(l,lp)) * 0.5_dp
             out(2 * (l - m), 2 * (lp - mp) + 1)     = (-ai(l,lp) +bi(l,lp) -ci(l,lp) +di(l,lp)) * 0.5_dp
             out(2 * (l - m) + 1, 2 * (lp - mp))     = (+ai(l,lp) +bi(l,lp) -ci(l,lp) -di(l,lp)) * 0.5_dp
             out(2 * (l - m) + 1, 2 * (lp - mp) + 1) = (+ar(l,lp) -br(l,lp) -cr(l,lp) +dr(l,lp)) * 0.5_dp
          end do
       end do

    end if
  end subroutine matrix_m_block_complex_to_real

  subroutine compute_real_Yh_D_Y_block_on_diagonal(m, lmax, ntheta, thetas, phase_map, out) bind(c)
    integer(i4b), value :: m, lmax, ntheta
    real(dp), dimension(1:ntheta) :: thetas
    complex(dpc), dimension(0:2 * lmax, 1:ntheta) :: phase_map
    real(dp), dimension(0:merge(lmax - m, 2 * (lmax - m) + 1, m == 0), 0:merge(lmax - m, 2 * (lmax - m) + 1, m == 0)) :: out
    !--
    real(dp), allocatable, dimension(:, :) :: ar, ai, br, bi, cr, ci
    integer(i4b) :: nl

    nl = lmax - m

    allocate(ar(0:nl, 0:nl), ai(0:nl, 0:nl), br(0:nl, 0:nl), bi(0:nl, 0:nl), cr(0:nl, 0:nl), ci(0:nl, 0:nl))
    call compute_complex_Yh_D_Y_block(m, m, lmax, ntheta, thetas, phase_map, ar, ai)
    call compute_complex_Yh_D_Y_block(m, -m, lmax, ntheta, thetas, phase_map, br, bi)

    ! We assume that mp=-m; and compute_complex_Yh_D_Y_block only makes use of abs(m), abs(mp), and abs(m - mp).
    cr = transpose(br)
    ci = transpose(bi)

    call matrix_m_block_complex_to_real(lmax, m, m, ar, ai, br, bi, cr, ci, ar, ai, out)

  end subroutine compute_real_Yh_D_Y_block_on_diagonal

  subroutine construct_banded_preconditioner(lmax, ncomp, ntheta, thetas, phase_map, mixing_scalars, bl, out) bind(c)
    integer(i4b), value :: lmax, ncomp, ntheta
    real(dp), dimension(1:ntheta) :: thetas
    real(dp), dimension(0:lmax) :: bl

    complex(dpc), dimension(0:2 * lmax, 1:ntheta) :: phase_map
    real(dp), dimension(ncomp) :: mixing_scalars

    !-- out: we *add* to out; so it should be initialized to zero (or something that should
    !-- be added to) on input;
    real(sp), dimension(1:5 * ncomp, 0:ncomp * (lmax + 1)**2 - 1), intent(out) :: out


    !--
    integer(i4b) :: m, neg, odd, delta, block_col, j, l, fac, k, kp, iband
    real(dp), allocatable, dimension(:, :) :: mblock
    real(dp) :: val

    ! precompute offsets since we're too lazy to figure out the formulas, and we want to parallelize loop
    integer(i4b), dimension(0:lmax) :: offsets
    block_col = 0
    do m = 0, lmax
       offsets(m) = block_col
       block_col = block_col + merge(1, 2, m == 0) * (lmax + 1 - m)
    end do

    !$OMP parallel default(none) &
    !$OMP     shared(out,lmax,offsets,bl,thetas,phase_map,ntheta,ncomp,mixing_scalars) &
    !$OMP     private(m,neg,odd,val,block_col,mblock,j,l,fac,iband)
    !$OMP do schedule(dynamic,1)
    do m = 0, lmax
       block_col = offsets(m)

       allocate(mblock(0:merge(lmax - m, 2 * (lmax - m) + 1, m == 0), &
                       0:merge(lmax - m, 2 * (lmax - m) + 1, m == 0)))
       call compute_real_Yh_D_Y_block_on_diagonal(m, lmax, ntheta, thetas, phase_map, mblock)

       fac = merge(1, 2, m == 0)
       do neg = 0, 1
          if (m == 0 .and. neg == 1) cycle
          do odd = 0, 1
             do l = m, lmax, 2
                if (l + odd > lmax) cycle
                j = (l + odd) - m
                do delta = 0, 4
                   if (j + 2 * delta > lmax - m) cycle
                   ! We are now located at the matrix ncomp-by-ncomp block we want to copy
                   ! to the banded representation, at (l + 2 * delta, l). The index changes required
                   ! to move from block to banded is a bit tricky; assume a lower-triangular block
                   ! matrix of 3x3 blocks it looks like this, with numbers indicating band, . indicates
                   ! stuff outside the banded representation, and 0 are zeroes that end up in the banded
                   ! representation that is not present in the input blocks:
                   !
                   !
                   ! [1..]
                   ! [21.]
                   ! [321]
                   ! -----
                   ! [432][1..]
                   ! [543][21.]
                   ! [654][321]
                   ! ----------
                   ! [.00][432]
                   ! [..0][543]
                   ! [...][654]
                   !
                   ! Since we assume that `out` has been zero-initialized we don't need to explicitly
                   ! insert the zeroes in index locations that are not hit. So we iterate over the blocks,
                   ! and figure out which band the elements fit in (iband)

                   do k = 1, ncomp
                      do kp = 1, ncomp

                         iband = delta * ncomp + (k - 1) - (kp - 1) + 1
                         if (iband < 1) cycle ! we are above the diagonal, the '.' values above diagonal in diagram above

                         val = mblock(fac * (j + 2 * delta) + neg, fac * j + neg) * mixing_scalars(k) * mixing_scalars(kp)

                         val = val * bl(l + odd) * bl(l + odd + 2 * delta)
                         out(iband, block_col * ncomp + (kp - 1)) = &
                              out(iband, block_col * ncomp + (kp - 1)) + real(val, kind=sp)
                      end do
                   end do
                end do
                block_col = block_col + 1
             end do
          end do
       end do

       deallocate(mblock)
    end do
    !$OMP end do
    !$OMP end parallel

    contains
      function k_kp_idx(k, kp)
        integer(i4b) :: k, kp, k_kp_idx
        !--
        integer(i4b) :: low_k, hi_k
        low_k = min(k, kp)
        hi_k = max(k, kp)
        k_kp_idx = ((hi_k - 1)*hi_k) / 2 + low_k
      end function k_kp_idx
  end subroutine construct_banded_preconditioner


  subroutine factor_banded_preconditioner(lmax, ncomp, data, global_info) bind(c)
    integer(i4b), value :: lmax, ncomp
    real(sp), dimension(5 * ncomp, ncomp * (lmax + 1)**2), intent(inout) :: data
    integer(i4b), intent(out) :: global_info
    !--
    integer(i4b) :: idx, neg, odd, len, info, m, i

    ! precompute offsets since we're too lazy to figure out the formulas, and we want to parallelize loop
    integer(i4b), dimension(0:lmax) :: offsets
    idx = 1
    do m = 0, lmax
       offsets(m) = idx
       idx = idx + merge(1, 2, m == 0) * (lmax + 1 - m) * ncomp
    end do

    global_info = 0

    !$OMP parallel default(none) &
    !$OMP     shared(data,global_info,lmax,offsets,ncomp) &
    !$OMP     private(m,neg,idx,info,len)
    !$OMP do schedule(dynamic,1)
    do m = 0, lmax
       idx = offsets(m)
       do neg = 0, 1
          if (m /= 0 .or. neg == 0) then ! avoid doing the negative case for m=0
             do odd = 0, 1
                len = (lmax + 1 - m) / 2
                if (odd == 0 .and. mod(lmax + 1 - m, 2) /= 0) then
                   ! odd number of coefficients; even case is one longer
                   len = len + 1
                end if


                if (len == 0) cycle

                len = ncomp * len
                call SPBTRF('L', len, 5 * ncomp - 1, data(:, idx:idx + len - 1), 5 * ncomp, info)
                if (info /= 0) then
                   print *, 'm', m, 'neg', neg, 'odd', odd, 'len', len, 'info != 0:', info
                   do i = 1, 5 * ncomp
                      print *, data(i, idx:idx + len - 1)
                   end do
                   print *, '^^ bands of singular matrix ===='
                   global_info = info
                end if
                idx = idx + len
             end do
          end if
       end do
    end do
    !$OMP end do
    !$OMP end parallel

  end subroutine factor_banded_preconditioner

  subroutine solve_banded_preconditioner(lmax, ncomp, data, x) bind(c)
    integer(i4b), value :: lmax, ncomp
    real(sp), dimension(5 * ncomp, 0:ncomp * (lmax + 1)**2-1), intent(in) :: data
    real(sp), dimension((lmax + 1)**2, 0:ncomp - 1), intent(inout) :: x
    !--
    integer(i4b) :: x_idx, data_idx, neg, info, m, even_len, odd_len, fac, global_info, k
    real(sp), dimension(:), allocatable :: buf

    ! precompute offsets since we're too lazy to figure out the formulas, and we want to parallelize loop
    integer(i4b), dimension(0:lmax) :: offsets
    x_idx = 0
    do m = 0, lmax
       fac = merge(1, 2, m == 0)
       offsets(m) = x_idx
       x_idx = x_idx + fac * (lmax + 1 - m)
    end do

    global_info = 0
    !$OMP parallel default(none) &
    !$OMP     shared(x,data,global_info,lmax,offsets,ncomp) &
    !$OMP     private(m,fac,neg,odd_len,even_len,buf,data_idx,x_idx,info)
    allocate(buf(0:ncomp * (lmax / 2 + 1) - 1))

    !$OMP do schedule(dynamic,1)
    do m = 0, lmax
       x_idx = offsets(m) + 1
       data_idx = offsets(m) * ncomp
       fac = merge(1, 2, m == 0)
       do neg = 0, 1
          if (m == 0 .and. neg == 1) cycle

          odd_len = (lmax + 1 - m) / 2
          if (mod(lmax + 1 - m, 2) /= 0) then
             even_len = odd_len + 1
          else
             even_len = odd_len
          end if

          ! even part
          do k = 0, ncomp - 1
             buf(k:k + ncomp * even_len - 1:ncomp) = x(x_idx + neg:x_idx + neg + 2 * fac * (even_len - 1):2 * fac, k)
          end do

          call SPBTRS('L', ncomp * even_len, ncomp * 5 - 1, 1, &
              data(:, data_idx:data_idx + ncomp * even_len - 1), 5 * ncomp, buf, even_len * ncomp, info)
          if (info /= 0) then
             print *, 'm', m, 'even case',  ' info != 0:', info
             stop
          end if
          data_idx = data_idx + even_len * ncomp

          do k = 0, ncomp - 1
             x(x_idx + neg:x_idx + neg + 2 * fac * (even_len - 1):2 * fac, k) = buf(k:k + ncomp * even_len - 1:ncomp)
          end do

          ! odd part

          if (odd_len /= 0) then
             do k = 0, ncomp - 1
                buf(k:k + ncomp * odd_len - 1:ncomp) = x(x_idx + neg + fac:x_idx + neg + fac + 2 * fac * (odd_len - 1):2 * fac, k)
             end do

             call SPBTRS('L', odd_len * ncomp, 5 * ncomp - 1, 1, &
                  data(:, data_idx:data_idx + odd_len * ncomp - 1), 5 * ncomp, buf, odd_len * ncomp, info)
             if (info /= 0) then
                print *, 'm', m, 'odd case',  ' info != 0:', info
                global_info = info
             end if
             data_idx = data_idx + odd_len * ncomp

             do k = 0, ncomp - 1
                x(x_idx + neg + fac:x_idx + neg + fac + 2 * fac * (odd_len - 1):2 * fac, k) = buf(k:k + ncomp * odd_len - 1:ncomp)
             end do
          end if
       end do
    end do
    !$OMP end do

    deallocate(buf)
    !$OMP end parallel

    if (global_info /= 0) stop

  end subroutine solve_banded_preconditioner


end module harmonic_preconditioner
