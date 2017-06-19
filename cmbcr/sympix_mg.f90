module sympix_mg
  use iso_c_binding
  use iso_fortran_env
  use types_as_c
  use constants
  implicit none

  interface
     subroutine sharp_legendre_transform_recfac(recfac, lmax) bind(c)
       use iso_c_binding, only: c_ptrdiff_t, c_double
       integer(c_ptrdiff_t), value :: lmax
       real(c_double), dimension(0:lmax) :: recfac
     end subroutine sharp_legendre_transform_recfac

     subroutine sharp_legendre_transform(bl, recfac, lmax, x, out, n) bind(c)
       use iso_c_binding, only: c_ptrdiff_t, c_double
       integer(c_ptrdiff_t), value :: lmax, n
       real(c_double), dimension(0:lmax) :: bl, recfac
       real(c_double), dimension(1:n) :: x, out
     end subroutine
  end interface

contains

  subroutine rescale_bl(bl, rescaled_bl, lmax) bind(c, name='sympix_mg_rescale_bl')
    integer(i4b), value :: lmax
    real(dp), dimension(0:lmax) :: bl, rescaled_bl
    !--
    integer(i4b) :: l
    do l = 0, ubound(bl, 1)
       rescaled_bl(l) = bl(l) * real(real(2 * l + 1, dp) / (4.0_dp * pi), sp)
    end do
  end subroutine rescale_bl

  function ang2vec(theta, phi) result(v)
    real(dp), value :: theta, phi
    real(dp), dimension(3) :: v
    v(1) = sin(theta) * cos(phi)
    v(2) = sin(theta) * sin(phi)
    v(3) = cos(theta)
  end function ang2vec

  subroutine cumsum_i4b(arr, out)
    integer(i4b), dimension(:), intent(in) :: arr
    integer(i4b), dimension(:), intent(out) :: out
    !--
    integer(i4b) :: i

    if (ubound(arr, 1) == 0) then
       return
    endif
    out(1) = arr(1)
    do i = 2, ubound(arr, 1)
       out(i) = out(i - 1) + arr(i)
    end do
  end subroutine cumsum_i4b


  ! Computes a single block of Y D Yt.
  !
  ! The parameters first describe two domains on the sphere, D1 (with
  ! n_D1 x n_D1 pixels, each row on thetas_D1(:), each column
  ! equispaced in increments of by dphi1) and D2 (with n_D2 x n_D2
  ! pixels, each row on thetas_D2(:), each column seperated by
  ! phi0_D2). The location of D2 w.r.t. D1 is given by phi0_D2,
  ! considered phi of the first pixel in D2. phi of the first pixel
  ! of D1 is hardcoded to 0 without loss of generality of the routine.
  !
  ! The D matrix is given as rescaled_bl, which is the result of calling
  ! rescale_bl on the bls. Recursion factors recfac should be provided, call
  ! legendre_transform_recfac to get them.
  !
  ! The output is indexed as out(i1, j1, i2, j2).
  subroutine compute_YDYt_block(n_D1, n_D2, dphi_D1, dphi_D2, &
       thetas_D1, thetas_D2, phi0_D2, lmax, rescaled_bl, recfac, out) &
       bind(c, name='sympix_mg_compute_YDYt_block')
    integer(i4b), value :: n_D1, n_D2, lmax
    real(dp), value :: phi0_D2, dphi_D1, dphi_D2
    real(dp), dimension(n_D1) :: thetas_D1
    real(dp), dimension(n_D2) :: thetas_D2
    real(dp), dimension(n_D1, n_D1, n_D2, n_D2) :: out
    real(dp), dimension(lmax + 1) :: rescaled_bl, recfac
    !--
    integer(i4b) :: i1, j1, i2, j2
    real(dp) :: phi1, phi2
    real(dp), dimension(3) :: v1, v2
    real(dp), allocatable, dimension(:, :, :, :) :: xs

    allocate(xs(n_D1, n_D1, n_D2, n_D2))

    do j2 = 1, n_D2
       do i2 = 1, n_D2
          do j1 = 1, n_D1
             do i1 = 1, n_D1

                phi1 = (j1 - 1) * dphi_D1
                phi2 = phi0_D2 + (j2 - 1) * dphi_D2

                v1 = ang2vec(thetas_D1(i1), phi1)
                v2 = ang2vec(thetas_D2(i2), phi2)

                xs(i1, j1, i2, j2) = real(sum(v1 * v2), sp)
             end do
          end do
       end do
    end do

    call sharp_legendre_transform(rescaled_bl, recfac, int(lmax, c_ptrdiff_t), xs, out, &
        int(n_D1**2 * n_D2**2, c_ptrdiff_t))
  end subroutine compute_YDYt_block


  ! Computes a set of blocks sampling a beam-like operator (Y D Yt).
  !
  ! Provided as input are 2 arrays of length `count`:
  ! `indices1` and `indices2`. We compute the tile-to-tile coupling block
  ! between pixel i and pixel j in the sympix pixelization provided.
  !
  ! The resulting `out_blocks` then have one block per label.
  subroutine compute_many_YDYt_blocks(&
       nblocks, &
       tilesize1, bandcount1, thetas1, tilecounts1, tileindices1, &
       tilesize2, bandcount2, thetas2, tilecounts2, tileindices2, &
       lmax, bl, ierr, out_blocks) &
       bind(c, name='sympix_mg_compute_many_YDYt_blocks')

    integer(i4b), value :: nblocks, bandcount1, bandcount2, lmax, tilesize1, tilesize2
    integer(i4b), dimension(nblocks) :: tileindices1, tileindices2
    real(dp), dimension(0:bandcount1 * tilesize1 - 1) :: thetas1
    real(dp), dimension(0:bandcount2 * tilesize2 - 1) :: thetas2
    integer(i4b), dimension(0:bandcount1 - 1) :: tilecounts1
    integer(i4b), dimension(0:bandcount2 - 1) :: tilecounts2
    real(dp), dimension(lmax + 1) :: bl
    real(dp), dimension(tilesize1, tilesize1, tilesize2, tilesize2, nblocks) :: out_blocks
    integer(i4b) :: ierr  ! out-argument
    !--
    real(dp), dimension(:), allocatable :: rescaled_bl, recfac
    integer(i4b) :: i, private_ierr
    real(dp) :: tile_dphi1, tile_dphi2, tile_phi1, tile_phi2
    real(dp), dimension(tilesize1) :: tile_thetas1
    real(dp), dimension(tilesize2) :: tile_thetas2
    integer(i4b), dimension(0:bandcount1) :: offsets1
    integer(i4b), dimension(0:bandcount2) :: offsets2

    allocate(recfac(0:lmax))
    allocate(rescaled_bl(0:lmax))
    call rescale_bl(bl, rescaled_bl, lmax)
    call sharp_legendre_transform_recfac(recfac, int(lmax, c_ptrdiff_t))

    offsets1(0) = 0
    offsets2(0) = 0
    call cumsum_i4b(2 * tilecounts1, offsets1(1:))
    call cumsum_i4b(2 * tilecounts2, offsets2(1:))

    ierr = 0
    !$OMP parallel default(none) private(private_ierr,i,tile_thetas1,tile_thetas2,tile_phi1,tile_phi2,&
    !$OMP                                tile_dphi1,tile_dphi2) &
    !$OMP                        shared(ierr,nblocks,tilesize1,tilesize2,bandcount1,bandcount2,&
    !$OMP                               offsets1,offsets2,thetas1,thetas2,&
    !$OMP                               tileindices1,tileindices2,out_blocks,rescaled_bl,recfac,lmax)
    !$OMP do schedule(static,8)
    do i = 1, nblocks
       if (ierr == 0) then
          call lookup_tile(tilesize1, bandcount1, offsets1, thetas1, tileindices1(i), &
               tile_phi1, tile_dphi1, tile_thetas1, private_ierr)
          if (private_ierr /= 0) then
             ierr = private_ierr
             cycle
          end if
          call lookup_tile(tilesize2, bandcount2, offsets2, thetas2, tileindices2(i), &
               tile_phi2, tile_dphi2, tile_thetas2, private_ierr)
          if (private_ierr /= 0) then
             ierr = private_ierr
             cycle
          end if
          call compute_YDYt_block(tilesize1, tilesize2, tile_dphi1, tile_dphi2, &
               tile_thetas1, tile_thetas2, tile_phi2 - tile_phi1, &
               lmax, rescaled_bl, recfac, out_blocks(:, :, :, :, i))
       end if
    end do
    !$OMP end do
    !$OMP end parallel
  end subroutine compute_many_YDYt_blocks

  subroutine lookup_tile(tilesize, bandcount, offsets, thetas, itile, &
       tile_phi, tile_dphi, tile_thetas, ierr)
    integer(i4b), value :: tilesize, bandcount, itile
    integer(i4b), dimension(0:), intent(in) :: offsets
    real(dp), dimension(0:), intent(in) :: thetas
    real(dp), intent(out) :: tile_phi, tile_dphi
    real(dp), dimension(tilesize), intent(out) :: tile_thetas
    integer(i4b), intent(out) :: ierr
    !--
    integer(i4b) :: iband, j, ringlen
    logical(lgt) :: is_north
    real(dp) :: phi0
    ! Do a simple linear search to find our band, for our number of
    ! items this is probably quicker than binary, not tested..
    ierr = 0  ! ok return
    do iband = 0, bandcount - 1
       if (itile < offsets(iband + 1)) then
          ! Found our band number iband. Find j, horizontal index within band.
          ringlen = (offsets(iband + 1) - offsets(iband)) / 2
          j = itile - offsets(iband)
          is_north = (j < ringlen)
          if (.not. is_north) then
             j = j - ringlen
          end if
          phi0 = pi / (ringlen * tilesize)
          tile_dphi = 2_dp * pi / (ringlen * tilesize)
          tile_phi = phi0 + j * tilesize * tile_dphi
          tile_thetas(:) = thetas((iband * tilesize):(iband + 1) * tilesize - 1)
          if (.not. is_north) then
             tile_thetas(:) = pi - tile_thetas(:)
          end if
          return
       end if
    end do
    ! Never found bounding offset, tile index too high
    ierr = 1
  end subroutine lookup_tile

end module
