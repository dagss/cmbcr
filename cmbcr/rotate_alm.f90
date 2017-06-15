module rotate_alm
  use iso_c_binding
  use types_as_c

  implicit none

contains
    subroutine exit_with_status (code, msg)
      ! ===========================================================
      integer, intent(in) :: code
      character (len=*), intent(in), optional :: msg
      ! ===========================================================

      if (present(msg)) print *,trim(msg)
      print *,'program exits with exit code ', code

      stop
    end subroutine exit_with_status

  subroutine assert_alloc (stat,code,arr)
!!    integer, intent(in) :: stat
    integer(i4b), intent(in) :: stat
    character(len=*), intent(in) :: code, arr

    if (stat==0) return

    print *, trim(code)//'> cannot allocate memory for array: '//trim(arr)
    call exit_with_status(1)
  end subroutine assert_alloc


  subroutine fatal_error (msg)
    character(len=*), intent(in) :: msg
       print *,'Fatal error: ', trim(msg)
    call exit_with_status(1)
  end subroutine fatal_error


  !========================================================
  subroutine rotate_alm_d(lmax, alm, psi, theta, phi)
    !=========================================================
    !Input: Complex array alm(p,l,m) with (l,m) in [0,lmax]^2, and p in [1,nd]
    !Euler rotation angles psi, theta, phi in radians
    !Output: Rotated array alm(p, l,m)
    !
    ! Euler angle convention  is right handed, active rotation
    ! psi is the first rotation about the z-axis (vertical), in [-2pi,2pi]
    ! then theta about the ORIGINAL (unrotated) y-axis, in [-2pi,2pi]
    ! then phi  about the ORIGINAL (unrotated) z-axis (vertical), in [-2pi,2pi]
    !
    ! Equivalently
    ! phi is the first rotation about the z-axis (vertical)
    ! then theta  about the NEW   y-axis (line of nodes)
    ! then psi    about the FINAL z-axis (figure axis)
    ! ---
    ! the recursion on the Wigner d matrix is inspired from the very stable
    ! double sided one described in Risbo (1996, J. of Geodesy, 70, 383)
    ! based on equation (4.4.1) in Edmonds (1957).
    ! the Risbo's scatter scheme has been repladed by a gather scheme for
    ! better computing efficiency
    ! the size of the matrix is divided by 2 using Edmonds Eq.(4.2.5) 
    ! to speed up calculations
    ! the loop on j has been unrolled for further speed-up
    ! EH, March--April 2005
    !=========================================================
    integer(I4B),   intent(in) :: lmax
    complex(DPC), intent(inout), dimension(1:,0:,0:) :: alm
    real(DP),       intent(in) :: psi, theta, phi
    ! local variables
    complex(DPC), dimension(0:lmax) :: exppsi, expphi
    complex(DPC), dimension(:,:), allocatable :: alm1, alm2
    real(DP),     dimension(:,:), allocatable :: d, dd
    real(DP),     dimension(:),   allocatable :: sqt, rsqt
    real(DP),     dimension(:),   allocatable :: tsign
    integer(I4B) :: status
    integer(I4B) :: mm, ll, na1, na2, nd
    integer(I4B) :: i, j, k, kd, hj
    real(DP)     :: p, q, pj, qj, fj, temp
    character(len=*), parameter :: code = 'ROTATE_ALM'
    !==========================================================
    


    nd = size(alm,1)
    na1 = size(alm,2)
    na2 = size(alm,3)
    if (na1 < (lmax+1) .or. na2 < (lmax+1)) then
       call fatal_error(code//': unconsistent alm array size and lmax')
    endif

    allocate(d (-1:2*lmax,   -1:lmax),   stat = status)
    call assert_alloc(status,code,'d')
    allocate(dd(-1:2*lmax, -1:lmax), stat = status)
    call assert_alloc(status,code,'dd')
    allocate(sqt(0:2*lmax), rsqt(0:2*lmax), stat = status)
    call assert_alloc(status,code,'sqt & rsqt')
    allocate(alm1(1:nd,0:lmax), alm2(1:nd,0:lmax), stat = status)
    call assert_alloc(status,code,'alm1 & alm2')
    allocate(tsign(0:lmax+1), stat = status)
    call assert_alloc(status,code,'tsign')
    
    do i=0, lmax,2
       tsign(i)   =  1.0_dp
       tsign(i+1) = -1.0_dp
    enddo
    !     initialization of square-root  table
    do i=0,2*lmax
       sqt(i) = SQRT(DBLE(i))
    enddo

    ! initialisation of exponential table
    exppsi(0)=cmplx(1, 0, kind=DPC)
    expphi(0)=cmplx(1, 0, kind=DPC)

    do i=1,lmax
       exppsi(i)= cmplx(cos(psi*i), -sin(psi*i), kind=DPC)
       expphi(i)= cmplx(cos(phi*i), -sin(phi*i), kind=DPC)
    enddo

    ! Note: theta has the correct sign.
    p = sin(theta/2.d0)
    q = cos(theta/2.d0)

    d  = 0.0_dp ! very important for gather scheme
    dd = 0.0_dp
    do ll=0,lmax

       ! ------ build d-matrix of order l ------
       if (ll == 0) then
          d(0,0) = 1.d0
          goto 2000
       endif
       if (ll == 1) then
          !     initialize d-matrix degree 1/2
          dd(0,0)  =  q
          dd(1,0)  = -p
          dd(0,1)  =  p
          dd(1,1)  =  q
          goto 1000
       endif

       !  l - 1 --> l - 1/2
       j = 2*ll - 1
       rsqt(0:j) = sqt(j:0:-1)
       fj = DBLE(j)
       qj = q / fj
       pj = p / fj
!$OMP parallel default(none) &
!$OMP   shared(j, fj, d, dd, rsqt, sqt, q, p, qj, pj) &
!$OMP   private(k)
!$OMP do schedule(dynamic,100)
       do k = 0, j/2 ! keep only m' <= -1/2
          dd(0:j,k) = rsqt(0:j) * ( d(0:j,k)      * (sqt(j-k)  * qj)   &
               &                  + d(0:j,k-1)    * (sqt(k)    * pj) ) &
               &    +  sqt(0:j) * ( d(-1:j-1,k-1) * (sqt(k)    * qj)   &
               &                  - d(-1:j-1,k)   * (sqt(j-k)  * pj) )
       enddo ! loop on k
!$OMP end do
!$OMP end parallel
       ! l=half-integer, reconstruct m'= 1/2 by symmetry
       hj = ll-1
       if (mod(ll,2) == 0) then
          do k = 0, j-1, 2
             dd(k,   ll) =   dd(j-k,   hj)
             dd(k+1, ll) = - dd(j-k-1, hj)
          enddo
       else
          do k = 0, j-1, 2
             dd(k,   ll) = - dd(j-k,   hj)
             dd(k+1, ll) =   dd(j-k-1, hj)
          enddo
       endif

1000   continue

       !  l - 1/2 --> l
       j = 2*ll
       rsqt(0:j) = sqt(j:0:-1)
       fj = DBLE(j)
       qj = q / fj
       pj = p / fj
!$OMP parallel default(none) &
!$OMP   shared(j, fj, d, dd, rsqt, sqt, q, p, qj, pj) &
!$OMP   private(k)
!$OMP do schedule(dynamic,100)
       do k = 0, j/2 ! keep only m' <= 0
          d (0:j,k) = rsqt(0:j) * ( dd(0:j,k)      * (sqt(j-k)  * qj)   &
               &                  + dd(0:j,k-1)    * (sqt(k)    * pj) ) &
               &    +  sqt(0:j) * ( dd(-1:j-1,k-1) * (sqt(k)    * qj)   &
               &                  - dd(-1:j-1,k)   * (sqt(j-k)  * pj) )
       enddo ! loop on k
!$OMP end do
!$OMP end parallel

2000   continue
       ! ------- apply rotation matrix -------
       do kd = 1, nd
          alm1(kd,0:ll)  = alm(kd,ll,0:ll) * exppsi(0:ll)
       enddo

       ! m = 0
       do kd = 1, nd
          alm2(kd,0:ll) = alm1(kd,0) * d(ll:2*ll,ll)
       enddo

!$OMP parallel default(none) &
!$OMP   shared(d, alm1, alm2, tsign, nd, ll) &
!$OMP   private(mm, kd)
!$OMP do schedule(dynamic,100)
       do mm = 0, ll
          do kd = 1, nd
             alm2(kd, mm) = alm2(kd,mm) + sum(alm1(kd,1:ll) *                d(ll-1:0:-1,ll-mm)) &
                  &                +conjg(sum(alm1(kd,1:ll) * (tsign(1:ll) * d(ll+1:2*ll,ll-mm))))
          enddo
       enddo
!$OMP end do
!$OMP end parallel

       ! new alm for ll
       do kd = 1,nd
          alm(kd,ll,0:ll) = alm2(kd,0:ll)*expphi(0:ll)
       enddo

    enddo ! loop on ll

    deallocate(d)
    deallocate(dd)
    deallocate(sqt, rsqt)
    deallocate(alm1, alm2)

  end subroutine rotate_alm_d

  subroutine rotate_alm_cwrapper(lmax, alm, psi, theta, phi) bind(c)


    integer(i4b), value :: lmax
    real(dp), dimension(1:(lmax + 1)**2) :: alm
    real(dp), value :: psi, theta, phi
    !--
    integer(i4b) :: l, m, idx
    complex(dp), dimension(:, :, :), allocatable :: alm_buf

    allocate(alm_buf(1:1, 0:lmax, 0:lmax))

    ! m = 0
    idx = 1
    do l = 0, lmax
       alm_buf(1, l, 0) = alm(idx)
       idx = idx + 1
    end do
    do m = 1, lmax
      do l = m, lmax

         alm_buf(1, l, m) = sqrt(.5) * cmplx(alm(idx), alm(idx + 1), kind=dp)
         idx = idx + 2
      end do
   end do
     

   call rotate_alm_d(lmax, alm_buf, psi, theta, phi)

    ! m = 0
    idx = 1
    do l = 0, lmax
       alm(idx) = alm_buf(1, l, 0)
       idx = idx + 1
    end do
    do m = 1, lmax
      do l = m, lmax

         alm(idx) = sqrt(2.0_dp) * real(alm_buf(1, l, m))
         alm(idx + 1) = sqrt(2.0_dp) * aimag(alm_buf(1, l, m))
         idx = idx + 2
      end do
   end do


 end subroutine rotate_alm_cwrapper

end module rotate_alm
