module block_matrix
  use iso_c_binding
  use types_as_c
  use constants

  implicit none

  interface
     SUBROUTINE DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO )
       CHARACTER          UPLO
       INTEGER            INFO, LDA, LDB, N, NRHS
       DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
     END SUBROUTINE DPOTRS

     SUBROUTINE SPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO )
       CHARACTER          UPLO
       INTEGER            INFO, LDA, LDB, N, NRHS
       REAL               A( LDA, * ), B( LDB, * )
     END SUBROUTINE SPOTRS

     SUBROUTINE DPOTRF( UPLO, N, A, LDA, INFO )
       CHARACTER          UPLO
       INTEGER            INFO, LDA, N
       DOUBLE PRECISION   A( LDA, * )
     END SUBROUTINE DPOTRF

     SUBROUTINE DPPTRF( UPLO, N, AP, INFO )
      CHARACTER      UPLO
      INTEGER        INFO, N
      DOUBLE PRECISION AP( * )
     END SUBROUTINE DPPTRF

     SUBROUTINE DPPTRS( UPLO, N, NRHS, AP,B, LDB, INFO )
      CHARACTER UPLO
      INTEGER INFO, LDB, N, NRHS
      DOUBLE PRECISION AP( * ), B( LDB, * )
     END SUBROUTINE DPPTRS

     SUBROUTINE SPOTRF( UPLO, N, A, LDA, INFO )
       CHARACTER          UPLO
       INTEGER            INFO, LDA, N
       REAL               A( LDA, * )
     END SUBROUTINE SPOTRF

     SUBROUTINE DTRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
       DOUBLE PRECISION ALPHA
       INTEGER LDA,LDB,M,N
       CHARACTER DIAG,SIDE,TRANSA,UPLO
       DOUBLE PRECISION A(LDA,*),B(LDB,*)
     END SUBROUTINE DTRSM

     SUBROUTINE STRSM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
       REAL ALPHA
       INTEGER LDA,LDB,M,N
       CHARACTER DIAG,SIDE,TRANSA,UPLO
       REAL A(LDA,*),B(LDB,*)
     END SUBROUTINE STRSM

     SUBROUTINE DTRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
       INTEGER INCX,LDA,N
       CHARACTER DIAG,TRANS,UPLO
       DOUBLE PRECISION A(LDA,*),X(*)
     END SUBROUTINE DTRSV

     SUBROUTINE STRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
       INTEGER INCX,LDA,N
       CHARACTER DIAG,TRANS,UPLO
       REAL A(LDA,*),X(*)
     END SUBROUTINE STRSV

     SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
       DOUBLE PRECISION ALPHA,BETA
       INTEGER K,LDA,LDB,LDC,M,N
       CHARACTER TRANSA,TRANSB
       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
     END SUBROUTINE DGEMM

     SUBROUTINE SGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
       REAL ALPHA,BETA
       INTEGER K,LDA,LDB,LDC,M,N
       CHARACTER TRANSA,TRANSB
       REAL A(LDA,*),B(LDB,*),C(LDC,*)
     END SUBROUTINE SGEMM

     SUBROUTINE DSYRK(UPLO,TRANS,N,K,ALPHA,A,LDA,BETA,C,LDC)
       DOUBLE PRECISION ALPHA,BETA
       INTEGER K,LDA,LDC,N
       CHARACTER TRANS,UPLO
       DOUBLE PRECISION A(LDA,*),C(LDC,*)
     END SUBROUTINE DSYRK

     SUBROUTINE SSYRK(UPLO,TRANS,N,K,ALPHA,A,LDA,BETA,C,LDC)
       REAL ALPHA,BETA
       INTEGER K,LDA,LDC,N
       CHARACTER TRANS,UPLO
       REAL A(LDA,*),C(LDC,*)
     END SUBROUTINE SSYRK

     SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
       DOUBLE PRECISION ALPHA,BETA
       INTEGER INCX,INCY,LDA,M,N
       CHARACTER TRANS
       DOUBLE PRECISION A(LDA,*),X(*),Y(*)
     END SUBROUTINE DGEMV

     SUBROUTINE SGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
       REAL ALPHA,BETA
       INTEGER INCX,INCY,LDA,M,N
       CHARACTER TRANS
       REAL A(LDA,*),X(*),Y(*)
     END SUBROUTINE SGEMV

  end interface

contains

  subroutine assert_(cnd, msg)
    logical :: cnd
    character(len=*), optional :: msg
    if (.not. cnd) then
       if (present(msg)) then
          print *, 'pix_matrices.f90: ASSERTION FAILED:', msg
       else
          print *, 'pix_matrices.f90: ASSERTION FAILED'
    end if
    end if
  end subroutine assert_

  ! Takes sorted lower-triangular CSC indices and produces *unsorted* full matrix CSC indices
  subroutine mirror_csc_indices(n, indptr_in, indices_in, indptr_out, indices_out) bind(c)
    integer(i4b), value :: n
    integer(i4b), dimension(0:n) :: indptr_in, indptr_out
    integer(i4b), dimension(0:indptr_in(n) - 1) :: indices_in
    integer(i4b), dimension(0:2 * indptr_in(n) - n - 1) :: indices_out
    !--
    integer(i4b), dimension(:), allocatable :: entries_per_row, col_lens
    integer(i4b) :: j, iptr, k, kptr, i

    allocate(entries_per_row(0:n - 1))

    ! Count entries per row outside of diagonal
    entries_per_row = 0
    do j = 0, n - 1
       do iptr = indptr_in(j) + 1, indptr_in(j + 1) - 1
          i = indices_in(iptr)
          entries_per_row(i) = entries_per_row(i) + 1
       end do
    end do

    ! Set up indptr_out. indices_out should contain the indices in
    ! the column j in indptr/indices_in, + the indices in row j
    ! (across the columns in indptr/indices_in).
    indptr_out(0) = 0
    do j = 0, n - 1
       k = entries_per_row(j) + (indptr_in(j + 1) - indptr_in(j)) ! row + (col & diagonal)
       indptr_out(j + 1) = indptr_out(j) + k
    end do

    deallocate(entries_per_row)
    allocate(col_lens(0:n - 1))

    ! copy data
    col_lens = 0
    do j = 0, n - 1
       ! diagonal entry
       kptr = indptr_out(j) + col_lens(j)
       indices_out(kptr) = j
       col_lens(j) = col_lens(j) + 1
       ! loop over sub-diagonal entries
       do iptr = indptr_in(j) + 1, indptr_in(j + 1) - 1
          i = indices_in(iptr)
          ! Copy entry from row to corresponding column and increase column len
          kptr = indptr_out(i) + col_lens(i)
          indices_out(kptr) = j
          col_lens(i) = col_lens(i) + 1
          ! Copy column entry
          kptr = indptr_out(j) + col_lens(j)
          indices_out(kptr) = i
          col_lens(j) = col_lens(j) + 1
       end do
    end do
  end subroutine mirror_csc_indices

  subroutine scale_rows(diag, buf)
    real(dp), dimension(:) :: diag
    real(dp), dimension(:,:) :: buf
    !--
    integer(i8b) :: i, j
    do j = 1, ubound(buf, 2)
       do i = 1, ubound(buf, 1)
          buf(i, j) = buf(i, j) * diag(i)
       end do
    end do
  end subroutine scale_rows

  subroutine zero_upper(buf)
    real(dp), dimension(:,:) :: buf
    !--
    integer(i8b) :: i, j
    do j = 1, ubound(buf, 2)
       do i = 1, j - 1
          buf(i, j) = 0.0_dp
       end do
    end do
  end subroutine zero_upper

  subroutine block_At_D_B( bs_left, bs_mid, bs_right, &
                           A_n, A_indptr, A_indices, A_labels, A_blocks, &
                           B_n, B_indptr, B_indices, B_labels, B_blocks, &
                           C_indptr, C_indices, C_blocks, D) bind(c,name='block_At_D_B')

    ! Matrix multiplication of tiled operators, C = C + A^T * D * B, where
    ! A, B, C are sparse block matrices and D is a diagonal matrix.
    ! Only the blocks given in the output matrix C are computed so the routine
    ! can either be used for complete or incomplete multiplication.

    integer(i4b), value                          :: bs_left, bs_mid, bs_right, A_n, B_n
    real(dp), dimension(bs_mid, 0:A_n - 1)       :: D
    integer(i4b), dimension(0:A_n)               :: A_indptr
    integer(i4b), dimension(0:B_n)               :: B_indptr, C_indptr
    integer(i4b), dimension(0:A_indptr(A_n) - 1) :: A_indices
    integer(i4b), dimension(0:B_indptr(B_n) - 1) :: B_indices
    integer(i4b), dimension(0:C_indptr(B_n) - 1) :: C_indices  ! C has same number of cols as B
    integer(i4b), dimension(0_i8b:int(A_indptr(A_n), i8b) - 1_i8b) :: A_labels
    integer(i4b), dimension(0_i8b:int(B_indptr(B_n), i8b) - 1_i8b) :: B_labels
    real(dp), dimension(1_i8b:int(bs_mid, i8b), 1_i8b:int(bs_left, i8b), 0_i8b:*) :: A_blocks
    real(dp), dimension(1_i8b:int(bs_mid, i8b), 1_i8b:int(bs_right, i8b), 0_i8b:*) :: B_blocks
    real(dp), dimension(1_i8b:int(bs_left, i8b), 1_i8b:int(bs_right, i8b), 0_i8b:int(C_indptr(B_n), i8b) - 1_i8b) :: C_blocks

    ! Variables
    real(dp), dimension(:,:), allocatable :: B_buffer
    integer(i4b)                   :: i, j, k, iptr, kptr, lptr

    !$OMP parallel default(none) &
    !$OMP     shared(A_indptr, A_indices, A_labels, A_blocks, &
    !$OMP            B_indptr, B_indices, B_labels, B_blocks, &
    !$OMP            C_indptr, C_indices, C_blocks, &
    !$OMP            D, B_n, bs_left, bs_mid, bs_right) &
    !$OMP     private(i, j, k, iptr, kptr, lptr, B_buffer)

    allocate(B_buffer(bs_mid, bs_right))

    ! Multiply all blocks in sparsity pattern
    ! (i, j) are tile indices in C
    ! (k, j) are tile indices in A
    ! (l, j) are tile indices in B

    !$OMP do schedule(dynamic,1)
    do j = 0, B_n - 1
       do iptr = C_indptr(j), C_indptr(j + 1) - 1
          i = C_indices(iptr)
          do kptr = A_indptr(i), A_indptr(i + 1) - 1
             k = A_indices(kptr)
             do lptr = B_indptr(j), B_indptr(j + 1) - 1
                if (B_indices(lptr) == k) then
                   ! If resulting block lies within sparsity pattern,
                   ! multiply blocks to get C[i,j] = A[k,j] * D * B[l,j]
                   B_buffer = B_blocks(:, :, int(B_labels(lptr), i8b))
                   call scale_rows(D(:, k), B_buffer) ! (D * B)
                   call DGEMM('T', 'N', bs_left, bs_right, bs_mid, 1.0_dp, &
                              A_blocks(:, :, int(A_labels(kptr), i8b)), bs_mid, &
                              B_buffer, bs_mid, 1.0_dp, &
                              C_blocks(:, :, iptr), bs_left)
                end if
             end do
          end do
       end do
    end do
    !$OMP end do

    deallocate(B_buffer)
    !$OMP end parallel

  end subroutine block_At_D_B


  subroutine block_At_D_A( bs_left, bs_right, n, &
                           A_indptr, A_indices, A_labels, A_blocks, D, &
                           C_indptr, C_indices, C_blocks) bind(c,name='block_At_D_A')

    ! Do sparse multiplication of diagonal matrix D with matrix A (RHS) and its
    ! transpose A^T (LHS);
    !
    !     C = C + A^T D A
    !
    ! The result is symmetric, so only the lower triangular
    ! part of the sparsity pattern is returned.
    !
    ! There is an indirection feature where the same blocks in A can be reused
    ! again and again; the blocks of A is A_blocks(A_labels(idx)).
    !
    ! The sparsity pattern is defined for some "tiled map". Each input operator
    ! must have been tiled according to the same tiling pattern. The block
    ! sparse matrix multiplication then proceeds according to the specified
    ! tilings.
    !
    ! Parameters
    ! ----------
    !
    ! bs_left, bs_right : int
    !
    !   No. pixels on each side of the operator matrix within each block.
    !   A has shape (bs_left, bs_right) (so A^T has shape (bs_right, bs_left))
    !   C has shape (bs_right, bs_right)
    !
    ! n : int
    !
    !   Number of pixels in tiled map.
    !
    ! A_indptr, A_indices, C_indptr, C_indices : CSC index arrays
    !
    !   Row pointers and row indices of non-zero blocks in the sparsity pattern.
    !   (See csc_24_neighbours(), csc_neighbours_lower())
    !
    !   A can be full or lower triangular sparsity pattern (depending on
    !   whether A is symmetric).
    !
    !   C must be the lower triangular sparsity pattern (since C is symmetric)
    !
    ! C_blocks, D : tiled operator matrices/maps
    !
    !   Input/output operators, that have been tiled according to the sparsity
    !   pattern. (See compute_csc_beam_matrix() for matrix operators, and
    !   convert_ring_dp_to_tiled_sp() for vector operators)
    !
    !   Blocks (tiled operators) have dimensions:
    !
    !     C_blocks:  (bs_right, bs_right, nnz(C))
    !     D:         (bs_left, n)
    !
    ! A_blocks, A_labels : blocks of A matrix
    !
    !   Blocks of A with an indirection array for reuse of blocks; the CSC index
    !   is looked up in A_labels which then forward to corresponding block in
    !   A_blocks. The indices are 0-based. Pass 0..nnz(A)-1 as A_labels will
    !   have the effect of disabling the indirection, and A_blocks turns into
    !   a regular CSC blocks array.
    !
    !     A_blocks:  (bs_left, bs_right, 0..max(A_labels))
    !     A_labels:  (nnz(A))


    integer(i4b), value                        :: bs_left, bs_right, n
    real(dp), dimension(bs_left, 0:n - 1)      :: D
    integer(i4b), dimension(0:n)               :: A_indptr, C_indptr
    integer(i4b), dimension(0_i8b:int(A_indptr(n), i8b) - 1_i8b) :: A_labels
    integer(i4b), dimension(0:A_indptr(n) - 1) :: A_indices
    integer(i4b), dimension(0:C_indptr(n) - 1) :: C_indices
    integer(i4b)                               :: iptr, i, j, kptr, k, lptr
    real(dp), dimension(:,:), allocatable      :: buf
    real(dp), dimension(1_i8b:int(bs_left, i8b), 1_i8b:int(bs_right, i8b), 0:*) :: A_blocks
    real(dp), dimension(1_i8b:int(bs_right, i8b), 1_i8b:int(bs_right, i8b), &
                        0_i8b:int(C_indptr(n), i8b) - 1_i8b) :: C_blocks
    ! Assume in comments that indptr/indices are given in CSC ordering.
    ! The full matrix is given for A, the lower matrix for C.

    !$OMP parallel default(none) &
    !$OMP     shared(C_indptr,C_indices,C_blocks,A_indptr,A_indices,A_labels,A_blocks, &
    !$OMP            n,bs_left,bs_right,D) &
    !$OMP     private(i, j, k, iptr, kptr, lptr, buf)

    allocate(buf(bs_left, bs_right))

    ! (i, j) are tile indices in C
    ! (k, j) are tile indices in A
    ! (l, j) are tile indices in B

    !$OMP do schedule(dynamic,1)
    do j = 0, n - 1
       iptr = C_indptr(j)
       call assert_(C_indices(iptr) == j, 'Diagonal not in expected place in C_indices')

       ! Diagonal C block; i=j. We could use SYRK, though only if D
       ! is non-negative, and it would require taking the sqrt of D,
       ! so it's not obvious it will be faster for small blocksizes
       ! (?).  So we just zero the upper half manually for now.
       do kptr = A_indptr(j), A_indptr(j + 1) - 1
          k = A_indices(kptr)
          buf = A_blocks(:, :, A_labels(kptr))
          call scale_rows(D(:, k), buf)
          call DGEMM('T', 'N', bs_right, bs_right, bs_left, 1.0_dp, &
               A_blocks(:, :, int(A_labels(kptr), i8b)), bs_left, &
               buf, bs_left, &
               1.0_dp, C_blocks(:, :, iptr), bs_right)
       end do
       call zero_upper(C_blocks(:, :, iptr))

       ! Sub-diagonal blocks, i>j.
       do iptr = C_indptr(j) + 1, C_indptr(j + 1) - 1
          i = C_indices(iptr)
          do kptr = A_indptr(i), A_indptr(i + 1) - 1
             k = A_indices(kptr)
             do lptr = A_indptr(j), A_indptr(j + 1) - 1
                if (A_indices(lptr) == k) then
                   buf = A_blocks(:, :, int(A_labels(lptr), i8b))
                   call scale_rows(D(:, k), buf)
                   call DGEMM('T', 'N', bs_right, bs_right, bs_left, 1.0_dp, &
                       A_blocks(:, :, int(A_labels(kptr), i8b)), bs_left, &
                       buf, bs_left, &
                       1.0_dp, C_blocks(:, :, iptr), bs_right)
                end if
             end do
          end do
       end do
    end do
    !$OMP end do

    deallocate(buf)

    !$OMP end parallel

  end subroutine block_At_D_A


  subroutine block_A_x( trans, bs_left, bs_right, n, &
                        A_indptr, A_indices, A_labels, A_blocks, x, y) bind(c,name='block_A_x')

    ! Do sparse multiplication of block matrix A and a vector x;
    ! y = y + A * x, or y = y + A^T * x.
    !
    ! There is an indirection feature where the same blocks in A can be reused
    ! again and again; the blocks of A is A_blocks(A_labels(idx)).
    !
    ! Parameters
    ! ----------
    !
    ! trans : int
    !   1 for transpose, 0 for non-transpose
    !
    ! bs_left, bs_right : int
    !
    !   No. pixels on each side of the operator matrix within each block.
    !   A has shape (bs_left, bs_right) (so A^T has shape (bs_right, bs_left))
    !   C has shape (bs_right, bs_right)
    !
    ! n : int
    !
    !   Number of tiles in tiled map.
    !
    ! A_indptr, A_indices : CSC index arrays
    !
    !   Row pointers and row indices of non-zero blocks in the sparsity pattern.
    !
    ! A_blocks, A_labels : blocks of A matrix
    !
    !   Blocks of A with an indirection array for reuse of blocks; the CSC index
    !   is looked up in A_labels which then forward to corresponding block in
    !   A_blocks. The indices are 0-based. Pass 0..nnz(A)-1 as A_labels will
    !   have the effect of disabling the indirection, and A_blocks turns into
    !   a regular CSC blocks array.
    !
    !     A_blocks:  (bs_left, bs_right, 0..max(A_labels))
    !     A_labels:  (nnz(A))
    !
    ! x, y : input and output vector
    !
    !   Dimensions:
    !
    !     x:  (bs_right, n / bs_right)
    !     y:  (bs_left, n / bs_left)

    integer(i4b), value :: trans
    integer(i4b), value                        :: bs_left, bs_right, n
    integer(i4b), dimension(0:n)               :: A_indptr
    integer(i4b), dimension(0_i8b:int(A_indptr(n), i8b) - 1_i8b) :: A_labels
    integer(i4b), dimension(0:A_indptr(n) - 1) :: A_indices
    integer(i4b)                               :: iptr, i, j
    real(dp), dimension(1_i8b:int(bs_left, i8b), 1_i8b:int(bs_right, i8b), 0:*) :: A_blocks
    real(dp), dimension(1_i8b:int(merge(bs_right, bs_left, trans == 0), i8b), 0:int(n - 1, i8b)) :: x
    real(dp), dimension(1_i8b:int(merge(bs_left, bs_right, trans == 0), i8b), 0:int(n-1, i8b)) :: y
    ! Assume in comments that indptr/indices are given in CSC ordering.
    ! The full matrix is given for A, the lower matrix for C.

    !$OMP parallel default(none) &
    !$OMP     shared(A_indptr,A_indices,A_labels,A_blocks, &
    !$OMP            n,bs_left,bs_right,x,y,trans) &
    !$OMP     private(i, j, iptr)

    ! (i, j) are tile indices in A

    if (trans == 0) then
       !$OMP do schedule(dynamic,1)
       do j = 0, n - 1
          do iptr = A_indptr(j), A_indptr(j + 1) - 1
             i = A_indices(iptr)
             call DGEMV('N', bs_left, bs_right, 1.0_dp, &
                  A_blocks(:, :, A_labels(iptr)), bs_left, &
                  x(:, j), 1, &
                  1.0_dp, y(:, i), 1)
          end do
       end do
       !$OMP end do
    else if (trans == 1) then
       ! For transpose case i, j still refer to blocks (i, j) in non-transposed A
       !$OMP do schedule(dynamic,1)
       do j = 0, n - 1
          do iptr = A_indptr(j), A_indptr(j + 1) - 1
             i = A_indices(iptr)
             call DGEMV('T', bs_left, bs_right, 1.0_dp, &
                  A_blocks(:, :, A_labels(iptr)), bs_left, &
                  x(:, i), 1, &
                  1.0_dp, y(:, j), 1)
          end do
       end do
       !$OMP end do
    end if

    !$OMP end parallel

  end subroutine block_A_x

  subroutine block_incomplete_cholesky_factor(bs, n, indptr, indices, blocks, ridge, info) bind(c)
    integer(i4b), value :: bs, n
    integer(i4b), dimension(0:n) :: indptr
    integer(i4b), dimension(0:indptr(n) - 1) :: indices
    real(dp), dimension(1_i8b:int(bs, i8b), 1_i8b:int(bs, i8b), 0_i8b:int(indptr(n), i8b)) :: blocks
    real(dp), value :: ridge
    integer(i4b), intent(out) :: info
    !--
    integer(i4b) :: iptr, diagptr, i, j, kptr, k, lptr, t
    logical :: found
    real(dp), dimension(bs, bs) :: buf
    ! Follow notation of FLAME paper; for every step, we have
    ! the remaining bottom-right matrix partitioned as
    ! [ A11 A12 ]
    ! [ A21 A22 ]
    ! NOTE: Note sure about IKJ vs KIJ and so on, the indices here are probably invented
    ! rather than taking on their traditional meaning.
    ! TODO: k refers to source, j to modified dest, so k and j should switch meanings.
    info = 0

    ! Add ridge to diagonal
    ! TODO: Move to inner loop to avoid memory transfer
    if (ridge /= 0.0_dp) then
       do j = 0, n - 1
          iptr = indptr(j)
          do t = 1, bs
             blocks(t, t, iptr) = blocks(t, t, iptr) + ridge
          end do
       end do
    end if

    do j = 0, n - 1
       ! Factor A11
       diagptr = indptr(j)
       call DPOTRF('L', bs, blocks(:, :, diagptr), bs, info)
       if (info /= 0) then
          return
       end if
       do iptr = indptr(j) + 1, indptr(j + 1) - 1
          ! A21 <- A21 * L^-T
          call DTRSM('Right', 'Lower', 'Transpose', 'Not unit triangular', &
                     bs, bs, 1.0_dp, blocks(:, :, diagptr), bs, blocks(:, :, iptr), bs)
          ! Handle diagonal part of A22 <- A22 - A21 * A21^T update right away
          i = indices(iptr)
          kptr = indptr(i)
          call DSYRK('L', 'N', bs, bs, &
                     -1.0_dp, blocks(:, :, iptr), bs, &
                     1.0_dp, blocks(:, :, kptr), bs)
       end do
       ! "Barrier", need to complete above loop for A21 <- A21 * L^-T.

       ! Now handle non-diagonal outer-product blocks to complete A22 <- A22 - A21 * A21^T
       do iptr = indptr(j) + 1, indptr(j + 1) - 1
          i = indices(iptr)
          ! We're treating block stored on row i; iterate over blocks above row i in
          ! same column for blocks we should outer-product with
          do kptr = indptr(j) + 1, iptr - 1
             k = indices(kptr)
             ! Should update A22 -= A21(iptr) * A21(kptr)^T; search for target block
             ! below diagonal in row i, column k
             found = .false.
             do lptr = indptr(k) + 1, indptr(k + 1) - 1
                if (indices(lptr) == i) then
                   ! Update
                   call DGEMM('N', 'T', bs, bs, bs, &
                        -1.0_dp, blocks(:, :, iptr), bs, blocks(:, :, kptr), bs, &
                        1.0_dp, blocks(:, :, lptr), bs)
                   found = .true.
                   exit
                end if
             end do
             if ((.not. found) .and. .false.) then
                ! MILU step: Add discarded elements to diagonal of corresponding row instead
                lptr = indptr(i) ! diagonal on row/col i
                ! gemm -> buf
                call DGEMM('N', 'T', bs, bs, bs, &
                     -1.0_dp, blocks(:, :, kptr), bs, blocks(:, :, iptr), bs, &
                     0.0_dp, buf, bs)
                ! sum each row of buf and add to diagonal of blocks(:,:,lptr)
                do t = 1, bs!
                   blocks(t, t, lptr) = blocks(t, t, lptr) + sum(buf(t,:))
                end do
             end if
          end do
       end do
    end do
  end subroutine block_incomplete_cholesky_factor

  subroutine block_triangular_solve(trans, bs, n, indptr, indices, blocks, x) bind(c)
    integer(i4b), value :: trans ! 1 if transpose, 0 otherwise
    integer(i4b), value :: bs, n
    integer(i4b), dimension(0_i8b:int(n, i8b)) :: indptr
    integer(i4b), dimension(0_i8b:int(indptr(n), i8b)) :: indices
    real(dp), dimension(1_i8b:int(bs, i8b), 1_i8b:int(bs, i8b), 0_i8b:int(indptr(n), i8b)) :: blocks
    real(dp), dimension(1_i8b:int(bs, i8b), 0_i8b:int(n, i8b)-1_i8b) :: x
    !--
    integer(i4b) :: jptr, j, iptr, i

    if (trans == 0) then
       ! CSC ordering

       do j = 0, n - 1
          ! Solve for diagonal block, x_j <- A_jj^-1 x_j
          call DTRSV('Lower', 'Not transposed', 'Not unit triangular', &
               bs, blocks(:, :, indptr(j)), bs, x(:, j), 1)
          ! Push updates for blocks in this column, x_i <- x_i - A_ij x_j
          do iptr = indptr(j) + 1, indptr(j + 1) - 1
             i = indices(iptr)
             call DGEMV('Not transposed', bs, bs, -1.0_dp, blocks(:, :, iptr), bs, &
                  x(:, j), 1, 1.0_dp, x(:, i), 1)
          end do
       end do

    else
       ! Comments refer to A_ij as if it is transposed, i.e., just assume CSR
       ! ordering instead of CSC
       do i = n - 1, 0, -1
          ! Pull updates from blocks in this row, x_i <- x_i - A_ij x_j
          do jptr = indptr(i) + 1, indptr(i + 1) - 1
             j = indices(jptr)
             call DGEMV('Transposed', bs, bs, -1.0_dp, blocks(:, :, jptr), bs, &
                  x(:, j), 1, 1.0_dp, x(:, i), 1)
          end do

          ! Solve for diagonal block, x_i <- A_ii^-1 x_i
          call DTRSV('Lower', 'Transposed', 'Not unit triangular', &
               bs, blocks(:, :, indptr(i)), bs, x(:, i), 1)
       end do

    end if
  end subroutine block_triangular_solve


  subroutine compute_block_norms(bs, n, blocks, norms) bind(c)
    integer(i4b), value :: bs, n
    real(dp), dimension(1_i8b:int(bs, i8b), 1_i8b:int(bs, i8b), 1_i8b:int(n, i8b)) :: blocks
    real(dp), dimension(1_i8b:int(n, i8b)) :: norms
    !--
    integer(i4b) :: i
    !$OMP parallel default(none) shared(blocks,norms,n) private(i)
    !$OMP do schedule(static)
    do i = 1, n
       norms(i) = sqrt(sum(blocks(:, :, i) * blocks(:, :, i)))
    end do
    !$OMP end do
    !$OMP end parallel

  end subroutine compute_block_norms

  subroutine block_diagonal_factor(bs, n, blocks, info) bind(c)
    integer(i4b), value :: bs, n
    integer(i4b), intent(out) :: info
    real(dp), dimension(1_i8b:int(bs, i8b), 1_i8b:int(bs, i8b), 1_i8b:int(n, i8b)) :: blocks
    !--
    integer(i8b) :: i
    integer(i4b) :: linfo
    info = 0

    !$OMP parallel default(none) shared(n,bs,info,blocks) private(i,linfo)
    !$OMP do schedule(static)
    do i = 1, n
       if (info == 0) then
          call DPOTRF('L', bs, blocks(:, :, i), bs, linfo)
          if (linfo /= 0) then
             info = linfo
          end if
       end if
    end do
    !$OMP end do
    !$OMP end parallel
  end subroutine block_diagonal_factor

  subroutine block_diagonal_solve(bs, n, blocks, x) bind(c)
    integer(i4b), value :: bs, n
    real(dp), dimension(1_i8b:int(bs, i8b), 1_i8b:int(bs, i8b), 1_i8b:int(n, i8b)) :: blocks
    real(dp), dimension(1_i8b:int(bs, i8b), 1_i8b:int(n, i8b)) :: x
    !--
    integer(i8b) :: i
    integer(i4b) :: info

    !$OMP parallel default(none) shared(n,bs,blocks,x) private(i,info)
    !$OMP do schedule(static)
    do i = 1, n
       call DPOTRS('L', bs, 1, blocks(:, :, i), bs, x(:, i), bs, info)
       if (info /= 0) then
          print *, info, 'should not happen'
          stop
       end if
    end do
    !$OMP end do
    !$OMP end parallel
  end subroutine block_diagonal_solve


  ! Takes a CSC block matrix and a set of clusters (given as a list of cluster sizes,
  ! and a permutation vector) and creates a packed representation suitable for used
  ! with LAPACK packed Cholesky. We also do Cholesky factorization of each block as we
  ! go.
  !
  ! If nofactor != 0, drop the Cholesky.
  subroutine csc_to_factored_compressed_block_diagonal( &
       bs, n, indptr, indices, blocks, &
       cluster_count, cluster_offsets, permutation, out, nofactor, startridge, info) bind(c)
    integer(i4b), value :: bs, n, cluster_count, nofactor
    integer(i4b), dimension(0_i8b:int(n, i8b)) :: indptr
    integer(i4b), dimension(0_i8b:int(indptr(n), i8b)) :: indices
    real(dp), dimension(1_i8b:int(bs, i8b), 1_i8b:int(bs, i8b), 0_i8b:int(indptr(n), i8b)) :: blocks
    integer(i4b), dimension(1_i8b:cluster_count + 1_i8b) :: cluster_offsets
    integer(i4b), dimension(0_i8b:int(n - 1, i8b)) :: permutation
    real(dp), dimension(*) :: out
    integer(i4b), intent(inout) :: info
    real(dp), value :: startridge
    !--
    integer(i4b) :: ic, permuted_i, permuted_j, iptr, i, j, x, out_idx, subinfo, m
    integer(i4b) :: csize, csize_max, matlen
    logical :: found
    integer(i4b), dimension(:), allocatable :: out_offsets
    real(dp), dimension(:), allocatable  :: buf, s, invs
    real(dp) :: ridge, bufmax, maxridge

    allocate(out_offsets(cluster_count + 1))

    info = 0
    if (cluster_count == 0) return
    out_offsets(1) = 1
    csize_max = 0
    do ic = 2, cluster_count + 1
       csize = cluster_offsets(ic) - cluster_offsets(ic - 1)
       csize_max = max(csize, csize_max)
       out_offsets(ic) = out_offsets(ic - 1) + (bs * csize * ((bs * csize) + 1)) / 2
    end do

    !$OMP parallel default(none) &
    !$OMP          shared(cluster_count,bs,cluster_offsets,out_offsets,info,&
    !$OMP                 permutation,nofactor,indptr,indices,blocks,csize_max,startridge) &
    !$OMP          private(ic,out_idx,permuted_i,permuted_j,i,j,x,&
    !$OMP                  subinfo,csize,found,buf,matlen,ridge,bufmax,s,invs,m) &
    !$OMP          reduction(max:maxridge)
    allocate(buf(0:(bs * csize_max * ((bs * csize_max) + 1)) / 2 - 1))
    allocate(s(0:bs * csize_max - 1), invs(0:bs * csize_max - 1))
    maxridge = 0_dp
    !$OMP do schedule(dynamic)
    do ic = 1, cluster_count
       out_idx = out_offsets(ic)

       if (info /= 0) cycle

       ! Check that permutation is monotonically increasing within every cluster
       if (any((permutation(cluster_offsets(ic) + 1:cluster_offsets(ic + 1) - 1) .le. &
                permutation(cluster_offsets(ic):cluster_offsets(ic + 1) - 2)))) then
          info = -1
       end if

       do permuted_j = cluster_offsets(ic), cluster_offsets(ic + 1) - 1
          j = permutation(permuted_j)
          ! we use x, y to refer to offsets within block, as in (i * bs + x).
          do x = 1, bs
             ! Diagonal block
             iptr = indptr(j)
             if (indices(iptr) /= j) then
                info = -2  ! illegal CSC matrix for this routine, not ordered and lower-triangular
             end if

             out(out_idx:out_idx + bs - x) = blocks(x:bs, x, iptr)
             out_idx = out_idx + bs - x + 1

             ! Sub-diagonal. Match up indices and permutations.
             do permuted_i = permuted_j + 1, cluster_offsets(ic + 1) - 1
                i = permutation(permuted_i)
                found = .false.
                do iptr = indptr(j) + 1, indptr(j + 1) - 1
                   if (indices(iptr) == i) then
                      found = .true.
                      exit
                   end if
                end do
                if (found) then
                   out(out_idx:out_idx + bs - 1) = blocks(:, x, iptr)
                else
                   out(out_idx:out_idx + bs - 1) = 0_dp
                end if
                out_idx = out_idx + bs
             end do
          end do
       end do

       if (nofactor == 0) then
          subinfo = 0
          csize = cluster_offsets(ic + 1) - cluster_offsets(ic)
          m = csize * bs
          matlen = (m * (m + 1)) / 2
          ! Copy to buf, factor, copy back -- in case it fails and we need to add ridge
          bufmax = maxval(out(out_offsets(ic):out_offsets(ic) + matlen - 1))
          ridge = startridge
          do
             ! Copy into buf
             buf(0:matlen - 1) = out(out_offsets(ic):out_offsets(ic) + matlen - 1)
             !buf(:) = 1
             ! Find scale and add ridge
             x = 0
             do i = 0, m - 1
                s(i) = 1_dp!sqrt(buf(x))
                invs(i) = 1_dp!sqrt(1 / buf(x))
                buf(x) = invs(i)**2 * buf(x) + ridge * bufmax  !! * bufmax !! (1.0_dp + ridge)! * bufmax
                x = x + m - i
             end do
             ! scale offdiagonal cols and rows by invs
             x = 0
             do i = 0, m - 1
                buf(x + 1:x + m - i - 1) = (buf(x + 1:x + m - i - 1) * &
                                            invs(i) * invs(i + 1:i + m - i - 1))
                x = x + m - i
             end do

             ! Attempt to factor
             call dpptrf('L', m, buf(0:matlen - 1), subinfo)
             if (subinfo == 0) then
                ! Success -- now scale resulting factor L along rows by s
                print *, 'cluster', ic, 'got ridge', ridge
                x = 0
                do i = 0, m - 1
                   buf(x:x + m - i - 1) = (buf(x:x + m - i - 1) * s(i:i + m - i - 1))
                   x = x + m - i
                end do
                exit
             else
                print *, 'cluster', ic, 'trided ridge', ridge, 'FAILED'
             end if
             ! Compute new ridge
             if (ridge == 0_dp) then  ! in case 0 was passed as startridge
                ridge = 1e-6_dp
             else
                ridge = 2_dp * ridge
             end if
             if (ridge > 0.5) then
                info = -3
                exit
             end if
          end do
          maxridge = max(ridge, maxridge)
          if (ridge > 0.5_dp) then
             print *, 'Ridged cluster ', ic, 'with', ridge
          end if
          out(out_offsets(ic):out_offsets(ic) + matlen - 1) = buf(0:matlen - 1)
       end if
    end do
    !$OMP end do
    deallocate(buf,s,invs)
    !$OMP end parallel
    print *, 'Max ridge', maxridge
  end subroutine csc_to_factored_compressed_block_diagonal

  subroutine csc_compressed_block_diagonal_solve( &
       bs, cluster_count, cluster_offsets, permutation, matrix, x) bind(c)
    integer(i4b), value :: bs, cluster_count
    integer(i4b), dimension(1_i8b:cluster_count + 1_i8b) :: cluster_offsets
    integer(i4b), dimension(0_i8b:int(cluster_offsets(cluster_count + 1) - 1, i8b)) :: permutation
    real(dp), dimension(0:*) :: matrix, x
    !--
    integer(i4b) :: ic, csize, info, max_csize, j, k
    integer(i4b), dimension(:), allocatable :: matrix_offsets
    real(dp), dimension(:), allocatable :: buf

    allocate(matrix_offsets(cluster_count + 1))

    if (cluster_count == 0) return
    matrix_offsets(1) = 0
    max_csize = 0
    do ic = 2, cluster_count + 1
       csize = cluster_offsets(ic) - cluster_offsets(ic - 1)
       max_csize = max(max_csize, csize)
       matrix_offsets(ic) = matrix_offsets(ic - 1) + (bs * csize * ((bs * csize) + 1)) / 2
    end do

    !$OMP parallel default(none) &
    !$OMP          shared(cluster_count,bs,cluster_offsets,matrix_offsets,&
    !$OMP                 permutation,max_csize) &
    !$OMP          private(buf,info,csize,j,k)
    allocate(buf(0:max_csize * bs - 1))
    !$OMP do schedule(static)
    do ic = 1, cluster_count
       csize = cluster_offsets(ic + 1) - cluster_offsets(ic)
       do k = 0, csize - 1
          j = permutation(cluster_offsets(ic) + k)
          buf(k * bs:(k + 1) * bs - 1) = x(j * bs:(j + 1) * bs - 1)
       end do
       call dpptrs('L', csize * bs, 1, matrix(matrix_offsets(ic):matrix_offsets(ic + 1) - 1), &
                   buf(0:csize * bs - 1), csize * bs, info)
       if (info /= 0) then
          print *, 'info != 0:', info
          stop
       end if
       do k = 0, csize - 1
          j = permutation(cluster_offsets(ic) + k)
          x(j * bs:(j + 1) * bs - 1) = buf(k * bs:(k + 1) * bs - 1)
       end do
    end do
    !$OMP end do
    deallocate(buf)
    !$OMP end parallel
  end subroutine csc_compressed_block_diagonal_solve

  subroutine csc_make_clusters(eps, n, indptr, indices, norms, &
                               permutation, cluster_size, cluster_count) bind(c)
    real(dp), value                                             :: eps
    integer(i4b), value                                         :: n
    integer(i4b), dimension(0_i8b:int(n, i8b))                  :: indptr
    integer(i4b), dimension(0_i8b:int(indptr(n), i8b))          :: indices
    real(dp), dimension(0_i8b:int(indptr(n), i8b))              :: norms
    integer(i4b), dimension(0_i8b:int(n - 1, i8b)), intent(out) :: permutation
    integer(i4b), dimension(0_i8b:int(n - 1, i8b)), intent(out) :: cluster_size
    integer(i4b), intent(out)                                   :: cluster_count
    !--
    ! We store the clusters as singly-linked lists, because we expect them to have very small
    ! sizes with a majority of single-node clusters. The cluster is given the number of the
    ! node with smallest index, and we loop through nodes with higher indices to change their
    ! cluster number.
    !
    ! While clustering, cluster_size(ic) and norms(ic) hold the size and norm of the cluster,
    ! while cluster(in) holds the cluster index of node i.
    !
    ! zero-based indexing.

    real(dp), dimension(:), allocatable                         :: cluster_norms  ! norm of cluster
    integer(i4b), dimension(:), allocatable                     :: cluster  ! node -> cluster
    integer(i4b), dimension(:), allocatable                     :: first    ! cluster -> first node
    integer(i4b), dimension(:), allocatable                     :: next     ! link to next node

    integer(i4b) :: best_size, best_i, best_j
    real(dp) :: best_relnorm, best_norm

    integer(i4b) :: ic, in, ip, ip_start, i, j, iptr, cc, newsize
    real(dp) :: relnorm_up, relnorm_right, relnorm
    logical :: ok

    allocate(cluster(0:n-1), next(0:n-1), first(0:n-1), cluster_norms(0:n-1))
    ! Initialize every node to a single cluster
    cluster_size(:) = 1
    next(:) = -1  ! sentinel value
    do ic = 0, n - 1
       cluster(ic) = ic
       first(ic) = ic
    end do

    do j = 0, n - 1
       cluster_norms(j) = norms(indptr(j))
       if (indices(indptr(j)) /= j) then
          print *, 'ASSERTION FAILED'
          stop
       end if
    end do

    ! We run through the matrix once every time two clusters are
    ! merged, finding the best one to merge every time, making the
    ! (implementation of the) algorithm O(nnz^2) :-(
    best_norm = 0_dp  ! avoid warning
    best_i = 0  ! avoid warning
    best_j = 0  ! avoid warning
    ok = .false.
    do while (.not. ok)
       ! First, we put all off-diagonal blocks in a priority queue
       ok = .true.
       best_size = 10000000
       best_relnorm = 0_dp

       do j = 0, n - 1
          do iptr = indptr(j) + 1, indptr(j + 1) - 1
             i = indices(iptr)
             if (cluster(i) == cluster(j)) cycle

             relnorm_up = norms(iptr) / cluster_norms(cluster(j))
             relnorm_right = norms(iptr) / cluster_norms(cluster(i))
             relnorm = max(relnorm_up, relnorm_right)
             newsize = cluster_size(cluster(i)) + cluster_size(cluster(j))
             if ((relnorm > eps) .and. (newsize < best_size) .or. &
                 (newsize == best_size .and. relnorm > best_relnorm)) then
                ok = .false.
                best_relnorm = relnorm
                best_norm = norms(iptr)
                best_size = cluster_size(cluster(i)) + cluster_size(cluster(j))
                best_i = i
                best_j = j
             end if
          end do
       end do

       
       ! We merge clusters; adding two times the norm of off-diagonal block
       ! to norm of cluster. This may be inaccurate, if you merge two 2-size
       ! clusters there may be another coupling block involved, but should
       ! be close enough and is conservative.
       if (.not. ok) then
          call merge_clusters_of_nodes(best_i, best_j, best_norm)
       end if
    end do

    ! Write permutation by following all linked lists.
    cluster_count = 0
    ip = 0
    do ic = 0, n - 1
       if (cluster_size(ic) == 0) cycle
       ip_start = ip
       in = first(ic)
       cc = 0  ! purely as an assertion, count the length of linked list and check it
       do while (in /= -1)
          cc = cc + 1
          permutation(ip) = in
          ip = ip + 1
          in = next(in)
       end do
       if (cc /= cluster_size(ic)) then
          print *, 'cc /= cluster_size(ic)', cc, cluster_size(ic)
          stop
       end if
       ! Now we need to sort this part of permutation, because we do not
       ! know that linked lists are in sorted order, and further code depends
       ! on this (+ good for locality)
       call insertion_sort(permutation(ip_start:ip - 1))
       ! Finally, compact the cluster_size array; this just means dropping 0 elements
       ! from it and so it's safe to do it in-place in loop
       cluster_size(cluster_count) = cluster_size(ic)
       cluster_count = cluster_count + 1
    end do


    contains
      subroutine merge_clusters_of_nodes(i, j, addnorm)
        integer(i4b), value :: i, j
        real(dp), value     :: addnorm
        !--

        ! We kill cluster(j) and keep cluster(i).
        cluster_norms(cluster(i)) = sqrt(cluster_norms(cluster(i))**2 &
             + cluster_norms(cluster(j))**2 + 2 * addnorm**2)
        cluster_size(cluster(i)) = cluster_size(cluster(i)) + cluster_size(cluster(j))
        cluster_size(cluster(j)) = 0

        ! Scan to end of cluster(i)-list and point it on to first node of cluster(j)
        do
           if (next(i) == -1) then
              next(i) = first(cluster(j))
              exit
           end if
           i = next(i)
        end do

        ! Rewrite cluster() for all nodes in cluster(j)
        j = first(cluster(j))
        do
           cluster(j) = cluster(i)
           j = next(j)
           if (j == -1) exit
        end do
      end subroutine merge_clusters_of_nodes

  end subroutine csc_make_clusters

  subroutine insertion_sort(lst)
    integer(i4b), dimension(:) :: lst
    !--
    integer(i4b) :: i, j, to_insert, next, n

    ! Insertion sort
    n = 0
    do i = 1, size(lst)
       to_insert = lst(i)
       ! Find location to insert in lst(1:n)
       do j = 1, n
          if (lst(j) > to_insert) exit
       end do
       ! Move remaining elements in lst(1:n) up, overwrites lst(n+1)
       do j = j, n + 1
          next = lst(j)
          lst(j) = to_insert
          to_insert = next
       end do
       n = n + 1
    end do
  end subroutine insertion_sort

end module block_matrix
