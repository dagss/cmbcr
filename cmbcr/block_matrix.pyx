from libc.stdint cimport int32_t, int8_t, int64_t
from libc.math cimport sqrt, log10, pow
import numpy as np
cimport numpy as cnp
cimport cython

class BlockMatrix:

    def __init__(self, indptr, indices, blocks=None, labels=None, blockshape=None, dtype=np.double):
        self.indptr = indptr
        self.indices = indices
        if blocks is None:
            blocks = np.zeros(blockshape + (indices.shape[0],), dtype=dtype, order='F')
        if labels is None:
            labels = np.arange(blocks.shape[2], dtype=np.int32)
        self.blocks = blocks
        self.labels = labels

    def to_dense(self, mirror=False):
        return blocks_to_dense(self.indptr, self.indices, self.blocks[:, :, self.labels], mirror=mirror)

    def diagonal(self):
        bs = self.blocks.shape[0]
        if bs != self.blocks.shape[1]:
            raise ValueError()
        n = (self.indptr.shape[0] - 1) * bs
        diag = np.zeros(n, dtype=self.blocks.dtype)
        for i, iptr in enumerate(self.indptr[:-1]):
            diag[i * bs:(i + 1) * bs] = self.blocks[:, :, self.labels[iptr]].diagonal()
        return diag


class NotPosDefError(Exception):
    pass


cdef extern:
    void mirror_csc_indices_ "mirror_csc_indices"(
        int32_t n, int32_t *indptr_in, int32_t *indices_in,
        int32_t *indptr_out, int32_t *indices_out) nogil

    void block_incomplete_cholesky_factor_ "block_incomplete_cholesky_factor"(
        int32_t bs, int32_t n, int32_t *indptr, int32_t *indices,
        double *blocks, double alpha, int32_t *info) nogil

    void block_triangular_solve_ "block_triangular_solve"(
        int32_t trans, int32_t bs, int32_t n,
        int32_t *indptr, int32_t *indices,
        double *blocks, double *x) nogil

    void block_symm_At_D_A_ "block_symm_At_D_A"(
        int32_t bs, int32_t n,
        int32_t *A_indptr, int32_t *A_indices, double *A_blocks,
        double *D, int32_t *C_indptr, int32_t *C_indices, double *C_blocks) nogil

    void block_At_D_B_ "block_At_D_B"(
        int32_t bs_left, int32_t bs_mid, int32_t bs_right,
        int32_t A_n, int32_t *A_indptr, int32_t *A_indices, int32_t *A_labels, double *A_blocks,
        int32_t B_n, int32_t *B_indptr, int32_t *B_indices, int32_t *B_labels, double *B_blocks,
        int32_t *C_indptr, int32_t *C_indices, double *C_blocks,
        double *D) nogil

    void block_At_D_A_ "block_At_D_A"(
        int32_t bs_left, int32_t bs_right, int32_t n,
        int32_t *A_indptr, int32_t *A_indices, int32_t *A_labels, double *A_blocks,
        double *D, int32_t *C_indptr, int32_t *C_indices, double *C_blocks) nogil

    void block_A_x_ "block_A_x"(
        char trans,
        int32_t bs_left, int32_t bs_right, int32_t n,
        int32_t *A_indptr, int32_t *A_indices, int32_t *A_labels, double *A_blocks,
        double *x, double *y) nogil

    void block_diagonal_factor_ "block_diagonal_factor"(
        int32_t bs, int32_t n, double *blocks, int32_t *info) nogil
    void block_diagonal_solve_ "block_diagonal_solve"(
        int32_t bs, int32_t n, double *blocks, double *x) nogil

    void csc_to_factored_compressed_block_diagonal_ "csc_to_factored_compressed_block_diagonal"(
        int32_t bs, int32_t n, int32_t *indptr, int32_t *indices, double *blocks,
        int32_t cluster_count, int32_t *cluster_offsets, int32_t *permutation, double *out,
        int32_t nofactor, double startridge, int32_t *info) nogil

    void csc_compressed_block_diagonal_solve_ "csc_compressed_block_diagonal_solve"(
        int32_t bs, int32_t cluster_count, int32_t *cluster_offsets, int32_t *permutation,
        double *matrix, double *x) nogil

    void csc_make_clusters_ "csc_make_clusters"(
        double eps, int32_t n, int32_t *indptr, int32_t *indices, double *norms,
        int32_t *permutation, int32_t *cluster_size, int32_t *cluster_count) nogil

    void compute_block_norms_ "compute_block_norms"(
        int32_t bs, int32_t n, double *blocks, double *norms) nogil


def block_incomplete_cholesky_factor(cnp.ndarray[int32_t, mode='fortran'] indptr,
                                     cnp.ndarray[int32_t, mode='fortran'] indices,
                                     cnp.ndarray[double, ndim=3, mode='fortran'] blocks,
                                     double alpha=0):
    cdef int32_t bs = blocks.shape[0], n = indptr.shape[0] - 1, info
    if blocks.shape[1] != blocks.shape[0]:
        raise ValueError('invalid shape on blocks arg')
    with nogil:
        block_incomplete_cholesky_factor_(bs, n, &indptr[0], &indices[0], &blocks[0,0,0],
                                          alpha, &info)
    if info != 0:
        raise NotPosDefError("DPOTRF or DPOSV failed with code %d" % info)


def probe_cholesky_ridging(cnp.ndarray[int32_t, mode='fortran'] indptr,
                           cnp.ndarray[int32_t, mode='fortran'] indices,
                           cnp.ndarray[double, ndim=3, mode='fortran'] blocks,
                           double ridge,
                           double eps_log10=0.1):
    """Retries block_incomplete_cholesky_factor until the lowest term to add to the diagonal
    for success is found (using binary search); continues until `highest-lowest/lowest < eps).

    Returns (alpha, ncalls). Does NOT alter input blocks, you should call
    block_incomplete_cholesky_factor again with the given alpha, usually after adding a bit extra.
    """
    cdef cnp.ndarray[double, ndim=3, mode='fortran'] out_blocks
    cdef double lower, upper, mid
    upper = np.max(blocks) + ridge
    ncalls = 0

    # May be that we succeed right away...
    try:
        out_blocks = blocks.copy('F')
        ncalls += 1
        block_incomplete_cholesky_factor(indptr, indices, out_blocks, alpha=ridge)
    except NotPosDefError:
        pass
    else:
        return ridge, ncalls

    # OK, validate the highest end of the bracket indeed works
    try:
        out_blocks = blocks.copy('F')
        ncalls += 1
        block_incomplete_cholesky_factor(indptr, indices, out_blocks, alpha=upper)
    except NotPosDefError:
        raise NotImplementedError('Did not work even by adding the maximum matrix val to diagonal')

    # Binary search, using geometric mean. Use log10 to match eps given.
    lower = upper * 1e-6
    lower = log10(lower)
    upper = log10(upper)
    while upper - lower > eps_log10:
        mid = .5 * (lower + upper)
        out_blocks = blocks.copy('F')
        try:
            success = False
            ncalls += 1
            block_incomplete_cholesky_factor(indptr, indices, out_blocks, alpha=pow(10, mid))
        except NotPosDefError:
            print '%.2e' % pow(10, mid), 'Failed'
            lower = mid
        else:
            print '%.2e' % pow(10, mid), 'Success'
            success = True
            upper = mid
    return pow(10, upper), ncalls


def block_triangular_solve(transpose,
                           cnp.ndarray[int32_t, mode='fortran'] indptr,
                           cnp.ndarray[int32_t, mode='fortran'] indices,
                           cnp.ndarray[double, ndim=3, mode='fortran'] blocks,
                           cnp.ndarray[double, ndim=2, mode='fortran'] x):
    cdef int32_t bs = blocks.shape[0], n = indptr.shape[0] - 1, info
    cdef int32_t trans_int
    if not (blocks.shape[0] == blocks.shape[1] == x.shape[0]):
        raise ValueError('invalid shape on blocks arg')
    if transpose == 'N':
        trans_int = 0
    elif transpose == 'T':
        trans_int = 1
    else:
        raise ValueError('transpose not in ("N", "T")')
    with nogil:
        block_triangular_solve_(trans_int, bs, n, &indptr[0], &indices[0], &blocks[0,0,0], &x[0,0])


def block_At_D_B(A, B, cnp.ndarray[double, ndim=2, mode='fortran'] D, C):
    """
    C = C + A^T D B

    A, B, C are BlockMatrix
    """

    cdef int32_t bs_left = A.blocks.shape[1]
    cdef int32_t bs_mid  = B.blocks.shape[0]
    cdef int32_t bs_right = B.blocks.shape[1]
    cdef int32_t A_n = A.indptr.shape[0] - 1
    cdef int32_t B_n = B.indptr.shape[0] - 1

    if A.blocks.shape[:2] != (bs_mid, bs_left):
         raise ValueError()
    if B.blocks.shape[:2] != (bs_mid, bs_right):
         raise ValueError()
    if C.blocks.shape[:2] != (bs_left, bs_right):
        raise ValueError()
    if (<object>D).shape[:2] != (bs_mid, A_n):
        raise ValueError()
 
    cdef cnp.ndarray[double, ndim=3, mode='fortran'] A_blocks, B_blocks, C_blocks
    cdef cnp.ndarray[int32_t, ndim=1, mode='fortran'] A_indices, A_indptr, A_labels, B_indices, B_indptr, B_labels, C_indices, C_indptr

    A_indptr = A.indptr; A_indices = A.indices; A_labels = A.labels; A_blocks = A.blocks;
    B_indptr = B.indptr; B_indices = B.indices; B_labels = B.labels; B_blocks = B.blocks;
    C_indptr = C.indptr; C_indices = C.indices; C_blocks = C.blocks;



    with nogil:
        block_At_D_B_(bs_left, bs_mid, bs_right,
                      A_n, &A_indptr[0], &A_indices[0], &A_labels[0], &A_blocks[0,0,0],
                      B_n, &B_indptr[0], &B_indices[0], &B_labels[0], &B_blocks[0,0,0],
                      &C_indptr[0], &C_indices[0], &C_blocks[0,0,0], &D[0, 0])
    return C


def block_At_D_A(cnp.ndarray[int32_t, mode='fortran'] A_indptr,
                 cnp.ndarray[int32_t, mode='fortran'] A_indices,
                 cnp.ndarray[double, ndim=3, mode='fortran'] A_blocks,
                 cnp.ndarray[double, ndim=2, mode='fortran'] D,
                 cnp.ndarray[int32_t, mode='fortran'] C_indptr,
                 cnp.ndarray[int32_t, mode='fortran'] C_indices,
                 cnp.ndarray[double, ndim=3, mode='fortran'] C_blocks=None,
                 cnp.ndarray[int32_t, mode='fortran'] A_labels=None):
    cdef int32_t bs_left = A_blocks.shape[0], bs_right = A_blocks.shape[1], n = A_indptr.shape[0] - 1
    if A_labels is None:
        A_labels = np.arange(A_indptr[-1], dtype=np.int32)
    if C_blocks is None:
        C_blocks = np.zeros((bs_right, bs_right, C_indices.shape[0]), np.double, order='F')
    with nogil:
        block_At_D_A_(bs_left, bs_right, n, &A_indptr[0], &A_indices[0], &A_labels[0], &A_blocks[0,0,0],
                      &D[0, 0], &C_indptr[0], &C_indices[0], &C_blocks[0,0,0])
    return C_blocks


def block_A_x(cnp.ndarray[int32_t, mode='fortran'] A_indptr,
              cnp.ndarray[int32_t, mode='fortran'] A_indices,
              cnp.ndarray[double, ndim=3, mode='fortran'] A_blocks,
              cnp.ndarray[double, ndim=1, mode='fortran'] x,
              cnp.ndarray[double, ndim=1, mode='fortran'] y=None,
              cnp.ndarray[int32_t, mode='fortran'] A_labels=None,
              transpose=False):
    cdef int32_t bs_left = A_blocks.shape[0], bs_right = A_blocks.shape[1], n = A_indptr.shape[0] - 1
    cdef int32_t transint = int(transpose)
    if A_labels is None:
        A_labels = np.arange(A_indptr[-1], dtype=np.int32)
    if y is None:
        y = np.zeros((bs_right * n) if transpose else (bs_left * n), np.double, order='F')
    if x.shape[0] != ((bs_left * n) if transpose else (bs_right * n)):
        raise ValueError('x has wrong shape')
    with nogil:
        block_A_x_(transint,
                   bs_left, bs_right, n, &A_indptr[0], &A_indices[0], &A_labels[0],
                   &A_blocks[0,0,0], &x[0], &y[0])
    return y


def block_diagonal_factor(cnp.ndarray[double, ndim=3, mode='fortran'] blocks):
    cdef int32_t info
    if blocks.shape[0] != blocks.shape[1]:
        raise ValueError("Illegal shape")
    with nogil:
        block_diagonal_factor_(blocks.shape[0], blocks.shape[2], &blocks[0,0,0], &info)
    if info != 0:
        raise Exception("SPOTRF code: %d" % info)


def block_diagonal_solve(cnp.ndarray[double, ndim=3, mode='fortran'] blocks,
                         cnp.ndarray[double, ndim=2, mode='fortran'] x):
    if (blocks.shape[0] != blocks.shape[1] or
        blocks.shape[0] != x.shape[0] or
        blocks.shape[2] != x.shape[1]):
        raise ValueError("Illegal shape or does not conform")
    with nogil:
        block_diagonal_solve_(blocks.shape[0], blocks.shape[2], &blocks[0,0,0], &x[0,0])


def blocks_to_dense(indptr, indices, blocks, mirror=False):
    n = indptr.shape[0] - 1
    bs_left = blocks.shape[0]
    bs_right = blocks.shape[1]
    if mirror and bs_left != bs_right:
        raise ValueError('cannot mirror non-symmetric matrix')
    out = np.zeros((n*bs_left,n*bs_right))
    for j in range(n):
        for iptr in range(indptr[j], indptr[j + 1]):
            i = indices[iptr]

            if not mirror or i != j:
                b = blocks[:, :, iptr]
            else:
                b = np.tril(blocks[:, :, iptr]) + np.tril(blocks[:, :, iptr], -1).T
            out[i * bs_left:(i+1)*bs_left,j*bs_right:(j+1)*bs_right] = b
            if mirror and i != j:
                assert bs_left == bs_right
                out[j * bs_left:(j+1)*bs_left,i*bs_left:(i+1)*bs_left] = b.T
    return out

def csc_to_factored_compressed_block_diagonal(A,
                                              cnp.ndarray[int32_t, mode='fortran'] cluster_offsets,
                                              cnp.ndarray[int32_t, mode='fortran'] permutation,
                                              factor=True, double startridge=1e-3):
    cdef int32_t bs = A.blocks.shape[0]
    cluster_sizes = cluster_offsets[1:] - cluster_offsets[:-1]
    cdef int32_t bufsize = ((cluster_sizes * bs * ((cluster_sizes * bs) + 1)) // 2).sum()

    cdef int32_t info
    cdef int32_t nofactor = 0 if factor else 1
    cdef cnp.ndarray[double, mode='fortran'] out = np.empty(bufsize, dtype=np.double)

    if permutation.shape[0] != A.indptr.shape[0] - 1:
        raise ValueError('permutation array and A does not match')
    if A.blocks.shape[0] != A.blocks.shape[1]:
        raise ValueError('not a square block size, does not make sense to Cholesky')

    cdef cnp.ndarray[double, ndim=3, mode='fortran'] A_blocks
    cdef cnp.ndarray[int32_t, ndim=1, mode='fortran'] A_indices, A_indptr
    A_indptr = A.indptr; A_indices = A.indices; A_blocks = A.blocks;

    with nogil:
        csc_to_factored_compressed_block_diagonal_(
            bs=bs,
            n=A_indptr.shape[0] - 1,
            indptr=&A_indptr[0],
            indices=&A_indices[0],
            blocks=&A_blocks[0,0,0],
            cluster_count=cluster_offsets.shape[0] - 1,
            cluster_offsets=&cluster_offsets[0],
            permutation=&permutation[0],
            out=&out[0],
            nofactor=nofactor,
            startridge=startridge,
            info=&info)
    if info != 0:
        raise Exception('csc_to_factored_compressed_block_diagonal error code: %d' % info)
    return out

def csc_compressed_block_diagonal_solve(cnp.ndarray[int32_t, mode='fortran'] cluster_offsets,
                                        cnp.ndarray[int32_t, mode='fortran'] permutation,
                                        cnp.ndarray[double, mode='fortran'] matrix,
                                        cnp.ndarray[double, mode='fortran'] x):
    nblocks = cluster_offsets[-1]
    if x.shape[0] % nblocks != 0:
        raise ValueError()
    if permutation.shape[0] != nblocks:
        raise ValueError('permutation array and x shape does not match')
    cdef int32_t bs = x.shape[0] // nblocks

    with nogil:
        csc_compressed_block_diagonal_solve_(
            bs=bs,
            cluster_count=cluster_offsets.shape[0] - 1,
            cluster_offsets=&cluster_offsets[0],
            permutation=&permutation[0],
            matrix=&matrix[0],
            x=&x[0])
    return x

def csc_make_clusters(double eps, object csc_norm_matrix):
    cdef cnp.ndarray[int32_t, mode='fortran'] indptr = csc_norm_matrix.indptr
    cdef cnp.ndarray[int32_t, mode='fortran'] indices = csc_norm_matrix.indices
    cdef cnp.ndarray[double, mode='fortran'] norms = csc_norm_matrix.data
    cdef int32_t n = indptr.shape[0] - 1

    cdef cnp.ndarray[int32_t, mode='fortran'] permutation = np.empty(n, dtype=np.int32)
    cdef cnp.ndarray[int32_t, mode='fortran'] cluster_size = np.empty(n, dtype=np.int32)
    cdef int32_t cluster_count

    with nogil:
        csc_make_clusters_(eps, n, &indptr[0], &indices[0], &norms[0],
                           &permutation[0], &cluster_size[0], &cluster_count)
    return permutation, cluster_size[:cluster_count]

def compute_block_norms(cnp.ndarray[double, ndim=3, mode='fortran'] blocks):
    cdef cnp.ndarray[double, mode='fortran'] norms = np.empty(blocks.shape[2])
    if blocks.shape[0] != blocks.shape[1]:
        raise ValueError()
    with nogil:
        compute_block_norms_(blocks.shape[0], blocks.shape[2], &blocks[0,0,0], &norms[0])
    return norms

