
from libc.stdint cimport int32_t, int8_t, int64_t
from libc.math cimport sqrt, log10, pow
import numpy as np
cimport numpy as cnp
cimport cython

def k_kp_idx(k, kp):
    k, kp = max(k, kp), min(k, kp)
    return ((k+1)*(k))/2 + kp


cdef extern:
     void compsep_assemble_U_ "compsep_assemble_u"(
         int32_t nobs, int32_t ncomp, int32_t lmax, int32_t *lmax_per_comp, double *mixing_scalars,
         double *bl, double *wl, double *dl, double *alpha, double *U) nogil
     void compsep_apply_U_block_diagonal_ "compsep_apply_u_block_diagonal"(
         int32_t nobs, int32_t ncomp, int32_t lmax, int32_t transpose,
         double *blocks, double *x) nogil


def compsep_assemble_U(lmax_per_comp, mixing_scalars, bl, wl, dl, alpha):
    cdef int32_t nobs = bl.shape[0]
    cdef int32_t ncomp = wl.shape[0]
    cdef int32_t lmax = wl.shape[1] - 1
    if alpha.shape[0] != nobs:
        raise ValueError()
    if bl.shape[1] != lmax + 1:
        raise ValueError()
    if mixing_scalars.shape != (nobs, ncomp):
        raise ValueError()
    U = np.zeros((nobs + ncomp, ncomp, lmax + 1), dtype=np.double, order='F')

    cdef cnp.ndarray[double, ndim=3, mode='fortran'] U_ = U
    cdef cnp.ndarray[int32_t, ndim=1, mode='fortran'] lmax_per_comp_ = np.asarray(lmax_per_comp, dtype=np.int32, order='F')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] mixing_scalars_ = mixing_scalars
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] bl_ = bl
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] wl_ = wl
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] dl_ = dl
    cdef cnp.ndarray[double, ndim=1, mode='fortran'] alpha_ = alpha

    if lmax_per_comp_.shape[0] != ncomp:
        raise ValueError()

    with nogil:
        compsep_assemble_U_(
            nobs, ncomp, lmax, &lmax_per_comp_[0], &mixing_scalars_[0, 0],
            &bl_[0, 0], &wl_[0, 0], &dl_[0, 0], &alpha_[0], &U_[0, 0, 0])
    return U


def compsep_apply_U_block_diagonal(int32_t lmax, blocks, x, transpose):
    cdef cnp.ndarray[double, ndim=3, mode='fortran'] blocks_ = blocks
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] x_ = x
    cdef int32_t trans_c = (1 if transpose else 0)
    if x.shape[0] != (lmax + 1)**2:
        raise ValueError()
    if x.shape[1] != blocks.shape[0]:
        raise ValueError()
    if blocks_.shape[1] >= blocks.shape[0]:
        raise ValueError()
    with nogil:
        compsep_apply_U_block_diagonal_(
            blocks_.shape[0] - blocks_.shape[1], blocks_.shape[1], lmax, trans_c, &blocks_[0, 0, 0], &x_[0, 0])