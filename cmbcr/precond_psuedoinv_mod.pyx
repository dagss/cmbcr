
from libc.stdint cimport int32_t, int8_t, int64_t
from libc.math cimport sqrt, log10, pow
import numpy as np
cimport numpy as cnp
cimport cython

def k_kp_idx(k, kp):
    k, kp = max(k, kp), min(k, kp)
    return ((k+1)*(k))/2 + kp


cdef extern:
     void compsep_apply_U_block_diagonal_ "compsep_apply_U_block_diagonal"(
         int32_t nobs, int32_t ncomp, int32_t lmax,
         float *blocks, float *x_comp, float *x_obs, char transpose) nogil

def compsep_apply_U_block_diagonal(int32_t lmax, blocks, x_comp, x_obs, transpose):
    cdef cnp.ndarray[float, ndim=3, mode='fortran'] blocks_ = blocks
    cdef cnp.ndarray[float, ndim=2, mode='fortran'] x_comp_ = x_comp
    cdef cnp.ndarray[float, ndim=2, mode='fortran'] x_obs_ = x_obs
    cdef char trans_c
    if transpose:
        trans_c = 'T'
    else:
        trans_c = 'N'
    with nogil:
        compsep_apply_U_block_diagonal_(
            blocks_.shape[0], blocks_.shape[1], lmax, &blocks_[0, 0, 0], &x_comp_[0, 0], &x_obs_[0, 0], trans_c)