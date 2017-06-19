from libc.stdint cimport int32_t, int8_t, int64_t
from libc.math cimport sqrt, log10, pow
import numpy as np
cimport numpy as cnp
cimport cython

def k_kp_idx(k, kp):
    k, kp = max(k, kp), min(k, kp)
    return ((k+1)*(k))/2 + kp


cdef extern:
     void construct_banded_preconditioner_ "construct_banded_preconditioner"(
          int32_t lmax, int32_t ncomp, int32_t ntheta, double *thetas,
          double complex *phase_map, double *mixing_scalars, double *bl, float *out) nogil
     void factor_banded_preconditioner_ "factor_banded_preconditioner"(int32_t lmax, int32_t ncomp, float *data, int32_t *info) nogil
     void solve_banded_preconditioner_ "solve_banded_preconditioner"(int32_t lmax, int32_t ncomp, float *data, float *x) nogil

     void compute_real_Yh_D_Y_block_on_diagonal_ "compute_real_yh_d_y_block_on_diagonal"(int32_t m, int32_t lmax,
         int32_t ntheta, double *thetas, double complex *phase_map, double *out) nogil


def compute_real_Yh_D_Y_block_on_diagonal(
        int32_t m, int32_t lmax,
        cnp.ndarray[double, ndim=1, mode='fortran'] thetas,
        cnp.ndarray[double complex, ndim=2, mode='fortran'] phase_map):
    nl = (lmax + 1 - m) if m == 0 else 2 * (lmax + 1 - m)
    out = np.zeros((nl, nl), order='fortran')
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] out_ = out

    if phase_map.shape[0] != 2 * lmax + 1 or phase_map.shape[1] != thetas.shape[0]:
        raise ValueError()
    with nogil:
        compute_real_Yh_D_Y_block_on_diagonal_(m, lmax, thetas.shape[0], &thetas[0], &phase_map[0,0], &out_[0,0])
    return out


def construct_banded_preconditioner(
        int32_t lmax,
        int32_t ncomp,
        cnp.ndarray[double, ndim=1, mode='fortran'] thetas,
        cnp.ndarray[double complex, ndim=2, mode='fortran'] phase_map,
        cnp.ndarray[double, ndim=1, mode='fortran'] mixing_scalars,
        cnp.ndarray[double, ndim=1, mode='fortran'] bl,
        out=None):
    cdef cnp.ndarray[float, ndim=2, mode='fortran'] out_
    if out is None:
        out = np.zeros((5 * ncomp, ncomp * (lmax + 1)**2), dtype=np.float32, order='F')
    else:
        if out.shape[0] != 5 * ncomp or out.shape[1] != ncomp * (lmax + 1)**2:
            raise ValueError()
    out_ = out
    if mixing_scalars.shape[0] != ncomp:
        raise ValueError('dl wrong shape')
    with nogil:
        construct_banded_preconditioner_(lmax, ncomp, thetas.shape[0], &thetas[0], &phase_map[0, 0], &mixing_scalars[0], &bl[0], &out_[0, 0])
    return out


def factor_banded_preconditioner(int32_t lmax, int32_t ncomp, cnp.ndarray[float, ndim=2, mode='fortran'] data):
    cdef int32_t info
    with nogil:
        factor_banded_preconditioner_(lmax, ncomp, &data[0, 0], &info)
    if info != 0:
        raise ValueError('factor_banded_preconditioner: SPBTRF error: %d' % info)
    return data


def solve_banded_preconditioner(int32_t lmax, int32_t ncomp, cnp.ndarray[float, ndim=2, mode='fortran'] data, x):
    cdef cnp.ndarray[float, ndim=2, mode='fortran'] x_ = x
    with nogil:
        solve_banded_preconditioner_(lmax, ncomp, &data[0, 0], &x_[0,0])
    return x
