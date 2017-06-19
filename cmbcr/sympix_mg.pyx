from libc.stdint cimport int32_t, int8_t, int64_t
from libc.math cimport sqrt, log10, pow
import numpy as np
cimport numpy as cnp
cimport cython

cdef extern:
    void sharp_legendre_transform_recfac(double *recfac, ptrdiff_t lmax)
    void rescale_bl "sympix_mg_rescale_bl"(double *bl, double *rescaled_bl, int32_t lmax)

    void compute_YDYt_block_ "sympix_mg_compute_YDYt_block"(
        int32_t n1, int32_t n2, double dphi1, double dphi2,
        double *thetas1, double *thetas2, double phi0_2,
        int32_t lmax, double *rescaled_bl, double *recfac,
        double *out)

    void compute_many_YDYt_blocks_ "sympix_mg_compute_many_YDYt_blocks"(
       int32_t nblocks,
       int32_t tilesize1, int32_t bandcount1, double *thetas1, int32_t *tilecounts1, int32_t *tileindices1,
       int32_t tilesize2, int32_t bandcount2, double *thetas2, int32_t *tilecounts2, int32_t *tileindices2,
       int32_t lmax, double *bl, int32_t *ierr, double *out_blocks) nogil


def compute_YDYt_block(
    cnp.ndarray[double, mode='c'] thetas1,
    cnp.ndarray[double, mode='c'] thetas2,
    double dphi1, double dphi2, double phi0_D2,
    cnp.ndarray[double, mode='c'] bl):

    cdef int32_t lmax = bl.shape[0] - 1, n1 = thetas1.shape[0], n2 = thetas2.shape[0]
    cdef cnp.ndarray[double, mode='c'] rescaled_bl = np.empty_like(bl)
    cdef cnp.ndarray[double, mode='c'] recfac = np.empty_like(bl)
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] out = np.empty((n1 * n1, n2 * n2), np.double,
                                                                   order='F')

    rescale_bl(&bl[0], &rescaled_bl[0], lmax)
    sharp_legendre_transform_recfac(&recfac[0], lmax)
    compute_YDYt_block_(n1, n2, dphi1, dphi2, &thetas1[0], &thetas2[0], phi0_D2,
                        lmax, &rescaled_bl[0], &recfac[0], &out[0, 0])
    return out

def compute_many_YDYt_blocks(grid1, grid2,
                             cnp.ndarray[double, mode='c'] bl,
                             cnp.ndarray[int32_t, mode='c'] indices1,
                             cnp.ndarray[int32_t, mode='c'] indices2):
    cdef int32_t lmax = bl.shape[0] - 1, nblocks = indices1.shape[0], ierr
    cdef cnp.ndarray[double, ndim=3, mode='fortran'] out = (
        np.empty((grid1.tilesize**2, grid2.tilesize**2, nblocks), np.double, order='F'))
    cdef cnp.ndarray[double, ndim=1, mode='c'] thetas1 = grid1.thetas, thetas2 = grid2.thetas
    cdef cnp.ndarray[int32_t, ndim=1, mode='c'] tile_counts1 = grid1.tile_counts
    cdef cnp.ndarray[int32_t, ndim=1, mode='c'] tile_counts2 = grid2.tile_counts

    if indices1.shape[0] != indices2.shape[0]:
        raise ValueError()
    compute_many_YDYt_blocks_(
        nblocks,
        grid1.tilesize, grid1.band_pair_count, &thetas1[0], &tile_counts1[0], &indices1[0],
        grid2.tilesize, grid2.band_pair_count, &thetas2[0], &tile_counts2[0], &indices2[0],
        lmax, &bl[0], &ierr, &out[0, 0, 0])
    if ierr != 0:
        msg = 'unknown'
        if ierr == 1:
            msg = 'Illegal pixel index'
        raise Exception(msg)
    return out


def compute_single_diagonal_YDYt_block(cnp.ndarray[double, mode='c'] thetas,
                                       cnp.ndarray[double, mode='c'] bl,
                                       int32_t tile_count):
    cdef int32_t lmax = bl.shape[0] - 1
    cdef int32_t tilesize = thetas.shape[0]
    cdef int32_t zero_index = 0
    cdef int32_t ierr
    cdef cnp.ndarray[double, ndim=2, mode='fortran'] out = (
        np.empty((tilesize**2, tilesize**2), np.double, order='F'))
    
    with nogil:
        compute_many_YDYt_blocks_(
            1,
            tilesize, 1, &thetas[0], &tile_count, &zero_index,
            tilesize, 1, &thetas[0], &tile_count, &zero_index,
            lmax, &bl[0], &ierr, &out[0, 0])
    if ierr != 0:
        msg = 'unknown'
        if ierr == 1:
            msg = 'Illegal pixel index'
        raise Exception(msg)
    return out
