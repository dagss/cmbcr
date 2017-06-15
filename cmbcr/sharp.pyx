#cython: embedsignature=True
from __future__ import division

"""
:mod:`commander.sphere.sharp`
----------------------------

Interface to libsharp

"""
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cimport cython

cimport numpy as cnp
import numpy as np

from cpython.pycapsule cimport PyCapsule_New

from .mmajor cimport lm_to_idx
from .mmajor import lm_count, find_lmax
from . import healpix

#cdef extern from "mpi.h":
#    ctypedef long long MPI_Comm

cdef extern from "stddef.h":
    ctypedef int ptrdiff_t

cdef extern from "sharp.h":

    ctypedef struct sharp_alm_info:
        ptrdiff_t *mvstart

    ctypedef struct sharp_ringinfo:
        double theta, phi0, weight, cth, sth
        ptrdiff_t ofs
        int nph, stride

    ctypedef struct sharp_ringpair:
        sharp_ringinfo r1, r2

    ctypedef struct sharp_geom_info:
        sharp_ringpair *pair
        int npairs

#    ctypedef struct sharp_sympix_info:
#        pass

    ctypedef struct sharpd_joblist:
        pass

    void sharp_make_alm_info (int lmax, int mmax, int stride,
                             ptrdiff_t *mvstart, sharp_alm_info **alm_info)

    void sharp_make_geom_info (int nrings, int *nph, ptrdiff_t *ofs,
                              int *stride, double *phi0, double *theta,
                              double *wgt, sharp_geom_info **geom_info)

    void sharp_destroy_alm_info(sharp_alm_info *info)
    void sharp_destroy_geom_info(sharp_geom_info *info)

    ctypedef enum sharp_jobtype:
        SHARP_YtW
        SHARP_Yt
        SHARP_WY
        SHARP_Y

    ctypedef enum:
        SHARP_DP
        SHARP_ADD


    void sharp_execute(sharp_jobtype type_,
                       int spin,
                       void *alm,
                       void *map,
                       sharp_geom_info *geom_info,
                       sharp_alm_info *alm_info,
                       int ntrans,
                       int flags,
                       double *time,
                       unsigned long long *opcnt)

cdef extern from "sharp_geomhelpers.h":

    void sharp_make_gauss_geom_info (int nrings, int nphi, double phi0,
      int stride_lon, int stride_lat, sharp_geom_info **geom_info)

cdef extern from "sharp_almhelpers.h":

    void sharp_make_mmajor_real_packed_alm_info (int lmax, int stride, int nm, int *ms, sharp_alm_info **alm_info)

#    void sharp_make_sympix_geom_info (int nbands_half, int tilesize, sharp_sympix_info **sympix_info,
#                                      sharp_geom_info **geom_info)

#def sympix_geom_info(nbands_half, tilesize):
#    cdef sharp_sympix_info* sympix_info
#    cdef sharp_geom_info* geom_info
#    sharp_make_sympix_geom_info(nbands_half, tilesize, &sympix_info, &geom_info)

## cdef extern from "sharp_mpi.h":
##     void sharp_execute_mpi(MPI_Comm comm,
##                            sharp_jobtype type_,
##                            int spin,
##                            void *alm,
##                            void *map,
##                            sharp_geom_info *geom_info,
##                            sharp_alm_info *alm_info,
##                            int ntrans,
##                            int flags,
##                            double *time,
##                            unsigned long long *opcnt)



if sizeof(ptrdiff_t) == 4:
    ptrdiff_dtype = np.int32
elif sizeof(ptrdiff_t) == 8:
    ptrdiff_dtype = np.int64
else:
    assert False

cdef double sqrt_one_half = np.sqrt(.5), sqrt_two = np.sqrt(2)
cdef double pi = np.pi

def _mirror_weights(weights, nside, nrings, ring_start, ring_stop):
    """
    Take weights provided for the northern hemisphere and mirror
    them to the southern as well.
    """
    if weights is None:
        return None
    weights = np.asarray(weights)
    if weights.shape[0] != 2 * nside:
        raise ValueError('len(weights) != 2 * nside')
    new_weights = np.empty(nrings)
    new_weights[:(nrings + 1) // 2] = weights[ring_start:ring_stop]
    new_weights[nrings // 2:] = weights[ring_start:ring_stop][::-1]
    return new_weights

cdef sharp_jobtype str_to_jobtype(jobtype):
    if jobtype == 'Y':
        return SHARP_Y
    elif jobtype == 'Yt':
        return SHARP_Yt
    elif jobtype == 'YtW':
        return SHARP_YtW
    elif jobtype == 'WY':
        return SHARP_WY

cdef class BaseRealMmajorPlan:
    cdef sharp_alm_info *alm_info
    cdef sharp_geom_info *geom_info
    cdef readonly int lmax, mmin, mmax, nsh_local, npix_local, npix_global
#    cdef MPI_Comm comm
    cdef bint use_mpi

    def __init__(self, int lmax, mmin=0, mmax=None, object comm=None):
        # Subclasses should set up geom_info and npix_local!

        # mstart is the "hypothetical" a_{0m}. It's copied by
        # sharp_make_alm_info, so we only need it in constructor
        if mmax is None:
            mmax = lmax

        cdef ptrdiff_t[::1] mvstart = np.empty(mmax - mmin + 1, dtype=ptrdiff_dtype)
        cdef int[::1] ms = np.arange(mmin, mmax + 1, dtype=np.intc)
        cdef int m

        # Set up alm_info
        if mmin != 0:
            raise NotImplementedError()

        sharp_make_mmajor_real_packed_alm_info(lmax, 1, mmax + 1, NULL, &self.alm_info)
        self.lmax = lmax
        self.mmin = mmin
        self.mmax = mmax
        self.nsh_local = lm_count(0, lmax, mmin, mmax)
        self.npix_local = -1 # subclasses
        self.npix_global = -1 # subclasses
        self.use_mpi = (comm is not None)
#        if self.use_mpi:
#            from . import _mpi4py_bridge
#            capsule = PyCapsule_New(&self.comm, "_mpi4py_bridge_MPI_Comm", NULL)
#            _mpi4py_bridge.mpi4py_Comm_to_handle(comm, capsule)
#        else:
#        self.comm = 0

    def __dealloc__(self):
        if self.alm_info != NULL:
            sharp_destroy_alm_info(self.alm_info)
        if self.geom_info != NULL:
            sharp_destroy_geom_info(self.geom_info)

    cdef _execute(self, sharp_jobtype jobtype, void *alm, void *map):
        if self.use_mpi:
            raise NotImplementedError()
            ## sharp_execute_mpi(self.comm, jobtype, 0, &alm, &map, self.geom_info,
            ##                   self.alm_info, 1,
            ##                   SHARP_DP | SHARP_REAL_HARMONICS,
            ##                   NULL, NULL)
        else:
            sharp_execute(jobtype, 0, &alm, &map, self.geom_info,
                          self.alm_info, 1,
                          SHARP_DP,
                          NULL, NULL)

    def adjoint_synthesis(self, map, out=None):
        return self.analysis(map, out, jobtype='Yt')

    def adjoint_analysis(self, alm, out=None):
        return self.synthesis(alm, out, jobtype='WY')

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def synthesis(self, double[::1] alm, double[::1] out=None, jobtype='Y'):
        cdef ptrdiff_t i, i_start
        cdef ptrdiff_t lmax = self.lmax
        cdef ptrdiff_t nsh = self.nsh_local

        if self.geom_info == NULL:
            raise NotImplementedError('subclass did not initialize self.geom_info')
        if jobtype not in ('Y', 'WY'):
            raise ValueError('Invalid jobtype')

        # Shape validation
        if alm.shape[0] != nsh:
            raise ValueError('alm.shape does not match lmax')
        if out is None:
            out = np.zeros(self.npix_local, np.double) * np.nan
        elif out.shape[0] != self.npix_local:
            raise ValueError('map has wrong number of pixels')
        # Finally, do the SHT...
        self._execute(str_to_jobtype(jobtype), &alm[0], &out[0])
        return np.asarray(out)

    def analysis(self, double[::1] map, double[::1] out=None, jobtype='YtW'):
        cdef ptrdiff_t i, i_start
        cdef ptrdiff_t lmax = self.lmax
        cdef ptrdiff_t nsh = self.nsh_local

        if self.geom_info == NULL:
            raise NotImplementedError('subclass did not initialize self.geom_info')
        if jobtype not in ('YtW', 'Yt'):
            raise ValueError('Invalid jobtype')

        # Shape validation
        if map.shape[0] != self.npix_local:
            raise ValueError('map has wrong number of pixels')
        if out is None:
            out = np.zeros(nsh, np.double) * np.nan
        elif out.shape[0] != nsh:
            raise ValueError('out.shape does not match lmax')
        self._execute(str_to_jobtype(jobtype), &out[0], &map[0])
        return np.asarray(out)

cdef class RealMmajorHealpixPlan(BaseRealMmajorPlan):
    cdef readonly int nside

    def __init__(self, nside, lmax,
                 cnp.ndarray[double, mode='c'] weights=None,
                 comm=None, mmin=0, mmax=None, ring_start=None, ring_stop=None):

        BaseRealMmajorPlan.__init__(self, lmax, mmin, mmax, comm)

        # Set up geom_info
        if ring_start is None:
            ring_start = 0
        if ring_stop is None:
            ring_stop = 2 * nside
        # Since we may only be using a subset of the rings, we make our own rather than
        # using sharp_make_healpix_geom_info.
        cdef int[:] irings = healpix.get_irings(nside, ring_start, ring_stop)
        cdef int nrings = irings.shape[0]
        cdef int[:] pixel_counts = healpix.get_ring_pixel_counts(nside, irings)
        cdef ptrdiff_t[:] offsets = np.hstack([[0], np.cumsum(pixel_counts)[:-1]]).astype(ptrdiff_dtype)
        cdef double[:] phi0 = healpix.get_phi0(nside, irings)
        cdef double[:] theta = healpix.get_ring_theta(nside, irings)

        weights = _mirror_weights(weights, nside, nrings, ring_start, ring_stop)

        cdef int[:] strides = np.ones(nrings, dtype=np.intc)
        sharp_make_geom_info(nrings, &pixel_counts[0], &offsets[0], &strides[0],
                             &phi0[0], &theta[0],
                             &weights[0] if weights is not None else NULL,
                             &self.geom_info)

        self.nside = nside
        self.npix_local = healpix.get_npix(nside, ring_start, ring_stop)
        self.npix_global = 12 * nside * nside

cdef class RealMmajorGridPlan(BaseRealMmajorPlan):
    """
    Plan with constant nphi and phi0=0; just pass in theta and weights.
    """
    cdef readonly int nrings, nphi

    def __init__(self, double[::1] theta, nphi, lmax,
                 cnp.ndarray[double, mode='c'] weights=None):
        BaseRealMmajorPlan.__init__(self, lmax, 0, lmax, None)

        # TODO MPI
        cdef int nrings = theta.shape[0]
        self.npix_local = nphi * nrings
        self.npix_global = nphi * nrings
        self.nrings = nrings
        self.nphi = nphi

        cdef int[:] pixel_counts = np.empty(nrings, dtype=np.intc)
        pixel_counts[:] = nphi
        cdef ptrdiff_t[:] offsets = np.hstack([[0], np.cumsum(pixel_counts)[:-1]]).astype(ptrdiff_dtype)
        cdef double[:] phi0 = np.zeros(nrings)

        cdef int[:] strides = np.ones(nrings, dtype=np.intc)
        sharp_make_geom_info(nrings, &pixel_counts[0], &offsets[0], &strides[0],
                             &phi0[0], &theta[0],
                             &weights[0] if weights is not None else NULL,
                             &self.geom_info)

cdef class SymPixGridPlan(BaseRealMmajorPlan):
    """
    SymPix grid; pass in a grid descriptor from the sympix module.
    """
    cdef readonly int nrings

    def __init__(self, grid, lmax):
        BaseRealMmajorPlan.__init__(self, lmax, 0, lmax, None)

        self.npix_local = grid.npix
        self.npix_global = grid.npix
        self.nrings = grid.nrings

        # Pass on information from the plan, but repeat all arrays so we
        # get both north and south halves

        tilesize = grid.tilesize

        # Following are set up in loop below
        cdef int[:] ring_lengths = np.zeros(self.nrings, np.intc)
        cdef ptrdiff_t[:] offsets = np.zeros(self.nrings + 1, ptrdiff_dtype)
        cdef double[::1] thetas = np.zeros(self.nrings, np.double)
        cdef double[::1] weights = np.zeros(self.nrings, np.double)
        cdef int[:] strides = np.zeros(self.nrings, dtype=np.intc)
        cdef double[:] phi0 = np.zeros(self.nrings)

        i = 0  # incremented as we go; index of ring as seen by libsharp
        for iband in range(grid.band_pair_count):
            # pixel offset of start of tile-band
            band_offset = grid.band_offsets[iband]
            ringlen = tilesize * grid.tile_counts[iband]

            # North half
            for iring in range(tilesize):  # ring within band
                iring_north = iband * tilesize + iring
                thetas[i] = grid.thetas[iring_north]
                weights[i] = grid.weights[iring_north]
                offsets[i] = band_offset + iring
                strides[i] = tilesize
                ring_lengths[i] = ringlen
                phi0[i] = grid.phi0s[iring_north]
                i += 1

            # South half -- we set offsets to *last* pixel in each ring and use a negative
            # stride in order to reverse phi on this hemisphere
            band_size = tilesize * ringlen
            for iring in range(tilesize):
                iring_north = iband * tilesize + iring
                thetas[i] = np.pi - grid.thetas[iring_north]
                weights[i] = grid.weights[iring_north]
                offsets[i] = band_offset + band_size + iring
                strides[i] = tilesize
                ring_lengths[i] = ringlen
                phi0[i] = grid.phi0s[iring_north]  # note, phi0 of 'last' pixel of ring, stride is negative
                i += 1

        offsets[-1] = grid.npix
        sharp_make_geom_info(self.nrings, &ring_lengths[0], &offsets[0], &strides[0],
                             &phi0[0], &thetas[0], &weights[0], &self.geom_info)
        # ^ copies all the values into memory allocated in self.geom_info, so it's OK for
        # our arrays to go out of scope now


def gauss_legendre_grid(lmax):
    from libsharp import legendre_roots
    nrings = lmax + 1
    nphi = 2 * nrings
    x, weights = legendre_roots(nrings)
    theta = np.arccos(-x)
    weights *= 2 * np.pi / nphi
    return theta, nphi, weights

def get_gauss_npix(lmax_grid):
    nrings = lmax_grid + 1
    nphi = 2 * nrings
    return nrings * nphi

cdef class RealMmajorGaussPlan(BaseRealMmajorPlan):
    cdef readonly int nrings, nphi

    cdef int npix
    def __init__(self, lmax_grid, lmax=None):
        if lmax is None:
            lmax = lmax_grid
        BaseRealMmajorPlan.__init__(self, lmax, 0, lmax, None)

        # TODO MPI
        self.nrings = lmax_grid + 1
        self.nphi = 2 * self.nrings
        self.npix_local = self.npix_global = self.nphi * self.nrings

        sharp_make_gauss_geom_info(self.nrings, self.nphi, 0.0, 1, self.nphi, &self.geom_info)

#
# HEALPix shorthands
#
def sh_synthesis(nside, alm, out=None):
    lmax = find_lmax(0, alm.shape[0])
    return RealMmajorHealpixPlan(nside, lmax).synthesis(alm, out)

def sh_adjoint_synthesis(lmax, map, out=None):
    nside = healpix.npix_to_nside(map.shape[0])
    return RealMmajorHealpixPlan(nside, lmax).adjoint_synthesis(map, out)

def sh_analysis(lmax, map, out=None):
    from .healpix_data import get_ring_weights_T
    nside = healpix.npix_to_nside(map.shape[0])
    weights = get_ring_weights_T(nside)
    return RealMmajorHealpixPlan(nside, lmax, weights=weights).analysis(map, out)

def sh_adjoint_analysis(nside, alm, out=None):
    from .healpix_data import get_ring_weights_T
    lmax = find_lmax(0, alm.shape[0])
    weights = get_ring_weights_T(nside)
    return RealMmajorHealpixPlan(nside, lmax, weights=weights).adjoint_analysis(alm, out)

#
# Gauss-Legendre shorthands
#


def sh_synthesis_gauss(lmax_grid, alm, out=None, lmax_sh=None):
    return RealMmajorGaussPlan(lmax_grid, lmax_sh).synthesis(alm, out)

def sh_analysis_gauss(lmax_grid, map, out=None, lmax_sh=None):
    return RealMmajorGaussPlan(lmax_grid, lmax_sh).analysis(map, out)

def sh_adjoint_analysis_gauss(lmax_grid, alm, out=None, lmax_sh=None):
    return RealMmajorGaussPlan(lmax_grid, lmax_sh).adjoint_analysis(alm, out)

def sh_adjoint_synthesis_gauss(lmax_grid, map, out=None, lmax_sh=None):
    return RealMmajorGaussPlan(lmax_grid, lmax_sh).adjoint_synthesis(map, out)
