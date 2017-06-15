#cython: embedsignature=True
"""
:mod:`commander.sphere.healpix`
-------------------------------

Utilities for working with the HEALPix grid.

"""
from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdint cimport int32_t, int8_t, int64_t

cimport healpix_lib

cdef double pi = np.pi

cdef int64_t imin(int64_t a, int64_t b):
    return a if a < b else b

cdef int64_t imax(int64_t a, int64_t b):
    return a if a > b else b


def healpix_pixel_size(int32_t nside):
    return np.sqrt(4 * pi / (12 * nside**2))

def is_power_of_two(n):
    return n & (n - 1) == 0

def get_ring_z(int32_t nside, int32_t[:] irings=None):
    """Returns the z-locations of rings in the HEALPix grid

    Parameters
    ----------

    nside : int
        Parameter of the HEALPix grid
    irings : int32_t[:], optional
        Which rings to find the locations of; by default, return
        all ``4 * nside - 1`` rings.

    """
    if irings is None:
        irings = np.arange(4 * nside - 1, dtype=np.int32)
    cdef double[:] z = np.empty(irings.shape[0], np.double)
    cdef int32_t i, iring, nrings

    nrings = 4 * nside - 1 # in total, not queried
    for i in range(irings.shape[0]):
        iring = irings[i]
        if iring < nside:
            # northern hemisphere
            z[i] = 1 - (iring + 1)**2 / (3 * nside**2)
        elif iring < 3 * nside:
            # middle band
            z[i] = (4/3) - (2 * (iring + 1)) / (3 * nside)
        else:
            # southern hemisphere
            iring = nrings - iring - 1
            z[i] = -1 + (iring + 1)**2 / (3 * nside**2)
    return np.asarray(z)

def get_ring_theta(nside, irings=None):
    """Returns the locations of rings in the HEALPix grid in `theta` (radians)

    Parameters
    ----------

    nside : int
        Parameter of the HEALPix grid
    irings : array of integers, optional
        Which rings to find the locations of; by default, return
        all ``4 * nside - 1`` rings.

    """
    return np.arccos(get_ring_z(nside, irings))

def npix_to_nside(npix):
    nside = int(np.sqrt(npix // 12))
    if 12 * nside**2 != npix:
        raise ValueError('%d not a valid number of HEALPix pixels')
    return nside

def nside_of(arr):
    return npix_to_nside(arr.shape[0])

cpdef int32_t get_npix_above(int32_t nside, int32_t ring_stop):
    """Returns number of pixels above `ring_stop` on northern hemisphere

    Note that the maximum value returned by this function is ``npix // 2``
    """
    # Number of pixels before ring ring_stop on northern hemisphere
    cdef int32_t cap_stop, cap_npix, ncenterband, band_npix
    cap_stop = imin(ring_stop, nside)
    cap_npix = 2 * (cap_stop + 1) * cap_stop
    ncenterband = imax(ring_stop - nside, 0)
    band_npix = ncenterband * 4 * nside
    return cap_npix + band_npix    

cpdef get_npix(int32_t nside, int32_t ring_start=0, int32_t ring_stop=-1):
    """Returns number of pixels in a (sliced) HEALPix grid.

    The `ring_start` and `ring_stop` arguments specify a slice of rings on
    the northern hemisphere; the dual rings on the southern is automatically
    included. Thus the total number of rings is either
    ``2 * (ring_stop - ring_start)`` or ``2 * (ring_stop - ring_start) - 1``,
    depending on whether ``ring_stop`` specifies the equatorial ring.

    Parameters
    ----------

    nside : int
        Parameter of HEALPix grid
    ring_start, ring_stop : int, optional
        When given, only include rings in the range ``ring_start,...,ring_stop-1``
        on the northern hemisphere, and the corresponding sibling rings on the
        southern hemisphere, in the pixel count.
    """
    if ring_stop == -1:
        ring_stop = 2 * nside
    if ring_start == 2 * nside:
        return 0

    if ring_start == 0 and ring_stop == 2 * nside:
        return 12 * nside**2

    cdef int32_t npix
    npix = get_npix_above(nside, ring_stop) - get_npix_above(nside, ring_start)
    npix *= 2 # include other hemisphere
    if ring_stop == 2 * nside:
        npix -= 4 * nside # compensate for the equatorial band
    return npix

def get_irings(int32_t nside, int32_t ring_start, int32_t ring_stop):
    """Return the ring indices for given `ring_start`, `ring_stop` arguments.

    This is not completely trivial because by `ring_start`, `ring_stop`
    one indicate pairs of rings on the northern and southern hemisphere,
    and there is an equatorial ring that should only be included once.
    """
    if not 0 <= ring_start <= ring_stop <= 2 * nside:
        raise ValueError("Invalid ring_start, ring_stop")
    cdef int32_t nrings = 2 * (ring_stop - ring_start)
    if ring_stop == 2 * nside:
        nrings -= 1 # equatorial ring
    cdef int32_t[:] irings = np.zeros(nrings, dtype=np.int32)
    cdef int32_t i
    for i in range((nrings + 1) // 2):
        # the ordering of the next two statements is important
        irings[nrings - i - 1] = (4 * nside - 1) - (ring_start + i) - 1
        irings[i] = ring_start + i
    return np.asarray(irings)    

def get_phi0(int32_t nside, int32_t[:] irings=None):
    if irings is None:
        irings = np.arange(4 * nside - 1, dtype=np.int32)
    cdef double[:] phi = np.zeros(irings.shape[0])

    cdef int32_t i, iring, nrings
    nrings = 4 * nside - 1
    for i in range(irings.shape[0]):
        iring = irings[i]
        if iring < nside - 1:
            phi[i] = pi / (4 * (iring + 1))
        elif iring < 3 * nside:
            phi[i] = (pi / (4 * nside) if iring % 2 == 1
                      else 0)
        else:
            iring = nrings - iring - 1
            phi[i] = pi / (4 * (iring + 1))
    return np.asarray(phi)

def get_ring_pixel_counts(int32_t nside, int32_t[:] irings=None):
    if irings is None:
        irings = np.arange(4 * nside - 1, dtype=np.int32)
    cdef int32_t[:] ringlens = np.zeros(irings.shape[0], np.int32)
    cdef int32_t iring, i, nrings
    nrings = 4 * nside - 1
    for i in range(irings.shape[0]):
        iring = irings[i]
        if iring < nside:
            ringlens[i] = 4 * (iring + 1)
        elif iring < 3 * nside:
            ringlens[i] = 4 * nside
        else:
            iring = nrings - iring - 1
            ringlens[i] = 4 * (iring + 1)
    return np.asarray(ringlens)

def healpix_scatter_ring_weights_to_map(nside, weights):
    map = np.ones(12 * nside**2)
    counts = get_ring_pixel_counts(nside)
    i = 0
    weights = np.hstack([weights, weights[:-1][::-1]])
    for count, w in zip(counts, weights):
        map[i:i + count] = w
        i += count
    return map
