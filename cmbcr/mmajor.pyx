#cython: embedsignature=True

"""
:mod:`commander.sphere.mmajor`
------------------------------

Utilities for working with the :math:`m`-major ordering scheme.
See the library guide for description of the scheme.

"""

from __future__ import division


cimport cython
from libc.math cimport sqrt as c_sqrt

cimport numpy as cnp
import numpy as np

cdef inline index_t imax(index_t a, index_t b) nogil:
    return a if a > b else b

cdef inline index_t triangle_side(index_t area) nogil:
    return (-1 + <index_t>c_sqrt(1 + 8 * area)) // 2
    
cdef inline index_t triangle_area(index_t side) nogil:
    return (side * (side + 1)) // 2


cpdef index_t lm_to_idx(index_t lmin,
                        index_t lmax,
                        index_t mmin,
                        index_t l,
                        index_t m) nogil:
    cdef index_t start
    start = 0 if mmin == 0 else lm_to_idx_all_m(lmin, lmax, imax(lmin, mmin), mmin)
    return lm_to_idx_all_m(lmin, lmax, l, m) - start

cdef index_t lm_to_idx_all_m(index_t lmin,
                             index_t lmax,
                             index_t l,
                             index_t m) nogil:
    """
    Finds the array index of (l, m) in mmajor ordering (as described in the
    documentation).

    The routine supports arbitrary `lmin`/`lmax`. `m` should be within the
    range except for the following exceptions::

        # Pass out-of-range l and m in order to get number of coefficients
        lm_to_idx(lmin, lmax, lmax + 1, lmax + 1)

    As to `l`, there are no restrictions on its value; thus the following
    idiom is valid (and faster than calling `lm_to_idx` again
    for every iteration)::

        # Cache virtual offset for this m
        idx0 = lm_to_idx(lmin, lmax, 0, m)
        for l in range(a, b):
            idx = idx0 + l # find index of (l, m)

    """
    cdef:
        index_t n, abs_m, m_is_neg, lp, mp, whole_triangle_area, remainder_triangle_area, idx
    abs_m = -m if m < 0 else m

    # The remainder of the comments regards finding an idx for the ordering
    # where you only have m>=0.
    
    # The indices can be partitioned in two; when m < lmin, indices are given
    # by a normal rectangular m-major array, whereas for m >= lmin we subtract
    # the rectangle and solve for the remaining triangle

    n = lmax - lmin + 1 # side-length of triangle part

    if abs_m < lmin:
        # We are in the rectangular part
        idx =  n * abs_m + (l - lmin)
    else:
        # We are in the triangular part; add contribution for rectangular part
        # to idx; and deal with (lp, mp) = shifted l and m
        idx = n * lmin
        lp = l - lmin
        mp = abs_m - lmin

        whole_triangle_area = triangle_area(n)
        remainder_triangle_area = triangle_area(n - mp)
        idx += whole_triangle_area - remainder_triangle_area + lp - mp

    # Now, use the sign and zero-ness of m to map the found index into the
    # ordering for both positive and negative m. The m=0 is a special case
    # since there's no "negative" counterpart.
    if m == 0:
        return idx
    else:
        idx *= 2 # make room for negative m's
        idx -= (lmax + 1 - lmin) # ...though not for m=0
        if m < 0:
            idx += 1
        return idx


@cython.cdivision(True)
cdef void idx_to_lm_fast(index_t lmin,
                         index_t lmax,
                         index_t mmin,
                         index_t idx,
                         index_t *out_l,
                         index_t *out_m) nogil:
    cdef:
        index_t n, rectangle_area, k, mp, lp, l, m, negative_m
        index_t whole_triangle_area, less_than_small_triangle_area
        index_t small_triangle_area, idx_of_m_eq_l

    # First compensate for "removed" m's first, then we can ignore mmin in the
    # remainder
    if mmin != 0:
        idx += lm_to_idx(lmin, lmax, 0, imax(mmin, lmin), mmin)

    # Deal with the special m==0 case first
    if idx < lmax + 1 - lmin:
        l = idx + lmin
        m = 0
    else:
        # Now, reduce to the case of packing only non-negative m
        idx += lmax + 1 - lmin # first insert padding for "negative m=0"
        negative_m = idx % 2
        idx /= 2 # and pack to non-negative m only

        n = lmax - lmin + 1 # side-length of triangle part
        rectangle_area = n * lmin
        if idx < rectangle_area:
            # we are in the rectangular part
            l = idx % n + lmin
            m = idx // n
        else:
            # Shift index so we only have triangle part; then let l index
            # rows and m index colums, and look at the (lower-left) n-by-n
            # triangle. The index counts this triangle in column-major order.
            idx -= rectangle_area 

            whole_triangle_area = triangle_area(n)
            # find the triangle *to the right* of our position; denote its
            # side-length by k
            less_than_small_triangle_area = whole_triangle_area - (idx + 1)
            k = triangle_side(less_than_small_triangle_area)
            small_triangle_area = triangle_area(k)

            # idx is in the first column (=mp) not covered by this triangle
            mp = n - k - 1
            # Find l by first finding the index where m==l
            idx_of_m_eq_l = whole_triangle_area - small_triangle_area - (k + 1)
            lp = mp + (idx - idx_of_m_eq_l)

            l = lp + lmin
            m = mp + lmin

        if negative_m:
            m = -m
            
    # Return
    out_l[0] = l
    out_m[0] = m

def idx_to_lm(lmin, lmax, mmin, idx):
    cdef index_t l, m
    idx_to_lm_fast(lmin, lmax, mmin, idx, &l, &m)
    return l, m

def lm_count(lmin, lmax, mmin=None, mmax=None):
    mmin = mmin or 0
    mmax = mmax or lmax
    stop = lm_to_idx_all_m(lmin, lmax, mmax + 1, mmax + 1)
    start = lm_to_idx_all_m(lmin, lmax, imax(lmin, mmin), mmin)
    return stop - start

def find_lmax(index_t lmin, index_t ncoefs):
    # Solve the equation
    #     ncoefs == (lmax + 1)**2 - lmin**2
    return <index_t>c_sqrt(ncoefs + lmin * lmin) - 1

def lmax_of(obj):
    lmax = find_lmax(0, obj.shape[0])
    if obj.shape[0] != (lmax + 1)**2:
        raise ValueError('%d is not a valid number of SH coefficients' % obj.shape[0])
    return lmax

def scatter_l_to_lm(cnp.ndarray[double] data_l, double[:] out=None, index_t lmin=0):
    lmax = lmin + data_l.shape[0] - 1

    cdef index_t n
    
    n = lm_count(lmin, lmax)
    if out is None:
        out = np.empty(n, dtype=np.double)
    elif out.shape[0] != n:
        raise ValueError("Invalid out array, needs %d coefficients" % lm_count)

    cdef index_t idx, m, l, i0, i

    # m=0
    for l in range(lmin, lmax + 1):
        out[l - lmin] = data_l[l - lmin]
    # m != 0
    for m in range(1, lmax + 1):
        i0 = lm_to_idx_all_m(lmin, lmax, 0, m)
        for l in range(imax(m, lmin), lmax + 1):
            out[i0 + 2 * l] = data_l[l - lmin]
            out[i0 + 2 * l + 1] = data_l[l - lmin]
    return np.asarray(out)

def truncate_alm(alm, lmax_from, lmax_to):
    s = np.zeros(lmax_from + 1)
    s[:lmax_to + 1] = 1
    return alm[scatter_l_to_lm(s) == 1]

def compute_power_spectrum(index_t lmin, index_t lmax, double[:] alm, double[:] Cl=None):
    cdef index_t l, abs_m, i
    cdef double s, a
    if Cl is None:
        Cl = np.zeros(lmax + 1 - lmin)
    # TODO: This traversal scheme is simple but very inefficient for large arrays
    for l in range(lmin, lmax + 1):
        a = alm[lm_to_idx_all_m(lmin, lmax, l, 0)]
        s = a * a        
        for abs_m in range(1, l + 1):
            i = lm_to_idx_all_m(lmin, lmax, l, abs_m)
            a = alm[i]
            s += a * a
            a = alm[i + 1]
            s += a * a
        Cl[l - lmin] = s / (2 * l + 1)
    return np.asarray(Cl)

def norm_by_l(x):
    l = lmax_of(x)
    p = compute_power_spectrum(0, l, x)
    return np.sqrt(p)

def stats_by_l(double[:] alm):
    """
    Returns three arrays min, max, mean, with min, max, mean for each l.
    """
    cdef index_t l, abs_m, i
    cdef double s, a
    lmax = lmax_of(alm)
    cdef double[:] minarr = np.zeros(lmax + 1)
    cdef double[:] maxarr = np.zeros(lmax + 1)
    cdef double[:] meanarr = np.zeros(lmax + 1)
    # TODO: This traversal scheme is simple but very inefficient for large arrays
    for l in range(lmax + 1):
        minarr[l] = maxarr[l] = s = alm[lm_to_idx_all_m(0, lmax, l, 0)]
        for abs_m in range(1, l + 1):
            i = lm_to_idx_all_m(0, lmax, l, abs_m)
            a = alm[i]
            minarr[l] = minarr[l] if minarr[l] < a else a
            maxarr[l] = maxarr[l] if maxarr[l] > a else a
            s += a
            a = alm[i + 1]
            minarr[l] = minarr[l] if minarr[l] < a else a
            maxarr[l] = maxarr[l] if maxarr[l] > a else a
            s += a
        meanarr[l] = s / (2 * l + 1)
    return np.asarray(minarr), np.asarray(maxarr), np.asarray(meanarr)
