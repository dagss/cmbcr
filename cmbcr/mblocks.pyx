from __future__ import division
import numpy as np
import libsharp

from . import mmajor
from . import sharp
from . import healpix

cimport cython

def ring_map_to_phases(ring_map, phi0s, ring_offsets, mmax, aliasing=True):
    nrings = phi0s.shape[0]
    phase_rings = np.zeros((mmax + 1, nrings), dtype=np.complex)
    for iring in range(phi0s.shape[0]):
        start, stop = ring_offsets[iring], ring_offsets[iring + 1]
        ring = ring_map[start:stop]

        # It is a real FFT, but the complex FFT automatically does the necesarry
        # extension that's necesarry when repeating coefficients below and so
        # is a lot more convenient
        phases = np.fft.fft(ring)
        m = np.arange(mmax + 1)
        shifts = np.exp(m * phi0s[iring] * -1j)
        if aliasing:
            i = 0
            for m in range(mmax + 1):
                phase_rings[m, iring] = phases[m % phases.shape[0]] * shifts[m]
        else:
            n = min(phases.shape[0], mmax + 1)
            phase_rings[:n, iring] = phases[:n] * shifts[:n]
    return phase_rings

def compute_complex_Yh_D_Y_block(m, mp, lmax_left, lmax_right, thetas, phase_map, out):
    P_l = libsharp.normalized_associated_legendre_table(lmax_left, abs(m), thetas)
    if mp != m:
        Pp_l = libsharp.normalized_associated_legendre_table(lmax_left, abs(mp), thetas)
    else:
        Pp_l = P_l
    d = phase_map[abs(m - mp), :]
    if m - mp < 0:
        d = d.conjugate()
    # real
    dr_Pp_l = Pp_l * d.real[:, None]
    out[0, :, :] = np.dot(P_l.T, dr_Pp_l)
    # imag
    di_Pp_l = Pp_l; del Pp_l
    di_Pp_l *= d.imag[:, None]
    out[1, :, :] = np.dot(P_l.T, di_Pp_l)

@cython.wraparound(False)
@cython.boundscheck(False)
def complex_blocks_to_real_block(int abs_m, int abs_mp,
                                 double[:, :,:,::1] complex_blocks,
                                 double[:,::1] out=None):
    """

    ::
        [ a b ]    [ (+m,+mp)  (+m,-mp) ]
        [ c d ] == [ (-m,+mp)  (-m,-mp) ]

    Parameters
    ----------

    complex_blocks : double[:,:,:,::1]
        Indexed by (block, is_imag, l, lp), where `is_imag` is 1 for the imaginary part
        and `block` is 0 for the ``a``/``(+m,+mp)`` block, 1 for ``b``, and so on,
        see diagram above.
        
    

    Implementation notes
    --------------------

    The formulas can be had by this SymPy script::

        from sympy import var, I, Matrix, simplify, re

        var('ar br cr dr ai bi ci di')

        A22 = Matrix([[ar + I * ai, br + I * bi],
                      [cr + I * ci, dr + I * di]])

        A12 = Matrix([[ar + I * ai, br + I * bi]])
        A21 = Matrix([[ar + I * ai], [cr + I * ci]])

        U_even = Matrix([[1, 1], [-I, I]])
        U_odd = Matrix([[-I, -I], [-1, 1]])

        def f(e):
            print simplify(e).subs({I:0})
            print 

        for U in [U_even, U_odd]:
            f(U * A22 * U.conjugate().T)
            f(A12 * U.conjugate().T)
            f(U * A21)
    """
    cdef int AIDX = 0, BIDX = 1, CIDX = 2, DIDX = 3, REAL = 0, IMAG = 1
    cdef int i, j
    cdef double sqrt_half = np.sqrt(.5)

    nrows_r = complex_blocks.shape[2] if abs_m == 0 else 2 * complex_blocks.shape[2]
    ncols_r = complex_blocks.shape[3] if abs_mp == 0 else 2 * complex_blocks.shape[3]

    if out.shape[0] != nrows_r or out.shape[1] != ncols_r:
        raise ValueError('out.shape does not conform with complex_blocks.shape')

    if abs_m == abs_mp == 0:
        out[:, :] = complex_blocks[AIDX, REAL, :, :]
    elif abs_m > 0 and abs_mp == 0:
        for i in range(complex_blocks.shape[2]):
            for j in range(complex_blocks.shape[3]):
                ar = complex_blocks[AIDX, REAL, i, j]
                ai = complex_blocks[AIDX, IMAG, i, j]
                cr = complex_blocks[CIDX, REAL, i, j]
                ci = complex_blocks[CIDX, IMAG, i, j]
                out[2 * i, j]     = (+ar +cr) * sqrt_half
                out[2 * i + 1, j] = (+ai -ci) * sqrt_half
    elif abs_m == 0 and abs_mp > 0:
        for i in range(complex_blocks.shape[2]):
            for j in range(complex_blocks.shape[3]):
                ar = complex_blocks[AIDX, REAL, i, j]
                ai = complex_blocks[AIDX, IMAG, i, j]
                br = complex_blocks[BIDX, REAL, i, j]
                bi = complex_blocks[BIDX, IMAG, i, j]
                out[i, 2 * j]     = (+ar +br) * sqrt_half
                out[i, 2 * j + 1] = (-ai +bi) * sqrt_half
    else:
        for i in range(complex_blocks.shape[2]):
            for j in range(complex_blocks.shape[3]):
                ar = complex_blocks[AIDX, REAL, i, j]
                ai = complex_blocks[AIDX, IMAG, i, j]
                br = complex_blocks[BIDX, REAL, i, j]
                bi = complex_blocks[BIDX, IMAG, i, j]
                cr = complex_blocks[CIDX, REAL, i, j]
                ci = complex_blocks[CIDX, IMAG, i, j]
                dr = complex_blocks[DIDX, REAL, i, j]
                di = complex_blocks[DIDX, IMAG, i, j]

                out[2 * i, 2 * j]         = (+ar +br +cr +dr) * .5
                out[2 * i, 2 * j + 1]     = (-ai +bi -ci +di) * .5
                out[2 * i + 1, 2 * j]     = (+ai +bi -ci -di) * .5
                out[2 * i + 1, 2 * j + 1] = (+ar -br -cr +dr) * .5
    return out

def compute_real_Yh_D_Y_block(m, mp, lmax_left, lmax_right, thetas, phase_map):
    nrows_c = lmax_left - m + 1
    nrows_r = nrows_c if m == 0 else 2 * nrows_c
    ncols_c = lmax_right - mp + 1
    ncols_r = ncols_c if mp == 0 else 2 * ncols_c
 
    complex_block = np.zeros((4, 2, nrows_c, ncols_c))
    compute_complex_Yh_D_Y_block(m, mp, lmax_left, lmax_right, thetas, phase_map, complex_block[0, ...])
    compute_complex_Yh_D_Y_block(m, -mp, lmax_left, lmax_right, thetas, phase_map, complex_block[1, ...])
    compute_complex_Yh_D_Y_block(-m, mp, lmax_left, lmax_right, thetas, phase_map, complex_block[2, ...])
    compute_complex_Yh_D_Y_block(-m, -mp, lmax_left, lmax_right, thetas, phase_map, complex_block[3, ...])

    out = np.zeros((nrows_r, ncols_r))
    complex_blocks_to_real_block(m, mp, complex_block, out)
    return out
    

def compute_Yh_D_Y_from_phase_map(lmax_left, lmax_right, thetas, phase_map):
    out = np.zeros(((lmax_left + 1)**2, (lmax_right + 1)**2))
    for m in range(lmax_left + 1):
        print m, 'of', lmax_left
        for mp in range(lmax_right + 1):
            block = compute_real_Yh_D_Y_block(m, mp, lmax_left, lmax_right, thetas, phase_map)

            i_start = mmajor.lm_to_idx(0, lmax_left, 0, m, m)
            i_stop = mmajor.lm_to_idx(0, lmax_left, 0, m + 1, m + 1)
            j_start = mmajor.lm_to_idx(0, lmax_right, 0, mp, mp)
            j_stop = mmajor.lm_to_idx(0, lmax_right, 0, mp + 1, mp + 1)

            out[i_start:i_stop, j_start:j_stop] = block
    return out


def compute_Yh_D_Y_healpix(lmax_left, lmax_right, map):
    nside = healpix.npix_to_nside(map.shape[0])
    thetas = healpix.get_ring_theta(nside)
    counts = healpix.get_ring_pixel_counts(nside)
    phi0s = healpix.get_phi0(nside)
    offsets = np.cumsum(np.hstack([[0], counts]))
    phase_map = ring_map_to_phases(map, phi0s, offsets, 2 * max([lmax_left, lmax_right]))
    return compute_Yh_D_Y_from_phase_map(lmax_left, lmax_right, thetas, phase_map)

def compute_Yh_D_Y_gauss(lmax_pix, lmax_left, lmax_right, map):
    thetas, nphi, weights = sharp.gauss_legendre_grid(lmax_pix)
    nrings = thetas.shape[0]
    phi0s = np.zeros(nrings)
    offsets = np.arange(nrings + 1) * nphi
    phase_map = ring_map_to_phases(map, phi0s, offsets, 2 * max([lmax_left, lmax_right]))
    block = compute_Yh_D_Y_from_phase_map(lmax_left, lmax_right, thetas, phase_map)
    return block


def healpix_ring_map_to_phase_map(map, lmax):
    nside = healpix.npix_to_nside(map.shape[0])
    thetas = healpix.get_ring_theta(nside)
    counts = healpix.get_ring_pixel_counts(nside)
    phi0s = healpix.get_phi0(nside)
    offsets = np.cumsum(np.hstack([[0], counts]))
    return ring_map_to_phases(map, phi0s, offsets, 2 * lmax), thetas


def gauss_ring_map_to_phase_map(map, lmax_pix, lmax):
    thetas, nphi, weights = sharp.gauss_legendre_grid(lmax_pix)
    nrings = thetas.shape[0]
    phi0s = np.zeros(nrings)
    offsets = np.arange(nrings + 1) * nphi
    phase_map = ring_map_to_phases(map, phi0s, offsets, 2 * lmax)
    return phase_map, thetas
