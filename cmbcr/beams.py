from __future__ import division
import numpy as np
import re
import libsharp

def beam_by_theta(bl, thetas):
    from libsharp import legendre_transform

    lmax = bl.shape[0] - 1
    l = np.arange(lmax + 1)
    scaled_bl = bl * (2 * l + 1)
    
    return legendre_transform(np.cos(thetas), scaled_bl) / (4. * np.pi)


def beam_by_cos_theta(bl, cos_thetas):
    from libsharp import legendre_transform

    lmax = bl.shape[0] - 1
    l = np.arange(lmax + 1)
    scaled_bl = bl * (2 * l + 1)
    
    return legendre_transform(cos_thetas, scaled_bl) / (4. * np.pi)



def standard_needlet_by_l(B, lmax, minval=None):
    """
    Returns the spherical harmonic profile of a standard needlet on the sphere,
    following the recipe of arXiv:1004.5576. Instead of providing j, you provide
    lmax, and a j is computed so that the function reaches 0 at lmax.

    If `minval` is provided, then lmax is gradually increased so as to avoid values
    smaller than `minval` in the result.
    """
    from scipy.integrate import quad
    
    def f(t):
        if -1 < t < 1:
            return np.exp(-1 / (1 - t*t))
        else:
            return 0

    phi_norm = 1 / quad(f, -1, 1)[0]
    
    def phi(u):
        return quad(f, -1, u)[0] * phi_norm

    def phi2(t, B):
        if t <= 1 / B:
            return 1
        elif t < 1:
            return phi(1 - (2 * B) / (B - 1) * (t - 1 / B))
        else:
            return 0

    def b(eta, B):
        if eta < 1 / B:
            return 0
        elif eta < B:
            b_sq = phi2(eta / B, B) - phi2(eta, B)
            if b_sq <= 0:
                return 0
            else:
                return np.sqrt(b_sq)
        else:
            return 0

    L = lmax
    while True:
        j = (np.log(L) - np.log(B)) / np.log(B)
        C = float(B)**j
        result = np.asarray([b(l / C, B) for l in range(lmax + 1)])
        if minval is None or result[-1] >= minval:
            return result
        L += 1


def fwhm_to_sigma(fwhm):
    fwhm = as_radians(fwhm)
    return fwhm / np.sqrt(8 * np.log(2))


_angle_re = re.compile(r"([0-9.]+)\s*(deg|min|sec|\*|''|')")
def as_radians(x):
    if isinstance(x, str):
        matches = _angle_re.findall(x)
        degrees = 0
        for val, unit in matches:
            val = float(val)
            if unit in ('deg', '*'):
                pass
            elif unit in ('min', "'"):
                val /= 60
            elif unit in ('sec', "''"):
                val /= 3600
            degrees += val
        return degrees / 180 * np.pi
    else:
        return float(x)


def gaussian_beam_by_l(lmax, fwhm):
    """
    fwhm : float
        Full-width half-max of Gaussian beam, in radians
    """
    sigma = fwhm_to_sigma(fwhm)
    ls = np.arange(lmax + 1)
    return np.exp(-0.5 * ls * (ls+1) * sigma*sigma)


def mhwavelet_beam_by_angle(theta, fwhm, a=3., b=.75):
    """
    Modified Mexican Hat Wavelet beam in real space, defined as 
    :math:`B(\theta) = \exp(-\frac{1}{2}(\theta/\sigma)^2) 
    [3 + \frac{3}{4}(\theta/\sigma)^2]`.
    
    This beam is designed to be more localised than a Gaussian, and resembles 
    (but is not exactly the same as) a harmonic space beam 
    :math:`b_l \propto \exp(-\frac{1}{2}\ell^2 (\ell +1)^2 / \sigma^2)`
    """
    sigma = 2.*fwhm_to_sigma(fwhm)
    norm = (9/8)**2. * 2. * np.pi * sigma**2.
    return np.exp(-0.5*(theta/sigma)**2.) * (1 - b*(theta/sigma)**2.) / norm

def mhwavelet_beam_by_l(lmax, fwhm, a=3, b=.75):
    """
    Find the Legendre coefficients :math:`b_\ell` for a modified Mexican Hat 
    Wavelet beam, where :math:`b_\ell` is defined such that 
    :math:`B(\theta) = \sum_\ell (2\ell + 1) b_\ell P_\ell(\cos \theta)`.
    
    Arguments:
      lmax -- Max. l to calculate :math:`b_\ell` up to
      fwhm -- FWHM of MH Wavelet (in radians)
    
    Returns:
      b_l -- 1D array of coefficients, for the l-range [0, lmax].
    """
    params = (fwhm, a, b)
    bl = arbitrary_beam_by_l(mhwavelet_beam_by_angle, params, lmax)
    bl /= bl[0]
    return bl

def arbitrary_beam_by_l(beam_function, params, lmax):
    """
    Find the Legendre coefficients :math:`b_\ell` for an arbitrary (symmetric) 
    real-space beam function, such that 
    :math:`B(\theta) = \sum_\ell (2\ell + 1) b_\ell P_\ell(\cos \theta)`.
    
    Arguments:
      beam_function -- Python function which takes arguments fn(theta, params)
      params -- Tuple containing parameters for the beam function (e.g. FWHM)
      lmax -- Max. l to calculate :math:`b_\ell` up to
    
    Returns:
      b_l -- 1D array of coefficients, for the l-range [0, lmax].
    """
    order = lmax
    l = np.arange(lmax + 1)
    
    # Get zeros and weights of the Legendre polynomials
    # (Zeros are symmetric about x=0, so only keep the nonzero ones)
    xi, wi = libsharp.legendre_roots(2*order)
    xi = xi[order:]; wi = 2.*wi[order:]
    
    # Get Legendre polynomials and normalise them
    leg = libsharp.normalized_associated_legendre_table(lmax, 0, np.arccos(xi))
    leg *= np.sqrt(4 * np.pi) / np.sqrt(2 * l + 1)
    
    # Perform Gauss-Legendre sum
    f = np.atleast_2d( wi * beam_function(np.arccos(xi), *params) ).T * leg
    return 0.25 * np.sum(f, axis=0)

def fourth_order_beam(lmax, ltreshold, epstreshold=0.04):
    """
    Squared Gaussian beam in SH space.
    """
    rhs = 2 * np.log(epstreshold)
    sigma = -rhs / (ltreshold**2 * (ltreshold + 1)**2)
    l = np.arange(lmax + 1)
    cl = np.exp(-l**2 * (l + 1)**2 * sigma)
    return np.sqrt(cl)
