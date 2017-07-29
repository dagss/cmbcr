from __future__ import division
import numpy as np
import re

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



def standard_needlet_by_l(B, lmax):
    """
    Returns the spherical harmonic profile of a standard needlet on the sphere,
    following the recipe of arXiv:1004.5576. Instead of providing j, you provide
    lmax, and a j is computed so that the function reaches 0 at lmax.
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

    j = (np.log(lmax) - np.log(B)) / np.log(B)
    C = float(B)**j
    return np.asarray([b(l / C, B) for l in range(lmax + 1)])


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
