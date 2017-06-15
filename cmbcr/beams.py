from __future__ import division
import numpy as np
import re


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
