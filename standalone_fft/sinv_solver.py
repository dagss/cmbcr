from __future__ import division

import sys
import os

from matplotlib.pyplot import *
import numpy as np
from cg import cg_generator
import scipy.ndimage


def load_Cl_cmb(filename):
    lmax = 10000
    dat = np.loadtxt(filename)
    assert dat[0,0] == 0 and dat[1,0] == 1 and dat[2,0] == 2
    Cl = dat[:, 1][:lmax + 1]
    ls = np.arange(2, lmax + 1)
    Cl[2:] /= ls * (ls + 1) / 2 / np.pi
    Cl[0] = Cl[1] = Cl[2]
    return Cl

Cl_cmb = load_Cl_cmb('camb_11229992_scalcls.dat')


rng = np.random.RandomState(1)

# Resolution parameters (for *top* level). The 2D grid shape is
# (nrings, 2*nrings); which actually corresponds to the sphere (2*pi,
# pi) on a Gauss-Legendre or equiangular grid, and can be transformed
# back to the sphere. For the purposes of this solver we can probably
# ignore anything far away from the equator, e.g., make it (nrings//2,
# 2 * nrings), to save computing, but did not bother so far.

nrings = 128  # use a power of 2
lmax = nrings - 1

# Set up the power spectrum -- it should extend to at least sqrt(2)*lmax to avoid a singular 2F FFT power spectrum

# The solver needs to work for two kinds of power spectrum

if 0:
    ## Use-case 1: Function of l.(The starting point at 1 is not important, should just be something invertible, not 0..)
    ## The solver should work with "1.0", "1.5", etc. instead of just "2.5".
    l = np.arange(1, 2 * lmax + 1).astype(float)
    power_spectrum = l**2.5
else:
    ## Use-case 2: The inverse CMB power spectrum
    power_spectrum = 1 / Cl_cmb



#
# Load mask -- simply take every i-th pixel for downgraded mask
#
dgfactor = 4096 // nrings
mask = np.load('mask.npy').reshape(4096, 2 * 4096)
mask = mask[::dgfactor, ::dgfactor]
pick = (mask == 0)
n_mask = int(pick.sum())



#
# Utilities
#


def pickvec(u):
    """
    Convert from 2D map to 1D vector of system coefficients (i.e. cuts away everything outside mask)
    """
    return u[pick]


def padvec(u):
    """
    Convert from 1D vector of system coefficients to a 2D map with pixels inside the mask, and 0 outside
    """
    u_pad = np.zeros((nrings, 2 * nrings))
    u_pad[pick] = u.real
    return u_pad


#
# Convert spherical power spectrum "power_spectrum" to 2D power spectrum
#
def cl_to_flatsky(cl, nx, ny, nl):
    out = np.zeros((nx, ny))

    l = np.sqrt(
        (np.fft.fftfreq(nx, 1. / (2 * nl))[:, None])**2
        + (np.fft.fftfreq(ny, 1. / (2 * nl))[None, :])**2)

    cl_flat = scipy.ndimage.map_coordinates(cl, l[None, :, :], mode='constant', order=1)
    return cl_flat

power_spectrum_fft = cl_to_flatsky(power_spectrum, nrings, 2 * nrings, lmax + 1)

# There's also a constant normalization factor which we can just ignore in this sub-solver, that's
# only needed when plugging it into the final solver


#
# Linear system. The synthesis/adjoint_synthesis etc. names corresponds with similar operators
# in the spherical case...
#
def flatsky_synthesis(u):
    return np.fft.ifftn(u) * np.prod(u.shape)

def flatsky_adjoint_synthesis(u):
    return np.fft.fftn(u)

def flatsky_analysis(u):
    return np.fft.fftn(u) / np.prod(u.shape)

def flatsky_adjoint_analysis(u):
    return np.fft.ifftn(u)


def matvec(u):
    u = padvec(u)
    u = flatsky_adjoint_synthesis(u)
    u *= power_spectrum_fft
    u = flatsky_synthesis(u)
    u = pickvec(u.real)
    return u


#
# Preconditioner -- no MG for now, just use FFT with the inverse power spectrum on the full grid
#

power_spectrum_fft_inv = 1 / power_spectrum_fft

def precond(u):
    u = padvec(u)
    u = flatsky_analysis(u)
    u *= power_spectrum_fft_inv
    u = flatsky_adjoint_analysis(u)
    u = pickvec(u.real)
    return u


#
# Set up a test
#

x0 = rng.normal(size=n_mask)
b = matvec(x0)
x = x0 * 0

norm0 = np.linalg.norm(x0)
errlst = []

solver = cg_generator(matvec, b, M=precond)

for i, (x, r, delta_new) in enumerate(solver):

    errvec = x0 - x
    errlst.append(np.linalg.norm(errvec) / norm0)

    if i > 50:
        break



if 0:
    # To show error at the end
    clf();
    imshow(padvec(x-x0), interpolation='none');
    colorbar()
    draw()
else:
    # To show convergence
    semilogy(errlst, '-o')
    draw()

