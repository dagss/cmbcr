from __future__ import division
import numpy as np

from matplotlib.pyplot import *

import logging
logging.basicConfig(level=logging.DEBUG)



import cmbcr
import cmbcr.utils
reload(cmbcr.beams)
reload(cmbcr.cr_system)
reload(cmbcr.precond_sh)
reload(cmbcr.precond_psuedoinv)
reload(cmbcr.precond_diag)
reload(cmbcr.precond_pixel)
reload(cmbcr.utils)
reload(cmbcr.multilevel)
reload(cmbcr)
from cmbcr.utils import *

from cmbcr import sharp
from healpy import mollzoom, mollview
from scipy.sparse import csc_matrix
#reload(cmbcr.main)

from cmbcr.cr_system import load_map_cached

import sys
from cmbcr.cg import cg_generator

import scipy.ndimage


def cl_to_flatsky(cl, nx, ny, nl):
    out = np.zeros((nx, ny))

    l = np.sqrt(
        (np.fft.fftfreq(nx, 1. / (2 * nl))[:, None])**2
        + (np.fft.fftfreq(ny, 1. / (2 * nl))[None, :])**2)

    cl_flat = scipy.ndimage.map_coordinates(cl, l[None, :, :], mode='constant', order=1)
    return cl_flat

def flatsky_analysis(u):
    return np.fft.fftn(u) / np.prod(u.shape)

def flatsky_adjoint_analysis(u):
    return np.fft.ifftn(u)

def flatsky_synthesis(u):
    return np.fft.ifftn(u) * np.prod(u.shape)

def flatsky_adjoint_synthesis(u):
    return np.fft.fftn(u)

from scipy.sparse import dok_matrix

def coarsen_matrix():
    R = dok_matrix((nphi // 2 * ntheta // 2, ntheta * nphi))

    def add(coarse_i, coarse_j, fine_i, fine_j, weight):
        # wrap around fine_i and fine_j..
        fine_i = fine_i % ntheta
        fine_j = fine_j % nphi

        coarse_ind = coarse_i * (nphi // 2) + coarse_j
        fine_ind = fine_i * nphi + fine_j
        R[coarse_ind, fine_ind] = weight
    
    for i in range(ntheta // 2):
        for j in range(nphi // 2):
            # row above
            add(i, j, 2 * i - 1, 2 * j - 1, 1/16.)
            add(i, j, 2 * i - 1, 2 * j, 1/8.)
            add(i, j, 2 * i - 1, 2 * j + 1, 1/16.)

            # center row
            add(i, j, 2 * i, 2 * j - 1, 1/8.)
            add(i, j, 2 * i, 2 * j, 1/4.)
            add(i, j, 2 * i, 2 * j + 1, 1/8.)

            # row below
            add(i, j, 2 * i + 1, 2 * j - 1, 1/16.)
            add(i, j, 2 * i + 1, 2 * j, 1/8.)
            add(i, j, 2 * i + 1, 2 * j + 1, 1/16.)

    return R.tocsr()


lmax = 128 - 1
lmax_sh = int(np.sqrt(2) * lmax)

l = np.arange(1, lmax_sh + 2).astype(float)
dl = l**2.5

nrings = lmax + 1

nphi = 2 * nrings
ntheta = 40

start_ring = nrings // 2 - ntheta // 2
stop_ring = nrings // 2 + ntheta // 2


def padvec(u):
    u_pad = np.zeros((nrings, 2 * nrings))
    u_pad[start_ring:stop_ring, :] = u.reshape((ntheta, nphi))
    return u_pad

def padvec_H(u):
    u_pad = np.zeros((nrings // 2, 2 * nrings // 2))
    u_pad[start_ring // 2:stop_ring // 2, :] = u.reshape((ntheta // 2, nphi // 2))
    return u_pad

def matvec_fft(dl, u):
    u = flatsky_adjoint_synthesis(u.reshape((ntheta, nphi)))
    u *= dl_fft
    u = flatsky_synthesis(u)
    return u.real.reshape(nphi * ntheta)

def matvec_sht(dl, u):
    u_pad = np.zeros((nrings, 2 * nrings))
    u_pad[start_ring:stop_ring, :] = u.reshape((ntheta, nphi))

    u = sharp.sh_adjoint_synthesis_gauss(lmax, u_pad.reshape(2 * nrings**2), lmax_sh=lmax_sh)
    u *= scatter_l_to_lm(dl)
    u = sharp.sh_synthesis_gauss(lmax, u, lmax_sh=lmax_sh).reshape((nrings, 2 * nrings))
    
    return u[start_ring:stop_ring, :].reshape(nphi * ntheta)





u = np.zeros((ntheta, nphi))
u[ntheta // 2, 0] = 1


def image_to_powspec(u, x):
    FtW_x = flatsky_analysis(x)
    Ft_u = flatsky_adjoint_synthesis(u)
    dl_fft = np.abs(FtW_x / Ft_u)
    return dl_fft

x = matvec_sht(dl, u.reshape(ntheta * nphi)).reshape(ntheta, nphi)
dl_fft = image_to_powspec(u, x)


R = coarsen_matrix()

def matvec_fft_coarse(u):
    u = R.T * u
    u = matvec_fft(dl_fft, u)
    u = R * u
    return u



u = np.zeros((ntheta // 2, nphi // 2))
u[ntheta//4, nphi//4] = 1
Au = matvec_fft_coarse(u.reshape(ntheta * nphi // 4)).reshape(ntheta // 2, nphi // 2)

dl_fft_H = image_to_powspec(Au)

def matvec_fft_coarse_approx(u):
    u = flatsky_adjoint_synthesis(u.reshape(ntheta // 2, nphi // 2))
    u *= dl_fft_H
    u = flatsky_synthesis(u)
    return u.real.reshape(ntheta * nphi // 4)
    
Au_approx = matvec_fft_coarse_approx(u.reshape(ntheta * nphi // 4)).reshape(ntheta // 2, nphi // 2)


X1 = hammer(matvec_fft_coarse, ntheta * nphi // 4)
X2 = hammer(matvec_fft_coarse_approx, ntheta * nphi // 4)




#clf()
#imshow(padvec_H(Au))

#draw()

#rng = np.random.RandomState(3)
#x = rng.normal(size=ntheta * nphi)

#x_f = matvec_fft(dl_fft, x.copy())
#x_s = matvec_sht(dl, x.copy())



1/0






if 0:

    # First scale diagonal accordingly to get a first estimate
    u = np.zeros(ntheta * nphi)
    i = u.shape[0] // 2
    u[i] = 1
    scaling_factor = matvec_sht(dl, u)[i] / matvec_fft(dl, u)[i]

    #dl_fft = dl * scaling_factor


    rng = np.random.RandomState(3)
    x = rng.normal(size=ntheta * nphi)
    #x = np.zeros(ntheta * nphi)
    #x[x.shape[0] // 2] = 1

    attempts = []

    def f(dl_fft):
        attempts.append(dl_fft.copy())
        print 'it', np.linalg.norm(dl_fft)
        x_f = matvec_fft(dl_fft, x.copy())
        x_s = matvec_sht(dl, x.copy())
        return np.linalg.norm(x_f - x_s)

    from scipy.optimize import minimize
    res = minimize(f, x0=dl * scaling_factor, options=dict(disp=True), method='Nelder-Mead')
    #clf()
    #for x in attempts[::100]:
    #    plot(x)
    #draw()

    clf()
    plot(dl)
    plot(res.x)
    draw()
