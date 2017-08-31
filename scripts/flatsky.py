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

import sys
from cmbcr.cg import cg_generator




import scipy.ndimage

config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))

w = 1

nside = 128 * w
factor = 2048 // nside * w


def padvec(u):
    x = np.zeros(12 * nside**2)
    x[pick] = u
    return x

full_res_system = cmbcr.CrSystem.from_config(config, udgrade=nside, mask_eps=0.8)

full_res_system.prepare_prior()

system = cmbcr.downgrade_system(full_res_system, 1. / factor)

lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (0, 0, 0)

system.set_params(
    lmax_ninv=lmax_ninv,
    rot_ang=rot_ang,
    flat_mixing=False,
    )

system.prepare_prior()


lmax = system.lmax_list[0]

nrings = lmax + 1
pw = np.pi / nrings


ring_data_theta = pw * np.arange(2 * nrings) #np.zeros(2 * nrings)
ring_data = cmbcr.beam_by_theta(system.dl_list[0], ring_data_theta)

from numpy.fft import fft, ifft

ring_data_fft = np.abs(ifft(ring_data))

clf()
plot(ring_data_fft)

1/0

#plot(system.dl_list[0] * (ring_data_fft[0] / system.dl_list[0][0]))



#plot(ring_data)

def cl_to_flatsky(cl, nx, ny, nl):
    out = np.zeros((nx, ny))

    l = np.sqrt(
        (np.fft.fftfreq(nx, 1. / (2 * nl))[:, None])**2
        + (np.fft.fftfreq(ny, 1. / (2 * nl))[None, :])**2)

    cl_flat = scipy.ndimage.map_coordinates(cl, l[None, :, :], mode='constant', order=1)
    return cl_flat

x = np.concatenate([ring_data_fft[:ring_data_fft.shape[0] // 2], ring_data_fft * 0, ring_data_fft[:ring_data_fft.shape[0]:]])
dl_fft = cl_to_flatsky(x, nrings, 2 * nrings, lmax + 1)


def flatsky_synthesis(u):
    return np.fft.ifftn(u) * np.prod(u.shape)

def flatsky_adjoint_synthesis(u):
    return np.fft.fftn(u)


def flatsky_matvec(u):
    u = flatsky_adjoint_synthesis(u)
    u *= v1_fft#dl_fft
    u = flatsky_synthesis(u).real
    return u


def matvec(u):
    ulm = sharp.sh_adjoint_synthesis_gauss(lmax, u.reshape(2 * nrings**2))
    ulm *= scatter_l_to_lm(system.dl_list[0])
    u = sharp.sh_synthesis_gauss(lmax, ulm).reshape((nrings, 2 * nrings))
    return u



u = np.zeros((nrings, 2 * nrings))

u[nrings // 2, nrings] = 1
v1 = matvec(u)



v1_fft = np.abs(ifft(v1))




v2 = flatsky_matvec(u)
clf()
imshow(v2, interpolation='none')
colorbar()
draw()

