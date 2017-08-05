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

nside = 2048 * w
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
system.prepare(use_healpix=True)


rng = np.random.RandomState(1)

x0 = [
    scatter_l_to_lm(1. / system.dl_list[k]) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)


if 1:
    lmax = system.lmax_list[0]
    l = np.arange(1, 2 * lmax + 1).astype(float)
    dl = l**2.5
else:
    dl = system.dl_list[0]
    if 1:
        nl = cmbcr.standard_needlet_by_l(1.5, 2 * dl.shape[0] - 1)
        i = nl.argmax()
        dl = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])
    #lmax = dl.shape[0] - 1
    lmax = system.lmax_list[0]
    
lmax_sh = dl.shape[0] - 1
nrings = lmax + 1

lmax = 4096 - 1

if 1:
    mask_lm = sharp.sh_analysis(lmax, system.mask)
    mask_gauss = sharp.sh_synthesis_gauss(lmax, mask_lm)
    mask_gauss[mask_gauss < 0.9] = 0
    mask_gauss[mask_gauss >= 0.9] = 1
else:
    mask_gauss = np.ones((nrings, 2 * nrings))
    k = -2
    mask_gauss[(5*nrings)//12 - k :(7*nrings)//12 + k] = 0
    mask_gauss = mask_gauss.reshape(2 * nrings**2)
1/0
#mask_gauss[:] = 0
    
#l = np.arange(lmax + 1)


# Figure out the maximum value of the operator when working in SHTs...
z = np.zeros(2 * nrings**2)
z[nrings**2 + nrings] = 1
z = sharp.sh_adjoint_synthesis_gauss(lmax, z)
z *= scatter_l_to_lm(dl[:lmax + 1])
z = sharp.sh_synthesis_gauss(lmax, z)
estimated_max_sht = z[nrings**2 + nrings]


# Set up FFT operator

def cl_to_flatsky(cl, nx, ny, nl):
    out = np.zeros((nx, ny))

    l = np.sqrt(
        (np.fft.fftfreq(nx, 1. / (2 * nl))[:, None])**2
        + (np.fft.fftfreq(ny, 1. / (2 * nl))[None, :])**2)

    cl_flat = scipy.ndimage.map_coordinates(cl, l[None, :, :], mode='constant', order=1)
    return cl_flat




#import libsharp
#lambda_lm = np.zeros((lmax + 1, lmax + 1))
#for m in range(lmax + 1):
#    lambda_lm[m:, m] = libsharp.normalized_associated_legendre_table(lmax, m, np.pi / 2)[0]
#dl_m = np.sum(dl[:, None] * lambda_lm**2, axis=1)
#dl_m = dl

dl_fft = cl_to_flatsky(dl, nrings, 2 * nrings, lmax + 1)

def flatsky_synthesis(u):
    return np.fft.ifftn(u) * np.prod(u.shape)

def flatsky_adjoint_synthesis(u):
    return np.fft.fftn(u)

def flatsky_analysis(u):
    return np.fft.fftn(u) / np.prod(u.shape)

def flatsky_adjoint_analysis(u):
    return np.fft.ifftn(u)


flatsky_matvec_ratio = 1

def flatsky_matvec(u):
    u = flatsky_adjoint_synthesis(u)
    u *= dl_fft
    u = flatsky_synthesis(u)
    return u


u = np.zeros((nrings, 2 * nrings))
u[nrings // 2, nrings] = 1
u_out = flatsky_matvec(u)
estimated_max_fft = u_out.max().real

flatsky_matvec_ratio = r = estimated_max_sht / estimated_max_fft

dl_fft *= flatsky_matvec_ratio


pick = (mask_gauss == 0)
n_mask = int(pick.sum())
    
def matvec_mask_basis(u):
    u_pad = np.zeros(2 * nrings**2)
    u_pad[pick] = u

    u_pad = flatsky_matvec(u_pad.reshape(nrings, 2 * nrings)).reshape(2 * nrings**2)
    return u_pad[pick]
        

#dl_fft_inv = cl_to_flatsky(1 / dl, nrings, 2 * nrings, np.pi, 2 * np.pi)
dl_fft_inv = dl_fft.copy()
dl_fft_inv[dl_fft_inv != 0] = 1 / dl_fft_inv[dl_fft_inv != 0]
#dl_fft += dl_fft.max() * 1e-4


def precond_fullsky(u):
    u_pad = np.zeros(2 * nrings**2)
    u_pad[pick] = u
    u_pad = u_pad.reshape(nrings, 2 * nrings)
    u_pad = flatsky_analysis(u_pad)
    u_pad *= dl_fft_inv
    u_pad = flatsky_adjoint_analysis(u_pad)
    u_pad = u_pad.reshape(2 * nrings**2).real
    return u_pad[pick]

import scipy.ndimage
def coarsen(u):
    w = np.asarray([
        [(1./16), (1./8), (1./16)],
        [(1./8), (1/.4), (1./8)],
        [(1./16), (1./8), (1./16)]])
    print u.shape
    print w.shape
    c = scipy.ndimage.convolve(u, w, mode='wrap')
    return c[::2, ::2]




#inv_cl_patch = cl_to_flatsky(1 / (dl * flatsky_matvec_ratio), K, K, lmax)


def precond_patch(u):
    u_pad = padvec(u)

    # pick out patch 8x8
    x = u_pad[28:28+K,0:K]

    x = flatsky_analysis(x)
    x *= inv_cl_patch
    x = flatsky_adjoint_analysis(x)

    u_pad[:] = 0
    u_pad[28:28+K,0:K] = x.real
    return pickvec(u_pad)

precond = precond_fullsky



if 'eig' in sys.argv:

    A = hammer(matvec_mask_basis, n_mask)
    M = hammer(precond_fullsky, n_mask)
    from scipy.linalg import eigvals, eigvalsh
    clf()
    semilogy(sorted(eigvalsh(A)))
    semilogy(sorted(eigvalsh(M))[::-1])
    semilogy(sorted(eigvalsh(np.dot(M, A))))
    draw()
    1/0



def pickvec(u):
    return u.reshape(2 * nrings**2)[pick]

def padvec(u):
    u_pad = np.zeros(2 * nrings**2)
    u_pad[pick] = u.real
    return u_pad.reshape(nrings, 2 * nrings)

rng = np.random.RandomState(12)
x0 = rng.normal(size=n_mask)
b = matvec_mask_basis(x0)
x = x0 * 0


1/0

norm0 = np.linalg.norm(x0)
errlst = []

solver = cg_generator(
    matvec_mask_basis,
    b,
    M=precond
    )

for i, (x, r, delta_new) in enumerate(solver):

#for i in range(1000):
#    r = b - matvec_mask_basis(x)
#    x = x + 0.1 * precond(r)

    errvec = x0 - x
    #err_its.append(errvec)
    #x_its.append(x)
    
    errlst.append(np.linalg.norm(errvec) / norm0)

    print 'it', i
    if i > 100:
        break



#clf();
#imshow(padvec(x-x0), interpolation='none');
semilogy(errlst, '-o')
#colorbar()
draw()

