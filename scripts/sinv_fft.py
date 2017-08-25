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
from scipy.sparse import csc_matrix, dok_matrix
#reload(cmbcr.main)

from cmbcr.cr_system import load_map_cached

import sys
from cmbcr.cg import cg_generator

import scipy.ndimage

lmax = 256 - 1
lmax_sh = int(1 * lmax)
OMEGA = 1

SQRTSPLITFULLSOLVE = True
INNER_ITS = 1

def load_Cl_cmb(lmax, filename='camb_11229992_scalcls.dat'):
    #dat = np.loadtxt()
    dat = np.loadtxt(filename)
    assert dat[0,0] == 0 and dat[1,0] == 1 and dat[2,0] == 2
    Cl = dat[:, 1][:lmax + 1]
    ls = np.arange(2, lmax + 1)
    Cl[2:] /= ls * (ls + 1) / 2 / np.pi
    Cl[0] = Cl[1] = Cl[2]
    return Cl
Cl_cmb = load_Cl_cmb(10000)


l = np.arange(1, lmax_sh + 2).astype(float)
#dl = np.exp(0.01 * l)
dl = l**4
#dl = 1 / Cl_cmb[::16][:lmax_sh + 1]

nl = cmbcr.standard_needlet_by_l(2, 2 * dl.shape[0] - 1)
i = nl.argmax()
dl = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])

lmax_sh = dl.shape[0] - 1

#1.2 #1.5




nrings = lmax + 1


mask_healpix = load_map_cached('mask_galactic_band_2048.fits')

#mask_hires[:] = 0

def cl_to_flatsky(cl, nx, ny, nl):
    1/0
    out = np.zeros((nx, ny))

    l = np.sqrt(
        (np.fft.fftfreq(nx, 1. / (2 * nl))[:, None])**2
        + (np.fft.fftfreq(ny, 1. / (2 * nl))[None, :])**2)

    cl_flat = scipy.ndimage.map_coordinates(cl, l[None, :, :], mode='constant', order=1)
    return cl_flat


class DenseSmoother(object):
    def __init__(self, level):
        self.M = np.linalg.inv(hammer(level.matvec, level.n_mask))

    def apply(self, u):
        return np.dot(self.M, u)

    
def image_to_powspec(u, x):
    # u: unit-vector in flatsky basis
    # x: image of operator
    FtW_x = flatsky_analysis(x)
    Ft_u = flatsky_adjoint_synthesis(u)
    dl_fft = np.abs(FtW_x / Ft_u)
    return dl_fft

def flatsky_analysis(u):
    return np.fft.fftn(u) / np.prod(u.shape)

def flatsky_adjoint_analysis(u):
    return np.fft.ifftn(u)

def flatsky_synthesis(u):
    return np.fft.ifftn(u) * np.prod(u.shape)

def flatsky_adjoint_synthesis(u):
    return np.fft.fftn(u)


def full_coarsen_matrix(ntheta, nphi):
    coarse_ntheta = ntheta // 2
    coarse_nphi = nphi // 2
    R = dok_matrix((coarse_ntheta * coarse_nphi, ntheta * nphi))

    def add(coarse_i, coarse_j, fine_i, fine_j, weight):
        # wrap around fine_i and fine_j..
        fine_i = fine_i % ntheta
        fine_j = fine_j % nphi

        coarse_ind = coarse_i * coarse_nphi + coarse_j
        fine_ind = fine_i * nphi + fine_j
        R[coarse_ind, fine_ind] = weight
    
    for i in range(coarse_ntheta):
        for j in range(coarse_nphi):
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



mask_lm = sharp.sh_analysis(lmax, mask_healpix)
mask_gauss = sharp.sh_synthesis_gauss(lmax, mask_lm)
mask_gauss[mask_gauss < 0.9] = 0
mask_gauss[mask_gauss >= 0.9] = 1

        

start_ring = nrings // 4
stop_ring = 3 * nrings // 4

mask = mask_gauss.reshape((nrings, 2 * nrings))[start_ring:stop_ring, :]


ntheta = stop_ring - start_ring
nphi = 2 * nrings

unitvec = np.zeros((ntheta, nphi))
unitvec[ntheta // 2, nphi // 2] = 1

# 'image' the spherical harmonic operator in pixel domain in order to transfer it to FFT
u_pad = np.zeros((nrings, 2 * nrings))
u_pad[start_ring:stop_ring, :] = unitvec.reshape((ntheta, nphi))
u = sharp.sh_adjoint_synthesis_gauss(lmax, u_pad.reshape(2 * nrings**2), lmax_sh=lmax_sh)
u *= scatter_l_to_lm(dl)
u = sharp.sh_synthesis_gauss(lmax, u, lmax_sh=lmax_sh).reshape((nrings, 2 * nrings))
image_of_operator = u[start_ring:stop_ring, :]


outer_dl_fft = image_to_powspec(unitvec, image_of_operator)

if SQRTSPLITFULLSOLVE:
    inner_dl_fft = np.sqrt(outer_dl_fft)
else:
    inner_dl_fft = outer_dl_fft


class Level(object):
    def __init__(self, dl_fft, mask):
        self.mask = mask
        self.dl_fft = dl_fft
        self.ntheta, self.nphi = dl_fft.shape
        self.pick = (mask.reshape(self.ntheta * self.nphi) == 0)
        self.n = int(self.pick.sum())
        self.R = full_coarsen_matrix(self.ntheta, self.nphi)
        self.ntheta_H = self.ntheta // 2
        self.nphi_H = self.nphi // 2

    def pickvec(self, u):
        return u.reshape(self.ntheta * self.nphi)[self.pick]

    def padvec(self, u):
        u_pad = np.zeros(self.ntheta * self.nphi)
        u_pad[self.pick] = u.real
        return u_pad.reshape(self.ntheta, self.nphi)

    def matvec_padded(self, u):
        u = flatsky_adjoint_synthesis(u)
        u *= self.dl_fft
        u = flatsky_synthesis(u).real
        return u
    
    def matvec(self, u):
        u = self.padvec(u)
        u = self.matvec_padded(u)
        u = self.pickvec(u)
        return u

    def matvec_coarsened(self, u):
        # do matvec on the next, coarser level. This is just done once, to create the operator on the next level
        return self.coarsen_padded(self.matvec_padded(self.interpolate_padded(u)))

    def coarsen_padded(self, u):
        return (self.R * u.reshape(self.ntheta * self.nphi)).reshape(self.ntheta_H, self.nphi_H)

    def interpolate_padded(self, u):
        return (self.R.T * u.reshape(self.ntheta_H * self.nphi_H)).reshape(self.ntheta, self.nphi)



class DiagonalSmoother(object):

    def __init__(self, level):
        self.level = level

        # hammer operator on a tilesize x tilesize patch...
        u = np.zeros((level.ntheta, level.nphi))
        u[level.ntheta // 2, 0] = 1
        u = level.matvec_padded(u).real
        ## u = flatsky_adjoint_synthesis(u)
        ## u *= level.dl_fft
        ## u = flatsky_synthesis(u).real
        self.diag = u[level.ntheta // 2, 0] #* (1 + 1e-1)
        self.inv_diag = 1 / self.diag


    def apply(self, u):
        return OMEGA * self.inv_diag * u


def v_cycle(ilevel, levels, smoothers, b):
    if ilevel == len(levels) - 1:
        return smoothers[ilevel].apply(b)
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]

        x = b * 0
        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))

        for i in range(2):
            r_h = b - level.matvec(x)

            r_H = coarsen(level, next_level, r_h)

            c_H = v_cycle(ilevel + 1, levels, smoothers, r_H)

            c_h = interpolate(level, next_level, c_H)
            
            x += c_h

        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))
        return x

def coarsen(level, next_level, u):
    return next_level.pickvec(level.coarsen_padded(level.padvec(u)))

def interpolate(level, next_level, u):
    return level.pickvec(level.interpolate_padded(next_level.padvec(u)))
    

def coarsen_level(level):
    # produce next coarser level
    ntheta_H = level.ntheta // 2
    nphi_H = level.nphi // 2
    mask_H = level.coarsen_padded(level.mask)
    mask_H[mask_H < 0.5] = 0
    mask_H[mask_H != 0] = 1

    unitvec = np.zeros((ntheta_H, nphi_H))
    unitvec[ntheta_H // 2, nphi_H // 2] = 1
    image_of_operator = level.matvec_coarsened(unitvec)
    dl_fft_H = image_to_powspec(unitvec, image_of_operator)
    return Level(dl_fft_H, mask_H)


cur_level = root_level = Level(inner_dl_fft, mask)
levels = [cur_level]

while cur_level.n > 500:
    cur_level = coarsen_level(cur_level)
    levels.append(cur_level)
    
smoothers = [DiagonalSmoother(lev) for lev in levels]



matvec_count = 0

def inner_precond(b):
    x = v_cycle(0, levels, smoothers, b)
    return x

inner_matvec = root_level.matvec


def outer_matvec(u):
    global matvec_count
    matvec_count += 1
    u = root_level.padvec(u)
    u = flatsky_adjoint_synthesis(u)
    u *= outer_dl_fft
    u = flatsky_synthesis(u)
    u = root_level.pickvec(u).real
    return u


if SQRTSPLITFULLSOLVE:
    
    def jacobi(matvec, precond, b, n):
        x = precond(b)
        for i in range(n - 1):
            x += precond(b - matvec(x))
        return x

    def outer_precond(b):
        x = jacobi(inner_matvec, inner_precond, b, INNER_ITS)
        #return x
        x = jacobi(inner_matvec, inner_precond, x, INNER_ITS)
        return x

    
else:
    outer_precond = inner_precond


rng = np.random.RandomState(12)
x0 = rng.normal(size=root_level.n)
b = outer_matvec(x0)



solver = cg_generator(
    outer_matvec,
    b,
    M=outer_precond
    )

matvec_counts = []
errlst = []
norm0 = np.linalg.norm(x0)
for i, (x, r, delta_new) in enumerate(solver):
    errvec = x0 - x
    matvec_counts.append(matvec_count)
    errlst.append(np.linalg.norm(errvec) / norm0)

    print 'it', i
    if i > 20:
        break



#clf();
#imshow(padvec(x-x0), interpolation='none');
#semilogy(matvec_counts, errlst, '-o')
semilogy(errlst, '-o')
#colorbar()
draw()

## if 1:
##     A = hammer(level.matvec, level.n)
##     M = hammer(smoother.apply, level.n)

##     lam = np.asarray(sorted(np.linalg.eigvals(np.dot(M, A))))
##     clf()
##     semilogy(np.linalg.eigvalsh(A))
##     semilogy(np.linalg.eigvalsh(M)[::-1])
##     semilogy(lam)
##     print lam.max()
##     draw()
##     1/0

