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

lmax = 400
lmax_sh = 2 * lmax

l = np.arange(1, lmax_sh).astype(float)
dl = l**1.5
    
lmax_sh = dl.shape[0] - 1
nrings = lmax + 1



mask_hires = load_map_cached('mask_galactic_band_2048.fits')

def cl_to_flatsky(cl, nx, ny, nl):
    out = np.zeros((nx, ny))

    l = np.sqrt(
        (np.fft.fftfreq(nx, 1. / (2 * nl))[:, None])**2
        + (np.fft.fftfreq(ny, 1. / (2 * nl))[None, :])**2)

    cl_flat = scipy.ndimage.map_coordinates(cl, l[None, :, :], mode='constant', order=1)
    return cl_flat


class Level(object):

    def __init__(self, lmax, mask, dl):
        self.lmax = lmax
        self.nrings = nrings = lmax + 1
        
        mask_lm = sharp.sh_analysis(lmax, mask)
        mask_gauss = sharp.sh_synthesis_gauss(lmax, mask_lm)
        mask_gauss[mask_gauss < 0.9] = 0
        mask_gauss[mask_gauss >= 0.9] = 1
        self.mask = mask_gauss
        self.pick = (mask_gauss == 0)
        self.n_mask = int(self.pick.sum())
        
        self.ind_map = np.zeros(2 * nrings**2, dtype=int)
        self.ind_map[:] = -1
        self.ind_map[self.pick] = np.arange(self.n_mask)
        self.ind_map = self.ind_map.reshape((nrings, 2 * nrings))

        self.dl_fft = cl_to_flatsky(dl, nrings, 2 * nrings, lmax + 1)

        self.dl_fft_inv = 1 / self.dl_fft

        self.nrows = self.nrings
        self.ncols = 2 * self.nrings
        
        unitvec = np.zeros((nrings, 2 * nrings))
        unitvec[nrings // 2, nrings] = 1
        
        z = sharp.sh_adjoint_synthesis_gauss(lmax, unitvec.reshape(2 * nrings**2))
        z *= scatter_l_to_lm(dl[:lmax + 1])
        z = sharp.sh_synthesis_gauss(lmax, z)
        estimated_max_sht = z.reshape((nrings, 2 * nrings))[nrings // 2, nrings]

        z = self.flatsky_adjoint_synthesis(unitvec)
        z *= self.dl_fft
        z = self.flatsky_synthesis(z).real
        estimated_max_fft = z[nrings // 2, nrings]

        self.dl_fft *= estimated_max_sht / estimated_max_fft
        self.dense = False

    def compute_dense(self):
        self.dense = True
        self.M = np.linalg.inv(hammer(self.matvec, self.n_mask))

    def pickvec(self, u):
        return u.reshape(2 * self.nrings**2)[self.pick]

    def padvec(self, u):
        u_pad = np.zeros(2 * self.nrings**2)
        u_pad[self.pick] = u.real
        return u_pad.reshape(self.nrings, 2 * self.nrings)

    def flatsky_synthesis(self, u):
        return np.fft.ifftn(u) * np.prod(u.shape)

    def flatsky_adjoint_synthesis(self, u):
        return np.fft.fftn(u)

    def flatsky_analysis(self, u):
        return np.fft.fftn(u) / np.prod(u.shape)

    def flatsky_adjoint_analysis(self, u):
        return np.fft.ifftn(u)

    def matvec(self, u):
        u = self.padvec(u)
        u = self.flatsky_adjoint_synthesis(u)
        u *= self.dl_fft
        u = self.flatsky_synthesis(u)
        u = self.pickvec(u).real
        return u

    def precond(self, u):
        if self.dense:
            return np.dot(self.M, u)
        else:
            u = self.padvec(u)
            u = self.flatsky_analysis(u)
            u *= self.dl_fft_inv
            u = self.flatsky_adjoint_analysis(u)
            return self.pickvec(u).real

    
from scipy.sparse import dok_matrix

def coarsen_matrix(coarse_level, fine_level):
    R = dok_matrix((coarse_level.n_mask, fine_level.n_mask))

    def add(coarse_i, coarse_j, fine_i, fine_j, weight):
        # wrap around fine_i and fine_j..
        fine_i = fine_i % fine_level.nrows
        fine_j = fine_j % fine_level.ncols

        coarse_ind = coarse_level.ind_map[coarse_i, coarse_j]
        fine_ind = fine_level.ind_map[fine_i, fine_j]
        if coarse_ind != -1 and fine_ind != -1:
            R[coarse_ind, fine_ind] = weight
    
    for i in range(coarse_level.nrows):
        for j in range(coarse_level.ncols):
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


def v_cycle(ilevel, levels, b):
    if ilevel == len(levels) - 1:
        return levels[ilevel].precond(b)
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]

        x = b * 0
        for i in range(1):
            x += level.precond(b - level.matvec(x))

        for i in range(1):
            r_h = b - level.matvec(x)

            r_H = coarseners[ilevel] * r_h

            c_H = v_cycle(ilevel + 1, levels, r_H)

            c_h = coarseners[ilevel].T * c_H
            
            x += c_h

        for i in range(1):
            x += level.precond(b - level.matvec(x))
        return x

def precond(b):
    return v_cycle(0, levels, b)



levels = []
lmax_H = lmax
while True:
    lev = Level(lmax_H, mask_hires, dl)
    levels.append(lev)
    if lev.n_mask < 1000:
        lev.compute_dense()
        break
    lmax_H //= 2

    
coarseners = []
for i in range(len(levels) - 1):
    coarseners.append(coarsen_matrix(levels[i + 1], levels[i]))

level = levels[0]


rng = np.random.RandomState(12)
x0 = rng.normal(size=level.n_mask)
b = level.matvec(x0)
x = x0 * 0


norm0 = np.linalg.norm(x0)
errlst = []

solver = cg_generator(
    level.matvec,
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
    if i > 20:
        break



#clf();
#imshow(padvec(x-x0), interpolation='none');
semilogy(errlst, '-o')
#colorbar()
draw()

