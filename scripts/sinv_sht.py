from __future__ import division
import numpy as np

from matplotlib.pyplot import *

import logging
logging.basicConfig(level=logging.DEBUG)


import healpy
import cmbcr
import cmbcr.utils
reload(cmbcr.cg)
reload(cmbcr.beams)
reload(cmbcr.cr_system)
reload(cmbcr.precond_sh)
reload(cmbcr.precond_psuedoinv)
reload(cmbcr.precond_diag)
reload(cmbcr.precond_pixel)
reload(cmbcr.utils)
reload(cmbcr.multilevel)
reload(cmbcr.masked_solver)
reload(cmbcr)
from cmbcr.utils import *
from cmbcr.healpix import nside_of
from cmbcr import sharp
from healpy import mollzoom, mollview
from scipy.sparse import csc_matrix, dok_matrix
#reload(cmbcr.main)

from cmbcr.cr_system import load_map_cached

import sys
from cmbcr.cg import cg_generator

import scipy.ndimage

nside = 256
lmax = 3 * nside



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
Cl_cmb = Cl_cmb[:6000]

from scipy.interpolate import interp1d
Cl_func = interp1d(np.arange(Cl_cmb.shape[0]), Cl_cmb)
Cl = Cl_func(np.linspace(0, Cl_cmb.shape[0] - 1, lmax + 1))


#l = np.arange(1, lmax + 2).astype(float)
#dl = l**6

dl = 1 / Cl
dl_orig = 1 / Cl

rl = cmbcr.fourth_order_beam(lmax, lmax // 2, 0.1)
rl = np.sqrt(rl)
dl *= rl**2


mask_healpix = load_map_cached('mask_galactic_band_2048.fits')

mask = healpy.ud_grade(mask_healpix, order_in='RING', order_out='RING', nside_out=nside, power=0)
mask[mask != 0] = 1
mask[:] = 0

pw = 1.8

class Level(object):

    def __init__(self, dl, mask):
        self.mask = mask
        self.lmax = dl.shape[0] - 1
        self.nside = nside_of(mask)
        self.npix = 12 * self.nside**2
        self.dl = dl

        self.pick = (mask == 0)
        self.n = int(self.pick.sum())

    def compute_diagonal(self):
        u = np.zeros(self.npix)
        u[self.npix // 2] = 1
        u = self.matvec_padded(u)
        return u[self.npix // 2]

    def pickvec(self, u):
        return u[self.pick]

    def padvec(self, u):
        u_pad = np.zeros(self.npix)
        u_pad[self.pick] = u
        return u_pad

    def matvec_padded(self, u):
        u = sharp.sh_adjoint_synthesis(self.lmax, u)
        u *= scatter_l_to_lm(self.dl)
        u = sharp.sh_synthesis(self.nside, u)
        return u

    def matvec(self, u):
        return self.pickvec(self.matvec_padded(self.padvec(u)))

    def matvec_coarsened(self, u):
        # do matvec on the next, coarser level. This is just done once, to create the operator on the next level
        return self.coarsen_padded(self.matvec_padded(self.interpolate_padded(u)))

    def coarsen_padded(self, u):
        alm = sharp.sh_analysis(self.lmax, u)
        alm *= scatter_l_to_lm(cmbcr.gaussian_beam_by_l(self.lmax, 2 * np.pi / (4 * self.nside) * pw))
        u = sharp.sh_synthesis(self.nside, alm)
        return healpy.ud_grade(u, order_in='RING', order_out='RING', nside_out=self.nside // 2, power=0)

    def interpolate_padded(self, u):
        u = healpy.ud_grade(u, order_in='RING', order_out='RING', nside_out=self.nside, power=0) * 0.25
        alm = sharp.sh_adjoint_synthesis(self.lmax, u)
        alm *= scatter_l_to_lm(cmbcr.gaussian_beam_by_l(self.lmax, 2 * np.pi / (4 * self.nside) * pw))
        u = sharp.sh_adjoint_analysis(self.nside, alm)
        return u


class DiagonalSmoother(object):
    def __init__(self, level):
        self.level = level

        self.diag = level.compute_diagonal()
        self.inv_diag = 1 / self.diag

    def apply(self, u):
        return 0.3 * self.inv_diag * u


class DenseSmoother(object):
    def __init__(self, level):
        self.matrix = hammer(level.matvec, level.n)
        self.inv_matrix = np.linalg.inv(self.matrix)

    def apply(self, u):
        return np.dot(self.inv_matrix, u)
    

def operator_image_to_power_spectrum(lmax, unitvec, opimage):
    # unitvec: unit-vector in flatsky basis
    # x: image of operator
    YtW_x = sharp.sh_analysis(lmax, opimage)
    Yt_u = sharp.sh_adjoint_synthesis(lmax, unitvec)
    return YtW_x[:lmax + 1] / Yt_u[:lmax + 1]



def coarsen_level(level):
    nside_H = level.nside // 2
    unitvec = np.zeros(12 * nside_H**2)
    unitvec[6 * nside_H**2 + 2 * nside_H] = 1

    image_of_operator = level.matvec_coarsened(unitvec)

    YtW_x = sharp.sh_analysis(lmax, image_of_operator)
    Yt_u = sharp.sh_adjoint_synthesis(lmax, unitvec)
    dl_H = operator_image_to_power_spectrum(lmax // 2, unitvec, image_of_operator)

    mask_H = level.coarsen_padded(level.mask)
    mask_H[mask_H < 0.5] = 0
    mask_H[mask_H != 0] = 1
    return Level(dl_H, mask_H)



#dl[:] = 1
root_level = Level(dl, mask)
levels = [root_level]

cur_level = root_level
while cur_level.n > 100:
    cur_level = coarsen_level(cur_level)
    levels.append(cur_level)


smoothers = [DiagonalSmoother(level) for level in levels[:-1]]
smoothers.append(DenseSmoother(levels[-1]))




# Test

rng = np.random.RandomState(11)


if 0:
    nside_H = nside // 2
    u = np.zeros(12* (nside // 2)**2)
    u[6 * (nside // 2)**2 + 20] = 1
    #u  = rng.normal(size=12 * (nside_H)**2)

    alm = sharp.sh_analysis(lmax, u)
    alm *= scatter_l_to_lm(cmbcr.gaussian_beam_by_l(lmax, '10 deg'))
    u = sharp.sh_synthesis(nside_H, alm)
    #u[:] = 1


    Au1 = levels[0].matvec_coarsened(u)
    Au2 = levels[1].matvec_padded(u)

    clf()
    mollview(Au1, sub=211, fig=gcf().number)
    mollview(Au2, sub=212, fig=gcf().number)
    draw()
    1/0



def coarsen(level, next_level, u):
    return next_level.pickvec(level.coarsen_padded(level.padvec(u)))

def interpolate(level, next_level, u):
    return level.pickvec(level.interpolate_padded(next_level.padvec(u)))    


def v_cycle(ilevel, levels, smoothers, b):
    if ilevel == len(levels) - 1:
        return smoothers[ilevel].apply(b)
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]

        x = b * 0
        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))

        for i in range(1):
            r_h = b - level.matvec(x)

            r_H = coarsen(level, next_level, r_h)

            c_H = v_cycle(ilevel + 1, levels, smoothers, r_H)

            c_h = interpolate(level, next_level, c_H)
            
            x += c_h

        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))
        return x

    


TOP_LEVEL_SH = True


errlst = []

#if x0 is not None:
#    x0_norm = np.linalg.norm(x0)

clf()
maxit = 10


#x0 = rng.normal(size=root_level.n)
#b = root_level.matvec(x0)

#x0 = rng.norm

if TOP_LEVEL_SH:

    def precond(b):
        x = scatter_l_to_lm(1 / dl_orig) * b

        b_pix = levels[0].pickvec(sharp.sh_synthesis(nside, b * scatter_l_to_lm(rl)))
        x_pix = v_cycle(0, levels, smoothers, b_pix)
        lo_corr = sharp.sh_adjoint_synthesis(lmax, levels[0].padvec(x_pix)) * scatter_l_to_lm(rl)

        x += lo_corr
        return x


    def matvec(u):
        return u * scatter_l_to_lm(dl_orig)


    
    x0 = rng.normal(size=(lmax + 1)**2)
    x0_l = cmbcr.norm_by_l(x0)
else:
    matvec = levels[0].matvec

    def precond(b):
        return v_cycle(0, levels, smoothers, b)
    
    x0 = rng.normal(size=levels[0].n)
    x0_l = cmbcr.norm_by_l(sharp.sh_analysis(lmax, levels[0].padvec(x0)))

    

b = matvec(x0)

cgsolver = cg_generator(
    matvec,
    b,
    M=precond
    )



rlm = scatter_l_to_lm(rl)

x0_norm = np.linalg.norm(x0)

#x = np.zeros_like(x0)
#for i in range(10):

#    x += precond(b - matvec(x))

    
for i, (x, r, delta_new) in enumerate(cgsolver):
    #r = np.linalg.norm(r) / b_norm
    #reslst.append(r)
    
    e = np.linalg.norm(x0 - x) / x0_norm
    errlst.append(e)
    print 'OUTER iteration {}, err={}'.format(i, e)

    elm = x - x0
    if not TOP_LEVEL_SH:
        elm = sharp.sh_analysis(lmax, levels[0].padvec(elm))
    semilogy(cmbcr.norm_by_l(elm) / x0_l)
    
    if i > maxit:
        break

draw()

def errmap():
    clf()
    mollview(sharp.sh_synthesis(nside, x0 - x), fig=gcf().number)
    draw()
1/0    
    
clf()
semilogy(errlst, '-o')
draw()
