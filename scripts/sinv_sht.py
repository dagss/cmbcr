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

from cmbcr.cr_system import load_map_cached
import healpy
from cmbcr import sharp
from healpy import mollzoom, mollview
from scipy.sparse import csc_matrix
#reload(cmbcr.main)

import sys
from cmbcr.cg import cg_generator

import scipy.ndimage

config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))

nside = 64
lmax_sh = int(4 * nside)

npix = 12 * nside**2

l = np.arange(1, lmax_sh + 2).astype(float)
dl = l**2.5


mask_hires = load_map_cached('mask_galactic_band_2048.fits')

class Level(object):
    def __init__(self, nside, dl, mask_hires, precond_lmin, omega=0.2):
        self.nside = nside
        self.lmax_sh = 4 * nside
        self.dl = dl[:self.lmax_sh + 1]

        self.inv_dl = dl.copy()[:self.lmax_sh + 1]
        self.inv_dl[:] = 1 / self.inv_dl
        self.inv_dl[:precond_lmin] = 0
        
        self.npix = 12 * nside**2
        self.mask = healpy.ud_grade(mask_hires, nside, order_in='RING', order_out='RING', power=0)
        t = 1
        self.mask[self.mask < t] = 0
        self.mask[self.mask >= t] = 1
        self.pick = (self.mask == 0)
        self.n_mask = int(self.pick.sum())
        self.omega = omega

    def padvec(self, u):
        u_pad = np.zeros(self.npix)
        u_pad[self.pick] = u
        return u_pad

    def pickvec(self, u):
        return u[self.pick]

    def matvec(self, u):
        u = self.padvec(u)
        x = sharp.sh_adjoint_synthesis(self.lmax_sh, u)
        x *= scatter_l_to_lm(self.dl)
        u = sharp.sh_synthesis(self.nside, x)
        return self.pickvec(u)
        

    def error_smoother(self, u):
        u = self.padvec(u)
        x = sharp.sh_analysis(self.lmax_sh, u)
        x *= scatter_l_to_lm(self.inv_dl)
        u = sharp.sh_adjoint_analysis(self.nside, x)
        return self.omega * self.pickvec(u)


class DenseLevel(Level):
    def __init__(self, *args, **kw):
        Level.__init__(self, *args, **kw)
        self.M = np.linalg.inv(hammer(self.matvec, self.n_mask))

    def error_smoother(self, u):
        return self.omega * np.dot(self.M, u)

        
        
    
def coarsen(level, next_level, u):
    u_pad = healpy.ud_grade(level.padvec(u), next_level.nside, order_in='RING', order_out='RING', power=0)
    u = next_level.pickvec(u_pad)
    return u

def interpolate(level, next_level, u):
    u_pad = healpy.ud_grade(next_level.padvec(u), level.nside, order_in='RING', order_out='RING', power=0)
    return level.pickvec(u_pad)

def v_cycle(ilevel, levels, b):
    if ilevel == len(levels) - 1:
        return levels[ilevel].error_smoother(b)
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]

        x = b * 0
        for i in range(3):
            x += level.error_smoother(b - level.matvec(x))

        for i in range(1):
            r_h = b - level.matvec(x)
            r_H = coarsen(level, next_level, r_h)

            c_H = v_cycle(ilevel + 1, levels, r_H)

            c_h = interpolate(level, next_level, c_H)
            x += c_h

        for i in range(3):
            x += level.error_smoother(b - level.matvec(x))
        return x

        
def precond(b):
    return v_cycle(0, levels, b)
        

omega = 0.1

levels = []

nside_H = nside
while nside_H > 32:
    levels.append(Level(nside_H, dl, mask_hires, 4 * nside_H // 8, omega=omega))
    nside_H //= 2
    

levels.append(DenseLevel(nside_H, dl, mask_hires, nside_H // 2, omega=1))


root_level = levels[0]


rng = np.random.RandomState(12)
x0 = rng.normal(size=root_level.n_mask)
b = root_level.matvec(x0)
x = x0 * 0


if 0:
    A = hammer(root_level.matvec, root_level.n_mask)
    M = hammer(root_level.error_smoother, root_level.n_mask)
    clf()
    
    semilogy(sorted(np.linalg.eigvalsh(A)))
    semilogy(sorted(np.linalg.eigvalsh(M))[::-1])

    w, v = np.linalg.eig(np.dot(M, A))
    i = np.argsort(w)
    w = w[i]
    print w.max()
    v = v[:, i]
    #1/0
    
    #semilogy(sorted(np.linalg.eigvals(np.dot(M, A))))
    semilogy(w)
    draw()
    1/0
 
if 0:
    clf();
    #mollview(levels[0].padvec( x-x0 ), hold=True, xsize=1000)

    #x0[:] = 1
    x0_dg = rng.normal(size=levels[1].n_mask)
    x0 = interpolate(levels[0], levels[1], x0_dg)
    mollview(levels[1].padvec(x0_dg), sub=211, xsize=1000)
    mollview(levels[0].padvec(x0), sub=212, xsize=1000)
    

    1/0


norm0 = np.linalg.norm(x0)
errlst = []

solver = cg_generator(
    root_level.matvec,
    b,
    M=precond,
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

