from __future__ import division
import numpy as np

from matplotlib.pyplot import *

import logging
logging.basicConfig(level=logging.DEBUG)

import healpy
from scipy.sparse import csc_matrix

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

#reload(cmbcr.main)

import sys
from cmbcr.cg import cg_generator



def csc_neighbours(nside, pick):
    
    pixels = pick.nonzero()[0]
    pixels_reverse = np.zeros(pick.shape, dtype=int)
    pixels_reverse[pixels] = np.arange(pixels.shape[0])

    length = pixels.shape[0]
    indices = np.zeros(9 * length, dtype=np.int)
    indptr = np.zeros(length + 1, dtype=np.int)
    neighbours = healpy.get_all_neighbours(nside, pixels, nest=False)
    idx = 0
    for j, ipix in enumerate(pixels):
        indptr[j] = idx
        neighlst = neighbours[:, j]
        neighlst = neighlst[(neighlst != -1) & pick[neighlst]]
        n = neighlst.shape[0]
        indices[idx] = j

        i_arr = pixels_reverse[neighlst]
        indices[idx + 1:idx + 1 + n] = i_arr

        #data[idx] = 1.0
        #data[idx + 1:idx + 1 + n] = x[j] * x[i_arr] + y[j] * y[i_arr] + z[j] * z[i_arr]
        
        idx += n + 1

    indptr[-1] = idx
    indices = indices[:idx]
    data = np.ones(idx)
    return csc_matrix((data, indices, indptr), shape=(length, length))


def make_Si_sparse_matrix(Si_pattern, dl, ridge, pixels):
    x, y, z = healpy.pix2vec(nside, pixels, nest=False)

    data = np.zeros_like(Si_pattern.data)
    for j in range(pixels.shape[0]):
        i_arr = Si_pattern.indices[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]]
        data[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]] = cmbcr.beam_by_cos_theta(
            dl,
            (x[j] * x[i_arr] + y[j] * y[i_arr] + z[j] * z[i_arr]))
        diag_ind = Si_pattern.indptr[j] + (i_arr == j).nonzero()[0][0]
        data[diag_ind] += ridge

    return csc_matrix((data, Si_pattern.indices, Si_pattern.indptr), shape=Si_pattern.shape)





config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))


w = 1

nside = 128 * w
factor = 2048 // nside * w



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


mask = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='RING', power=0)
mask[mask != 1] = 0
pick = (mask == 0)
n = int(pick.sum())


rng = np.random.RandomState(1)

x0 = [
    #scatter_l_to_lm(1. / dl) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)


from cmbcr.precond_psuedoinv import *

x = lstscale(0, b)

from cmbcr import sympix, block_matrix, sympix_mg


class SinvSolver(object):
    def __init__(self, system, nside, level0=None, diag_factor=1, ridge_factor=5e-2):
        self.level0 = level0
        self.system = system
        self.nside = nside

        self.mask = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='RING', power=0)
        self.mask[self.mask <= 0.9] = 0
        self.mask[self.mask > 0.9] = 1
        
        self.pickvec = (self.mask == 0)
        self.n = int(self.pickvec.sum())
        self.npix = self.mask.shape[0]

        self.Si_inv_dense = None

        if level0 is None:
            dl = system.dl_list[0]
            nl = cmbcr.standard_needlet_by_l(2, 2 * dl.shape[0] - 1)
            i = nl.argmax()
            dl = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])
            Si_pattern = csc_neighbours(nside, pick)
            Si_pattern = Si_pattern * Si_pattern * Si_pattern
            Si_pattern.sum_duplicates()
            lmax = dl.shape[0] - 1
            self.diag_val = cmbcr.beam_by_cos_theta(dl, np.ones(1))[0]
            self.Si_sparse = make_Si_sparse_matrix(Si_pattern, dl, self.diag_val * ridge_factor, self.pickvec.nonzero()[0])
        elif nside <= 8:
            self.Si_inv_dense = np.linalg.inv(hammer(self.matvec, self.n))
        else:
            self.diag_val = self.level0.diag_val * diag_factor


    def pad(self, x):
        x_pad = np.zeros(self.npix)
        x_pad[self.pickvec] = x
        return x_pad

    def pick(self, x):
        return x[self.pickvec]

    def norm_by_l(self, x):
        return cmbcr.norm_by_l(sharp.sh_analysis(self.system.lmax_list[0], self.pad(x)))

    def matvec(self, x):
        if self.level0 is None:
            # root level
            return (self.Si_sparse * x)
        else:
            # interpolate to root level and invoke root level
            x_pad = self.pad(x)
            x_pad = healpy.ud_grade(x_pad, self.level0.nside, order_in='RING', order_out='RING', power=0)
            x_pad = self.level0.pad(self.level0.matvec(self.level0.pick(x_pad)))
            x_pad = healpy.ud_grade(x_pad, self.nside, order_in='RING', order_out='RING', power=0)
            return self.pick(x_pad)
            

    def error_smoother(self, x):
        if self.Si_inv_dense is not None:
            return np.dot(self.Si_inv_dense, x)
        else:
            return x / self.diag_val



level0 = SinvSolver(system, nside)

## nside_H = nside // 8

## mask_H = healpy.ud_grade(system.mask, nside_H, order_in='RING', order_out='RING', power=0)
## mask_H[mask_H <= 0.5] = 0
## mask_H[mask_H > 0.5] = 1
## pick_H = (mask_H == 0)
## n_H = int(pick_H.sum())

## u = np.zeros(n_H)
## u[10] = 1

## u_pad = np.zeros(12*nside_H**2)
## u_pad[pick_H] = u
## u_pad = healpy.ud_grade(u_pad, nside, order_in='RING', order_out='RING', power=0)

## u_pad = level0.pad(level0.matvec(u_pad[pick]))

## u_pad = healpy.ud_grade(u_pad, nside_H, order_in='RING', order_out='RING', power=0)
## print nside_H, u_pad.max()


## #




levels = [
    level0,
    SinvSolver(system, nside // 2, diag_factor=2, level0=level0),
    SinvSolver(system, nside // 4, diag_factor=4, level0=level0), 
    SinvSolver(system, nside // 8, diag_factor=8, level0=level0), 
    SinvSolver(system, nside // 16, diag_factor=16, level0=level0), 
    #SinvSolver(system, nside//2),

]

#u = np.zeros(levels[-1].n)
#u[10] = 1

#clf()
#mollview(levels[-1].pad(levels[-1].matvec(u)), sub=111)
#draw()
#1/0

    
    
def vcycle(levels, ilevel, b):
    if ilevel == len(levels) - 1:
        return levels[ilevel].error_smoother(b)
    else:
        level, next_level = levels[ilevel], levels[ilevel + 1]

        # pre-smooth
        x = None
        for i in range(1):
            if x is None:
                x = level.error_smoother(b)
            else:
                r = b - level.matvec(x)
                x += level.error_smoother(r)

        # V
        for i in range(2):
            r_h = b - level.matvec(x)

            
            r_H = next_level.pick(healpy.ud_grade(level.pad(r_h), next_level.nside, order_in='RING', order_out='RING', power=0))
            c_H = vcycle(levels, ilevel + 1, r_H)
            c_h = level.pick(healpy.ud_grade(next_level.pad(c_H), level.nside, order_in='RING', order_out='RING', power=0))
            x += c_h

        # post-smooth
        for i in range(1):
            r = b - level.matvec(x)
            x += level.error_smoother(r)

        return x

    

if 0:
    from scipy.linalg import eigvals
    A = hammer(sinv_solver.matvec, sinv_solver.n)
    M = hammer(precond, sinv_solver.n)
    clf()
    semilogy(sorted(np.abs(eigvals(np.dot(M, A)))))
    draw()


else:

    

    rng = np.random.RandomState(1)
    x0 = rng.normal(size=levels[0].n)
    b = levels[0].matvec(x0)
    x = x0 * 0


    def precond(b):
        return vcycle(levels, 0, b)
    
    clf()

    solver = cg_generator(
     levels[0].matvec,
     b,
     M=precond,
     )

    x0norm = levels[0].norm_by_l(x0)
    
    #for i, (x, r, delta_new) in enumerate(solver):
    for i in range(2000):
        r = b - levels[0].matvec(x)
        x += precond(r)

        errvec = x0 - x
        semilogy(levels[0].norm_by_l(errvec) / x0norm, label=int(i))
        
        print i
        
        if i > 20:
            break

    draw()

