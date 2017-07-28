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

#reload(cmbcr.main)

import sys
from cmbcr.cg import cg_generator

config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))


w = 1

nside = 16 * w
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



dl = system.dl_list[0]


#dl = 1 / cmbcr.gaussian_beam_by_l(system.lmax_list[0], '10 deg')

rng = np.random.RandomState(1)

x0 = [
    scatter_l_to_lm(1. / dl) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)


from cmbcr.precond_psuedoinv import *

x = lstscale(0, b)

from cmbcr import sympix, block_matrix

lmax = 3 * nside
tilesize = 8
corner_factor = 0.6

grid = sympix.make_sympix_grid(lmax + 1, tilesize, n_start=8)

mask_lm = sharp.sh_analysis(lmax, system.mask)
plan = sharp.SymPixGridPlan(grid, lmax)
mask = plan.synthesis(mask_lm)
mask[mask <= 0.8] = 0
mask[mask > 0.8] = 1
if 1:
    imshow(sympix.sympix_plot(grid, mask), interpolation='none')
    draw()
    1/0

neighmat, (label_to_i, label_to_j) = sympix.sympix_csc_neighbours(grid, lower_only=False, corner_factor=corner_factor)
neighmat_lower, (label_to_i_lower, label_to_j_lower) = sympix.sympix_csc_neighbours(grid, lower_only=True, corner_factor=corner_factor)

#A_sparse = block_matrix.BlockMatrix(neighmat_lower.indptr, neighmat_lower.indices, blockshape=(tilesize**2, tilesize**2))
#mask_p = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='RING', power=0)
#mask_p[mask_p != 1] = 0
#pick = (mask_p == 0)


# estimate diagonal value for ridging -- just ignore the mask then...
n = int(pick.sum())

z = np.zeros_like(mask_p)
z[0] = 1
z = sharp.sh_adjoint_synthesis(system.lmax_list[0], z)
z *= scatter_l_to_lm(dl)
z = sharp.sh_synthesis(nside, z)
estimated_max = z[0]

ridge = 5e-4 * estimated_max


def f(u_in):
    u_pad = np.zeros_like(mask_p)
    u_pad[pick] = u_in
    u = sharp.sh_adjoint_synthesis(system.lmax_list[0], u_pad)
    u *= scatter_l_to_lm(dl)
    u_pad = sharp.sh_synthesis(nside, u)
    return u_pad[pick] + ridge * u_in


if 'op' in sys.argv:
    
    def op(i):
        u = np.zeros(12*nside**2)
        u[i] = 1
        alm = sharp.sh_adjoint_synthesis(system.lmax_list[0], u)
        alm *= scatter_l_to_lm(dl)
        return sharp.sh_synthesis(nside, alm)

    def doit(i):
        clf()
        u = op(i)
        mollview(u, fig=gcf().number, sub=111)
        draw()

    t = 10
    if 'up' in sys.argv:
        doit(6*nside**2 + 2 * nside - t * 4 * nside)
    else:
        doit(6*nside**2 + 2 * nside + t * 4 * nside)
    1/0


if 1:
    #Q = hammer(f, int(pick.sum()))
    #1/0
    def precond(u):
        return u / estimated_max #u / Q.diagonal()
else:
    def precond(u):
        u_pad = np.zeros_like(mask_p)
        u_pad[pick] = u
        u = sharp.sh_analysis(system.lmax_list[0], u_pad)
        u *= scatter_l_to_lm(1. / dl)
        u_pad = sharp.sh_adjoint_analysis(nside, u)
        return u_pad[pick]
    


def pad_vec(u):
    u_pad = np.zeros_like(mask_p)
    u_pad[pick] = u
    return u_pad
    
rng = np.random.RandomState(1)
x0 = rng.normal(size=n)
b = f(x0)



solver = cg_generator(
    f,
    b,
    M=precond,
    )


errlst = []
#clf()
for i, (x, r, delta_new) in enumerate(solver):
    errvec = x0 - x
    errlst.append(np.linalg.norm(errvec) / np.linalg.norm(x0))

    #ww = pad_vec(errvec)
    #semilogy(cmbcr.norm_by_l(sharp.sh_analysis(system.lmax_list[0], ww)), label=int(i))
    
    if i > 50:
        break

#clf()
#mollview(pad_vec(errvec), fig=gcf().number)
#legend()
semilogy(errlst, '-o')
draw()

