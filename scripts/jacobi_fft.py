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
reload(cmbcr.masked_solver)
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
system.prepare(use_healpix=True)

if 0:
    nl = cmbcr.standard_needlet_by_l(2, system.lmax_list[0])
    i = nl.argmax()
    nl[:i] = nl.max()
    nl /= nl.max()
    rl = nl
else:
    #rl = cmbcr.fourth_order_beam(system.lmax_list[0], int(0.5 * system.lmax_list[0]), epstreshold=0.5)
    #rl = np.ones(system.lmax_list[0] + 1)
    rl = cmbcr.gaussian_beam_by_l(system.lmax_list[0], '1.25 deg')

#system.dl_list[0] *= rl**2

#rl = np.sqrt(nl)
#rl = nl
#rl[:] = 1

if 'plot' in sys.argv:
    clf()
    system.plot()
    plot(system.dl_list[0] * rl**2)
    1/0

SQRTSPLIT = False

dl = system.dl_list[0]

if SQRTSPLIT:
    #dl = np.sqrt(dl) ## why does it converge faster with this in place?
    dl = np.sqrt(dl)

    
#sinv_solver = cmbcr.SinvSolver(dl, system.mask, b=1.2, lmax_factor=6, split=False)

udgrade = 2048
mask_for_sinv = np.ones(12*udgrade**2)
k = 0 * (2048//nside)
mask_for_sinv[int(5.5*udgrade**2) - 2*udgrade + k * 4 * nside:int(6.5*udgrade**2)+2*udgrade - k * 4 * udgrade] = 0



sinv_solver = cmbcr.SinvSolver(dl * rl**2, mask_for_sinv, split=False, rl=np.ones(dl.shape[0]), #, rl=rl,
                               nrings=(system.lmax_list[0] + 1) // 2)

#def solve_mask(b):
#    x = sinv_solver.solve_alm(b, repeat=2 if SQRTSPLIT else 1, single_v_cycle=False, rtol=1e-3, maxit=50)
#    #if SQRTSPLIT:
#    #    x = sinv_solver.solve_alm(x, single_v_cycle=False, rtol=1e-3, maxit=50)
#    return x

def solve_mask(b):
    self = sinv_solver
    x = b

    args = ()
    kw = {'rtol': 1e-3}
    
    x = x * scatter_l_to_lm(rl)
    x = self.pickvec(self.gauss_grid_to_equator(sharp.sh_synthesis_gauss(self.lmax, x, lmax_sh=self.lmax_sh)))
    x, _, _ = self.solve_mask(x, *args, **kw)

    if SQRTSPLIT:
        npix = 2 * self.nrings**2
        x *= npix / 4 / np.pi
    
        x, _, _ = self.solve_mask(x, *args, **kw)


    
    x = sharp.sh_adjoint_synthesis_gauss(self.lmax, self.equator_to_gauss_grid(self.padvec(x)), lmax_sh=self.lmax_sh)
    x = x * scatter_l_to_lm(rl)

    return x

rng = np.random.RandomState(1)

x0 = [
    #scatter_l_to_lm(1. / system.dl_list[k]) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)

from cmbcr.precond_psuedoinv import lstadd, lstsub, lstmul, lstscale

precond_1 = cmbcr.PsuedoInversePreconditioner(system)


if 0:
    nsh = (system.lmax_list[0] + 1)**2
    M2 = hammer(solve_mask, nsh)
    M1 = hammer(lambda x: precond_1.apply([x])[0], nsh)
    A = hammer(lambda x: system.matvec([x])[0], nsh)

    clf()
    lam_M2 = np.linalg.eigvalsh(M2)
    lam_M1 = np.linalg.eigvalsh(M1)
    lam_A = np.linalg.eigvalsh(A)

    semilogy(lam_M2)
    semilogy(lam_M1)
    semilogy(lam_A)
    
    1/0



if 0:
    x = np.zeros(12 * nside**2)
    x[6 * nside**2 + 2 * nside + 8 * 4 * nside] = 1
    u = sharp.sh_analysis(system.lmax_list[0], x)
    u *= scatter_l_to_lm(cmbcr.gaussian_beam_by_l(system.lmax_list[0], '2 deg'))

    v = system.matvec([u])[0]

    v1 = precond_1.apply([v])[0]
    v2 = solve_mask(v)
    
    clf()
    mollview(sharp.sh_synthesis(nside, v), sub=221, title='Au')
    mollview(sharp.sh_synthesis(nside, v1), sub=222, title='Psuedo-inv')
    mollview(sharp.sh_synthesis(nside, v2), sub=223, title='Mask solver')
    mollview(sharp.sh_synthesis(nside, v1 + v2), sub=224, title='Both solvers')
    draw()
    1/0
    
import healpy
mask_nside = nside
mask_deg = healpy.ud_grade(system.mask, mask_nside, order_in='RING', order_out='RING', power=-1)
mask_deg[mask_deg != 0] = 1

def filtered_precond1(u):
    u = u[0]
    u = sharp.sh_synthesis(mask_nside, u)
    u *= mask_deg
    u = sharp.sh_analysis(system.lmax_list[0], u)
    u = precond_1.apply([u])[0]
    u = sharp.sh_adjoint_analysis(mask_nside, u)
    u *= mask_deg
    u = sharp.sh_adjoint_synthesis(system.lmax_list[0], u)
    return [u]
    

sinv_pick = sinv_solver.mask.reshape(2 * sinv_solver.nrings**2) == 0
def restrict(x):
    self = sinv_solver
    x = x * scatter_l_to_lm(rl)
    #x = sharp.sh_synthesis_gauss(self.lmax, x, lmax_sh=self.lmax_sh)
    #x = x[sinv_pick]
    #return x
    x = self.pickvec(self.gauss_grid_to_equator(sharp.sh_synthesis_gauss(self.lmax, x, lmax_sh=self.lmax_sh)))
    return x

def prolong(x):
    self = sinv_solver
    x = sharp.sh_adjoint_synthesis_gauss(self.lmax, self.equator_to_gauss_grid(self.padvec(x)), lmax_sh=self.lmax_sh)
    #xpad = np.zeros((2 * self.nrings**2))
    #xpad[sinv_pick] = x
    #x = sharp.sh_adjoint_synthesis_gauss(self.lmax, xpad, lmax_sh=self.lmax_sh)
    x = x * scatter_l_to_lm(rl)
    return x

if 0:

    import healpy
    mask_nside = 16
    mask_deg = healpy.ud_grade(system.mask, mask_nside, order_in='RING', order_out='RING', power=-1)
    mask_deg[mask_deg != 0] = 1
    mask_deg_pick = (mask_deg == 0)
    def restrict(x):
        x = sharp.sh_synthesis(mask_nside, x)
        return x[mask_deg_pick]

    def prolong(x):
        xpad = np.zeros(12 * mask_nside**2)
        xpad[mask_deg_pick] = x
        return sharp.sh_adjoint_synthesis(system.lmax_list[0], xpad)


def ZAZ(x):
    x = prolong(x)
    x *= scatter_l_to_lm(system.dl_list[0])
    #x = system.matvec([x])[0]
    x = restrict(x)
    return x

#ZAZ_dense = hammer(ZAZ, sinv_solver.n)
#ZAZ_dense += np.eye(ZAZ_dense.shape[0]) * 1e-3 * ZAZ_dense.max()
#ZAZ_inv = np.linalg.inv(ZAZ_dense)


def mask_dense_solver(x):
    x = restrict(x)
    x = np.dot(ZAZ_inv, x)
    return prolong(x)
    
#solve_mask = mask_dense_solver
#1/0


def precond_both(b):

    if 0:
        return precond_1.apply(b)
    elif 0:
        x = [solve_mask(b[0])]
        return x
    elif 1:
        x = precond_1.apply(b)
        x = lstadd(x, [solve_mask(b[0])])
        return x
    elif 0:
        x = filtered_precond1(b)
        x = lstadd(x, [solve_mask(b[0])])
        return x
    else:
        x = b
        x = lstadd(x, lstscale(0.5, precond_1.apply(b)))
        r = lstsub(b, system.matvec(x))
        x = lstadd(x, [solve_mask(r[0])])

        r = lstsub(b, system.matvec(x))
        x = lstadd(x, lstscale(0.5, precond_1.apply(b)))
#        x = lstadd(x, precond_1.apply(r))

        return x

from cmbcr import norm_by_l    
errlst = []
x_its = []
err_its = []
norm0 = np.linalg.norm(x0_stacked)

err_its.append(x0)

if 'err' in sys.argv:
    clf()

solver = cg_generator(
    lambda x: system.stack(system.matvec(system.unstack(x))),
    system.stack(b),
    #x0=start_vec,
    M=lambda x: system.stack(precond_both(system.unstack(x))),
    )

for i, (x, r, delta_new) in enumerate(solver):
    x = system.unstack(x)

    errvec = lstsub(x0, x)
    err_its.append(errvec)
    x_its.append(x)
    
    errlst.append(np.linalg.norm(system.stack(errvec)) / norm0)

    if 'err' in sys.argv:
        semilogy(norm_by_l(x[0] - x0[0]) / norm_by_l(x0[0]))

    print 'it', i
    if i > 10:
        break

def errmap():
    clf()
    e = sharp.sh_synthesis(nside, x0[0]-x[0])
    mollview(e, sub=311, max=e.max(), min=e.min(), xsize=2000)
    mollview(e * mask_deg, sub=312, max=e.max(), min=e.min(), xsize=2000)
    mollview(e * (1 - mask_deg), sub=313, max=e.max(), min=e.min(), xsize=2000)
    draw()
    
#clf()
if 'err' not in sys.argv:
    semilogy(errlst, '-o')
draw()

#errmap()
