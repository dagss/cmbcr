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

nside = 1024
factor = 2048 // nside

full_res_system = cmbcr.CrSystem.from_config(config, udgrade=nside, mask_eps=0.8)

full_res_system.prepare_prior()

system = cmbcr.downgrade_system(full_res_system, 1. / factor)
#system = full_res_system

lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (0, 0, 0)

system.set_params(
    lmax_ninv=lmax_ninv,
    rot_ang=rot_ang,
    flat_mixing=False,
    )

system.prepare_prior()
l = np.arange(system.lmax_list[0] + 1)

system.prepare(use_healpix=True)

lmax = system.lmax_list[0]

dl = system.dl_list[0]
rl = cmbcr.fourth_order_beam(system.lmax_list[0], system.lmax_list[0]//2, 0.2)
rl = np.sqrt(rl)
dl_sinv = dl * rl**2

    

mask_for_sinv = system.mask_gauss_grid
sinv_solver = cmbcr.SinvSolver(dl_sinv, mask_for_sinv, split=False)


def solve_mask(b):
    self = sinv_solver

    args = ()
    kw = {'rtol': 1e-2, 'maxit': 5}

    b_restricted = self.restrict(b * scatter_l_to_lm(rl))

    

    if 0:
        def matvec_middle(x):
            x = self.prolong(x)
            #xpad = np.zeros((2 * self.nrings**2))
            #xpad[sinv_pick] = x

            #alm = sharp.sh_adjoint_synthesis_gauss(self.lmax, xpad)
            x *= scatter_l_to_lm(dl * rl**2)
            #xpad = alm = sharp.sh_synthesis_gauss(self.lmax, alm)
            #return xpad[sinv_pick]
            return self.restrict(x)

        
        solver = cg_generator(
            matvec_middle,
            b_restricted,
            #M=self.inner_precond,
            #M=self.inner_precond,
            M=lambda x: self.solve_mask(x, *args, **kw)[0]
            )
        
        b_norm = np.linalg.norm(b_restricted)
        for i, (x, r, delta_new) in enumerate(solver):
            res = np.linalg.norm(r) / b_norm
            print 'MIDDLE IT', i, res
            if res < 1e-4 or i > 1:
                break

    elif 1:
        x = sinv_solver.solve_mask(b_restricted, *args, **kw)[0]
    else:
        x = sinv_solver.inner_precond(b_restricted)
        #x = sinv_solver.solve_mask(b_restricted, *args, **kw)[0]
    
        
    x = self.prolong(x) * scatter_l_to_lm(rl)

    return x



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


def ZAZ(x):
    x = sinv_solver.prolong(x)
    #x *= scatter_l_to_lm(system.dl_list[0])
    x *= scatter_l_to_lm(dl * rl**2)
    x = sinv_solver.restrict(x)
    return x


if 'dense' in sys.argv:
    ZAZ_dense = hammer(ZAZ, sinv_solver.n)
    ZAZ_dense += np.eye(ZAZ_dense.shape[0])# * 1e-3 * ZAZ_dense.max()
    ZAZ_inv = np.linalg.pinv(ZAZ_dense)

    def mask_dense_solver(x):
        x = sinv_solver.restrict(x * scatter_l_to_lm(rl))
        x = np.dot(ZAZ_inv, x)
        x = sinv_solver.prolong(x) * scatter_l_to_lm(rl)
        return x
    
    solve_mask = mask_dense_solver
    
    
def precond_psuedo(b):
    x = precond_1.apply([b])[0]
    return 0.4 * x
    
def precond_both(b):
    if 0:
        r = b
        return precond_1.apply([r])[0] + solve_mask(r)
    else:
        x = np.zeros_like(b)
        for i in range(1):
            #r = b - system.matvec([x])[0]
            r = b
            x = precond_psuedo(r)

        x += solve_mask(r)

        for i in range(1):
            r = b - system.matvec([x])[0]
            x += precond_psuedo(r)

        return x


rng = np.random.RandomState(1)

x0 = rng.normal(size=(lmax + 1)**2).astype(np.float64)
b = system.matvec([x0])[0]
x0_stacked = x0.copy()

from cmbcr import norm_by_l    
errlst = []
x_its = []
err_its = []
norm0 = np.linalg.norm(x0_stacked)

err_its.append(x0)

if 'err' in sys.argv:
    clf()

solver = cg_generator(
    lambda x: system.matvec([x])[0],
    b,
    M=lambda x: precond_both(x),
    )

for i, (x, r, delta_new) in enumerate(solver):
    #x = system.unstack(x)

    #errvec = lstsub(x0, x)
    #err_its.append(errvec)
    #x_its.append(x)
    errvec = x0 - x
    
    errlst.append(np.linalg.norm(errvec) / norm0)

    if 'err' in sys.argv:
        semilogy(norm_by_l(x - x0) / norm_by_l(x0))

    print 'it', i
    if i > 10:
        break

def errmap():
    clf()
    e = sharp.sh_synthesis(nside, x0-x)
    mollview(e, sub=111, max=e.max(), min=e.min(), xsize=2000)
    #mollview(e * mask_deg, sub=312, max=e.max(), min=e.min(), xsize=2000)
    #mollview(e * (1 - mask_deg), sub=313, max=e.max(), min=e.min(), xsize=2000)
    draw()
    
#clf()
if 'err' not in sys.argv:
    semilogy(errlst, '-o')
draw()

#errmap()
