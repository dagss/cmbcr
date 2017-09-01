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

nside = 32 * w
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
l = np.arange(system.lmax_list[0] + 1)
#wl_list = [
#    np.exp(-0.006 * (l - 60)**2) * 1.2

#    #1 / np.sqrt(system.dl_list[0] + system.ni_approx_by_comp_lst[0])
#    ]
#system.set_wl_list(wl_list)


def needletify_dl(b, lmax_factor, dl):
    lmax = int(lmax_factor * dl.shape[0]) - 1
    if 1:
        nl = cmbcr.standard_needlet_by_l(b, lmax)
        i = nl.argmax()
        return np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])
    else:
        # FAILED!
        nl = fourth_order_beam(lmax, int(0.1 * lmax))
        return np.concatenate([dl, nl * dl[-1] / nl[0]])

system.prepare(use_healpix=True)

if 0:
    nl = cmbcr.standard_needlet_by_l(2, system.lmax_list[0])
    i = nl.argmax()
    nl[:i] = nl.max()
    nl /= nl.max()
    rl = nl
else:

    lmax_hi = int(1 * system.lmax_list[0])
    dl = system.dl_list[0]
    #l = np.arange(1, lmax_hi + 2).astype(float)
    
    ql = cmbcr.standard_needlet_by_l(1.65, lmax_hi, 0.05)
    i = ql.argmax()
    ql[:i] = 1
    
    #dl_new = l**6
    #dl_new *= dl[0] / dl_new[0]

    
    dl_ext = dl * ql
    rl = np.ones(dl_ext.shape[0])
    


if 'plot' in sys.argv:
    clf()
    system.plot()
    plot(system.dl_list[0] * rl**2)
    1/0

SQRTSPLIT = False


dl_sinv = dl_ext
if SQRTSPLIT:
    dl_sinv = np.sqrt(dl_sinv)

    

mask_for_sinv = system.mask_gauss_grid


sinv_solver = cmbcr.SinvSolver(dl_sinv, mask_for_sinv, split=False, # rl=np.ones(dl.shape[0]), #, rl=rl,
                               nrings=system.lmax_mixing_pix + 1,
                               ridge=0)


def solve_mask(b):
    self = sinv_solver

    args = ()
    kw = {'rtol': 1e-2, 'maxit': 1}

    b_restricted = restrict(b, lmax=lmax_hi)

    def matvec_middle(x):
        xpad = np.zeros((2 * self.nrings**2))
        xpad[sinv_pick] = x

        alm = sharp.sh_adjoint_synthesis_gauss(self.lmax, xpad)
        alm *= scatter_l_to_lm(dl_sinv)
        xpad = alm = sharp.sh_synthesis_gauss(self.lmax, alm)
        return xpad[sinv_pick]
    

    if 1:
        solver = cg_generator(
            matvec_middle,
            b_restricted,
            #M=self.inner_precond,
            M=self.inner_precond,#lambda x: self.solve_mask(x, *args, **kw)[0]
            )
        
        b_norm = np.linalg.norm(b_restricted)
        for i, (x, r, delta_new) in enumerate(solver):
            res = np.linalg.norm(r) / b_norm
            print 'MIDDLE IT', i, res
            if res < 1e-4 or i > 20:
                break

    else:
        x = sinv_solver.solve_mask(b_restricted, *args, **kw)[0]
    
        
    x = prolong(x, lmax=lmax_hi)

    return x

    ## if SQRTSPLIT:
    ##     npix = 2 * self.nrings**2
    ##     x *= npix / 4 / np.pi
    
    ##     x, _, _ = self.solve_mask(x, *args, **kw)



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
def restrict(x, lmax=None):
    self = sinv_solver
    lmax = lmax or self.lmax
    x = x * scatter_l_to_lm(rl)
    x = sharp.sh_synthesis_gauss(self.lmax, x, lmax_sh=lmax)
    x = x[sinv_pick]
    return x

def prolong(x, lmax=None):
    self = sinv_solver
    lmax = lmax or lmax_hi
    xpad = np.zeros((2 * self.nrings**2))
    xpad[sinv_pick] = x
    x = sharp.sh_adjoint_synthesis_gauss(self.lmax, xpad, lmax_sh=lmax)
    x = x * scatter_l_to_lm(rl)
    return x


import healpy
mask_nside = 32
mask_hp = healpy.ud_grade(system.mask, mask_nside, order_in='RING', order_out='RING', power=0)
mask_hp[mask_hp != 1] = 0
mask_hp_pick = (mask_hp == 0)
def restrict_healpix(x):
    x = x * scatter_l_to_lm(rl)
    x = sharp.sh_synthesis(mask_nside, x)
    return x[mask_hp_pick]

def prolong_healpix(x):
    xpad = np.zeros(12 * mask_nside**2)
    xpad[mask_hp_pick] = x
    x = sharp.sh_adjoint_synthesis(system.lmax_list[0], xpad)
    x = x * scatter_l_to_lm(rl)
    return x


def ZAZ(x):
    x = prolong(x, lmax=lmax_hi)
    #x *= scatter_l_to_lm(system.dl_list[0])
    x *= scatter_l_to_lm(dl_ext)
    x = restrict(x, lmax=lmax_hi)
    return x

if 0:  # USE healpix dense system instead

    prolong = prolong_healpix
    restrict = restrict_healpix
    N_MASK = int(mask_hp_pick.sum())
else:
    N_MASK = int(sinv_pick.sum())
    


if 'dense' in sys.argv:
    ZAZ_dense = hammer(ZAZ, N_MASK)
    ZAZ_dense += np.eye(ZAZ_dense.shape[0])# * 1e-3 * ZAZ_dense.max()
    ZAZ_inv = np.linalg.pinv(ZAZ_dense)

    def mask_dense_solver(x):
        x = x * lowpass_lm
        x = restrict(x, lmax_hi)
        x = np.dot(ZAZ_inv, x)
        x = prolong(x, lmax_hi)
        x = x * lowpass_lm
        return x
    
    solve_mask = mask_dense_solver
#1/0

lmax = system.lmax_list[0]
u = np.arange(2 * sinv_solver.nrings**2, dtype=float)#.reshape(sinv_solver.nrings, 2 * sinv_solver.nrings)
ush = sharp.sh_analysis_gauss(sinv_solver.nrings - 1, u, lmax_sh=lmax)


    

def precond_both(b):

    if 0:
        return precond_1.apply(b)
    elif 0:
        x = [solve_mask(b[0])]
        return x
    elif 0:
        b_lo = pad_or_truncate_alm(b, system.lmax_list[0])
        x_lo = precond_1.apply([b_lo])[0]
        x = pad_or_truncate_alm(x_lo, lmax_hi)

        #x += b * (1 - lowpass_lm) * scatter_l_to_lm(1 / dl_ext)
        
        x += solve_mask(b)
        return x
    elif 0:
        x = filtered_precond1(b)
        x = lstadd(x, [solve_mask(b[0])])
        return x
    else:
        x = precond_1.apply([b])[0]

        r = b - system.matvec([x])[0]
        x += solve_mask(r)
        
        r = b - system.matvec([x])[0]
        x += precond_1.apply([r])[0]

        return x



def matvec(x):
    x_lo = pad_or_truncate_alm(x, system.lmax_list[0])
    Ni_x = system.matvec([x_lo], skip_prior=True)[0]

    x = x * scatter_l_to_lm(dl_ext)
    x += pad_or_truncate_alm(Ni_x, lmax_hi)
    return x
    


lowpass_l = np.zeros(lmax_hi + 1)
lowpass_l[:lmax + 1] = 1
lowpass_lm = scatter_l_to_lm(lowpass_l)

rng = np.random.RandomState(1)

x0 = rng.normal(size=(lmax_hi + 1)**2).astype(np.float64)
x0 *= lowpass_lm
b = matvec(x0)
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
    lambda x: matvec(x),
    b,
    #x0=start_vec,
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
    if i > 30:
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
