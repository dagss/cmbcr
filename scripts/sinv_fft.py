from __future__ import division
import numpy as np

from matplotlib.pyplot import *

import logging
logging.basicConfig(level=logging.DEBUG)



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

from cmbcr import sharp
from healpy import mollzoom, mollview
from scipy.sparse import csc_matrix, dok_matrix
#reload(cmbcr.main)

from cmbcr.cr_system import load_map_cached

import sys
from cmbcr.cg import cg_generator

import scipy.ndimage

lmax = 32 - 1
lmax_sh = int(1 * lmax)


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

#dl = 1 / Cl

def make_dl_approx(lmax):
    l = np.arange(lmax + 1)
    lpivot = int(lmax * 1600. / 6000.)
    l_to_6 = 0.14 * (l / float(lpivot))**6
    l_to_3 = 0.14 * (l / float(lpivot))**3

    return np.concatenate([l_to_3[:lpivot], l_to_6[lpivot:]])

l = np.arange(1, lmax + 2).astype(float)
dl = l**4


#dl = make_dl_approx(lmax)# * cmbcr.gaussian_beam_by_l(lmax, '120 min')
#dl[0] = dl[1]
dl = needletify_dl(1.7, 3, dl)

#dlext = 0.000001 * dl2
#dlext[:lmax + 1] += dl
#dl = dlext

nrings = lmax + 1

mask_gauss = np.ones((nrings, 2 * nrings))
mask_gauss[(4 * nrings // 10):(6 * nrings // 10), :] = 0
mask_gauss = mask_gauss.reshape(2 * nrings**2)

#mask = np.zeros(
#mask_healpix = load_map_cached('mask_galactic_band_2048.fits')
#mask_lm = sharp.sh_analysis(lmax, mask_healpix)
#mask_gauss = sharp.sh_synthesis_gauss(lmax, mask_lm)
#mask_gauss[mask_gauss < 0.8] = 0
#mask_gauss[mask_gauss >= 0.8] = 1

solver = cmbcr.SinvSolver(dl, mask_gauss, split=False, nrings=lmax + 1)

if 0:
    A = hammer(solver.outer_matvec, solver.n)
    M = hammer(solver.outer_precond, solver.n)
    clf()
    lam_A = np.linalg.eigvalsh(A)
    lam_M = np.linalg.eigvalsh(M)
    print 'lam_M.min()', lam_M.min()
    semilogy(lam_A, label='A')
    semilogy(lam_M, label='M')
    #semilogy(lam_A * lam_B * 2)
    lam = np.abs(np.linalg.eigvals(np.dot(M, A)))
    semilogy(sorted(lam), label='MA')
    legend()
    draw()
    1/0



split_l = np.zeros_like(dl)
split_l[:lmax + 1] = 1
lowpass_lm = scatter_l_to_lm(split_l)
dlm_low_only = lowpass_lm * scatter_l_to_lm(dl)

def Z(u):
    u = sharp.sh_synthesis_gauss(solver.lmax, u, lmax_sh=lmax_hi)
    u *= (1 - mask_gauss)
    u = sharp.sh_analysis_gauss(solver.lmax, u, lmax_sh=lmax_hi)
    return u
    
def matvec_outer(u):
    u = Z(u)
    
    u = u * scatter_l_to_lm(dl) # dlm_low_only ##scatter_l_to_lm(dl)#  + u * 1e-10 #dlm_low_only

    u = sharp.sh_adjoint_analysis_gauss(solver.lmax, u, lmax_sh=lmax_hi)
    u *= (1 - mask_gauss)
    u = sharp.sh_adjoint_synthesis_gauss(solver.lmax, u, lmax_sh=lmax_hi)
    
    return u 


lmax_hi = solver.lmax_sh

def precond(u_in):
    u = solver.restrict(u_in, lmax=lmax_hi)
    u, reslst, errlst = solver.solve_mask(u)
    u = solver.prolong(u, lmax=lmax_hi)
    
    return u# + u_in * 1e-4
    
nside = (lmax + 1) // 2
rng = np.random.RandomState(11)
x0 = Z(rng.normal(size=(lmax_hi + 1)**2))
x0 *= lowpass_lm
b = matvec_outer(x0)

cgsolver = cg_generator(
    matvec_outer,
    b,
    M=precond
    )

errlst = []
if x0 is not None:
    x0_norm = np.linalg.norm(Z(x0))


maxit = 20
for i, (x, r, delta_new) in enumerate(cgsolver):
    #r = np.linalg.norm(r) / b_norm
    #reslst.append(r)
    e = np.linalg.norm(Z(x0 - x)) / x0_norm
    errlst.append(e)
    print 'OUTER iteration {}, err={}'.format(i, e)

    if i > maxit:
        break
clf()
semilogy(errlst, '-o')

def errmap():
    clf()
    e = sharp.sh_synthesis_gauss(nrings - 1, x0 -x, lmax_sh=lmax_hi)
    #e *= (1 - mask_gauss)
    imshow(e.reshape((nrings, 2 * nrings)), interpolation='none')
    colorbar()
    #mollview(e, sub=111, max=e.max(), min=e.min(), xsize=2000)
    #mollview(e * mask_deg, sub=312, max=e.max(), min=e.min(), xsize=2000)
    #mollview(e * (1 - mask_deg), sub=313, max=e.max(), min=e.min(), xsize=2000)
    draw()
