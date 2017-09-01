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

#lmax = 96 - 1
#lmax_sh = int(1 * lmax)


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


lmax = 1000 - 1
lmax_lo = 100 - 1

#Cl_cmb = load_Cl_cmb(lmax)

l = np.arange(1, lmax + 2).astype(float)
dl = l**6
#dl = Cl_cmb #1 / Cl_cmb

nrings = lmax + 1

unitvec = np.zeros((nrings, 2 * nrings))
unitvec[nrings // 2, nrings] = 1


u = sharp.sh_adjoint_synthesis_gauss(nrings - 1, unitvec.reshape(2 * nrings**2))
u *= scatter_l_to_lm(dl)
opimage = sharp.sh_synthesis_gauss(nrings - 1, u)

nrings_lo = lmax_lo + 1

opimage = opimage.reshape((nrings, 2 * nrings))
opimage_lo = opimage[(nrings - nrings_lo) // 2:(nrings + nrings_lo) // 2, nrings - nrings_lo:nrings + nrings_lo]
unitvec_lo = unitvec[(nrings - nrings_lo) // 2:(nrings + nrings_lo) // 2, nrings - nrings_lo:nrings + nrings_lo]

YtW_x = sharp.sh_analysis_gauss(lmax_lo, opimage_lo.reshape(2 * nrings_lo**2))
Yt_unit = sharp.sh_adjoint_synthesis_gauss(lmax_lo, unitvec_lo.reshape(2 * nrings_lo**2))

from cmbcr.masked_solver import operator_image_to_power_spectrum

q = operator_image_to_power_spectrum(unitvec_lo, opimage_lo)

#print opimage_lo.shape
#clf()
#plot(YtW_x / Yt_unit)
#1/0

#opimage_lo = sharp.sh_analysis_gauss(self.nrings - 1, opimage, lmax=)
#        opimage = self.gauss_grid_to_equator(sharp.sh_synthesis_gauss(self.lmax, u, lmax_sh=self.lmax_sh))

#sharp.sh_adjoint_synthesis_gauss



