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


nrings = 3000
lmax_sh = nrings - 1
pw = 2 * np.pi / (2 * nrings)

l = np.arange(1, lmax_sh + 2).astype(float)


dl = 1 / Cl_cmb[:lmax_sh + 1]


nl = cmbcr.standard_needlet_by_l(10, lmax_sh)
i = nl.argmax()
nl[:i] = 1

new_dl_1 = dl * nl


#nl = cmbcr.standard_needlet_by_l(2, 2 * dl.shape[0] - 1)
#i = nl.argmax()
new_dl_2 = new_dl_1 #np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])


lmax_sh = dl.shape[0] - 1


clf()
fig = gcf()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

N = 8
ipix = np.linspace(0, N, 1000)
ax1.plot(ipix, cmbcr.beam_by_theta(dl, ipix * pw))
ax1.plot(ipix, cmbcr.beam_by_theta(new_dl_1, ipix * pw))
ax1.plot(ipix, cmbcr.beam_by_theta(new_dl_2, ipix * pw))
for i in range(N):
    ax1.axvline(i, color='black', ls='dotted')

ax2.semilogy(ipix, np.abs(cmbcr.beam_by_theta(dl, ipix * pw)))
ax2.semilogy(ipix, np.abs(cmbcr.beam_by_theta(new_dl_1, ipix * pw)))
ax2.semilogy(ipix, np.abs(cmbcr.beam_by_theta(new_dl_2, ipix * pw)))
#for i in range(N):
#    ax2.axvline(i, color='black', ls='dotted')

    
ax3.semilogy(dl)
ax3.semilogy(new_dl_1)
ax3.semilogy(new_dl_2)
ax3.set_ylim((1e-3, dl.max() * 2))

draw()
