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

import healpy








1/0

def csc_neighbours(nside, pick):
    # The returned matrix will be npix-times-npix, but have zero entries outside the mask
    pixels = pick.nonzero()[0]
    pixels_reverse = np.zeros(pick.shape, dtype=int)
    pixels_reverse[pixels] = np.arange(pixels.shape[0])

    length = pixels.shape[0]
    indices = np.zeros(9 * length, dtype=np.int)
    indptr = np.zeros(12 * nside**2 + 1, dtype=np.int)
    neighbours = healpy.get_all_neighbours(nside, pixels, nest=True)

    npix = 12 * nside**2
    
    idx = 0

    for j in range(npix):
        indptr[j] = idx

        # if columns is outside mask it is just entirely empty, no elements stored
        if pick[j]:
            # column is inside mask
            k = pixels_reverse[j]
            neighlst = neighbours[:, k]
            neighlst = neighlst[(neighlst != -1) & pick[neighlst]]
            n = neighlst.shape[0]
            indices[idx] = j
            #i_arr = pixels_reverse[neighlst]
            indices[idx + 1:idx + 1 + n] = neighlst
            idx += n + 1

    indptr[-1] = idx
    indices = indices[:idx]
    data = np.ones(idx)
    return csc_matrix((data, indices, indptr), shape=(npix, npix))

def make_dist_matrix(Si_pattern):

    data = np.zeros_like(Si_pattern.data)
    for j in range(Si_pattern.shape[1]):
        i_arr = Si_pattern.indices[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]]

        if len(i_arr):
            # k is the offset of the diagonal entry
            k = (i_arr == j).nonzero()[0][0]
            
            x, y, z = healpy.pix2vec(nside, i_arr, nest=True)
            data[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]] = (x[k] * x + y[k] * y + z[k] * z)

    return csc_matrix((data, Si_pattern.indices, Si_pattern.indptr), shape=Si_pattern.shape)

nside = 64
lmax = 3 * nside


from scipy.sparse import csc_matrix
M = csc_neighbours(nside, np.ones(12 * nside**2, dtype=bool))

M = make_dist_matrix(M)
M.data[M.data > 1] = 1
pws = np.arccos(M.data)
pws = pws[pws > 0.01 * pws.mean()]

clf()
fig = gcf()
ax0 = fig.add_subplot(2,1,1)
ax1 = fig.add_subplot(2,1,2)





ls = np.arange(1, lmax + 1)
dl = ls**2.5


nl = cmbcr.standard_needlet_by_l(1.5, 2 * dl.shape[0] - 1)
i = nl.argmax()
dl_new = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])

pw = 2*np.pi / (4. * nside)


clf();
thetas = np.linspace(0, 0.3, 1000)
plot(thetas, cmbcr.beam_by_theta(dl_new, thetas))
plot(thetas, cmbcr.beam_by_theta(dl, thetas))
axvline(pws.min(), linestyle=':')
axvline(pws.max(), linestyle=':')
axvline(2 * pws.min(), linestyle=':')
axvline(2 * pws.max(), linestyle=':')
draw()
1/0
#1/0
