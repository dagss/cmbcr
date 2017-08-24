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


def make_Si_sparse_matrix(Si_pattern, dl, ridge):

    data = np.zeros_like(Si_pattern.data)
    for j in range(Si_pattern.shape[1]):
        i_arr = Si_pattern.indices[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]]

        if len(i_arr):
            # k is the offset of the diagonal entry
            k = (i_arr == j).nonzero()[0][0]
            
            x, y, z = healpy.pix2vec(nside, i_arr, nest=True)
            data[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]] = cmbcr.beam_by_cos_theta(
                dl,
                (x[k] * x + y[k] * y + z[k] * z))
            diag_ind = Si_pattern.indptr[j] + k
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


from cmbcr.precond_psuedoinv import *

from cmbcr import sympix, block_matrix, sympix_mg


# Put on a nice tail on dl
dl = system.dl_list[0]

#l = np.arange(1, dl.shape[0] + 1).astype(np.double)
#dl = l**1

#nl = cmbcr.standard_needlet_by_l(3, 2 * dl.shape[0] - 1)

nl = cmbcr.standard_needlet_by_l(3, 2 * dl.shape[0] - 1)

#dl = nl
i = nl.argmax()
dl = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])


# NOTE: nested ordering from this point!!
mask = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='NESTED', power=0)
mask[mask <= 0.9] = 0
mask[mask > 0.9] = 1
pick = (mask == 0)

Si_pattern = csc_neighbours(nside, pick)
Si_pattern = Si_pattern * Si_pattern #* Si_pattern
Si_pattern.sum_duplicates()
Si_pattern = Si_pattern.tocsc()
diag_val = cmbcr.beam_by_cos_theta(dl, np.ones(1))[0]
ridge_factor = 5e-2
Si_sparse = make_Si_sparse_matrix(Si_pattern, dl, diag_val * ridge_factor)


if 0:
    Si_dense = Si_sparse.toarray()[pick,:][:, pick]
    clf()
    semilogy(np.linalg.eigvalsh(Si_dense))
    draw()
    1/0
    


def healpix_coarsen_sparse_matrix(M):
    # sums 4 and 4 rows of M, which should be in CSC ordering. This is easily done simply
    # by dividing each row index by 4, and take every 4th indptr, then invoke sum_duplicates()

    indices = M.indices // 4
    indptr = M.indptr[::4].copy()
    
    result = csc_matrix((M.data.copy(), indices, indptr), shape=(M.shape[0] // 4, M.shape[1] // 4))
    result.sum_duplicates()

    result.data *= 0.25
 
    
    return result
        

def i_to_nest(i):
    u = np.zeros(12*nside**2)
    u[i] = 1
    u = healpy.reorder(u, r2n=True)
    return u.nonzero()[0][0]


kk = 16

class SparseJacobi(object):
    def __init__(self, M):
        self.M = M

        if kk == 1:
            self.diag_inv = M.diagonal().copy()
            self.diag_inv[self.diag_inv != 0] = 1. / self.diag_inv[self.diag_inv != 0]
        else:
            self.diag_blocks = []
            for i in range(0, M.shape[0], kk):
                block = M[i:i + kk, i:i + kk].toarray()
                self.diag_blocks.append(np.linalg.pinv(block))
        
    def matvec(self, x):
        return self.M * x

    def error_smoother(self, x):
        if kk == 1:
            return x * self.diag_inv
        else:
            out = np.zeros_like(x)
            for i, block in zip(range(0, x.shape[0], kk), self.diag_blocks):
                out[i:i + kk] = np.dot(block, x[i:i + kk])
            return out


class DenseSolve(object):
    def __init__(self, M):
        self.M = M

        self.pick = self.M.diagonal() != 0
        self.Minv = np.linalg.inv(M[self.pick, :][:, self.pick])

    def matvec(self, x):
        return np.dot(self.M, x)

    def error_smoother(self, x):
        r = np.zeros(self.M.shape[0])
        r[self.pick] = np.dot(self.Minv, x[self.pick])
        return r
    

class Noop(object):
    def matvec(self, x):
        return x
    def error_smoother(self, x):
        return x


nside_H = nside
Si_H = Si_sparse

#C0 = Si_sparse
#C1 = healpix_coarsen_sparse_matrix(C0)
#C2 = healpix_coarsen_sparse_matrix(C1)
#C3 = healpix_coarsen_sparse_matrix(C2)
#C4 = healpix_coarsen_sparse_matrix(C3)
#C5 = healpix_coarsen_sparse_matrix(C4)


## levels = [
##     SparseJacobi(C0),
##     SparseJacobi(C1),
##     SparseJacobi(C2), 
##     SparseJacobi(C3),
##     #SparseJacobi(C4),
##     DenseSolve(C4.toarray())
##     ]

levels = []

if 1:
    while nside_H > 8:
        levels.append(SparseJacobi(Si_H))
        nside_H = nside_H // 2
        Si_H = healpix_coarsen_sparse_matrix(Si_H)
    levels.append(DenseSolve(Si_H.toarray()))


    
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
        nside_h = nside / 2**ilevel
        nside_H = nside / 2**(ilevel + 1)
        for i in range(1):
            r_h = b - level.matvec(x)

            r_H = healpy.ud_grade(r_h, nside_H, order_in='NESTED', order_out='NESTED', power=0)
            c_H = vcycle(levels, ilevel + 1, r_H)
            c_h = healpy.ud_grade(c_H, nside_h, order_in='NESTED', order_out='NESTED', power=0)
            x += c_h

        # post-smooth
        for i in range(1):
            r = b - level.matvec(x)
            x += level.error_smoother(r)

        return x

    

def padvec(u):
    x = np.zeros(12 * nside**2)
    x[pick] = u
    return x

def matvec(x):
    return (Si_sparse * padvec(x))[pick]

def norm_by_l(x):
    x = padvec(x)
    x = healpy.reorder(x, n2r=True)
    return cmbcr.norm_by_l(sharp.sh_analysis(system.lmax_list[0], x))


N = int(pick.sum())

def precond(b):
    b = padvec(b)
    return vcycle(levels, 0, b)[pick]

if 0:
    from scipy.linalg import eigvals
    A = hammer(matvec, N)
    M = hammer(precond, N)
    clf()
    semilogy(np.linalg.eigvalsh(A))
    semilogy(np.linalg.eigvalsh(M)[::-1])
    semilogy(sorted(np.abs(eigvals(np.dot(M, A)))))
    draw()


else:
    rng = np.random.RandomState(1)
    x0 = rng.normal(size=N)
    b = matvec(x0)
    x = x0 * 0

    
    solver = cg_generator(
     matvec,
     b,
     M=precond,
     )

    x0norm = norm_by_l(x0)
    errlst = []
    for i, (x, r, delta_new) in enumerate(solver):
    #@for i in range(2000):
    #    r = b - matvec(x)
    #    x += precond(r)

        errvec = x0 - x
        errlst.append(np.linalg.norm(errvec) / np.linalg.norm(x0))
        #semilogy(norm_by_l(errvec) / x0norm, label=int(i))
        
        print i
        
        if i > 30:
            break

    semilogy(errlst)
    draw()

