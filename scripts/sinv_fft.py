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

lmax = 128 - 1
lmax_sh = 2 * lmax
OMEGA = 1 #0.05

SQRTSPLITFULLSOLVE = True


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


l = np.arange(1, lmax_sh).astype(float)
#dl = np.exp(0.01 * l)
dl = l**2.5


#1.2 #1.5

#dl = 1 / Cl_cmb[:lmax_sh]



nrings = lmax + 1



mask_hires = load_map_cached('mask_galactic_band_2048.fits')
mask_hires[:] = 0

def cl_to_flatsky(cl, nx, ny, nl):
    out = np.zeros((nx, ny))

    l = np.sqrt(
        (np.fft.fftfreq(nx, 1. / (2 * nl))[:, None])**2
        + (np.fft.fftfreq(ny, 1. / (2 * nl))[None, :])**2)

    cl_flat = scipy.ndimage.map_coordinates(cl, l[None, :, :], mode='constant', order=1)
    return cl_flat


class DenseSmoother(object):
    def __init__(self, level):
        self.M = np.linalg.inv(hammer(level.matvec, level.n_mask))

    def apply(self, u):
        return np.dot(self.M, u)


class DiagonalSmoother(object):

    def __init__(self, level):
        self.level = level

        # hammer operator on a tilesize x tilesize patch...
        u = np.zeros((level.nrows, level.ncols))
        u[0, 0] = 1
        u = flatsky_adjoint_synthesis(u)
        u *= np.sqrt(level.dl_fft)
        u = flatsky_synthesis(u).real
        self.inv_diag = 1 / u[0, 0]

    def apply(self, u):
        return self.inv_diag * u


def flatsky_analysis(u):
    return np.fft.fftn(u) / np.prod(u.shape)

def flatsky_adjoint_analysis(u):
    return np.fft.ifftn(u)

def flatsky_synthesis(u):
    return np.fft.ifftn(u) * np.prod(u.shape)

def flatsky_adjoint_synthesis(u):
    return np.fft.fftn(u)

class BlockSmoother(object):
    def __init__(self, level):
        self.level = level
        self.tilesize = 1

        # hammer operator on a tilesize x tilesize patch...
        u = np.zeros((level.nrows, level.ncols))
        block = np.zeros((self.tilesize**2, self.tilesize**2))
        for i in range(self.tilesize):
            for j in range(self.tilesize):
                idx = i * self.tilesize + j
                u[:, :] = 0
                u[i, j] = 1
                u = flatsky_adjoint_synthesis(u)
                u *= np.sqrt(level.dl_fft)
                u = flatsky_synthesis(u).real

                x = u.reshape((level.nrows // self.tilesize, self.tilesize, level.ncols // self.tilesize, self.tilesize))
                x = x[0, :, 0, :].reshape(self.tilesize**2)
                
                block[:, idx] = x#.reshape(level.nrows * level.ncols, order='c')[:self.tilesize**2]
        #clf()
        #imshow(block[:, 11].reshape((self.tilesize, self.tilesize)), interpolation='none')
        #colorbar()
        #imshow(np.log(np.abs(block)), interpolation='none')
        #draw()
        #1/0
        self.block_inv = np.linalg.inv(block)
        

    def apply(self, u):
        u = self.level.padvec(u)

        # pad u_in in order to wrap around
        tmp = np.hstack([u, u, u])
        u_in = np.vstack([tmp, tmp, tmp])

        u_out = np.zeros_like(u_in)

        overlap = 0
        
        # TODO pad a border around to wrap around.. for now forget about it...
        ##u_in 
        
        #u = np.transpose(
        #    u.reshape((self.level.nrows // self.tilesize, self.tilesize,
        #               self.level.ncols // self.tilesize, self.tilesize)),
        #    (0, 2, 1, 3))

        

        k = self.tilesize
        n, m = self.level.nrows, self.level.ncols
        for i in range(0, n + k - overlap, k):
            for j in range(0, m + k - 1, k):
                s = (
                    slice(n + i - overlap, n + i + k - overlap),
                    slice(m + j - overlap, m + j + k - overlap)
                    )
                tile = u_in[s]
                assert tile.shape == (k, k)
                tile = np.dot(self.block_inv, tile.reshape(k**2)).reshape(k, k)
                
                # downweight the edges of the tile, which overlaps with other tiles
                tile[:, :] *= 0.5
                tile[k:-k, k:-k] *= 2
                
                u_out[s] += tile

        u_out = u_out[n:2 * n, m:2 * m]
                
        ##         buf = np.zeros((4, 4))
        ##         if i == 0:
        ##             buf[ :] = 
        ##         u_out[i, j, :, :] = np.dot(
        ##             self.block_inv, u_in[i, j, :, :].reshape(self.tilesize**2)).reshape((self.tilesize, self.tilesize))
        ##         #x = flatsky_analysis(u[i, j, :, :])
        ##         #x *= self.inv_dl
        ##         #u[i, j, :, :] = flatsky_adjoint_analysis(x).real
        ## u = np.transpose(u, (0, 2, 1, 3)).reshape((self.level.nrows, self.level.ncols))
        u = OMEGA * self.level.pickvec(u_out)
        return u
    
        
class Level(object):

    def __init__(self, lmax, mask, dl, smoother_factory):
        self.lmax = lmax
        self.nrings = nrings = lmax + 1
        
        mask_lm = sharp.sh_analysis(lmax, mask)
        mask_gauss = sharp.sh_synthesis_gauss(lmax, mask_lm)
        mask_gauss[mask_gauss < 0.9] = 0
        mask_gauss[mask_gauss >= 0.9] = 1
        self.mask = mask_gauss
        self.pick = (mask_gauss == 0)
        self.n_mask = int(self.pick.sum())
        
        self.ind_map = np.zeros(2 * nrings**2, dtype=int)
        self.ind_map[:] = -1
        self.ind_map[self.pick] = np.arange(self.n_mask)
        self.ind_map = self.ind_map.reshape((nrings, 2 * nrings))

        self.dl = dl
        self.dl_fft = cl_to_flatsky(dl, nrings, 2 * nrings, lmax + 1)

        self.dl_fft_inv = 1 / self.dl_fft

        self.nrows = self.nrings
        self.ncols = 2 * self.nrings
        
        unitvec = np.zeros((nrings, 2 * nrings))
        unitvec[nrings // 2, nrings] = 1
        
        z = sharp.sh_adjoint_synthesis_gauss(lmax, unitvec.reshape(2 * nrings**2))
        z *= scatter_l_to_lm(dl[:lmax + 1])
        z = sharp.sh_synthesis_gauss(lmax, z)
        estimated_max_sht = z.reshape((nrings, 2 * nrings))[nrings // 2, nrings]

        z = self.flatsky_adjoint_synthesis(unitvec)
        z *= self.dl_fft
        z = self.flatsky_synthesis(z).real
        estimated_max_fft = z[nrings // 2, nrings]


        scale_factor = estimated_max_sht / estimated_max_fft
        self.dl_fft *= scale_factor
        
        self.inv_diag = 1 / np.sqrt(estimated_max_fft * scale_factor)
        
        self.smoother = smoother_factory(self)

    def compute_dense(self):
        self.dense = True

    def pickvec(self, u):
        return u.reshape(2 * self.nrings**2)[self.pick]

    def padvec(self, u):
        u_pad = np.zeros(2 * self.nrings**2)
        u_pad[self.pick] = u.real
        return u_pad.reshape(self.nrings, 2 * self.nrings)

    def flatsky_synthesis(self, u):
        return np.fft.ifftn(u) * np.prod(u.shape)

    def flatsky_adjoint_synthesis(self, u):
        return np.fft.fftn(u)

    def flatsky_analysis(self, u):
        return np.fft.fftn(u) / np.prod(u.shape)

    def flatsky_adjoint_analysis(self, u):
        return np.fft.ifftn(u)

    def matvec(self, u):
        with timed('matvec'):
            u = self.padvec(u)
            u = self.flatsky_adjoint_synthesis(u)
            u *= np.sqrt(self.dl_fft)
            u = self.flatsky_synthesis(u)
            u = self.pickvec(u).real
            return u

    def precond(self, u):
        with timed('precond'):
            return self.smoother.apply(u)

        
    
from scipy.sparse import dok_matrix

def coarsen_matrix(coarse_level, fine_level):
    R = dok_matrix((coarse_level.n_mask, fine_level.n_mask))

    def add(coarse_i, coarse_j, fine_i, fine_j, weight):
        # wrap around fine_i and fine_j..
        fine_i = fine_i % fine_level.nrows
        fine_j = fine_j % fine_level.ncols

        coarse_ind = coarse_level.ind_map[coarse_i, coarse_j]
        fine_ind = fine_level.ind_map[fine_i, fine_j]
        if coarse_ind != -1 and fine_ind != -1:
            R[coarse_ind, fine_ind] = weight
    
    for i in range(coarse_level.nrows):
        for j in range(coarse_level.ncols):
            # row above
            add(i, j, 2 * i - 1, 2 * j - 1, 1/16.)
            add(i, j, 2 * i - 1, 2 * j, 1/8.)
            add(i, j, 2 * i - 1, 2 * j + 1, 1/16.)

            # center row
            add(i, j, 2 * i, 2 * j - 1, 1/8.)
            add(i, j, 2 * i, 2 * j, 1/4.)
            add(i, j, 2 * i, 2 * j + 1, 1/8.)

            # row below
            add(i, j, 2 * i + 1, 2 * j - 1, 1/16.)
            add(i, j, 2 * i + 1, 2 * j, 1/8.)
            add(i, j, 2 * i + 1, 2 * j + 1, 1/16.)

    return R.tocsr()


def v_cycle(ilevel, levels, b):
    if ilevel == len(levels) - 1:
        return levels[ilevel].precond(b)
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]

        x = b * 0
        for i in range(1):
            x += level.precond(b - level.matvec(x))

        for i in range(1):
            r_h = b - level.matvec(x)

            r_H = coarseners[ilevel] * r_h

            c_H = v_cycle(ilevel + 1, levels, r_H)

            c_h = coarseners[ilevel].T * c_H
            
            x += c_h

        for i in range(1):
            x += level.precond(b - level.matvec(x))
        return x

def precond(b):
    x = v_cycle(0, levels, b)
    return x


def last_level(level):
    return level.n_mask < 2000

def smoother_factory(level):
    if last_level(level):
        return DenseSmoother(level)
    elif 'diagonal' in sys.argv:
        return DiagonalSmoother(level)
    else:
        return BlockSmoother(level)


if 1:
    levels = []
    lmax_H = lmax
    while True:
        print 'Level(', lmax_H
        lev = Level(lmax_H, mask_hires, dl, smoother_factory)
        levels.append(lev)
        if last_level(lev):
            break
        lmax_H //= 2
else:
    levels = [
        Level(lmax, mask_hires, dl, DiagonalSmoother),
        #Level(lmax // 2, mask_hires, dl, DiagonalSmoother),
        #Level(lmax // 4, mask_hires, dl, DiagonalSmoother),
        #Level(lmax // 4, mask_hires, dl, DenseSmoother),
    ]

level = levels[0]

coarseners = []
for i in range(len(levels) - 1):
    coarseners.append(coarsen_matrix(levels[i + 1], levels[i]))


def jacobi(matvec, precond, b, n):
    x = precond(b)
    for i in range(n - 1):
        x += precond(b - matvec(x))
    return x
    

    
if 0:

    A = hammer(level.matvec, level.n_mask)
    M = hammer(precond, level.n_mask)
    clf()
    semilogy(np.linalg.eigvalsh(A))
    semilogy(np.linalg.eigvalsh(M)[::-1])
    lam = np.linalg.eig(np.dot(M, A))[0].real
    semilogy(sorted(lam))
    print lam.max()
    #semilogy(np.linalg.eigvalsh(M)[::-1])
    draw()
    1/0


        
rng = np.random.RandomState(12)
x0 = rng.normal(size=level.n_mask)

matvec_count = 0

if SQRTSPLITFULLSOLVE:

    INNERITS = 1
    def precond2(b):
        
        solver = cg_generator(
            matvec,
            b,
            M=precond
            )

        for i, (x, r, delta_new) in enumerate(solver):
            if i > INNERITS:
                break

        solver = cg_generator(
            matvec,
            x,
            M=precond
            )

        for i, (x, r, delta_new) in enumerate(solver):
            if i > INNERITS:
                break
            
        return x

    def precond2_jac(b):
        x = precond(b)
        x = precond(x)
        return x

    def matvec2(u):
        print 'inc count 2'
        global matvec_count
        matvec_count += 1
        u = level.padvec(u)
        u = flatsky_adjoint_synthesis(u)
        u *= level.dl_fft
        u = flatsky_synthesis(u)
        u = level.pickvec(u).real
        return u

    b = matvec2(x0)
    solver = cg_generator(
        matvec2,
        b,
        M=precond2_jac
        )

    
else:
    b = level.matvec(x0)

    solver = cg_generator(
        level.matvec,
        b,
        M=precond
        )


norm0 = np.linalg.norm(x0)
errlst = []
matvec_counts = []

for i, (x, r, delta_new) in enumerate(solver):

#for i in range(1000):
#    r = b - matvec_mask_basis(x)
#    x = x + 0.1 * precond(r)

    errvec = x0 - x
    #err_its.append(errvec)
    #x_its.append(x)

    matvec_counts.append(matvec_count)
    errlst.append(np.linalg.norm(errvec) / norm0)

    print 'it', i
    if matvec_count > 20:
        break



#clf();
#imshow(padvec(x-x0), interpolation='none');
semilogy(matvec_counts, errlst, '-o')
#colorbar()
draw()

