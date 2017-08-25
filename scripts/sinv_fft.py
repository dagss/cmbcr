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

lmax = 256 - 1
lmax_sh = int(1 * lmax)
OMEGA = 1

SQRTSPLITFULLSOLVE = False


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


l = np.arange(1, lmax_sh + 2).astype(float)
#dl = np.exp(0.01 * l)
dl = l**3
#dl = 1 / Cl_cmb[::2][:lmax_sh + 1]

nl = cmbcr.standard_needlet_by_l(3, 2 * dl.shape[0] - 1)
i = nl.argmax()
dl = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])

lmax_sh = dl.shape[0] - 1

#1.2 #1.5




nrings = lmax + 1


mask_healpix = load_map_cached('mask_galactic_band_2048.fits')

#mask_hires[:] = 0

def cl_to_flatsky(cl, nx, ny, nl):
    1/0
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

    
def image_to_powspec(u, x):
    # u: unit-vector in flatsky basis
    # x: image of operator
    FtW_x = flatsky_analysis(x)
    Ft_u = flatsky_adjoint_synthesis(u)
    dl_fft = np.abs(FtW_x / Ft_u)
    return dl_fft

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
                
                block[:, idx] = x
        self.block_inv = np.linalg.inv(block)
        

    def apply(self, u):
        u = self.level.padvec(u)

        # pad u_in in order to wrap around
        tmp = np.hstack([u, u, u])
        u_in = np.vstack([tmp, tmp, tmp])

        u_out = np.zeros_like(u_in)

        overlap = 0
        
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
                if overlap:
                    tile[:, :] *= 0.5
                    tile[overlap:-overlap, overlap:-overlap] *= 2
                
                u_out[s] += tile

        u_out = u_out[n:2 * n, m:2 * m]
        u = OMEGA * self.level.pickvec(u_out)
        return u
    
        
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
            if 0:
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
            else:
                # row above
                add(i, j, 2 * i - 1, 2 * j, 1/8.)

                # center row
                add(i, j, 2 * i, 2 * j - 1, 1/8.)
                add(i, j, 2 * i, 2 * j, 1/2.)
                add(i, j, 2 * i, 2 * j + 1, 1/8.)

                # row below
                add(i, j, 2 * i + 1, 2 * j, 1/8.)

    return R.tocsr()


def full_coarsen_matrix(ntheta, nphi):
    coarse_ntheta = ntheta // 2
    coarse_nphi = nphi // 2
    R = dok_matrix((coarse_ntheta * coarse_nphi, ntheta * nphi))

    def add(coarse_i, coarse_j, fine_i, fine_j, weight):
        # wrap around fine_i and fine_j..
        fine_i = fine_i % ntheta
        fine_j = fine_j % nphi

        coarse_ind = coarse_i * coarse_nphi + coarse_j
        fine_ind = fine_i * nphi + fine_j
        R[coarse_ind, fine_ind] = weight
    
    for i in range(coarse_ntheta):
        for j in range(coarse_nphi):
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



mask_lm = sharp.sh_analysis(lmax, mask_healpix)
mask_gauss = sharp.sh_synthesis_gauss(lmax, mask_lm)
mask_gauss[mask_gauss < 0.9] = 0
mask_gauss[mask_gauss >= 0.9] = 1

        

start_ring = nrings // 4
stop_ring = 3 * nrings // 4

mask = mask_gauss.reshape((nrings, 2 * nrings))[start_ring:stop_ring, :]


ntheta = stop_ring - start_ring
nphi = 2 * nrings

unitvec = np.zeros((ntheta, nphi))
unitvec[ntheta // 2, nphi // 2] = 1

# 'image' the spherical harmonic operator in pixel domain in order to transfer it to FFT
u_pad = np.zeros((nrings, 2 * nrings))
u_pad[start_ring:stop_ring, :] = unitvec.reshape((ntheta, nphi))
u = sharp.sh_adjoint_synthesis_gauss(lmax, u_pad.reshape(2 * nrings**2), lmax_sh=lmax_sh)
u *= scatter_l_to_lm(dl)
u = sharp.sh_synthesis_gauss(lmax, u, lmax_sh=lmax_sh).reshape((nrings, 2 * nrings))
image_of_operator = u[start_ring:stop_ring, :]


dl_fft = image_to_powspec(unitvec, image_of_operator)


class Level(object):
    def __init__(self, dl_fft, mask):
        self.mask = mask
        self.dl_fft = dl_fft
        self.ntheta, self.nphi = dl_fft.shape
        self.pick = (mask.reshape(self.ntheta * self.nphi) == 0)
        self.n = int(self.pick.sum())
        self.R = full_coarsen_matrix(self.ntheta, self.nphi)
        self.ntheta_H = self.ntheta // 2
        self.nphi_H = self.nphi // 2

    def pickvec(self, u):
        return u.reshape(self.ntheta * self.nphi)[self.pick]

    def padvec(self, u):
        u_pad = np.zeros(self.ntheta * self.nphi)
        u_pad[self.pick] = u.real
        return u_pad.reshape(self.ntheta, self.nphi)

    def matvec_padded(self, u):
        u = flatsky_adjoint_synthesis(u)
        u *= self.dl_fft
        u = flatsky_synthesis(u).real
        return u
    
    def matvec(self, u):
        u = self.padvec(u)
        u = self.matvec_padded(u)
        u = self.pickvec(u)
        return u

    def matvec_coarsened(self, u):
        # do matvec on the next, coarser level. This is just done once, to create the operator on the next level
        return self.coarsen_padded(self.matvec_padded(self.interpolate_padded(u)))

    def coarsen_padded(self, u):
        return (self.R * u.reshape(self.ntheta * self.nphi)).reshape(self.ntheta_H, self.nphi_H)

    def interpolate_padded(self, u):
        return (self.R.T * u.reshape(self.ntheta_H * self.nphi_H)).reshape(self.ntheta, self.nphi)



class DiagonalSmoother(object):

    def __init__(self, level):
        self.level = level

        # hammer operator on a tilesize x tilesize patch...
        u = np.zeros((level.ntheta, level.nphi))
        u[level.ntheta // 2, 0] = 1
        u = level.matvec_padded(u).real
        ## u = flatsky_adjoint_synthesis(u)
        ## u *= level.dl_fft
        ## u = flatsky_synthesis(u).real
        self.diag = u[level.ntheta // 2, 0] #* (1 + 1e-1)
        self.inv_diag = 1 / self.diag


    def apply(self, u):
        return OMEGA * self.inv_diag * u


def v_cycle(ilevel, levels, smoothers, b):
    if ilevel == len(levels) - 1:
        return smoothers[ilevel].apply(b)
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]

        x = b * 0
        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))

        for i in range(2):
            r_h = b - level.matvec(x)

            r_H = coarsen(level, next_level, r_h)

            c_H = v_cycle(ilevel + 1, levels, smoothers, r_H)

            c_h = interpolate(level, next_level, c_H)
            
            x += c_h

        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))
        return x

def coarsen(level, next_level, u):
    return next_level.pickvec(level.coarsen_padded(level.padvec(u)))

def interpolate(level, next_level, u):
    return level.pickvec(level.interpolate_padded(next_level.padvec(u)))
    

def coarsen_level(level):
    # produce next coarser level
    ntheta_H = level.ntheta // 2
    nphi_H = level.nphi // 2
    mask_H = level.coarsen_padded(level.mask)
    mask_H[mask_H < 0.5] = 0
    mask_H[mask_H != 0] = 1

    unitvec = np.zeros((ntheta_H, nphi_H))
    unitvec[ntheta_H // 2, nphi_H // 2] = 1
    image_of_operator = level.matvec_coarsened(unitvec)
    dl_fft_H = image_to_powspec(unitvec, image_of_operator)
    return Level(dl_fft_H, mask_H)


cur_level = level = Level(dl_fft, mask)
levels = [cur_level]

while cur_level.n > 500:
    cur_level = coarsen_level(cur_level)
    levels.append(cur_level)
    
smoothers = [DiagonalSmoother(lev) for lev in levels]



rng = np.random.RandomState(12)
x0 = rng.normal(size=level.n)

matvec_count = 0

def precond(b):
    x = v_cycle(0, levels, smoothers, b)
    return x

if 1:
    b = level.matvec(x0)

    solver = cg_generator(
        level.matvec,
        b,
        M=precond
        )


norm0 = np.linalg.norm(x0)
errlst = []

for i, (x, r, delta_new) in enumerate(solver):
    errvec = x0 - x
    errlst.append(np.linalg.norm(errvec) / norm0)
    if i > 30:
        break



#clf();
#imshow(padvec(x-x0), interpolation='none');
#semilogy(matvec_counts, errlst, '-o')
semilogy(errlst, '-o')
#colorbar()
draw()





if 0:
    A_h = hammer(level.matvec, level.n_mask)
    R = coarseners[0].toarray()

    A_H = np.dot(R, np.dot(A_h, R.T))
    
    A_H_0 = hammer(levels[1].matvec, levels[1].n_mask)

    clf()
    plot(A_H.diagonal())
    plot(A_H_0.diagonal())
    draw()
    1/0
    


    

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

    INNERITS = 10

    def matvec(x):
        global matvec_count
        matvec_count += 1
        return level.matvec(x)
    
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
        x = jacobi(matvec, precond, b, INNERITS)
        x = jacobi(matvec, precond, x, INNERITS)
        return x

    def matvec2(u):
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
        M=precond2
        )

    #A = hammer(matvec2, level.n_mask)
    #M = hammer(precond2, level.n_mask)
    #w = np.linalg.eigvals(np.dot(M, A))
    #clf()
    #semilogy(sorted(w))
    #draw()
    
    
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
    if i > 5:
        break



#clf();
#imshow(padvec(x-x0), interpolation='none');
#semilogy(matvec_counts, errlst, '-o')
semilogy(errlst, '-o')
#colorbar()
draw()

## if 1:
##     A = hammer(level.matvec, level.n)
##     M = hammer(smoother.apply, level.n)

##     lam = np.asarray(sorted(np.linalg.eigvals(np.dot(M, A))))
##     clf()
##     semilogy(np.linalg.eigvalsh(A))
##     semilogy(np.linalg.eigvalsh(M)[::-1])
##     semilogy(lam)
##     print lam.max()
##     draw()
##     1/0

