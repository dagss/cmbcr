from __future__ import division
import numpy as np

from matplotlib.pyplot import *

import logging
logging.basicConfig(level=logging.DEBUG)


import healpy
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
from cmbcr.healpix import nside_of
from cmbcr import sharp
from healpy import mollzoom, mollview
from scipy.sparse import csc_matrix, dok_matrix
#reload(cmbcr.main)

from cmbcr.cr_system import load_map_cached

import sys
from cmbcr.cg import cg_generator

import scipy.ndimage

nside = 64
lmax = 3 * nside


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


class HealPixRestriction(object):

    def __init__(self, nside):
        self.nside_h = nside
        self.nside_H = nside // 2
        self.neighbour_matrix = csc_neighbours(self.nside_H, np.ones(12 * self.nside_H**2).astype(bool))
        self.alpha = 2


    def apply(self, u):
        u_H = healpy.ud_grade(u, order_in='RING', order_out='NESTED', nside_out=self.nside_H, power=0)
        return healpy.reorder(self.neighbour_matrix * u_H + self.alpha * u_H, n2r=True)

    def apply_transpose(self, u_H):
        u_H = healpy.reorder(u_H, r2n=True)
        u_H = self.neighbour_matrix * u_H + self.alpha * u_H
        u_h = healpy.ud_grade(u_H, order_in='NESTED', order_out='RING', nside_out=self.nside_h, power=0) * 0.25
        return u_h


if 0:
    u = np.zeros(12 * (nside//2)**2)
    u[6 * (nside//2)**2 + 2 * nside//2] = 1
    R = HealPixRestriction(nside)

    clf()
    mollview(R.apply_transpose(u), sub=111, hold=True, nest=False)
    draw()

    1/0




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


#l = np.arange(1, lmax + 2).astype(float)
#dl = l**6

dl = 1 / Cl
dl_orig = 1 / Cl

rl = cmbcr.fourth_order_beam(lmax, lmax // 2, 0.1)
rl = np.sqrt(rl)
dl *= rl**2


mask_healpix = load_map_cached('mask_galactic_band_2048.fits')

mask = healpy.ud_grade(mask_healpix, order_in='RING', order_out='RING', nside_out=nside, power=0)
#mask[mask != 0] = 1

mask[:] = 0

pw = 3.8



class Level(object):

    def __init__(self, dl, mask):
        self.mask = mask
        self.lmax = dl.shape[0] - 1
        self.nside = nside_of(mask)
        self.npix = 12 * self.nside**2
        self.dl = dl

        self.pick = (mask == 0)
        self.n = int(self.pick.sum())

        self.restrict_l = cmbcr.gaussian_beam_by_l(self.lmax, 2 * np.pi / (4 * self.nside) * pw)
        ##self.R = HealPixRestriction(self.nside)

        #neighbour_pattern = csc_neighbours(self.nside, self.pick)
        
        ##self.coarsen_matrix = csc_neighbours(self.nside, np.ones(12*nside**2).astype(bool))
        #self.coarsen_matrix = self.coarsen_matrix * self.coarsen_matrix
        ##self.coarsen_matrix.data *= (1/8.)
        ##self.coarsen_matrix += scipy.sparse.diags([np.ones(self.coarsen_matrix.shape[0])], [0]) * 1
        

        
        #Si_pattern = Si_pattern * Si_pattern# * Si_pattern * Si_pattern
        #Si_pattern.sum_duplicates()
        #Si_pattern = Si_pattern.tocsc()
        #diag_val = cmbcr.beam_by_cos_theta(dl, np.ones(1))[0]
        #ridge_factor = 5e-2
        #self.Si_sparse = make_Si_sparse_matrix(Si_pattern, self.dl, 0)#diag_val * ridge_factor)



    def compute_diagonal(self):
        u = np.zeros(self.npix)
        u[self.npix // 2] = 1
        u = self.matvec_padded(u)
        return u[self.npix // 2]

    def pickvec(self, u):
        return u[self.pick]

    def padvec(self, u):
        u_pad = np.zeros(self.npix)
        u_pad[self.pick] = u
        return u_pad

    def matvec_padded(self, u):
        u = sharp.sh_adjoint_synthesis(self.lmax, u)
        u *= scatter_l_to_lm(self.dl)
        u = sharp.sh_synthesis(self.nside, u)
        return u

    def matvec(self, u):
        #return self.Si_sparse * u
        return self.pickvec(self.matvec_padded(self.padvec(u)))

    def matvec_coarsened(self, u):
        # do matvec on the next, coarser level. This is just done once, to create the operator on the next level
        return self.coarsen_padded(self.matvec_padded(self.interpolate_padded(u)))

    def coarsen_padded(self, u):
        ##return self.R.apply(u)


        if 1:
            alm = sharp.sh_analysis(self.lmax, u)
            alm *= scatter_l_to_lm(self.restrict_l)
            u = sharp.sh_synthesis(self.nside // 2, alm)
        else:
            u = healpy.ud_grade(u, order_in='RING', order_out='RING', nside_out=self.nside // 2, power=0)
        return u

    def interpolate_padded(self, u):
        ##return self.R.apply_transpose(u)

        if 1:
            alm = sharp.sh_adjoint_synthesis(self.lmax, u)
            alm *= scatter_l_to_lm(self.restrict_l)
            u = sharp.sh_adjoint_analysis(self.nside, alm)
        else:
            u = healpy.ud_grade(u, order_in='RING', order_out='RING', nside_out=self.nside, power=0)
        ##u = healpy.reorder(self.coarsen_matrix.T * healpy.reorder(u, r2n=True), n2r=True)
        return u


class DiagonalSmoother(object):
    def __init__(self, level):
        self.level = level

        self.diag = level.compute_diagonal()
        self.inv_diag = 1 / self.diag

    def apply(self, u):
        return 0.2 * self.inv_diag * u


class DenseSmoother(object):
    def __init__(self, level):
        self.matrix = hammer(level.matvec, level.n)
        self.inv_matrix = np.linalg.pinv(self.matrix, rcond=0.05)

    def apply(self, u):
        return np.dot(self.inv_matrix, u)
    

def operator_image_to_power_spectrum(lmax, unitvec, opimage):
    # unitvec: unit-vector in flatsky basis
    # x: image of operator
    YtW_x = sharp.sh_analysis(lmax, opimage)
    Yt_u = sharp.sh_adjoint_synthesis(lmax, unitvec)
    return YtW_x[:lmax + 1] / Yt_u[:lmax + 1]



def coarsen_level(level):
    nside_H = level.nside // 2


    if 1:
        lmax_H = level.lmax // 2

        unitvec = np.zeros(12 * nside_H**2)
        unitvec[6 * nside_H**2 + 2 * nside_H] = 1


        image_of_operator = level.matvec_coarsened(unitvec)
        YtW_x = sharp.sh_analysis(lmax, image_of_operator)
        Yt_u = sharp.sh_adjoint_synthesis(lmax, unitvec)

        dl_H = YtW_x[:lmax_H + 1] / Yt_u[:lmax_H + 1]

    elif 0:

        lmax_H = level.lmax // 2
        nrings = lmax_H + 1
        unitvec = np.zeros(2 * nrings**2)
        unitvec[nrings**2] = 1

        x = unitvec

        # interpolate
        x = sharp.sh_adjoint_synthesis_gauss(lmax_H, x, lmax_sh=level.lmax)
        x *= scatter_l_to_lm(level.restrict_l)
        x = sharp.sh_adjoint_analysis_gauss(level.lmax, x)

        # matvec
        x = sharp.sh_adjoint_synthesis_gauss(level.lmax, x)
        x *= scatter_l_to_lm(level.dl)
        x = sharp.sh_synthesis_gauss(level.lmax, x)

        # restrict
        x = sharp.sh_analysis_gauss(level.lmax, x)
        x *= scatter_l_to_lm(level.restrict_l)
        x = sharp.sh_synthesis_gauss(lmax_H, x, lmax_sh=level.lmax)

        image_of_operator = x

        
        YtW_x = sharp.sh_analysis_sharp(lmax_H, image_of_operator)
        Yt_u = sharp.sh_adjoint_synthesis_sharp(lmax_H, unitvec)

        dl_H = YtW_x[:lmax_H + 1] / Yt_u[:lmax_H + 1]

        
        
        #x = level.interpolated_padded(x)
        #alm = sharp.sh_adjoint_synthesis_gauss(lmax_H, unitvec)
                               

    else:
        dl_H = (level.dl * level.restrict_l**2)[:level.lmax // 2 + 1]

    mask_H = level.coarsen_padded(level.mask)
    mask_H[mask_H < 0.5] = 0
    mask_H[mask_H != 0] = 1
    return Level(dl_H, mask_H)



#dl[:] = 1
root_level = Level(dl, mask)
levels = [root_level]

cur_level = root_level
while cur_level.n > 100:
    cur_level = coarsen_level(cur_level)
    levels.append(cur_level)



smoothers = [DiagonalSmoother(level) for level in levels]
#smoothers.append(DenseSmoother(levels[-1]))


#levels = levels[:1]
#smoothers = smoothers[:1]


# Test

rng = np.random.RandomState(11)


if 0:
    nside_H = nside // 2
    u = np.zeros(12* (nside // 2)**2)
    u[6 * (nside // 2)**2 + 20] = 1
    #u  = rng.normal(size=12 * (nside_H)**2)

    alm = sharp.sh_analysis(lmax, u)
    alm *= scatter_l_to_lm(cmbcr.gaussian_beam_by_l(lmax, '10 deg'))
    u = sharp.sh_synthesis(nside_H, alm)
    #u[:] = 1


    Au1 = levels[0].matvec_coarsened(u)
    Au2 = levels[1].matvec_padded(u)

    clf()
    mollview(Au1, sub=211, fig=gcf().number)
    mollview(Au2, sub=212, fig=gcf().number)
    draw()
    1/0



def coarsen(level, next_level, u):
    return next_level.pickvec(level.coarsen_padded(level.padvec(u)))

def interpolate(level, next_level, u):
    return level.pickvec(level.interpolate_padded(next_level.padvec(u)))    


def v_cycle(ilevel, levels, smoothers, b):
    if ilevel == len(levels) - 1:
        return smoothers[ilevel].apply(b)
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]

        x = b * 0
        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))

        for i in range(1):
            r_h = b - level.matvec(x)

            r_H = coarsen(level, next_level, r_h)

            c_H = v_cycle(ilevel + 1, levels, smoothers, r_H)

            c_h = interpolate(level, next_level, c_H)
            
            x += c_h

        for i in range(1):
            x += smoothers[ilevel].apply(b - level.matvec(x))
        return x

    

def v_cycle2(ilevel, levels, smoothers, b):
        

    
    M_b = smoothers[ilevel].apply(b)
    
    if ilevel == len(levels) - 1:
        return M_b
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]


        def YTWY(v):
            ##return v
            return sharp.sh_analysis(level.lmax, (1 - level.mask) * sharp.sh_synthesis(level.nside, v))

        def Yt_h(v):
            return sharp.sh_adjoint_synthesis(level.lmax, level.padvec(v))

        def Ytw_h(v):
            return sharp.sh_analysis(level.lmax, level.padvec(v))
        
        def WY_h(v):
            return level.pickvec(sharp.sh_adjoint_analysis(level.nside, v))

        def Ytw_H(v):
            return sharp.sh_analysis(level.lmax, next_level.padvec(v))
        
        def Yt_H(v):
            return sharp.sh_adjoint_synthesis(level.lmax, next_level.padvec(v))
        
        def Y_H(v):
            return next_level.pickvec(sharp.sh_synthesis(next_level.nside, v))
        
        def Y_h(v):
            return level.pickvec(sharp.sh_synthesis(level.nside, v))

        def R(v):
            return scatter_l_to_lm(level.restrict_l) * v
        
        def D(v):
            return scatter_l_to_lm(level.dl) * v
        
        
        x = M_b

        r_h = b - Y_h( D( Yt_h(x) ) )
        r_H = Y_H( R( Ytw_h(r_h) ) )

        c_H = v_cycle2(ilevel + 1, levels, smoothers, r_H)

        assert c_H.shape[0] == next_level.n
        c_h = WY_h ( R ( Yt_H ( c_H ) ) )

        x += c_h

        r_h = b - Y_h( D( Yt_h(x) ) )
        x += smoothers[ilevel].apply(r_h)

        
        return x
    
    


TOP_LEVEL_SH = True


errlst = []

#if x0 is not None:
#    x0_norm = np.linalg.norm(x0)

maxit = 40


#x0 = rng.normal(size=root_level.n)
#b = root_level.matvec(x0)

#x0 = rng.norm

if TOP_LEVEL_SH:

    hipass_l = np.zeros(lmax + 1)
    hipass_l[3 * lmax // 4:] = 1
    hipass_lm = scatter_l_to_lm(hipass_l)

    def matvec(u):
        return u * scatter_l_to_lm(dl_orig)

    if 0:
        LOWL = 5


        def matvec_lo(x):
            x = pad_or_truncate_alm(x, lmax)
            x = matvec(x)
            return pad_or_truncate_alm(x, LOWL)

        lowl_block = hammer(matvec_lo, (LOWL + 1)**2)
        low_l_inv = np.linalg.inv(lowl_block)
    
    def precond(b):
        x = scatter_l_to_lm((1 - rl) * (1 / dl_orig)) * b

        b_pix = levels[0].pickvec(sharp.sh_synthesis(nside, b * scatter_l_to_lm(rl)))
        x_pix = v_cycle2(0, levels, smoothers, b_pix)
        lo_corr = sharp.sh_adjoint_synthesis(lmax, levels[0].padvec(x_pix)) * scatter_l_to_lm(rl)

        x += lo_corr

        ###x += pad_or_truncate_alm(np.dot(low_l_inv, pad_or_truncate_alm(b, LOWL)), lmax)
        
        return x




    
    x0 = rng.normal(size=(lmax + 1)**2)
    x0_l = cmbcr.norm_by_l(x0)
else:
    matvec = levels[0].matvec

    def precond(b):
        #return b
        return v_cycle(0, levels, smoothers, b)
    
    x0 = rng.normal(size=levels[0].n)
    x0_l = cmbcr.norm_by_l(sharp.sh_analysis(lmax, levels[0].padvec(x0)))

    

b = matvec(x0)

cgsolver = cg_generator(
    matvec,
    b,
    M=precond
    )



rlm = scatter_l_to_lm(rl)

x0_norm = np.linalg.norm(x0)

#x = np.zeros_like(x0)
#for i in range(10):

#    x += precond(b - matvec(x))
#clf()

    
for i, (x, r, delta_new) in enumerate(cgsolver):
    #r = np.linalg.norm(r) / b_norm
    #reslst.append(r)
    
    e = np.linalg.norm(x0 - x) / x0_norm
    errlst.append(e)
    print 'OUTER iteration {}, err={}'.format(i, e)

    elm = x - x0
    if not TOP_LEVEL_SH:
        elm = sharp.sh_analysis(lmax, levels[0].padvec(elm))
    #semilogy(cmbcr.norm_by_l(elm) / x0_l)
    
    if i > maxit:
        break

    
#clf()
semilogy(errlst, '-o')
draw()
#draw()

def errmap():
    clf()
    q = x0 - x
    if TOP_LEVEL_SH:
        q = sharp.sh_synthesis(nside, q)
    else:
        q = levels[0].padvec(q)
    
    mollview(q, fig=gcf().number)
    draw()
1/0    
