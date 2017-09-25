from __future__ import division
import numpy as np
from scipy.sparse import dok_matrix
import scipy
import healpy

from .beams import standard_needlet_by_l, fourth_order_beam, gaussian_beam_by_l
from . import sharp
from .cg import cg_generator
from .utils import scatter_l_to_lm, hammer, pad_or_truncate_alm
from .cache import memory
from .healpix import nside_of


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


def v_cycle_combined_shts(ilevel, levels, smoothers, b):
    # like the above, but inline coarsen/interpolate in order to remove redundant SHTs

    M_b = smoothers[ilevel].apply(b)
    
    if ilevel == len(levels) - 1:
        return M_b
    else:
        level = levels[ilevel]
        next_level = levels[ilevel + 1]
        lmax_restrict = level.lmax // 2


        def Yt_h(v):
            return sharp.sh_adjoint_synthesis(level.lmax, level.padvec(v))

        def Ytw_h(v):
            # use lmax_restrict as the output is passed to R
            return sharp.sh_analysis(lmax_restrict, level.padvec(v))
        
        def WY_h(v):
            return level.pickvec(sharp.sh_adjoint_analysis(level.nside, v))

        def Ytw_H(v):
            return sharp.sh_analysis(level.lmax, next_level.padvec(v))
        
        def Yt_H(v):
            return sharp.sh_adjoint_synthesis(lmax_restrict, next_level.padvec(v))
        
        def Y_H(v):
            return next_level.pickvec(sharp.sh_synthesis(next_level.nside, v))
        
        def Y_h(v):
            return level.pickvec(sharp.sh_synthesis(level.nside, v))

        def R(v):
            v = pad_or_truncate_alm(v, lmax_restrict)
            return scatter_l_to_lm(level.restrict_l[:lmax_restrict + 1]) * v

        def Rt(v):
            v = scatter_l_to_lm(level.restrict_l[:lmax_restrict + 1]) * v
            return pad_or_truncate_alm(v, level.lmax)
        
        def D(v):
            return scatter_l_to_lm(level.dl) * v

        def M(v):
            return smoothers[ilevel].apply(v)

        def approx_I(v):
            return v
            #return Ytw_h(Y_h(v))
        
        x = M_b.copy()

        u = Yt_h(x)

        r_H = Y_H( R( Ytw_h(b) - pad_or_truncate_alm( D( u ) , lmax_restrict) ) )

        c_H = v_cycle_combined_shts(ilevel + 1, levels, smoothers, r_H)

        assert c_H.shape[0] == next_level.n
        
        v_lo = Rt ( Yt_H ( c_H ) )
        v = pad_or_truncate_alm(v_lo, level.lmax)
        x += M_b + WY_h ( v_lo ) - M( Y_h( D( u + approx_I(v) ) ))
        
        return x



class SinvSolver(object):

    def __init__(self, dl, mask):
        self.dl = dl
        self.lmax = self.dl.shape[0] - 1
        self.mask = mask
        self.nside = nside_of(mask)

        root_level = Level(dl, mask)
        self.levels = [root_level]

        cur_level = root_level
        while cur_level.n > 1000:
            cur_level = coarsen_level(cur_level)
            self.levels.append(cur_level)


        self.smoothers = [DiagonalSmoother(level) for level in self.levels[:-1]]
        self.smoothers.append(DenseSmoother(self.levels[-1]))

        self.n = int((self.mask == 0).sum())

    def restrict(self, u):
        return self.pickvec(sharp.sh_synthesis(self.nside, u))

    def prolong(self, u):
        return sharp.sh_adjoint_synthesis(self.lmax, self.padvec(u))
            
    def pickvec(self, u):
        return self.levels[0].pickvec(u)

    def padvec(self, u):
        return self.levels[0].padvec(u)

    def precond(self, b):
        #return v_cycle(0, self.levels, self.smoothers, b)
        return v_cycle_combined_shts(0, self.levels, self.smoothers, b)

    def solve_mask(self, b, x0=None, rtol=1e-6, maxit=50):
        """
        Returns (x, reslst, errlst)

        If x0 is supplied, compute the errors and return errlst; otherwise errlst is empt
        """
        solver = cg_generator(
            self.levels[0].matvec,
            b=b,
            M=self.precond,
            x0=np.zeros_like(b)
            )

        reslst = []
        errlst = []
        if x0 is not None:
            x0_norm = np.linalg.norm(x0)
        b_norm = np.linalg.norm(b)

        for i, (x, r, delta_new) in enumerate(solver):
            r = np.linalg.norm(r) / b_norm
            reslst.append(r)
            if x0 is not None:
                e = np.linalg.norm(x0 - x) / x0_norm
                errlst.append(e)
                print 'iteration {}, res={}, err={}'.format(i, r, e)
            else:
                pass
                print 'iteration {}, res={}'.format(i, r)
            if r < rtol or i > maxit:
                print 'breaking', r, repr(rtol), i, maxit
                break

        return x, reslst, errlst


    

class Level(object):

    def __init__(self, dl, mask):
        self.mask = mask
        self.lmax = dl.shape[0] - 1
        self.nside = nside_of(mask)
        self.npix = 12 * self.nside**2
        self.dl = dl

        self.pick = (mask == 0)
        self.n = int(self.pick.sum())

        pw = 2
        self.restrict_l = gaussian_beam_by_l(self.lmax, 2 * np.pi / (4 * self.nside) * pw)
        ##self.restrict_l = fourth_order_beam(self.lmax, self.lmax // 2, 0.05)

        
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
        return self.pickvec(self.matvec_padded(self.padvec(u)))

    def matvec_coarsened(self, u):
        # do matvec on the next, coarser level. This is just done once, to create the operator on the next level
        return self.coarsen_padded(self.matvec_padded(self.interpolate_padded(u)))

    def coarsen_padded(self, u):
        lmax_restrict = self.lmax // 2
        alm = sharp.sh_analysis(lmax_restrict, u)
        alm *= scatter_l_to_lm(self.restrict_l[:lmax_restrict + 1])
        u = sharp.sh_synthesis(self.nside // 2, alm)
        return u

    def interpolate_padded(self, u):
        lmax_restrict = self.lmax // 2
        alm = sharp.sh_adjoint_synthesis(lmax_restrict, u)
        alm *= scatter_l_to_lm(self.restrict_l[:lmax_restrict + 1])
        u = sharp.sh_adjoint_analysis(self.nside, alm)
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
        self.inv_matrix = np.linalg.pinv(self.matrix)

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
    if 0:
        unitvec = np.zeros(12 * nside_H**2)
        unitvec[6 * nside_H**2 + 2 * nside_H] = 1

        image_of_operator = level.matvec_coarsened(unitvec)
        YtW_x = sharp.sh_analysis(level.lmax, image_of_operator)
        Yt_u = sharp.sh_adjoint_synthesis(level.lmax, unitvec)
        dl_H = operator_image_to_power_spectrum(level.lmax // 2, unitvec, image_of_operator)
    else:
        dl_H = (level.restrict_l**2 * level.dl)[:level.lmax // 2 + 1]
    
    mask_H = healpy.ud_grade(level.mask, order_in='RING', order_out='RING', nside_out=nside_H, power=0)
    mask_H[mask_H != 0] = 1
    return Level(dl_H, mask_H)
