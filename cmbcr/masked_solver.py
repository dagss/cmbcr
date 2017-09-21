from __future__ import division
import numpy as np
from scipy.sparse import dok_matrix
import scipy
import healpy

from .beams import standard_needlet_by_l, fourth_order_beam, gaussian_beam_by_l
from . import sharp
from .cg import cg_generator
from .utils import scatter_l_to_lm, hammer
from .cache import memory
from .healpix import nside_of



class SinvSolver(object):

    def __init__(self, dl, mask):
        self.dl = dl
        self.lmax = self.dl.shape[0] - 1
        self.mask = mask
        self.nside = nside_of(mask)

        root_level = Level(dl, mask)
        self.levels = [root_level]

        cur_level = root_level
        while cur_level.n > 200:
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
        return v_cycle(0, self.levels, self.smoothers, b)

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
        return healpy.ud_grade(u, order_in='RING', order_out='RING', nside_out=self.nside // 2, power=0)

    def interpolate_padded(self, u):
        return healpy.ud_grade(u, order_in='RING', order_out='RING', nside_out=self.nside, power=0)


class DiagonalSmoother(object):
    def __init__(self, level):
        self.level = level

        self.diag = level.compute_diagonal()
        self.inv_diag = 1 / self.diag

    def apply(self, u):
        return 0.3 * self.inv_diag * u


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
    unitvec = np.zeros(12 * nside_H**2)
    unitvec[6 * nside_H**2 + 2 * nside_H] = 1

    image_of_operator = level.matvec_coarsened(unitvec)

    YtW_x = sharp.sh_analysis(level.lmax, image_of_operator)
    Yt_u = sharp.sh_adjoint_synthesis(level.lmax, unitvec)
    dl_H = operator_image_to_power_spectrum(level.lmax // 2, unitvec, image_of_operator)

    mask_H = level.coarsen_padded(level.mask)
    mask_H[mask_H < 0.5] = 0
    mask_H[mask_H != 0] = 1
    return Level(dl_H, mask_H)
