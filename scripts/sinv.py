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


def beam_by_theta(bl, thetas):
    from libsharp import legendre_transform

    lmax = bl.shape[0] - 1
    l = np.arange(lmax + 1)
    scaled_bl = bl * (2 * l + 1)
    
    return legendre_transform(np.cos(thetas), scaled_bl) / (4. * np.pi)

def standard_needlet_by_l(B, lmax):
    """
    Returns the spherical harmonic profile of a standard needlet on the sphere,
    following the recipe of arXiv:1004.5576. Instead of providing j, you provide
    lmax, and a j is computed so that the function reaches 0 at lmax.
    """
    from scipy.integrate import quad
    
    def f(t):
        if -1 < t < 1:
            return np.exp(-1 / (1 - t*t))
        else:
            return 0

    phi_norm = 1 / quad(f, -1, 1)[0]
    
    def phi(u):
        return quad(f, -1, u)[0] * phi_norm

    def phi2(t, B):
        if t <= 1 / B:
            return 1
        elif t < 1:
            return phi(1 - (2 * B) / (B - 1) * (t - 1 / B))
        else:
            return 0

    def b(eta, B):
        if eta < 1 / B:
            return 0
        elif eta < B:
            b_sq = phi2(eta / B, B) - phi2(eta, B)
            if b_sq <= 0:
                return 0
            else:
                return np.sqrt(b_sq)
        else:
            return 0

    j = (np.log(lmax) - np.log(B)) / np.log(B)
    C = float(B)**j
    return np.asarray([b(l / C, B) for l in range(lmax + 1)])


config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))


w = 1

nside = 64 * w
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



dl = system.dl_list[0]
nl = standard_needlet_by_l(10, 1 * dl.shape[0] - 1)
i = nl.argmax()
dl = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])

clf();
x = beam_by_theta(dl, np.linspace(0, 0.3, 1000))

plot(dl * x.max())
plot(x)
draw()
1/0
#1/0


#dl = 1 / cmbcr.gaussian_beam_by_l(system.lmax_list[0], '10 deg')

rng = np.random.RandomState(1)

x0 = [
    #scatter_l_to_lm(1. / dl) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)


from cmbcr.precond_psuedoinv import *

x = lstscale(0, b)

from cmbcr import sympix, block_matrix, sympix_mg


class SinvSolver(object):
    method = 'icc'
    
    def __init__(self, system, lmax_grid, rl=None, tilesize=8, ridge=5e-2):

        self.tilesize = tilesize
        if rl is None:
            self.lmax = system.lmax_list[0]
            rl = np.ones(self.lmax + 1)
        else:
            self.lmax = rl.shape[0] - 1
        self.rl = rl

        self.grid = sympix.make_sympix_grid(lmax_grid + 1, tilesize, n_start=8)
        self.plan = sharp.SymPixGridPlan(self.grid, self.lmax)

        self.dl = dl[:self.lmax + 1] * rl**2 #system.dl_list[0][:self.lmax + 1] * rl**2
        
        mask_lm = sharp.sh_analysis(self.lmax, system.mask)
        plan = sharp.SymPixGridPlan(self.grid, self.lmax)
        t = 0.8
        mask = plan.synthesis(mask_lm)
        mask[mask <= t] = 0
        mask[mask > t] = 1

        # Estimate max, in order to create ridge
        z = np.zeros_like(mask)
        z[z.shape[0]//2] = 1
        z = plan.adjoint_synthesis(z)
        z *= scatter_l_to_lm(self.dl)
        z = plan.synthesis(z)
        self.estimated_max = z[z.shape[0]//2]
        self.ridge = ridge

        self.bs = tilesize**2
        corner_factor = 0.6


        if 0:
            clf()
            imshow(sympix.sympix_plot(self.grid, mask), interpolation='none')
            draw()
            1/0

        neighmat, (label_to_i, label_to_j) = sympix.sympix_csc_neighbours(self.grid, lower_only=False, corner_factor=corner_factor)
        neighmat_lower, (label_to_i_lower, label_to_j_lower) = sympix.sympix_csc_neighbours(self.grid, lower_only=True, corner_factor=corner_factor)

        self.pickvec = (mask == 0)
        self.n = int(self.pickvec.sum())
        self.npix = mask.shape[0]

        mask_reshaped = mask.reshape(tilesize**2, self.npix / tilesize**2, order='F')

        Si = block_matrix.BlockMatrix(neighmat_lower.indptr, neighmat_lower.indices, blockshape=(tilesize**2, tilesize**2))
        Si_blocks = sympix_mg.compute_many_YDYt_blocks(
            self.grid, self.grid, self.dl, np.asarray(label_to_i_lower, dtype=np.int32), np.asarray(label_to_j_lower, dtype=np.int32))
        Si_blocks = Si_blocks[:, :, neighmat_lower.data]


        if self.method == 'icc':
            # For each of the blocks we should go through and mask it...
            print Si.indptr.shape
            print self.npix
            for j in range(self.npix // self.bs):
                for iptr in range(Si.indptr[j], Si.indptr[j + 1]):
                    i = Si.indices[iptr]

                    # apply the mask
                    Si_blocks[:, :, iptr] *= (1 - mask_reshaped[:, i])[:, None]
                    Si_blocks[:, :, iptr] *= (1 - mask_reshaped[:, j])[None, :]

                    if i == j:
                    ##    # diagonal block...
                        r = self.ridge * self.estimated_max
                        Si_blocks[:, :, iptr] += np.eye(Si_blocks.shape[0]) * r
                        Si_blocks[:, :, iptr] += np.diag(mask_reshaped[:, i])

            Si.blocks = Si_blocks

            with timed('Ridge probing and ICC'):
                alpha, ncalls = block_matrix.probe_cholesky_ridging(Si.indptr, Si.indices, Si.blocks, ridge=0, eps_log10=.1)
                print 'alpha', alpha / Si.blocks.max(), 'abs', '%e' % alpha

                block_matrix.block_incomplete_cholesky_factor(Si.indptr, Si.indices, Si.blocks, alpha=alpha * 1.5)

            self.Si = Si
            self.weight = 1

        else:
            return
            1/0
                #    block_incomplete_cholesky_factor(self.M.indptr, self.M.indices, self.M.blocks, alpha=alpha * 1.5)




            # only care about diagonal blocks now...
            self.precond_data = Si_blocks[:, :, A_sparse.labels[A_sparse.indptr[:-1]]].copy('F')
            for i in range(self.precond_data.shape[2]):
                self.precond_data[:, :, i] += np.eye(self.precond_data.shape[0]) * ridge * self.estimated_max
                self.precond_data[:, :, i] *= (1 - mask_reshaped[:, i])[:, None]
                self.precond_data[:, :, i] *= (1 - mask_reshaped[:, i])[None, :]
                self.precond_data[:, :, i] += np.diag(mask_reshaped[:, i])
            block_matrix.block_diagonal_factor(self.precond_data)

            self.weight = 0.3

    def pad(self, x):
        x_pad = np.zeros(self.npix)
        x_pad[self.pickvec] = x
        return x_pad

    def pick(self, x):
        return x[self.pickvec]

    def adjoint_synthesis(self, x):
        return self.plan.adjoint_synthesis(self.pad(x))

    def synthesis(self, x):
        return self.pick(self.plan.synthesis(x))

    def analysis(self, x):
        return self.plan.analysis(self.pad(x))

    def adjoint_analysis(self, x):
        return self.pick(self.plan.adjoint_analysis(x))

    def norm_by_l(self, x):
        return cmbcr.norm_by_l(self.plan.analysis(self.pad(x)))

    def error_smoother_diag(self, x):
        x = self.pad(x).reshape((self.bs, self.npix // self.bs), order='F')
        block_matrix.block_diagonal_solve(self.precond_data, x)
        assert not np.any(np.isnan(x))
        x = self.pick(x.reshape(self.npix, order='F'))
        return self.weight * x

    def error_smoother(self, x):
        if np.all(self.rl == 1) and False:
            ###return x / self.Si.blocks.max()
            ## print 'was hit'

            ql = 1 / self.dl
            ql[:2 * self.lmax//4] = 0
            
            x = self.analysis(x)
            x *= scatter_l_to_lm(ql)
            return self.adjoint_analysis(x) * 1
        else:        
            x_stacked = self.pad(x).reshape((self.bs, self.npix // self.bs), order='F')
            block_matrix.block_triangular_solve('N', self.Si.indptr, self.Si.indices, self.Si.blocks, x_stacked)
            block_matrix.block_triangular_solve('T', self.Si.indptr, self.Si.indices, self.Si.blocks, x_stacked)
            x = self.pick(x_stacked.reshape(self.npix, order='F'))
            return self.weight * x
        
    def matvec(self, x_in):
        x = self.adjoint_synthesis(x_in)
        x *= scatter_l_to_lm(self.dl)
        return self.synthesis(x) + self.ridge * self.estimated_max * x_in


def gaussian_beam_by_l_by_eps(lmax, eps=0.1):
    ls = np.arange(lmax + 1)
    sigma_sq = -2. * np.log(eps) / lmax / (lmax + 1)
    return np.exp(-0.5 * ls * (ls + 1) * sigma_sq)


lmax = dl.shape[0] - 1 #system.lmax_list[0]

ridge = 5e-2

level0 = SinvSolver(
    system,
    lmax,# + 100,
    np.ones(lmax + 1),#gaussian_beam_by_l_by_eps(lmax, 0.1),
    ridge=ridge)


# NOTES:
# - larger ridge and it solves on large scales
# - smaller rigde and it doesn't work on large scales
# - attempt 1: introduce dense solves
#
# for small scales, should have a look at dampening with needlet tail
#
#
#



eps = 0.4

levels = [
    level0,
        
    SinvSolver(
        system,
        lmax // 2,
        gaussian_beam_by_l_by_eps(lmax, eps),
        ridge=ridge),
        
    SinvSolver(
        system,
        lmax // 4,
        gaussian_beam_by_l_by_eps(lmax // 2, eps),
        ridge=ridge),
        
    SinvSolver(
        system,
        lmax // 8,
        gaussian_beam_by_l_by_eps(lmax // 4, eps),
        ridge=ridge),
        
    SinvSolver(
        system,
        lmax // 16,
        gaussian_beam_by_l_by_eps(lmax // 8, eps),
        ridge=ridge),

    ][:2]


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
        if 1:
            r_h = b - level.matvec(x)

            restrict_lm = scatter_l_to_lm(next_level.rl / level.rl[:next_level.lmax + 1])

            # todo optimize and remove sht back/forth
            r_H = level.analysis(r_h)
            r_H = truncate_alm(r_H, level.lmax, next_level.lmax)
            r_H *= restrict_lm
            r_H = next_level.synthesis(r_H)

            c_H = vcycle(levels, ilevel + 1, r_H)

            c_h = next_level.adjoint_synthesis(c_H)
            c_h *= restrict_lm
            c_h = pad_alm(c_h, next_level.lmax, level.lmax)
            c_h = level.adjoint_analysis(c_h)

            x += c_h

        # post-smooth
        for i in range(1):
            r = b - level.matvec(x)
            x += level.error_smoother(r)

        return x

def precond(b):
    return vcycle(levels, 0, b)

    
## def f(u_in):
##     u_pad = np.zeros_like(mask)
##     u_pad[pick] = u_in
##     u = plan.adjoint_synthesis(u_pad)
##     u *= scatter_l_to_lm(dl)
##     u_pad = plan.synthesis(u)
##     return u_pad[pick] + ridge * u_in



## if 1:
##     #Q = hammer(f, int(pick.sum()))
##     #1/0
##     def precond(u):
##         return u / estimated_max #u / Q.diagonal()
## else:
##     def precond(u):
##         u_pad = np.zeros_like(mask_p)
##         u_pad[pick] = u
##         u = sharp.sh_analysis(system.lmax_list[0], u_pad)
##         u *= scatter_l_to_lm(1. / dl)
##         u_pad = sharp.sh_adjoint_analysis(nside, u)
##         return u_pad[pick]




if 0:
    from scipy.linalg import eigvals
    A = hammer(sinv_solver.matvec, sinv_solver.n)
    M = hammer(precond, sinv_solver.n)
    clf()
    semilogy(sorted(np.abs(eigvals(np.dot(M, A)))))
    draw()


else:

    rng = np.random.RandomState(1)
    x0 = rng.normal(size=levels[0].n)
    b = levels[0].matvec(x0)
    x = x0 * 0


    clf()

    ## solver = cg_generator(
    ##     levels[0].matvec,
    ##     b,
    ##     M=sinv_solver.error_smoother,
    ##     )

    #for i, (x, r, delta_new) in enumerate(solver):

    x0norm = levels[0].norm_by_l(x0)

    for i in range(20):
        r = b - levels[0].matvec(x)
        x += precond(r)

        errvec = x0 - x
        semilogy(levels[0].norm_by_l(errvec) / x0norm, label=int(i))
        print i

    draw()

