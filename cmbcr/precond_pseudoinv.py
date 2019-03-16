import numpy as np
import healpy
from .utils import pad_or_truncate_alm, timed, pad_or_trunc
from .mmajor import scatter_l_to_lm
from .mblocks import gauss_ring_map_to_phase_map
from . import sharp, beams
from cmbcr.healpix import nside_of
from .precond_pseudoinv_mod import compsep_apply_U_block_diagonal, compsep_assemble_U
from .beams import fourth_order_beam
from .block_matrix import block_diagonal_factor, block_diagonal_solve
from . import sht


def pinv_block_diagonal(blocks):
    out = np.zeros(blocks.shape, dtype=blocks.dtype, order='F')
    for idx in range(blocks.shape[2]):
        out[:, :, idx] = np.linalg.pinv(blocks[:, :, idx]).T
    return out


def apply_block_diagonal_pinv(system, blocks, x):

    lmax = max(system.lmax_list)
    pad_x = np.zeros(((lmax + 1)**2, system.comp_count + system.band_count), order='F', dtype=np.float64)

    for i in range(system.band_count + system.comp_count):
        pad_x[:, i] = pad_or_truncate_alm(x[i], lmax)
 
    compsep_apply_U_block_diagonal(lmax, blocks, pad_x, transpose=True)

    result = []
    for k in range(system.comp_count):
        result.append(pad_or_truncate_alm(pad_x[:, k].astype(np.float64), system.lmax_list[k]))

    return result


def apply_block_diagonal_pinv_transpose(system, blocks, x):
    lmax = max(system.lmax_list)
    pad_x = np.zeros(((lmax + 1)**2, system.comp_count + system.band_count), order='F', dtype=np.float64)

    for k in range(system.comp_count):
        pad_x[:, k] = pad_or_truncate_alm(x[k], lmax)

    compsep_apply_U_block_diagonal(lmax, blocks, pad_x, transpose=False)

    result = []
    for i in range(system.band_count + system.comp_count):
        result.append(pad_x[:, i].astype(np.float64))
    return result

def create_mixing_matrix(system, lmax, alpha_lst):
    bl_arr = np.zeros((system.band_count, lmax + 1), order='F')
    wl_arr = np.zeros((system.comp_count, lmax + 1), order='F')
    dl_arr = np.zeros((system.comp_count, lmax + 1), order='F')

    for k in range(system.comp_count):
        wl_arr[k, :] = pad_or_trunc(system.wl_list[k], lmax + 1)
        dl_arr[k, :] = pad_or_trunc(np.sqrt(system.dl_list[k]) * system.wl_list[k], lmax + 1)
        
    for nu in range(system.band_count):
        bl_arr[nu, :] = pad_or_trunc(system.bl_list[nu], lmax + 1)

    U = compsep_assemble_U(
        lmax_per_comp=system.lmax_list,
        mixing_scalars=system.mixing_scalars.copy('F'),
        bl=bl_arr,
        dl=dl_arr,
        wl=wl_arr,
        alpha=np.asarray(alpha_lst, order='F'))

    return U


def lstscale(a, b):
    return [a * bx for bx in b]

def lstmul(a, b):
    return [ax * bx for ax, bx in zip(a, b)]

def lstadd(a, b):
    return [ax + bx for ax, bx in zip(a, b)]

def lstsub(a, b):
    return [ax - bx for ax, bx in zip(a, b)]



class PseudoInversePreconditioner(object):

    def __init__(self, system):
        self.system = system

        lmax = max(system.lmax_list)

        lmax = max(system.lmax_list)
        self.lmax = lmax
        self.plan = sharp.RealMmajorGaussPlan(system.lmax_ninv, lmax)

        self.alpha_lst = []
        for nu in range(system.band_count):
            # ninv_gauss has *W* included in it. To normalize it we want a map without the weights..
            ##p = sharp.RealMmajorGaussPlan(system.lmax_ninv, system.lmax_ninv)
            ##ninv_gauss_no_w = p.synthesis(system.winv_ninv_sh_lst[nu])
            # did some experimentation and indeed 1.0 is the optimal value below, just to verify the intuition;
            # when plotted this makes the diagonal of Y^T N^{-1} Y not center on 1, but I suppose that "power"
            # is in the rest of the matrix
            ##q = ninv_gauss_no_w[ninv_gauss_no_w > ninv_gauss_no_w.max() * 7e-4]
            
            ##alpha = np.sqrt((ninv_gauss_no_w**2).sum() / ninv_gauss_no_w.sum())

            tau = system.ninv_maps[nu] / (4 * np.pi / system.ninv_maps[nu].shape[0])
            alpha = np.sqrt((tau**2).sum() / tau.sum())
            
            self.alpha_lst.append(1 * alpha)

        self.U = create_mixing_matrix(system, lmax, self.alpha_lst)
        self.Uplus = pinv_block_diagonal(self.U)

        def make_inv_map(x):
            import healpy
            mask = system.mask_dg_map[nside_of(x)]
            
            m = 1. / x
            #m *= mask

            # blue: remove inside mask
            # orange: full-sky
            
            
            m[mask == 0] *= 10
            
            return m

        if self.system.use_healpix:
            self.inv_inv_maps = [make_inv_map(x) for x in system.ninv_maps]
        else:
            self.inv_inv_maps = [make_inv_map(x) for x in system.ninv_gauss_lst]

    def apply(self, x_lst):
        #x_lst = lstscale(1/10., x_lst)
        x_lst = apply_block_diagonal_pinv_transpose(self.system, self.Uplus, x_lst)
        c_h = (
            [self.inverse_noise_map(nu, x_lst[nu]) for nu in range(self.system.band_count)]
            + x_lst[self.system.band_count:]
            )
        x_lst = apply_block_diagonal_pinv(self.system, self.Uplus, c_h)
        return x_lst
            
    def inverse_noise_map(self, nu, u):
        u *= self.alpha_lst[nu]
        if self.system.use_healpix:
            n_map = self.inv_inv_maps[nu]
            u = sht.sh_adjoint_analysis(nside_of(n_map), u)
            u *= n_map
            u = sht.sh_analysis(self.lmax, u)
        else:
            u = self.plan.adjoint_analysis(u)
            u *= self.inv_inv_maps[nu]
            u = self.plan.analysis(u)
        u *= self.alpha_lst[nu]
        return u

    
class DiagonalPreconditioner2(object):
    def __init__(self, system):
        from .precond_diag import compute_Yh_D_Y_diagonal
        from .mblocks import gauss_ring_map_to_phase_map
        from commander.compute.cr.sh_integrals import compute_approximate_Yt_D_Y_diagonal_mblock, compute_approximate_Yt_D_Y_diagonal

        self.system = system

        lmax = max(self.system.lmax_list)

        U = create_mixing_matrix(system, lmax, [1.] * system.band_count)
        
        Ni_diag_lst = []
        for nu in range(self.system.band_count):

            if 1:
                nside = nside_of(system.ninv_maps[nu])
                lmax_drc3jj = 3 * nside
                with timed('map2alm'):
                    alm = sharp.sh_analysis(2 * lmax_drc3jj, system.ninv_maps[nu] * system.mask_dg_map[nside])
                Ni_diag = np.zeros((lmax_drc3jj + 1)**2, dtype=np.double)
                with timed('compute_Yh_D_Y_diagonal(drc3jj) for {}'.format(nu)):
                    Ni_diag = compute_approximate_Yt_D_Y_diagonal(12*nside**2, 0, lmax_drc3jj, alm, out=Ni_diag)
                    #compute_approximate_Yt_D_Y_diagonal_mblock(12*nside**2, 0, lmax_drc3jj, 0, lmax_drc3jj, alm, out=Ni_diag)
            else:
                with timed('compute_Yh_D_Y_diagonal for {}'.format(nu)):
                    ninv_phase, thetas = gauss_ring_map_to_phase_map(system.ninv_gauss_lst[nu], system.lmax_ninv, lmax)
                    Ni_diag = compute_Yh_D_Y_diagonal(lmax, ninv_phase, thetas)
            Ni_diag_lst.append(Ni_diag)

        blocks = np.zeros((self.system.comp_count, self.system.comp_count, (lmax + 1)**2), order='F')
        idx = 0
        for m in range(lmax + 1):
            print 'Precond construction, m=', m
            for l in range(m, lmax + 1):
                for neg in range(2):
                    if m == 0 and neg == 1:
                        continue

                    U_block = U[:, :, l].copy()
                    for nu in range(self.system.band_count):
                        U_block[nu, :] *= np.sqrt(Ni_diag_lst[nu][idx])

                    blocks[:, :, idx] = np.dot(U_block.T, U_block)
                    for k in range(self.system.comp_count):
                        # if l is larger than lmax_list[k], then the corresponding rows/columns
                        # in U_block will be zero. In this case just insert 1 so that the system can
                        # be inverted. The resulting coefficients in the inverted blocks will not be
                        # used anyway (due to padding/truncation)
                        if blocks[k, k, idx] == 0:
                            blocks[k, k, idx] = 1
                    idx += 1
        assert idx == (lmax + 1)**2
        block_diagonal_factor(blocks)
        self.blocks = blocks
        self.lmax = lmax

        

    def apply(self, x_lst):
        comp_count = self.system.comp_count

        buf = np.empty((comp_count, (self.lmax + 1)**2), order='F')
        for k in range(comp_count):
            buf[k, :] = pad_or_truncate_alm(x_lst[k], self.lmax)

        block_diagonal_solve(self.blocks, buf)
        
        result = [None] * comp_count
        for k in range(comp_count):
            result[k] = pad_or_truncate_alm(buf[k, :], self.system.lmax_list[k])

        return result
        
        
        
    

class PseudoInverseWithMaskPreconditioner(object):
    def __init__(self, system, flatsky=False, inner_its=5):
        self.pseudo_inv = PseudoInversePreconditioner(system)
        self.system = system

        self.rl_list = [
            fourth_order_beam(system.lmax_list[k], system.lmax_list[k] // 2, 0.05)
            for k in range(system.comp_count)
        ]
        self.inner_its = inner_its

        if self.system.mask is not None:
            if flatsky:
                from .masked_solver_fft import SinvSolver
                mask = system.mask_gauss_grid
            else:
                from .masked_solver import SinvSolver
                mask = system.mask_dg
            self.sinv_solvers = [
                SinvSolver(system.dl_list[k] * self.rl_list[k]**2, mask)
                for k in range(self.system.comp_count)
                ]

    def solve_component_under_mask(self, k, x):
        sinv_solver = self.sinv_solvers[k]
        x_pix = sinv_solver.restrict(x * scatter_l_to_lm(self.rl_list[k]))
        if self.inner_its == 0:
            x_pix = sinv_solver.precond(x_pix)
        else:
            x_pix, _, _ = sinv_solver.solve_mask(x_pix, rtol=1e-2, maxit=self.inner_its)
        x = sinv_solver.prolong(x_pix) * scatter_l_to_lm(self.rl_list[k])
        return x

    def apply(self, b_lst):
        x = self.pseudo_inv.apply(b_lst)
        if self.system.mask is not None:
            x_under_mask = [
                self.solve_component_under_mask(k, b_lst[k])
                for k in range(self.system.comp_count)
            ]
            x = lstadd(x, x_under_mask)
        return x
