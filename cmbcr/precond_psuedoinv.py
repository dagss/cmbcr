import numpy as np
import healpy
from .utils import pad_or_truncate_alm, timed, pad_or_trunc
from .mmajor import scatter_l_to_lm
from .mblocks import gauss_ring_map_to_phase_map
from . import sharp
from cmbcr.healpix import nside_of
from .precond_psuedoinv_mod import compsep_apply_U_block_diagonal, compsep_assemble_U

def compute_Yh_D_Y_diagonal(lmax, phase_map, thetas):
    from commander.sphere import legendre
    from commander.compute.cr.mblocks import compute_real_Yh_D_Y_block
    result = np.zeros((lmax + 1)**2)
    idx = 0
    for m in range(lmax + 1):
        block = compute_real_Yh_D_Y_block(m, m, lmax, lmax, thetas, phase_map)
        result[idx:idx + block.shape[0]] = block.diagonal()
        idx += block.shape[0]
    return result


def pinv_block_diagonal(blocks):
    out = np.zeros(blocks.shape, dtype=blocks.dtype, order='F')
    for idx in range(blocks.shape[2]):
        out[:, :, idx] = np.linalg.pinv(blocks[:, :, idx]).T
    return out


def apply_block_diagonal_pinv(system, blocks, x):

    lmax = max(system.lmax_list)
    pad_x = np.zeros(((lmax + 1)**2, system.comp_count + system.band_count), order='F', dtype=np.float32)

    for i in range(system.band_count + system.comp_count):
        pad_x[:, i] = pad_or_truncate_alm(x[i], lmax)

    compsep_apply_U_block_diagonal(lmax, blocks, pad_x, transpose=True)

    result = []
    for k in range(system.comp_count):
        result.append(pad_or_truncate_alm(pad_x[:, k].astype(np.float64), system.lmax_list[k]))

    return result


def apply_block_diagonal_pinv_transpose(system, blocks, x):
    lmax = max(system.lmax_list)
    pad_x = np.zeros(((lmax + 1)**2, system.comp_count + system.band_count), order='F', dtype=np.float32)

    for k in range(system.comp_count):
        pad_x[:, k] = pad_or_truncate_alm(x[k], lmax)

    compsep_apply_U_block_diagonal(lmax, blocks, pad_x, transpose=False)

    result = []
    for i in range(system.band_count + system.comp_count):
        result.append(pad_x[:, i].astype(np.float64))
    return result

def create_mixing_matrix(system, lmax, alpha_lst):
    bl_arr = np.zeros((system.band_count, lmax + 1), order='F')
    sqrt_invCl_arr = np.zeros((system.comp_count, lmax + 1), order='F')

    for k in range(system.comp_count):
        sqrt_invCl_arr[k, :] = pad_or_trunc(np.sqrt(system.dl_list[k]), lmax + 1)
    for nu in range(system.band_count):
        bl_arr[nu, :] = pad_or_trunc(system.bl_list[nu], lmax + 1)

    U = compsep_assemble_U(
        lmax_per_comp=system.lmax_list,
        mixing_scalars=system.mixing_scalars.copy('F'), bl=bl_arr, sqrt_Cl=sqrt_invCl_arr,
        alpha=np.asarray(alpha_lst, order='F'))

    return U


def lstmul(a, b):
    return [ax * bx for ax, bx in zip(a, b)]

def lstadd(a, b):
    return [ax + bx for ax, bx in zip(a, b)]

def lstsub(a, b):
    return [ax - bx for ax, bx in zip(a, b)]


class PsuedoInversePreconditioner(object):
    def __init__(self, system):
        self.system = system

        lmax = max(system.lmax_list)

        self.alpha_lst = []
        for nu in range(system.band_count):
            # ninv_gauss has *W* included in it. To normalize it we want a map without the weights..
            p = sharp.RealMmajorGaussPlan(system.lmax_ninv, system.lmax_ninv)
            ninv_gauss_no_w = p.synthesis(system.winv_ninv_sh_lst[nu])
            # did some experimentation and indeed 1.0 is the optimal value below, just to verify the intuition;
            # when plotted this makes the diagonal of Y^T N^{-1} Y not center on 1, but I suppose that "power"
            # is in the rest of the matrix
            q = ninv_gauss_no_w[ninv_gauss_no_w > ninv_gauss_no_w.max() * 7e-4]
            alpha = np.sqrt((ninv_gauss_no_w**2).sum() / ninv_gauss_no_w.sum())
            self.alpha_lst.append(1 * alpha)
            
        self.U = create_mixing_matrix(system, lmax, self.alpha_lst)
        self.Uplus = pinv_block_diagonal(self.U)

        #self.Uplus = np.zeros(
        #    (self.system.comp_count + self.system.band_count, self.system.comp_count, lmax + 1),
        #    dtype=np.float32, order='F')


        #for l in range(lmax + 1):
        #    self.Uplus[:, :, l] = self.Pi[:, :, l].T
        #del self.Pi
        #del self.P

        lmax = max(system.lmax_list)
        self.lmax = lmax
        self.plan = sharp.RealMmajorGaussPlan(system.lmax_ninv, lmax)

        def make_inv_map(x):
            x = x.copy()
            eps = x.max() * 1e-5#7e-4
            m = (x < eps)
            x[m] = 0
            x[~m] = 1. / x[~m]
            return x

        if self.system.use_healpix:
            self.inv_inv_maps = [make_inv_map(x) for x in system.ninv_maps]
        else:
            self.inv_inv_maps = [make_inv_map(x) for x in system.ninv_gauss_lst]

    def apply(self, x_lst):
        x_lst = apply_block_diagonal_pinv_transpose(self.system, self.Uplus, x_lst)
        c_h = []
        for nu in range(self.system.band_count):
            u = x_lst[nu]
            u *= self.alpha_lst[nu]
            if self.system.use_healpix:
                n_map = self.inv_inv_maps[nu]
                u = sharp.sh_adjoint_analysis(nside_of(n_map), u)
                u *= n_map
                u = sharp.sh_analysis(self.lmax, u)
            else:
                u = self.plan.adjoint_analysis(u)
                u *= self.inv_inv_maps[nu]
                u = self.plan.analysis(u)
            u *= self.alpha_lst[nu]
            c_h.append(u)
        for k in range(self.system.comp_count):
            c_h.append(x_lst[self.system.band_count + k])
        return apply_block_diagonal_pinv(self.system, self.Uplus, c_h)


class PsuedoInverseWithMaskPreconditioner(object):
    def __init__(self, system):
        self.psuedo_inv = PsuedoInversePreconditioner(system)
        self.system = system

        if self.system.mask is not None:
            # Make appropriate masks for each component
            self.filter_lst = []
            for k in range(self.system.comp_count):
                lmax = self.system.lmax_list[k]
                # round up to nearest power of 2, then divide by 2, to get nside
                nside = 1
                while nside < lmax:
                    nside *= 2
                nside //= 2
                mask_ud = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='RING', power=-1)
                mask_ud[mask_ud != 0] = 1
                self.filter_lst.append(1 - mask_ud)

    def filter_vec(self, k, x):
        # Applies a mask filter for component k to vector x
        f = self.filter_lst[k]
        return sharp.sh_analysis(self.system.lmax_list[k], sharp.sh_synthesis(nside_of(f), x) * f)

    def solve_under_mask(self, r_h_lst):
        c_h_lst = []
        for k in range(self.system.comp_count):
            # restrict
            r_H = self.filter_vec(k, r_h_lst[k])

            # solve
            r_H *= scatter_l_to_lm(1 / self.system.dl_list[k][:self.system.lmax_list[k] + 1])

            # prolong
            c_h = self.filter_vec(k, r_H)
            c_h_lst.append(c_h)
        return c_h_lst

    def apply_Pt_2(self, x_lst):
        return lstsub(x_lst, self.solve_under_mask(self.system.matvec(x_lst, scalar_mixing=True)))

    def apply_P_2(self, x_lst):
        return lstsub(x_lst, self.system.matvec(self.solve_under_mask(x_lst), scalar_mixing=True))

    def apply_Pt_1(self, x_lst):
        return lstsub(x_lst, self.psuedo_inv.apply(self.system.matvec(x_lst, scalar_mixing=True)))

    def apply_P_1(self, x_lst):
        return lstsub(x_lst, self.system.matvec(self.psuedo_inv.apply(x_lst), scalar_mixing=True))

    def starting_vector(self, b_lst):
        # P_1-form
        return self.psuedo_inv.apply(b_lst)

    def apply(self, b_lst):
        if self.system.mask is None:
            return self.psuedo_inv.apply(b_lst)
        else:
            return self.apply_MG_V(b_lst)
            #return self.apply_schwarz(b_lst)
            #return self.apply_bnn(b_lst)

    def apply_schwarz(self, b_lst):
        x_lst = self.psuedo_inv.apply(b_lst)
        x2_lst = self.solve_under_mask(b_lst)
        return lstadd(x_lst, x2_lst)

    #def apply_bnn(self, b_lst):
    #    return lstadd(
    #        self.apply_Pt_2(self.psuedo_inv.apply(self.apply_P_2(b_lst))),
    #        self.solve_under_mask(b_lst))

    def apply_bnn(self, b_lst):
        return self.apply_Pt_1(self.solve_under_mask(self.apply_P_1(b_lst)))

        return lstadd(
            self.apply_Pt_1(self.solve_under_mask(self.apply_P_1(b_lst))),
            self.psuedo_inv.apply(b_lst))
        #return self.apply_Pt(self.psuedo_inv.apply(b_lst))

    def apply_MG_V(self, b_lst):

        x_lst = self.psuedo_inv.apply(b_lst)

        # r = b - A x
        r_lst = lstsub(b_lst, self.system.matvec(x_lst, scalar_mixing=True))
        c_lst = self.solve_under_mask(r_lst)
        # x = x + Mdata r
        x_lst = lstadd(x_lst, c_lst)


        # r = b - A x
        r_lst = lstsub(b_lst, self.system.matvec(x_lst, scalar_mixing=True))
        c_lst = self.psuedo_inv.apply(r_lst)
        # x = x + Mdata r
        x_lst = lstadd(x_lst, c_lst)



        return x_lst
