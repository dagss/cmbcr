import numpy as np
import healpy
from .utils import pad_or_truncate_alm, timed, pad_or_trunc
from .mmajor import scatter_l_to_lm
from .mblocks import gauss_ring_map_to_phase_map
from . import sharp, beams
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
        wl_arr[k, :] = pad_or_trunc(np.sqrt(system.wl_list[k]), lmax + 1)
        dl_arr[k, :] = pad_or_trunc(np.sqrt(system.dl_list[k]), lmax + 1)
        
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
            eps = x.max() * 1e-3
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
    def __init__(self, system, method='add'):
        self.psuedo_inv = PsuedoInversePreconditioner(system)
        self.system = system
        self.method = method

        if self.system.mask is not None:
            # Make appropriate masks for each component
            self.filter_lst = []
            for k in range(self.system.comp_count):
                lmax = self.system.lmax_list[k]
                # round up to nearest power of 2, then divide by 2, to get nside
                nside = 1
                while nside < lmax:
                    nside *= 2
                #nside //= 2
                from healpy import mollzoom
                mask_ud = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='RING', power=0)
                # Tested out a few options here, but seems like having a mask with 0.5, 0.25, etc on the border
                # was the best choice
                self.filter_lst.append(1 - mask_ud)

    def filter_vec(self, k, x, neg=False):
        # Applies a mask filter for component k to vector x
        f = self.filter_lst[k]
        if neg:
            f = 1 - f
        return sharp.sh_analysis(self.system.lmax_list[k], sharp.sh_synthesis(nside_of(f), x) * f)

    def filter(self, x_lst, neg=False):
        return [self.filter_vec(k, x_lst[k], neg) for k in range(self.system.comp_count)]
    
    def solve_under_mask(self, r_h_lst):
        c_h_lst = []
        for k in range(self.system.comp_count):
            r_H = r_h_lst[k]

            # solve
            r_H *= (1. / scatter_l_to_lm(self.system.dl_list[k]))

            # prolong
            c_h = r_H
            c_h_lst.append(c_h)
        return c_h_lst

    def M1(self, u_lst):
        return self.psuedo_inv.apply(u_lst)
        return self.filter(self.psuedo_inv.apply(self.filter(u_lst, neg=True)), neg=True)

    def M2(self, u_lst):
        return self.solve_under_mask(u_lst)

    def apply_Q(self, u_lst):
        return self.M1(u_lst)
        #return self.M2(u_lst)

    def apply_Minv(self, u_lst):
        return self.M2(u_lst)
        #return self.M1(u_lst)

    #def v_end(self, b_lst, x_lst):
    #    return lstadd(self.apply_Q(b_lst), self.apply_Pt(x_lst))

    def apply_Pt(self, x_lst):
        return lstsub(x_lst, self.apply_Q(self.system.matvec(x_lst)))

    def apply_P(self, x_lst):
        return lstsub(x_lst, self.system.matvec(self.apply_Q(x_lst)))

    
    #def starting_vector(self, b_lst):
    #    #return [0 * u for u in b_lst]
        #return self.apply_Q(b_lst)

    #def apply_CG_M2(self, b_lst):
    #    return self.apply_Pt(b_lst)

    def apply(self, b_lst):
        if self.system.mask is None:
            return self.psuedo_inv.apply(b_lst)
        else:
            m = getattr(self, 'apply_{}'.format(self.method))
            return m(b_lst)

    def apply_add1(self, b_lst):
        return lstadd(
            self.psuedo_inv.apply(b_lst),
            lstscale(1,
                self.filter(self.solve_under_mask(self.filter(b_lst)))
                         ))

    def apply_add2(self, b_lst):
        return lstadd(
            self.filter(self.psuedo_inv.apply(self.filter(b_lst, neg=True)), neg=True),
            self.filter(self.solve_under_mask(self.filter(b_lst))))

    def apply_add3(self, b_lst):
        # diverges
        return lstadd(
            self.psuedo_inv.apply(b_lst),
            self.solve_under_mask(b_lst))
    
    def apply_add4(self, b_lst):
        # diverges
        return lstadd(
            self.filter(self.psuedo_inv.apply(self.filter(b_lst, neg=True)), neg=True),
            self.solve_under_mask(b_lst))
    
    def apply_v1(self, b_lst):

        def A_approx(u_lst):
            #u_lst = self.filter(u_lst, neg=True)
            u_lst = self.system.matvec(u_lst)
            #u_lst = self.filter(u_lst, neg=True)
            return u_lst

        # x = M_outer b
        x_lst = self.filter(self.psuedo_inv.apply(self.filter(b_lst, neg=True)), neg=True)

        # x = x + M_inner (b - A x)
        r_lst = lstsub(b_lst, A_approx(x_lst))
        c_lst = self.filter(self.solve_under_mask(self.filter(r_lst, neg=False)), neg=False)
        x_lst = lstadd(x_lst, c_lst)
        # x = x + M_outer (b - A x)
        r_lst = lstsub(b_lst, A_approx(x_lst))
        c_lst = self.filter(self.psuedo_inv.apply(self.filter(r_lst, neg=True)), neg=True)
        x_lst = lstadd(x_lst, c_lst)
        return x_lst
        
    def apply_v2(self, b_lst):

        def A_approx(u_lst):
            #u_lst = self.filter(u_lst, neg=True)
            u_lst = self.system.matvec(u_lst)
            #u_lst = self.filter(u_lst, neg=True)
            return u_lst

        # x = M_outer b
        x_lst = self.psuedo_inv.apply(b_lst)

        # x = x + M_inner (b - A x)
        r_lst = lstsub(b_lst, A_approx(x_lst))
        #c_lst = self.solve_under_mask(r_lst)
        c_lst = self.filter(self.solve_under_mask(self.filter(r_lst, neg=False)), neg=False)
        x_lst = lstadd(x_lst, c_lst)
        # x = x + M_outer (b - A x)
        r_lst = lstsub(b_lst, A_approx(x_lst))
        c_lst = self.psuedo_inv.apply(r_lst)
        x_lst = lstadd(x_lst, c_lst)
        return x_lst

    def apply_bnn(self, b_lst):
        return lstadd(
            self.apply_Pt(self.apply_Minv(self.apply_P(b_lst))),
            self.apply_Q(b_lst))
    

        

class SwitchPreconditioner(object):
    def __init__(self, first, second, n):
        self.first = first
        self.second = second
        self.n = n
        self.i = 0

    def apply(self, x):
        if self.i <= self.n:
            r = self.first.apply(x)
        else:
            r = self.second.apply(x)
        self.i += 1
        return r
            

class MGPreconditioner(object):
    def __init__(self, system):
        from .cr_system import restrict_system
        self.system_h = system
        self.system_H = restrict_system(system)

        self.precond_h = PsuedoInverseWithMaskPreconditioner(self.system_h, 'add1')
        self.precond_H = PsuedoInverseWithMaskPreconditioner(self.system_H, 'add1')

    def apply(self, b_lst):
        
        def restrict(u_lst):
            v_lst = []
            for k in range(self.system_h.comp_count):
                u = pad_or_truncate_alm(u_lst[k], self.system_H.lmax_list[k])
                u *= scatter_l_to_lm(self.system_H.rl_list[k])
                v_lst.append(u)
            return v_lst

        def prolong(u_lst):
            v_lst = []
            for k in range(self.system_h.comp_count):
                u = u_lst[k] * scatter_l_to_lm(self.system_H.rl_list[k])
                u = pad_or_truncate_alm(u, self.system_h.lmax_list[k])
                v_lst.append(u)
            return v_lst

        x_lst = lstscale(0, b_lst)

        for i in range(10):
            r_h_lst = lstsub(b_lst, self.system_h.matvec(x_lst))
            c_h_lst = self.precond_h.apply(r_h_lst)
            x_lst = lstadd(x_lst, lstscale(0.01, c_h_lst))
        
        #x_lst = self.precond_h.apply(b_lst)
        #r_h_lst = lstsub(b_lst, self.system_h.matvec(x_lst))
        #r_H_lst = restrict(r_h_lst)
        #c_H_lst = self.precond_H.apply(r_H_lst)
        #c_h_lst = prolong(c_H_lst)
        #x_lst = lstadd(x_lst, lstscale(0.01, c_h_lst))

        for i in range(10):
            r_h_lst = lstsub(b_lst, self.system_h.matvec(x_lst))
            c_h_lst = self.precond_h.apply(r_h_lst)
            x_lst = lstadd(x_lst, lstscale(0.01, c_h_lst))
        
        return x_lst
        
