import numpy as np
from .utils import pad_or_truncate_alm
from .mmajor import scatter_l_to_lm
from .mblocks import gauss_ring_map_to_phase_map
from . import sharp
from cmbcr.healpix import nside_of


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
    out = np.zeros((blocks.shape[1], blocks.shape[0], blocks.shape[2]))
    for idx in range(blocks.shape[2]):
        out[:, :, idx] = np.linalg.pinv(blocks[:, :, idx])
    return out


def apply_block_diagonal_pinv(system, blocks, x):
    lmax = max(system.lmax_list)
    pad_x = np.zeros((system.band_count + system.comp_count, (lmax + 1)**2))
    pad_y = np.zeros((system.comp_count, (lmax + 1)**2))

    for i in range(system.band_count + system.comp_count):
        pad_x[i, :] = x[i]

    for idx in range(blocks.shape[2]):
        pad_y[:, idx] = np.dot(blocks[:, :, idx], pad_x[:, idx])

    result = []
    for k in range(system.comp_count):
        result.append(pad_or_truncate_alm(pad_y[k], system.lmax_list[k]))

    return result


def apply_block_diagonal_pinv_transpose(system, blocks, x):
    lmax = max(system.lmax_list)
    pad_x = np.zeros((system.comp_count, (lmax + 1)**2))
    pad_y = np.zeros((system.band_count + system.comp_count, (lmax + 1)**2))

    for k in range(system.comp_count):
        pad_x[k, :] = pad_or_truncate_alm(x[k], lmax)

    for idx in range(blocks.shape[2]):
        pad_y[:, idx] = np.dot(blocks[:, :, idx].T, pad_x[:, idx])

    result = []
    for i in range(system.band_count + system.comp_count):
        result.append(pad_y[i, :])
    return result

def create_mixing_matrix(system, lmax, D_lm_list):
    
    result = np.zeros((system.band_count + system.comp_count, system.comp_count, (lmax + 1)**2), order='F')
    idx = 0
    for m in range(lmax + 1):
        for l in range(m, lmax + 1):
            for neg in [0, 1]:
                if m == 0 and neg == 1:
                    continue

                for nu in range(system.band_count):
                    for kp in range(system.comp_count):
                        if l <= system.lmax_list[kp]:
                            result[nu, kp, idx] = system.mixing_scalars[nu, kp] * system.bl_list[nu][l] * D_lm_list[nu][idx]
                        else:
                            result[nu, kp, idx] = 0
                for k in range(system.comp_count):
                    for kp in range(system.comp_count):
                        if k == kp and l <= system.lmax_list[kp]:
                            result[system.band_count + k, kp, idx] = np.sqrt(system.dl_list[k][l])
                        else:
                            result[system.band_count + k, kp, idx] = 0
                idx += 1
    return result


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

        self.D_lm_lst = []
        for nu in range(system.band_count):
            ##ninv_phase, thetas = gauss_ring_map_to_phase_map(system.ninv_gauss_lst[nu], system.lmax_ninv, lmax)
            ##D_lm = np.sqrt(compute_Yh_D_Y_diagonal(lmax, ninv_phase, thetas))


            # Rescale to make \Y^T N^{-1} Y as close as possible to identity matrix, for the psuedo-inverse precond

            # ninv_gauss has *W* included in it. To normalize it we want a map without the weights..
            p = sharp.RealMmajorGaussPlan(system.lmax_ninv, system.lmax_ninv)
            ninv_gauss_no_w = p.synthesis(system.winv_ninv_sh_lst[nu])
            # did some experimentation and indeed 1.0 is the optimal value below, just to verify the intuition;
            # when plotted this makes the diagonal of Y^T N^{-1} Y not center on 1, but I suppose that "power"
            # is in the rest of the matrix
            q = ninv_gauss_no_w[ninv_gauss_no_w > ninv_gauss_no_w.max() * 7e-4]
            alpha = 1.0 * ninv_gauss_no_w.sum() / (ninv_gauss_no_w**2).sum()

            D_lm = np.ones((lmax + 1)**2) * (1. / np.sqrt(alpha))
             # rescale happens in cr_system right now
            #/ np.sqrt(system.ninv_alphas[nu])
            self.D_lm_lst.append(D_lm)

        self.P = create_mixing_matrix(system, lmax, self.D_lm_lst)
        self.Pi = pinv_block_diagonal(self.P)
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
        x_lst = apply_block_diagonal_pinv_transpose(self.system, self.Pi, x_lst)
        c_h = []
        for nu in range(self.system.band_count):
            u = x_lst[nu]
            u *= self.D_lm_lst[nu]
            if self.system.use_healpix:
                n_map = self.inv_inv_maps[nu]
                u = sharp.sh_adjoint_analysis(nside_of(n_map), u)
                u *= n_map
                u = sharp.sh_analysis(self.lmax, u)
            else:
                u = self.plan.adjoint_analysis(u)
                u *= self.inv_inv_maps[nu]
                u = self.plan.analysis(u)
            u *= self.D_lm_lst[nu]
            c_h.append(u)
        for k in range(self.system.comp_count):
            c_h.append(x_lst[self.system.band_count + k])
        return apply_block_diagonal_pinv(self.system, self.Pi, c_h)
