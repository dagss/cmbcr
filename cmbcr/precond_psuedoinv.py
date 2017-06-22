import numpy as np
from .utils import pad_or_truncate_alm
from .mmajor import scatter_l_to_lm
from . import sharp


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

def create_mixing_matrix(system):
    lmax = max(system.lmax_list)

    
    
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
                            result[nu, kp, idx] = system.mixing_scalars[nu, kp] * system.bl_list[nu][l]
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
        self.P = create_mixing_matrix(system)
        self.Pi = pinv_block_diagonal(self.P)
        lmax = max(system.lmax_list)
        self.plan = sharp.RealMmajorGaussPlan(system.lmax_ninv, lmax)

        def make_inv_map(x):
            x = x.copy()
            eps = x.max() * 7e-4
            m = (x < eps)
            x[m] = 0
            x[~m] = 1. / x[~m]
            return x

        self.inv_inv_maps = [make_inv_map(x) for x in system.ninv_gauss_lst]

    def apply(self, x_lst):
        x_lst = apply_block_diagonal_pinv_transpose(self.system, self.Pi, x_lst)
        c_h = []
        for nu in range(self.system.band_count):
            r_H = self.plan.adjoint_analysis(x_lst[nu])
            r_H *= self.inv_inv_maps[nu]
            c_h.append(self.plan.analysis(r_H))
        for k in range(self.system.comp_count):
            c_h.append(x_lst[self.system.band_count + k])
        return apply_block_diagonal_pinv(self.system, self.Pi, c_h)
