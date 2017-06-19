import numpy as np
from .utils import pad_or_truncate_alm
from .mmajor import scatter_l_to_lm
from . import sharp


def invert_block_diagonal(blocks):
    out = np.zeros_like(blocks)
    n = blocks.shape[1]
    for idx in range(blocks.shape[2]):
        out[:, :, idx] = np.linalg.solve(blocks[:, :, idx], np.eye(n))
    return out


def create_mixing_matrix(system):
    lmax = max(system.lmax_list)
    result = np.zeros((system.band_count, system.comp_count, (lmax + 1)**2), order='F')
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
                        elif nu == kp:
                            result[nu, kp, idx] = 1
                        else:
                            result[nu, kp, idx] = 0
                idx += 1
    return result


def apply_block_diagonal(system, blocks, x_lst, transpose=False):
    lmax = max(system.lmax_list)
    pad_x = np.zeros((system.comp_count, (lmax + 1)**2))
    for k in range(system.comp_count):
        pad_x[k, :] = pad_or_truncate_alm(x_lst[k], lmax)

    for idx in range(blocks.shape[2]):
        block = blocks[:, :, idx]
        if transpose:
            block = block.T
        pad_x[:, idx] = np.dot(block, pad_x[:, idx])

    result = []
    for k in range(system.comp_count):
        result.append(pad_or_truncate_alm(pad_x[k, :], system.lmax_list[k]))
    return result


def lstmul(a, b):
    return [ax * bx for ax, bx in zip(a, b)]

def lstadd(a, b):
    return [ax + bx for ax, bx in zip(a, b)]

def lstsub(a, b):
    return [ax - bx for ax, bx in zip(a, b)]


class PsuedoInversePreconditioner(object):
    def __init__(self, system, hi_l_precond, mg=False):
        self.system = system
        self.P = create_mixing_matrix(system)
        self.Pi = invert_block_diagonal(self.P)
        self.plans = [
            sharp.RealMmajorGaussPlan(system.lmax_ninv, system.lmax_list[k])
            for k in range(system.comp_count)]
        assert self.system.band_count == self.system.comp_count

        def make_inv_map(x):
            x = x.copy()
            x[x < 0] = 0
            x[x != 0] = 1. / x[x != 0]
            return x
        
        self.inv_inv_maps = [make_inv_map(x) for x in system.ninv_gauss_lst]
        self.hi_l_precond = hi_l_precond


        def make_filter(k):
            lfilter = np.ones(system.lmax_list[k] + 1)
            lfilter[system.prior_list[k].lcross:] = 0
            return scatter_l_to_lm(lfilter)

        self.lo_filters = [make_filter(k) for k in range(system.comp_count)]
        self.hi_filters = [1 - f for f in self.lo_filters]
        self.mg = mg

    def apply(self, b_lst):
        if self.mg:
            x_lst = self.hi_l_precond.apply(b_lst)
            
            r_lst = lstsub(b_lst, self.system.matvec(x_lst))
            x_lst = lstadd(x_lst, self.apply_psuedo_inverse(r_lst))
            
            r_lst = lstsub(b_lst, self.system.matvec(x_lst))
            x_lst = lstadd(x_lst, self.hi_l_precond.apply(r_lst))
            
            return x_lst
        else:
            M_hi_b = lstmul(self.hi_filters, self.hi_l_precond.apply(lstmul(self.hi_filters, b_lst)))
            M_lo_b = lstmul(self.lo_filters, self.apply_psuedo_inverse(lstmul(self.lo_filters, b_lst)))
            return lstadd(M_hi_b, M_lo_b)

    def apply_psuedo_inverse(self, x_lst):
        x_lst = apply_block_diagonal(self.system, self.Pi, x_lst, transpose=True)
        c_h = []
        for nu in range(self.system.band_count):
            r_H = self.plans[nu].adjoint_analysis(x_lst[nu])
            r_H *= self.inv_inv_maps[nu]
            c_h.append(self.plans[nu].analysis(r_H))
        return apply_block_diagonal(self.system, self.Pi, c_h)
