import numpy as np

from .mblocks import gauss_ring_map_to_phase_map
from .harmonic_preconditioner import factor_banded_preconditioner
from .harmonic_preconditioner import solve_banded_preconditioner
from .harmonic_preconditioner import construct_banded_preconditioner
from .harmonic_preconditioner import k_kp_idx
from .utils import pad_or_trunc, timed, pad_or_truncate_alm, scatter_l_to_lm
from .cache import memory
from .mblocks import compute_real_Yh_D_Y_block

__all__ = ['DiagonalPreconditioner']


def compute_Yh_D_Y_diagonal(lmax, phase_map, thetas):
    result = np.zeros((lmax + 1)**2)
    idx = 0
    for m in range(lmax + 1):
        block = compute_real_Yh_D_Y_block(m, m, lmax, lmax, thetas, phase_map)
        result[idx:idx + block.shape[0]] = block.diagonal()
        idx += block.shape[0]
    return result


@memory.cache
def compute_diagonal_preconditioner(self):
    system = self.system
    lmax = max(system.lmax_list)
    
    precond_data = 0
    A_diag_lst = [0] * system.comp_count
    for nu in range(system.band_count):
        ninv_phase, thetas = gauss_ring_map_to_phase_map(system.ninv_gauss_lst[nu], system.lmax_ninv, lmax)
        Ni_diag = compute_Yh_D_Y_diagonal(lmax, ninv_phase, thetas) * scatter_l_to_lm(system.bl_list[nu][:lmax + 1])**2
        for k in range(system.comp_count):
            A_diag_lst[k] += (
                pad_or_truncate_alm(Ni_diag, system.lmax_list[k])
                * system.mixing_scalars[nu, k]**2
                * scatter_l_to_lm(system.wl_list[k])
                )

    for k, x in enumerate(A_diag_lst):
        x += scatter_l_to_lm(system.dl_list[k])
            
    return [1. / x for x in A_diag_lst]


class DiagonalPreconditioner(object):
    def __init__(self, system, diagonal=False, couplings=True, factor=True):
        self.system = system
        self.lmax = max(system.lmax_list)
        self.M_lst = compute_diagonal_preconditioner(self)


    def apply(self, x_lst):
        return [M * x for M, x in zip(self.M_lst, x_lst)]
