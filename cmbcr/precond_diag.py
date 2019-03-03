import numpy as np

from .mblocks import gauss_ring_map_to_phase_map
from .harmonic_preconditioner import factor_banded_preconditioner
from .harmonic_preconditioner import solve_banded_preconditioner
from .harmonic_preconditioner import construct_banded_preconditioner
from .harmonic_preconditioner import k_kp_idx
from .utils import pad_or_trunc, timed, pad_or_truncate_alm, scatter_l_to_lm
from .cache import memory
from .mblocks import compute_real_Yh_D_Y_block
from .healpix import nside_of
from . import sharp

__all__ = ['DiagonalPreconditioner']



from commander.compute.cr.sh_integrals import compute_approximate_Yt_D_Y_diagonal_mblock

def compute_Yh_D_Y_diagonal(lmax, phase_map, thetas):
    result = np.zeros((lmax + 1)**2)
    idx = 0
    for m in range(lmax + 1):
        block = compute_real_Yh_D_Y_block(m, m, lmax, lmax, thetas, phase_map)
        result[idx:idx + block.shape[0]] = block.diagonal()
        idx += block.shape[0]
    return result


#@memory.cache
def compute_diagonal_preconditioner(self):
    system = self.system
    lmax = max(system.lmax_list)
    
    precond_data = 0
    A_diag_lst = [0] * system.comp_count
    for nu in range(system.band_count):

        if 1:
            nside = nside_of(system.ninv_maps[nu])
            lmax = 3 * nside
            alm = sharp.sh_analysis(2 * lmax, system.ninv_maps[nu] * system.mask_dg_map[nside])
            Ni_diag = np.zeros((lmax + 1)**2, dtype=np.double)
            with timed('precond-diag-compute compute_Yh_D_Y_diagonal(drc3jj)'):
                compute_approximate_Yt_D_Y_diagonal_mblock(12*nside**2, 0, lmax, 0, lmax, alm, out=Ni_diag)
            Ni_diag *= scatter_l_to_lm(system.bl_list[nu][:lmax + 1])**2
        if 0:
            with timed('precond-diag-compute compute_Yh_D_Y_diagonal'):
                ninv_phase, thetas = gauss_ring_map_to_phase_map(system.ninv_gauss_lst[nu], system.lmax_ninv, lmax)
                Ni_diag_p = compute_Yh_D_Y_diagonal(lmax, ninv_phase, thetas) * scatter_l_to_lm(system.bl_list[nu][:lmax + 1])**2
            
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
