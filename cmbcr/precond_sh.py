import numpy as np

from .mblocks import gauss_ring_map_to_phase_map
from .harmonic_preconditioner import factor_banded_preconditioner
from .harmonic_preconditioner import solve_banded_preconditioner
from .harmonic_preconditioner import construct_banded_preconditioner
from .harmonic_preconditioner import k_kp_idx
from .utils import pad_or_trunc, timed, pad_or_truncate_alm, scatter_l_to_lm
from .cache import memory

__all__ = ['BandedHarmonicPreconditioner']

@memory.cache
def compute_banded_preconditioner(self, couplings, diagonal, factor):
    system = self.system
    lmax = max(system.lmax_list)

    precond_data = np.zeros((5 * system.comp_count, system.comp_count * (lmax + 1)**2), dtype=np.float32, order='F')
    Ni_diag = 0

    dl = np.zeros((lmax + 1, system.comp_count), order='F')
    for k in range(system.comp_count):
        dl[:, k] += pad_or_trunc(1. / np.sqrt(system.dl_list[k]), lmax + 1)

    for nu in range(system.band_count):

        ninv_phase_maps = np.zeros(
            (2 * lmax + 1, system.lmax_ninv + 1, (system.comp_count * (system.comp_count + 1)) // 2),
            order='F', dtype=np.complex128)

        ninv_phase, thetas = gauss_ring_map_to_phase_map(system.ninv_gauss_lst[nu], system.lmax_ninv, lmax)
        for k in range(system.comp_count):
            for kp in range(k + 1):
                if couplings or k == kp:
                    ninv_phase_maps[:, :, k_kp_idx(k, kp)] = (
                        ninv_phase * system.mixing_scalars[nu, k] * system.mixing_scalars[nu, kp])

        with timed('construct_banded_preconditioner {}'.format(nu)):
            construct_banded_preconditioner(
                lmax=lmax,
                ncomp=system.comp_count,
                thetas=thetas,
                bl=system.bl_list[nu][:lmax + 1],
                dl=dl,
                phase_map=ninv_phase_maps,
                #mixing_scalars=system.mixing_scalars[nu, :].copy(),
                out=precond_data)
        # Only add dl in first iteration
        #dl *= 0

    # prior
    precond_data[0, :] += 1
            
    if diagonal:
        precond_data[1:, :] = 0

    if factor:
        factor_banded_preconditioner(lmax, system.comp_count, precond_data)

    return precond_data


class BandedHarmonicPreconditioner(object):
    def __init__(self, system, diagonal=False, couplings=True, factor=True):
        self.system = system
        self.lmax = max(system.lmax_list)
        self.data = compute_banded_preconditioner(self, couplings, diagonal, factor)


    def apply(self, x_lst):
        comp_count = self.system.comp_count

        buf = np.empty(((self.lmax + 1)**2, comp_count), order='F', dtype=np.float32)
        for k in range(comp_count):
            buf[:, k] = pad_or_truncate_alm(x_lst[k], self.lmax)

        buf = solve_banded_preconditioner(self.lmax, comp_count, self.data, buf)

        result = [None] * comp_count
        for k in range(comp_count):
            result[k] = pad_or_truncate_alm(buf[:, k], self.system.lmax_list[k]).astype(np.double)

        return result
