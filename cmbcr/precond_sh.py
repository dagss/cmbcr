import numpy as np

from .mblocks import gauss_ring_map_to_phase_map
from .harmonic_preconditioner import factor_banded_preconditioner
from .harmonic_preconditioner import solve_banded_preconditioner
from .harmonic_preconditioner import construct_banded_preconditioner
from .harmonic_preconditioner import k_kp_idx
from .utils import pad_or_trunc, timed
from .mmajor import lmax_of, scatter_l_to_lm

__all__ = ['BandedHarmonicPreconditioner']


def truncate_alm(alm, lmax_from, lmax_to):
    s = np.zeros(lmax_from + 1)
    s[:lmax_to + 1] = 1
    return alm[scatter_l_to_lm(s) == 1]


def pad_alm(alm, lmax_from, lmax_to, fillval=0):
    out = np.ones((lmax_to + 1)**2) * fillval
    s = np.zeros(lmax_to + 1)
    s[:lmax_from + 1] = 1
    out[scatter_l_to_lm(s) == 1] = alm
    return out

def pad_or_truncate_alm(alm, to_lmax, fillval=0):
    alm = alm.copy()
    from_lmax = lmax_of(alm)
    if to_lmax == from_lmax:
        return alm
    elif to_lmax < from_lmax:
        return truncate_alm(alm, from_lmax, to_lmax)
    else:
        return pad_alm(alm, from_lmax, to_lmax, fillval=fillval)


class BandedHarmonicPreconditioner(object):
    def __init__(self, system, diagonal=False):
        lmax = max(system.lmax_list)
        
        dl = np.zeros((lmax + 1, system.comp_count), order='F')
        for k in range(system.comp_count):
            dl[:, k] = pad_or_trunc(system.dl_list[k], lmax + 1)

        precond_data = np.zeros((5 * system.comp_count, system.comp_count * (lmax + 1)**2), dtype=np.float32, order='F')
        Ni_diag = 0

        for nu in range(system.band_count):

            ninv_phase_maps = np.zeros(
                (2 * lmax + 1, system.lmax_ninv + 1, (system.comp_count * (system.comp_count + 1)) // 2),
                order='F', dtype=np.complex128)

            ninv_phase, thetas = gauss_ring_map_to_phase_map(system.ninv_gauss_lst[nu], system.lmax_ninv, lmax)
            assert thetas.shape[0] == ninv_phase_maps.shape[1]

            for k in range(system.comp_count):
                for kp in range(k + 1):
                    ninv_phase_maps[:, :, k_kp_idx(k, kp)] = (
                        ninv_phase * system.mixing_scalars[nu, k] * system.mixing_scalars[nu, kp])

            with timed('construct_banded_preconditioner {}'.format(nu)):
                construct_banded_preconditioner(
                    lmax,
                    system.comp_count,
                    thetas,
                    ninv_phase_maps,
                    bl=system.bl_list[nu][:lmax + 1],
                    dl=dl,
                    out=precond_data)
            # We want to make sure we only add dl once, so zero it after the first time...should change this API
            # to pass it in to factor instead...
            dl *= 0

        if diagonal:
            precond_data[1:, :] = 0

        factor_banded_preconditioner(lmax, system.comp_count, precond_data)

        self.system = system
        self.data = precond_data
        self.lmax = lmax


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
