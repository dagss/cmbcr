import numpy as np
import os
import logging

__all__ = ['CrSystem', 'downgrade_system']


from .data_utils import load_map, load_beam
from .cache import cached, memory
from .rotate_alm import rotate_alm
from .mmajor import scatter_l_to_lm
from . import sharp
from .utils import timed

load_map_cached = cached(lambda filename: load_map('raw', filename))
load_beam_cached = cached(load_beam)

@memory.cache
def rotate_ninv(lmax_ninv, ninv_map, rot_ang):
    winv_ninv_sh = sharp.sh_adjoint_synthesis(lmax_ninv, ninv_map)
    if rot_ang != (0, 0, 0):
        rotate_alm(lmax_ninv, winv_ninv_sh, *rot_ang)
    plan_dg = sharp.RealMmajorGaussPlan(lmax_ninv, lmax_ninv)
    ninv_gauss = plan_dg.adjoint_analysis(winv_ninv_sh)
    return winv_ninv_sh, ninv_gauss

@memory.cache
def rotate_mixing(lmax_pix, mixing_map, rot_ang):
    mixing_map_sh = sharp.sh_analysis(lmax_pix, mixing_map)
    if rot_ang != (0, 0, 0):
        rotate_alm(lmax_pix, mixing_map_sh, *rot_ang)
    p = sharp.RealMmajorGaussPlan(lmax_pix, lmax_pix)
    return p.synthesis(mixing_map_sh)


class HarmonicPrior(object):
    def __init__(self, beta, lcross, lmax):
        self.beta = beta
        self.lcross = lcross
        self.lmax = lmax

    def get_cl(self, system, k):
        l = np.arange(1, self.lmax + 2, dtype=np.float)

        amplitude = 1. / system.ni_approx_by_comp_lst[k][self.lcross]
        Cl = amplitude * (l / (self.lcross + 1))**self.beta
        return Cl


class CrSystem(object):
    def __init__(self, ninv_maps, bl_list, mixing_maps, prior_list):
        self.ninv_maps = ninv_maps
        self.bl_list = bl_list
        self.mixing_maps = mixing_maps
        self.prior_list = prior_list
        self.band_count = len(ninv_maps)
        self.comp_count = len(prior_list)
        self.lmax_list = [prior.lmax for prior in prior_list]
        self.sh_lengths = [(lmax + 1)**2 for lmax in self.lmax_list]
        self.sh_offsets = np.concatenate([[0], np.cumsum(self.sh_lengths)])

    def stack(self, x_lst):
        for k, x in enumerate(x_lst):
            assert self.sh_lengths[k] == x.shape[0]
        return np.concatenate(x_lst)

    def unstack(self, x):
        result = []
        for k in range(self.comp_count):
            result.append(x[self.sh_offsets[k]:self.sh_offsets[k + 1]])
        return result

    def copy_with(self, **kw):
        self_kw = dict(
            ninv_maps=self.ninv_maps,
            bl_list=self.bl_list,
            mixing_maps=self.mixing_maps)
        self_kw.update(**kw)
        return CrSystem(**self_kw)

    def set_params(self, lmax_ninv, rot_ang):
        self.lmax_mixed = self.lmax_mixing_pix = max(self.lmax_list)
        self.lmax_ninv = lmax_ninv
        self.rot_ang = rot_ang

    def prepare(self):
        # Make G-L ninv-maps, possibly rotated

        self.ninv_gauss_lst = []
        self.winv_ninv_sh_lst = []
        for ninv_map in self.ninv_maps:
            winv_ninv_sh, ninv_gauss = rotate_ninv(self.lmax_ninv, ninv_map, self.rot_ang)
            self.winv_ninv_sh_lst.append(winv_ninv_sh)
            self.ninv_gauss_lst.append(ninv_gauss)

        self.mixing_scalars = np.zeros((self.band_count, self.comp_count))
        for nu in range(self.band_count):
            for k in range(self.comp_count):
                self.mixing_scalars[nu, k] = self.mixing_maps[nu, k].mean()

        # Rescale prior vs. mixing_scalars and mixing_maps_ugrade and mixing_maps to avoid some numerical issues
        # We adjust mixing_scalars in-place. self.mixing_maps is kept as is but all derived quantities
        # computed in this routine are changed
        self.component_scale = 1. / np.sqrt(np.dot(self.mixing_scalars.T, self.mixing_scalars).diagonal())

        self.mixing_scalars *= self.component_scale[None, :]
        self.mixing_maps_ugrade = {}
        for nu in range(self.band_count):
            for k in range(self.comp_count):
                with timed('mixing'):
                    self.mixing_maps_ugrade[nu, k] = rotate_mixing(
                        self.lmax_mixing_pix, self.mixing_maps[nu, k], self.rot_ang) * self.component_scale[k]

        self.plan_outer_lst = [
            sharp.RealMmajorGaussPlan(self.lmax_mixing_pix, lmax)
            for lmax in self.lmax_list]
        self.plan_mixed = sharp.RealMmajorGaussPlan(self.lmax_mixing_pix, self.lmax_mixed) # lmax_mixing(pix) -> lmax_mixing(sh)
        self.plan_ninv = sharp.RealMmajorGaussPlan(self.lmax_ninv, self.lmax_mixed)

        # Estimates of Ni level for prior construction in demos
        self.ni_approx_by_comp_lst = []
        for k in range(self.comp_count):
            ni_approx = 0
            for nu in range(self.band_count):
                tau = self.ninv_maps[nu].mean() * self.ninv_maps[nu].shape[0] / (4 * np.pi)
                ni_approx += self.mixing_scalars[nu, k]**2 * tau * self.bl_list[nu][:self.lmax_list[k] + 1]**2
            self.ni_approx_by_comp_lst.append(ni_approx)

        # Prepare prior
        self.dl_list = []
        for k in range(self.comp_count):
            self.dl_list.append(self.component_scale[k]**2 / self.prior_list[k].get_cl(self, k))



    def matvec(self, x_lst):
        assert len(x_lst) == self.comp_count
        z_lst = [0] * self.comp_count

        for nu in range(self.band_count):
            # Mix components together
            y = np.zeros((self.lmax_mixed + 1)**2)
            for k in range(self.comp_count):
                u = self.plan_outer_lst[k].synthesis(x_lst[k])
                u *= self.mixing_maps_ugrade[nu, k]
                y += self.plan_mixed.analysis(u)
            # Instrumental beam
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            # Inverse noise weighting
            u = self.plan_ninv.synthesis(y)
            u *= self.ninv_gauss_lst[nu]
            y = self.plan_ninv.adjoint_synthesis(u)
            # Transpose our way out, accumulate result in z_list[icomp];
            # note that z_list will get result from all bands
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            for k in range(self.comp_count):
                u = self.plan_mixed.adjoint_analysis(y)
                u *= self.mixing_maps_ugrade[nu, k]
                z_lst[k] += self.plan_outer_lst[k].adjoint_synthesis(u)

        for k in range(self.comp_count):
            z_lst[k] += scatter_l_to_lm(self.dl_list[k]) * x_lst[k]
        return z_lst



    @classmethod
    def from_config(cls, config_doc):
        ninv_maps = []
        bl_list = []
        prior_list = []

        mixing_maps_template = config_doc['model']['mixing_maps_template']
        mixing_maps = {}

        nu = 0
        for dataset in config_doc['datasets']:
            path = dataset['path']

            for band in dataset['bands']:
                assert isinstance(band['name'], basestring), 'You need to surround band names with quotes'
                map_filename = os.path.join(path, dataset['map_template'].format(band=band))
                rms_filename = os.path.join(path, dataset['rms_template'].format(band=band))
                beam_filename = os.path.join(path, dataset['beam_template'].format(band=band))

                rms = load_map_cached(rms_filename)
                ninv_map = 1 / rms**2
                ninv_maps.append(ninv_map)

                bl = load_beam_cached(os.path.join(path, dataset['beam_template'].format(band=band)))
                bl_list.append(bl)

                for k, component in enumerate(config_doc['model']['components']):
                    mixing_maps[nu, k] = load_map_cached(mixing_maps_template.format(band=band, component=component))

                nu += 1

        for component in config_doc['model']['components']:
            prior_list.append(HarmonicPrior(
                beta=component['prior']['beta'],
                lcross=component['prior']['lcross'],
                lmax=component['lmax'],
                ))

        return cls(ninv_maps=ninv_maps, bl_list=bl_list, mixing_maps=mixing_maps, prior_list=prior_list)

    def plot(self):
        from matplotlib import pyplot as plt
        plt.clf()
        for k in range(self.comp_count):
            plt.semilogy(self.dl_list[k], 'k-')
            plt.semilogy(self.ni_approx_by_comp_lst[k], 'r-')
        plt.draw()


def downgrade_system(system, fraction):
    # Downgrade all the beams in the system to fraction*fwhm; using Gaussian beam approximations

    new_bl_list = []
    for bl in system.bl_list:
        # simply compute fwhm from bl...
        l = min(bl.shape[0] // 2, 1000)
        sigma = -2*np.log(bl[l]) / (l * (l + 1.))

        ls = np.arange(int(bl.shape[0] * fraction + 1), dtype=np.double)

        new_bl_list.append(np.exp(-0.5 * ls * (ls + 1) * (sigma / fraction)))

    new_prior_list = []
    for prior in system.prior_list:
        new_prior_list.append(HarmonicPrior(
            beta=prior.beta,
            lcross=int(prior.lcross * fraction + 1),
            lmax=int(prior.lmax * fraction + 1),
            ))

    return system.copy_with(bl_list=new_bl_list, prior_list=new_prior_list)
