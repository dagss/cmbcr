import numpy as np
import os
import logging

import healpy



from .data_utils import load_map, load_beam
from .cache import cached, memory
from .rotate_alm import rotate_alm
from .mmajor import scatter_l_to_lm
from . import sharp
from .utils import timed, pad_or_truncate_alm
from .healpix import nside_of

__all__ = ['CrSystem', 'downgrade_system']

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
    def __init__(self, lmax, spec):
        self.lmax = lmax
        self.spec = spec

    def downgrade(self, fraction):
        spec = dict(self.spec)
        spec.pop('l', None)  # if you use l-based spec, crash..
        spec.pop('l_eps', None)  # if you use l-based spec, crash, until these are fixed
        return HarmonicPrior(lmax=int(self.lmax * fraction + 1), spec=dict(self.spec))

    def get_Cl(self, system, k):
        t = self.spec['type']
        if t == 'power':
            l = np.arange(1, self.lmax + 2, dtype=np.float)

            nl = system.ni_approx_by_comp_lst[k]
            l_cross = (nl < nl.max() * self.spec['cross']).nonzero()[0][0]
            amplitude = 1. / nl[l_cross]
            Cl = amplitude * (l / l_cross)**self.spec['beta']
        elif t == 'file':
            dat = np.loadtxt(self.spec['filename'])
            assert dat[0,0] == 0 and dat[1,0] == 1 and dat[2,0] == 2
            Cl = dat[:, 1][:self.lmax + 1]
            ls = np.arange(2, self.lmax + 1)
            Cl[2:] /= ls * (ls + 1) / 2 / np.pi
            Cl[0] = Cl[1] = Cl[2]

            ni = system.ni_approx_by_comp_lst[k][self.spec['l']]
            if ni == 0:
                # mask-only; doesn't matter what the amplitude is
                amplitude = 1
            else:
                amplitude = 1. / (ni * Cl[self.spec['l']])
            Cl *= amplitude
        elif t == 'gaussian':
            ls = np.arange(self.lmax + 1)
            leps = self.spec['l_eps']
            sigma = np.sqrt(-2. * np.log(self.spec['eps']) / leps / (leps + 1))
            dl = np.exp(0.5 * ls * (ls + 1) * sigma**2)

        return Cl


class CrSystem(object):
    def __init__(self, ninv_maps, bl_list, mixing_maps, prior_list, mask=None):
        self.ninv_maps = ninv_maps
        self.bl_list = bl_list
        self.mixing_maps = mixing_maps
        self.prior_list = prior_list
        self.band_count = len(ninv_maps)
        self.comp_count = len(prior_list)
        self.lmax_list = [prior.lmax for prior in prior_list]
        self.x_lengths = [(lmax + 1)**2 for lmax in self.lmax_list]
        self.x_offsets = np.concatenate([[0], np.cumsum(self.x_lengths)])
        self.mask = mask

    def stack(self, x_lst):
        for k, x in enumerate(x_lst):
            assert self.x_lengths[k] == x.shape[0]
        return np.concatenate(x_lst)

    def unstack(self, x):
        result = []
        for k in range(self.comp_count):
            result.append(x[self.x_offsets[k]:self.x_offsets[k + 1]])
        return result

    def copy_with(self, **kw):
        self_kw = dict(
            ninv_maps=self.ninv_maps,
            bl_list=self.bl_list,
            mixing_maps=self.mixing_maps,
            prior_list=self.prior_list)
        self_kw.update(**kw)
        return CrSystem(**self_kw)

    def set_params(self, lmax_ninv, rot_ang, flat_mixing):
        self.lmax_mixed = max(self.lmax_list)
        self.lmax_ninv = lmax_ninv
        self.lmax_mixing_pix = self.lmax_mixed #lmax_ninv
        self.rot_ang = rot_ang
        self.flat_mixing = flat_mixing

    def prepare_prior(self, set_wl_dl=True):
        self.mixing_scalars = np.zeros((self.band_count, self.comp_count))
        for nu in range(self.band_count):
            for k in range(self.comp_count):
                q = self.mixing_maps[nu, k]
                self.mixing_scalars[nu, k] = (q[q != 0]).mean()
                #self.mixing_scalars[nu, k] = (q**2).sum() / q.sum()
        #print self.mixing_scalars


        # Estimates of Ni level for prior construction in demos; *note* that we *include* component_scale
        # here...
        self.ni_approx_by_comp_lst = []
        for k in range(self.comp_count):
            ni_approx = 0
            for nu in range(self.band_count):
                tau = self.ninv_maps[nu].mean() * self.ninv_maps[nu].shape[0] / (4 * np.pi)
                ni_approx += self.mixing_scalars[nu, k]**2 * tau * self.bl_list[nu][:self.lmax_list[k] + 1]**2
            self.ni_approx_by_comp_lst.append(ni_approx)
        # Prepare prior
        if set_wl_dl:
            self.dl_list = []
            self.wl_list = []
            self.Cl_list = []
            for k in range(self.comp_count):
                Cl = self.prior_list[k].get_Cl(self, k)
                self.Cl_list.append(Cl)
                self.dl_list.append(1. / Cl)
                self.wl_list.append(np.ones(self.lmax_list[k] + 1))

    def prepare(self, use_healpix=False):
        ## TODO DEBUG
        #for bl in self.bl_list:
        #    bl[:] = 1
        ## END DEBUG
        # Make G-L ninv-maps, possibly rotated
        self.ninv_gauss_lst = []
        self.winv_ninv_sh_lst = []
        self.use_healpix = use_healpix

        for nu, ninv_map in enumerate(self.ninv_maps):
            winv_ninv_sh, ninv_gauss = rotate_ninv(self.lmax_ninv, ninv_map, self.rot_ang)
            self.winv_ninv_sh_lst.append(winv_ninv_sh)
            self.ninv_gauss_lst.append(ninv_gauss)

        if not self.use_healpix:
            self.plan_ninv = sharp.RealMmajorGaussPlan(self.lmax_ninv, self.lmax_mixed)

        # Rescale prior vs. mixing_scalars and mixing_maps_ugrade and mixing_maps to avoid some numerical issues
        # We adjust mixing_scalars in-place. self.mixing_maps is kept as is but all derived quantities
        # computed in this routine are changed
        ## self.component_scale = np.ones(self.comp_count) # DEBUG

        self.component_scale = np.array([1])#1. / np.sqrt(np.dot(self.mixing_scalars.T, self.mixing_scalars).diagonal())

        self.mixing_scalars *= self.component_scale[None, :]
        for k in range(self.comp_count):
            self.dl_list[k] *= self.component_scale[k]**2
            self.ni_approx_by_comp_lst[k] *= self.component_scale[k]**2

        self.mixing_maps_ugrade = {}
        for nu in range(self.band_count):
            for k in range(self.comp_count):
                with timed('mixing'):
                    self.mixing_maps_ugrade[nu, k] = (
                        rotate_mixing(self.lmax_mixing_pix, self.mixing_maps[nu, k], self.rot_ang)
                        * self.component_scale[k])
                    self.mixing_maps[nu, k] *= self.component_scale[k]
                    if self.flat_mixing:
                        assert False
                        self.mixing_maps_ugrade[nu, k][:] = self.mixing_maps_ugrade[nu, k].mean()

        self.plan_outer_lst = [
            sharp.RealMmajorGaussPlan(self.lmax_mixing_pix, lmax)
            for lmax in self.lmax_list]
        self.plan_mixed = sharp.RealMmajorGaussPlan(self.lmax_mixing_pix, self.lmax_mixed) # lmax_mixing(pix) -> lmax_mixing(sh)

    def matvec(self, x_lst, skip_prior=False):
        assert len(x_lst) == self.comp_count

        nside_mixing = nside_of(self.mixing_maps[0, 0])
        x_pix_lst = [
            sharp.sh_synthesis(nside_mixing,
                x_lst[k] * scatter_l_to_lm(self.wl_list[k])
            ) for k in range(self.comp_count)]
        z_pix_lst = [0] * self.comp_count

        for nu in range(self.band_count):
            # Mix components together
            y = np.zeros(self.mixing_maps[0,0].shape[0])
            for k in range(self.comp_count):
                u = x_pix_lst[k] * self.mixing_maps[nu, k]
                y += u
            y = sharp.sh_analysis(self.lmax_mixed, y) #self.plan_mixed.analysis(y)
            # Instrumental beam
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            # Inverse noise weighting
            if self.use_healpix:
                u = sharp.sh_synthesis(nside_of(self.ninv_maps[nu]), y)
                u *= self.ninv_maps[nu]
                y = sharp.sh_adjoint_synthesis(self.lmax_mixed, u)
            else:
                # gauss-legendre mode
                u = self.plan_ninv.synthesis(y)
                u *= self.ninv_gauss_lst[nu]
                y = self.plan_ninv.adjoint_synthesis(u)
            # Transpose our way out, accumulate result in z_list[icomp];
            # note that z_list will get result from all bands
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            y = sharp.sh_adjoint_analysis(nside_mixing, y) #self.plan_mixed.adjoint_analysis(y)
            for k in range(self.comp_count):
                u = y * self.mixing_maps[nu, k]
                z_pix_lst[k] += u

        z_lst = [
            sharp.sh_adjoint_synthesis(self.lmax_list[k], z_pix_lst[k])
             * scatter_l_to_lm(self.wl_list[k])
            for k in range(self.comp_count)
            ]

        if not skip_prior:
            for k in range(self.comp_count):
                z_lst[k] += scatter_l_to_lm(self.dl_list[k]) * x_lst[k]
        return z_lst


    def matvec_scalar_mixing(self, x_lst):
        assert len(x_lst) == self.comp_count
        z_lst = [0] * self.comp_count

        for nu in range(self.band_count):
            # Mix components together
            y = np.zeros((self.lmax_mixed + 1)**2)
            for k in range(self.comp_count):
                y += pad_or_truncate_alm(x_lst[k], self.lmax_mixed) * self.mixing_scalars[nu, k]
            # Instrumental beam
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            # Inverse noise weighting
            if self.use_healpix:
                u = sharp.sh_synthesis(nside_of(self.ninv_maps[nu]), y)
                u *= self.ninv_maps[nu]
                y = sharp.sh_adjoint_synthesis(self.lmax_mixed, u)
            else:
                # gauss-legendre mode
                u = self.plan_ninv.synthesis(y)
                u *= self.ninv_gauss_lst[nu]
                y = self.plan_ninv.adjoint_synthesis(u)
            # Transpose our way out, accumulate result in z_list[icomp];
            # note that z_list will get result from all bands
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            for k in range(self.comp_count):
                z_lst[k] = pad_or_truncate_alm(y, self.lmax_list[k]) * self.mixing_scalars[nu, k]

        for k in range(self.comp_count):
            z_lst[k] += scatter_l_to_lm(self.dl_list[k]) * x_lst[k]
        return z_lst


    @classmethod
    def from_config(cls, config_doc, rms_treshold=1, mask_eps=0.1, mask=None, udgrade=None):
        ninv_maps = []
        bl_list = []
        prior_list = []

        mixing_maps_template = config_doc['model']['mixing_maps_template']
        mixing_maps = {}

        mask = config_doc['model'].get('mask', '')


        if mask:
            if 0:
                mask = np.zeros(12 * udgrade**2)
                mask[:] = 1
                nside = udgrade
                mask[6*udgrade**2:12*udgrade**2] = 0
            else:
                mask = load_map_cached(mask)
                mask = mask.copy()
                #mask[:] = 1
                #nside = nside_of(mask)
                #mask[6*nside**2:12*nside**2] = 0
        else:
            mask = None
        
        
        ## if mask:
        ##     mask = load_map_cached(mask)
        ##     mask = mask.copy()
        ##     mask[:] = 1
        ##     nside = nside_of(mask)
        ##     mask[6*nside**2:12*nside**2] = 0
        ## else:
        ##     mask = None

        nu = 0
        for dataset in config_doc['datasets']:
            path = dataset['path']

            for band in dataset['bands']:
                assert isinstance(band['name'], basestring), 'You need to surround band names with quotes'
                map_filename = os.path.join(path, dataset['map_template'].format(band=band))
                rms_filename = os.path.join(path, dataset['rms_template'].format(band=band))
                beam_filename = os.path.join(path, dataset['beam_template'].format(band=band))

                bl = load_beam_cached(os.path.join(path, dataset['beam_template'].format(band=band)))
                bl_list.append(bl)

                rms = load_map_cached(rms_filename)
                rms = rms.copy()
                ##print 'WARNING: mean rms'
                ##rms[:] = rms.mean() ## DEBUG

                if udgrade is not None:
                    rms = healpy.ud_grade(rms, order_in='RING', order_out='RING', nside_out=udgrade, power=1)

                alpha = np.percentile(rms, rms_treshold)
                rms[rms < alpha] = alpha
                ninv_map = 1 / rms**2

                # We don't deal with the mask before precompute, because if the system is downscaled
                # we want to
                nside = nside_of(ninv_map)

                if mask is not None:
                    # First, udgrade the mask to same resolution as ninv_map. Then, extend it with one beam-size.
                    mask_ud = healpy.ud_grade(mask, nside, order_in='RING', order_out='RING', power=-1)
                    mask_ud[mask_ud != 0] = 1

                    #mask_lm = sharp.sh_analysis(3 * nside, mask_ud)
                    #from .beams import gaussian_beam_by_l
                    #mask_lm *= scatter_l_to_lm(gaussian_beam_by_l(3 * nside, '10 deg'))
                    #mask_ext = sharp.sh_synthesis(nside, mask_lm)

                    #healpy.mollzoom(mask_ext - mask_ud)
                    #1/0

                    ninv_map *= mask_ud

                ninv_maps.append(ninv_map)

                for k, component in enumerate(config_doc['model']['components']):
                    mixing_maps[nu, k] = load_map_cached(mixing_maps_template.format(band=band, component=component))
                    mixing_maps[nu, k] = mixing_maps[nu, k].copy()
                    if mask is not None:
                        mask_ud = healpy.ud_grade(mask, nside_of(mixing_maps[nu, k]), order_in='RING', order_out='RING', power=0)
                        mask_ud[mask_ud != 0] = 1
                        mixing_maps[nu, k] *= mask_ud

                nu += 1

        for component in config_doc['model']['components']:
            prior_list.append(HarmonicPrior(component['lmax'], component['prior']))

        return cls(ninv_maps=ninv_maps, bl_list=bl_list, mixing_maps=mixing_maps, prior_list=prior_list, mask=mask)

    def plot(self, lmax=None):
        from matplotlib import pyplot as plt
        #plt.clf()
        for k in range(self.comp_count):
            L = lmax or self.lmax_list[k]
            scale = (1 / self.ni_approx_by_comp_lst[k].max())
            plt.semilogy(1 / self.Cl_list[k][:L + 1] * scale)
            plt.semilogy(self.ni_approx_by_comp_lst[k][:L + 1] * scale, linestyle='dotted')
        plt.draw()


def restrict_system(system):
    """
    Create a new system that has half the resolution.
    """
    lmax_list_dg = [L // 2 for L in system.lmax_list]

    wl_list_dg = []
    dl_list_dg = []
    rl_list = []
    
    for k in range(system.comp_count):
        # make restriction rl such that rl=0.1 (i.e., rl**2=0.01) at L
        L = lmax_list_dg[k]
        ls = np.arange(L + 1, dtype=np.double)
        sigma_sq = -2. * np.log(0.1) / L / (L + 1)
        rl = np.exp(0.5 * ls * (ls + 1) * sigma_sq)

        # update and truncate wl_list and dl_list
        wl_list_dg.append(system.wl_list[k][:L + 1] * rl)
        dl_list_dg.append(system.dl_list[k][:L + 1] * rl**2)
        rl_list.append(rl)

    system_dg = system.copy_with()
    system_dg.lmax_list = lmax_list_dg
    system_dg.wl_list = wl_list_dg
    system_dg.dl_list = dl_list_dg
    system_dg.rl_list = rl_list
    system_dg.set_params(system.lmax_ninv, system.rot_ang, system.flat_mixing)
    system_dg.prepare_prior(set_wl_dl=False)
    system_dg.prepare(use_healpix=True)
    return system_dg
        

def downgrade_system(system, fraction):
    # Downgrade all the beams in the system to fraction*fwhm; using Gaussian beam approximations

    new_bl_list = []
    for bl in system.bl_list:
        # simply compute fwhm from bl...
        l = min(bl.shape[0] // 2, 1000)
        sigma = -2*np.log(bl[l]) / (l * (l + 1.))

        ls = np.arange(int(bl.shape[0] * fraction + 1), dtype=np.double)

        new_bl = np.exp(-0.5 * ls * (ls + 1) * (sigma / fraction**2))
        new_bl_list.append(new_bl)

    new_prior_list = []
    for prior in system.prior_list:
        new_prior_list.append(prior.downgrade(fraction))

    return system.copy_with(bl_list=new_bl_list, prior_list=new_prior_list, mask=system.mask)
