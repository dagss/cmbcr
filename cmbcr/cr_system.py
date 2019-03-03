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
from .beams import fwhm_to_sigma, gaussian_beam_by_l

__all__ = ['CrSystem', 'downgrade_system']

load_map_cached = cached(lambda filename: load_map('raw', filename))
load_beam_cached = cached(load_beam)

def rotate_ninv(lmax_ninv, ninv_map, rot_ang):
    with timed('rotate_ninv'):
        with timed('winv adjoint_syntesis'):
            winv_ninv_sh = sharp.sh_adjoint_synthesis(lmax_ninv, ninv_map)
        if rot_ang != (0, 0, 0):
            rotate_alm(lmax_ninv, winv_ninv_sh, *rot_ang)
        plan_dg = sharp.RealMmajorGaussPlan(lmax_ninv, lmax_ninv)
        with timed('ninv_gauss adjoint_analysis'):
            ninv_gauss = plan_dg.adjoint_analysis(winv_ninv_sh)
        return winv_ninv_sh, ninv_gauss

def rotate_mixing(lmax_pix, mixing_map, rot_ang):
    1/0
    mixing_map_sh = sharp.sh_analysis(lmax_pix, mixing_map)
    if rot_ang != (0, 0, 0):
        rotate_alm(lmax_pix, mixing_map_sh, *rot_ang)
    p = sharp.RealMmajorGaussPlan(lmax_pix, lmax_pix)
    return p.synthesis(mixing_map_sh)


class HarmonicPrior(object):
    def __init__(self, lmax, spec, fullres_lmax=None):
        self.lmax = lmax
        self.spec = spec
        self.fullres_lmax = fullres_lmax or lmax

    def downgrade(self, fraction):
        spec = dict(self.spec)
        spec.pop('l', None)  # if you use l-based spec, crash..
        spec.pop('l_eps', None)  # if you use l-based spec, crash, until these are fixed
        new_lmax = int(self.lmax * fraction + 1)
        if new_lmax % 2 == 0:
            new_lmax += 1  # make nrings == lmax+1 be a pair number
        return HarmonicPrior(lmax=new_lmax, spec=dict(self.spec), fullres_lmax=self.lmax)

    def get_Cl(self, system, k):
        t = self.spec['type']
        if t == 'power':
            l = np.arange(1, self.lmax + 2, dtype=np.float)
            Cl = l**self.spec['beta']
        elif t == 'file':
            dat = np.loadtxt(self.spec['filename'])
            assert dat[0,0] == 0 and dat[1,0] == 1 and dat[2,0] == 2
            Cl = dat[:, 1]
            ls = np.arange(2, Cl.shape[0])
            Cl[2:] /= ls * (ls + 1) / 2 / np.pi
            Cl[0] = Cl[1] = Cl[2]

            if self.lmax < self.fullres_lmax and self.spec.get('compress', False):
                from scipy.interpolate import interp1d
                Cl_func = interp1d(np.arange(Cl.shape[0]), Cl)
                Cl = Cl_func(np.linspace(0, self.fullres_lmax, self.lmax + 1))
            else:
                Cl = Cl[:self.lmax + 1]
        elif t == 'gaussian':
            ls = np.arange(self.lmax + 1)
            sigma = fwhm_to_sigma(self.spec['fwhm'])
            sigma *= (self.fullres_lmax / float(self.lmax))
            ##sigma = np.sqrt(-2. * np.log(self.spec['beam_cross']) / self.lmax / (self.lmax + 1))
            Cl = np.exp(-0.5 * ls * (ls + 1) * sigma**2)
        elif t == 'none':
            Cl = None
            return Cl

        cross = self.spec.get('cross', None)
        if cross is None:
            # use amplitude from file
            assert t in ('file', 'gaussian')
            amplitude = 1
        else:
            nl = system.ni_approx_by_comp_lst[k]
            
            l_cross = (nl < nl.max() * self.spec['cross']).nonzero()[0]
            if len(l_cross):
                l_cross = l_cross[0]
            else:
                l_cross = self.lmax
            amplitude = 1. / nl[l_cross] / Cl[l_cross]
            print 'Adjusting Cl by ', amplitude
            
        Cl *= amplitude * self.spec.get('relamp', 1)
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

    def set_params(self, lmax_ninv, rot_ang, flat_mixing, use_mixing_mask):
        self.lmax_mixed = max(self.lmax_list)
        self.lmax_ninv = lmax_ninv
        self.lmax_mixing_pix = self.lmax_mixed #lmax_ninv
        self.rot_ang = rot_ang
        self.flat_mixing = flat_mixing
        self.use_mixing_mask = use_mixing_mask

    def prepare_prior(self, set_wl_dl=True, wl=None, scale_unity=False):
        self.mixing_scalars = np.zeros((self.band_count, self.comp_count))
        for nu in range(self.band_count):
            for k in range(self.comp_count):
                q = self.mixing_maps[nu, k]
                self.mixing_scalars[nu, k] = (q[q != 0]).mean()

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
                if scale_unity:
                    assert Cl is not None
                    self.wl_list.append(np.sqrt(Cl))
                else:
                    self.wl_list.append(np.ones(self.lmax_list[k] + 1))
                self.dl_list.append(1. / Cl if Cl is not None else np.zeros(self.lmax_list[k] + 1))

    def set_wl_list(self, wl_list):
        self.wl_list = wl_list

    def prepare(self, use_healpix=False, use_healpix_mixing=False, mixing_nside=None):
        # Make G-L ninv-maps, possibly rotated
        self.ninv_gauss_lst = []
        self.winv_ninv_sh_lst = []
        self.use_healpix = use_healpix
        self.use_healpix_mixing = use_healpix_mixing

        if 0:
        # this stuff is needed for diagonal preconditioner
            for nu, ninv_map in enumerate(self.ninv_maps):
                winv_ninv_sh, ninv_gauss = rotate_ninv(self.lmax_ninv, ninv_map, self.rot_ang)
                self.winv_ninv_sh_lst.append(winv_ninv_sh)
                self.ninv_gauss_lst.append(ninv_gauss)
                pass

        if not self.use_healpix:
            self.plan_ninv = sharp.RealMmajorGaussPlan(self.lmax_ninv, self.lmax_mixed)

        # Rescale prior vs. mixing_scalars and mixing_maps_ugrade and mixing_maps to avoid some numerical issues
        # We adjust mixing_scalars in-place. self.mixing_maps is kept as is but all derived quantities
        # computed in this routine are changed
        self.component_scale = 1. / np.sqrt(np.dot(self.mixing_scalars.T, self.mixing_scalars).diagonal())
        self.component_scale[:] = 1

        self.mixing_scalars *= self.component_scale[None, :]
        for k in range(self.comp_count):
            self.dl_list[k] *= self.component_scale[k]**2
            self.ni_approx_by_comp_lst[k] *= self.component_scale[k]**2

        self.mixing_maps_ugrade = {}

        if self.use_healpix_mixing:
            from cmbcr.healpix_data import get_ring_weights_T

            self.mask_dg_map = {} # nside : mask
            
            # Downgrade mask used for region partitioning in preconditioner
            if self.mask is not None:
                self.mask_dg = healpy.ud_grade(self.mask, order_in='RING', order_out='RING', nside_out=mixing_nside, power=0)
                ##self.mask_dg[self.mask_dg <= 0.5] = 0
                ##self.mask_dg[self.mask_dg != 0] = 1

                if 0:
                    # Apodize mask
                    alm = sharp.sh_analysis(3 * mixing_nside, self.mask_dg)
                    alm *= scatter_l_to_lm(gaussian_beam_by_l(3 * mixing_nside, '6 deg'))
                    self.mask_dg = sharp.sh_synthesis(mixing_nside, alm)
                    self.mask_dg[self.mask_dg > 0.8] = 1
                    self.mask_dg[self.mask_dg < 0.2] = 0
                else:
                    self.mask_dg[self.mask_dg <= 0.5] = 0
                    self.mask_dg[self.mask_dg != 0] = 1
                    

                

                
            for nu in range(self.band_count):
                # masks used in matvec 
                nside = nside_of(self.ninv_maps[nu])
                if nside not in self.mask_dg_map:
                    if self.mask is not None:
                        assert nside_of(self.mask_dg) == nside
                        m = self.mask_dg
                        ##m = healpy.ud_grade(self.mask, order_in='RING', order_out='RING', nside_out=nside, power=0)
                        ##m[self.mask_dg <= 0.5] = 0
                        ##m[self.mask_dg != 0] = 1
                        self.mask_dg_map[nside] = m
                    else:
                        self.mask_dg_map[nside] = np.ones(12*nside**2)                        

                for k in range(self.comp_count):

                    self.mixing_maps_ugrade[nu, k] = healpy.ud_grade(
                        self.mixing_maps[nu, k],
                        order_in='RING',
                        order_out='RING',
                        nside_out=mixing_nside,
                        power=0)

                if self.mask is not None and self.use_mixing_mask:
                    self.mixing_maps_ugrade[nu, k] *= self.mask_dg

            weights = get_ring_weights_T(mixing_nside)
            self.plan_outer_lst = [
                sharp.RealMmajorHealpixPlan(mixing_nside, lmax, weights=weights)
                for lmax in self.lmax_list]
            self.plan_mixed = sharp.RealMmajorHealpixPlan(mixing_nside, self.lmax_mixed, weights=weights)
        else:
            # Resample mask to Gauss-Legendre grid
            if self.mask is not None:
                mask_lm = sharp.sh_analysis(self.lmax_mixing_pix, self.mask)
                self.mask_gauss_grid = sharp.sh_synthesis_gauss(self.lmax_mixing_pix, mask_lm)
                self.mask_gauss_grid[self.mask_gauss_grid < 0.8] = 0
                self.mask_gauss_grid[self.mask_gauss_grid >= 0.8] = 1
            else:
                self.mask_gauss_grid = None

            for nu in range(self.band_count):
                for k in range(self.comp_count):
                    with timed('mixing'):
                        self.mixing_maps_ugrade[nu, k] = (
                            rotate_mixing(self.lmax_mixing_pix, self.mixing_maps[nu, k], self.rot_ang)
                            * self.component_scale[k])

                        if self.mask_gauss_grid is not None:
                            self.mixing_maps_ugrade[nu, k] *= self.mask_gauss_grid

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
        if self.flat_mixing:
            return self.matvec_scalar_mixing(x_lst)

        x_pix_lst = [
            plan.synthesis(x_lst[k] * scatter_l_to_lm(self.wl_list[k]))
            for k, plan in enumerate(self.plan_outer_lst)
            ]
        z_pix_lst = [0] * self.comp_count

        for nu in range(self.band_count):
            # Mix components together
            y = np.zeros(self.mixing_maps_ugrade[0,0].shape[0])
            for k in range(self.comp_count):
                u = x_pix_lst[k] * self.mixing_maps_ugrade[nu, k]
                y += u
            y = self.plan_mixed.analysis(y)
            # Instrumental beam
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            # Inverse noise weighting
            if self.use_healpix:
                u = sharp.sh_synthesis(nside_of(self.ninv_maps[nu]), y)
                u *= self.ninv_maps[nu]
                # mask mul
                if not self.use_mixing_mask:
                    u *= self.mask_dg_map[nside_of(u)]
                y = sharp.sh_adjoint_synthesis(self.lmax_mixed, u)
            else:
                1/0
                # gauss-legendre mode
                u = self.plan_ninv.synthesis(y)
                u *= self.ninv_gauss_lst[nu]
                y = self.plan_ninv.adjoint_synthesis(u)
            # Transpose our way out, accumulate result in z_list[icomp];
            # note that z_list will get result from all bands
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            y = self.plan_mixed.adjoint_analysis(y)
            for k in range(self.comp_count):
                u = y * self.mixing_scalars[nu, k] * self.mixing_maps_ugrade[nu, k]
                z_pix_lst[k] += u

        z_lst = [
            plan.adjoint_synthesis(z_pix_lst[k]) * scatter_l_to_lm(self.wl_list[k])
            for k, plan in enumerate(self.plan_outer_lst)
            ]

        if not skip_prior:
            for k in range(self.comp_count):
                z_lst[k] += scatter_l_to_lm(self.wl_list[k]**2 * self.dl_list[k]) * x_lst[k]

        return z_lst


    def matvec_scalar_mixing(self, x_lst):
        assert len(x_lst) == self.comp_count
        z_lst = [0] * self.comp_count
        if not self.use_mixing_mask and not self.use_healpix:
            print "ERROR: MASK WILL NOT BE APPLIED AS SCALAR MIXING IS TURNED ON..."
            1/0

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
                if not self.use_mixing_mask:
                    u *= self.mask_dg_map[nside_of(u)]
                y = sharp.sh_adjoint_synthesis(self.lmax_mixed, u)
            else:
                1/0
                # gauss-legendre mode
                u = self.plan_ninv.synthesis(y)
                u *= self.ninv_gauss_lst[nu]
                y = self.plan_ninv.adjoint_synthesis(u)
            # Transpose our way out, accumulate result in z_list[icomp];
            # note that z_list will get result from all bands
            y *= scatter_l_to_lm(self.bl_list[nu][:self.lmax_mixed + 1])
            for k in range(self.comp_count):
                z_lst[k] = z_lst[k] + pad_or_truncate_alm(y, self.lmax_list[k]) * self.mixing_scalars[nu, k]

        for k in range(self.comp_count):
            z_lst[k] += scatter_l_to_lm(self.dl_list[k]) * x_lst[k]
        return z_lst


    @classmethod
    def from_config(cls, config_doc, rms_treshold=0, mask_eps=0.1, mask=None, udgrade=None):
        ninv_maps = []
        bl_list = []
        prior_list = []

        mixing_maps_template = config_doc['model']['mixing_maps_template']
        mixing_maps = {}

        mask = config_doc['model'].get('mask', '')


        if mask:
            mask = load_map_cached(mask)
            mask = mask.copy()
        else:
            mask = None
        
        
        nu = 0
        for dataset in config_doc['datasets']:
            path = dataset['path']

            for band in dataset['bands']:
                assert isinstance(band['name'], basestring), 'You need to surround band names with quotes'
                ##map_filename = os.path.join(path, dataset['map_template'].format(band=band))
                rms_filename = os.path.join(path, dataset['rms_template'].format(band=band))
                beam_filename = os.path.join(path, dataset['beam_template'].format(band=band))

                bl = load_beam_cached(os.path.join(path, dataset['beam_template'].format(band=band)))
                bl_list.append(bl)

                rms = load_map_cached(rms_filename)
                rms = rms.copy()

                if udgrade is not None:
                    rms = healpy.ud_grade(rms, order_in='RING', order_out='RING', nside_out=udgrade, power=1)

                alpha = np.percentile(rms, rms_treshold)
                rms[rms < alpha] = alpha
                ninv_map = 1 / rms**2

                ninv_maps.append(ninv_map)

                for k, component in enumerate(config_doc['model']['components']):
                    cached_map = load_map_cached(mixing_maps_template.format(band=band, component=component))

                    mixing_smooth_fwhm = config_doc['model'].get('mixing_smooth', None)
                    
                    if 1==0 and mixing_smooth_fwhm is not None:
                        nside_mix = nside_of(cached_map)
                        lmax_mix = 3 * nside_mix
                        alm = sharp.sh_analysis(lmax_mix, cached_map)
                        alm *= scatter_l_to_lm(gaussian_beam_by_l(lmax_mix, mixing_smooth_fwhm))
                        mixing_maps[nu, k] = sharp.sh_synthesis(nside_mix, alm)
                    else:
                        mixing_maps[nu, k] = cached_map.copy()
                        #mixing_maps[nu, k][:] = cached_map.mean()

                nu += 1

        for component in config_doc['model']['components']:
            prior_list.append(HarmonicPrior(component['lmax'], component['prior']))

        return cls(ninv_maps=ninv_maps, bl_list=bl_list, mixing_maps=mixing_maps, prior_list=prior_list, mask=mask)

    def plot(self, lmax=None):
        from matplotlib import pyplot as plt
        #plt.clf()
        colors = ['red', 'green', 'blue']
        for k in range(self.comp_count):
            L = lmax or self.lmax_list[k]
            scale = scale = (1 / self.ni_approx_by_comp_lst[k].max())
            plt.semilogy(self.wl_list[k]**2 / self.Cl_list[k][:L + 1] * scale, color=colors[k])
            plt.semilogy(self.wl_list[k]**2 * self.ni_approx_by_comp_lst[k][:L + 1] * scale, linestyle='dotted', color=colors[k])
        plt.draw()
        

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
