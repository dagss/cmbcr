from __future__ import division
import numpy as np

from matplotlib.pyplot import *

import logging
logging.basicConfig(level=logging.DEBUG)



import cmbcr
import cmbcr.utils
reload(cmbcr.beams)
reload(cmbcr.cr_system)
reload(cmbcr.precond_sh)
reload(cmbcr.precond_psuedoinv)
reload(cmbcr.precond_diag)
reload(cmbcr.precond_pixel)
reload(cmbcr.utils)
reload(cmbcr.multilevel)
reload(cmbcr)
from cmbcr.utils import *

from cmbcr import sharp
from healpy import mollzoom, mollview

#reload(cmbcr.main)

import sys
from cmbcr.cg import cg_generator


config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))


w = 1

nside = 64 * w
factor = 2048 // nside * w



full_res_system = cmbcr.CrSystem.from_config(config, udgrade=nside, mask_eps=0.8)

full_res_system.prepare_prior()

system = cmbcr.downgrade_system(full_res_system, 1. / factor)

lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (0, 0, 0)

system.set_params(
    lmax_ninv=lmax_ninv,
    rot_ang=rot_ang,
    flat_mixing=False,
    )

system.prepare_prior()
system.prepare(use_healpix=True)



dl = system.dl_list[0]
nl = cmbcr.standard_needlet_by_l(2, 2 * dl.shape[0] - 1)
i = nl.argmax()
dl_new = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])

pw = 2*np.pi / (4. * nside)


if 0:
    clf();
    thetas = np.linspace(0, 0.3, 1000)
    plot(thetas, cmbcr.beam_by_theta(dl_new, thetas))
    plot(thetas, cmbcr.beam_by_theta(dl, thetas))
    axvline(pw, linestyle=':')
    axvline(2 * pw, linestyle=':')
    draw()
    1/0
    #1/0


else:

    # try to dgrade...

    dl = dl_new

    N = 8
    
    thetas_h = np.linspace(0, (N-1) * pw, N)
    thetas_H = np.linspace(0, (N-1) * pw, N // 2)

    x_h = cmbcr.beam_by_theta(dl, thetas_h)
    x_H_sub = x_h[::2]
    x_H_mean = x_h.reshape((x_h.shape[0] // 2, 2)).mean(axis=1)
    

    clf()
    plot(thetas_h, x_h, '-o')
    plot(thetas_H, x_H_sub, '-o', label='x_H_sub')
    plot(thetas_H, x_H_mean * 2, '-o', label='x_H_mean')
    plot(thetas_H, cmbcr.beam_by_theta(dl[::2], thetas_H) * 4, '-o', label='dl[::2]')
    legend()
    draw()
    #x_h = np.arange(500)
