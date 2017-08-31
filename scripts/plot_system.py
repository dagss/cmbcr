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

full_res_system = cmbcr.CrSystem.from_config(config)
full_res_system.prepare_prior()

clf()
full_res_system.plot()

def make_cl_approx(lmax):
    l = np.arange(lmax)
    lpivot = int(lmax * 1600. / 6000.)
    l_to_6 = 0.14 * (l / float(lpivot))**6
    l_to_3 = 0.14 * (l / float(lpivot))**3

    return np.concatenate([l_to_3[:lpivot], l_to_6[lpivot:]])

plot(make_cl_approx(6000))
#semilogy(make_cl_approx(6000))
