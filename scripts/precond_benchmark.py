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
reload(cmbcr.utils)
reload(cmbcr)
from cmbcr.utils import *

#reload(cmbcr.main)

from cmbcr.cg import cg_generator



config = cmbcr.load_config_file('input/basic.yaml')

system = cmbcr.CrSystem.from_config(config)
system = cmbcr.downgrade_system(system, 0.04)

print system.lmax_list

lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (-1.71526923, -0.97844199, -0.03666168)

system.set_params(lmax_ninv=lmax_ninv, rot_ang=rot_ang)
system.prepare()

preconditioner = cmbcr.BandedHarmonicPreconditioner(system, diagonal=False)


r0 = None
errlst = []
reslst = []


rng = np.random.RandomState(1)

x0 = [
    cmbcr.scatter_l_to_lm(system.dl_list[k]) * rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)

solver = cg_generator(
    lambda x: system.stack(system.matvec(system.unstack(x))),
    system.stack(b),
    M=lambda x: system.stack(preconditioner.apply(system.unstack(x))))

for i, (x, r, delta_new) in enumerate(solver):
    print 'it', i
    if r0 is None:
        r0 = np.linalg.norm(r)


    err = np.linalg.norm(x - x0_stacked) / np.linalg.norm(x0_stacked)
    #
    #x = system.unstack(x)
    #err = [np.linalg.norm(x[k] - x0[k]) / np.linalg.norm(x0[k]) for k in range(system.comp_count)]

    errlst.append(err)
    reslst.append(np.linalg.norm(r) / r0)
    if i >= 30:
        break

semilogy(errlst, '-o')
