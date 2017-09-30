from __future__ import division

"""
Benchmark script. Usage:

scripts/precond_benchmark.py input_file downgrade_resolution
"""

import numpy as np

from matplotlib.pyplot import *

import logging
logging.basicConfig(level=logging.DEBUG)



import cmbcr
import cmbcr.utils
reload(cmbcr.beams)
reload(cmbcr.cr_system)
reload(cmbcr.precond_sh)
reload(cmbcr.precond_pseudoinv)
reload(cmbcr.precond_diag)
reload(cmbcr.precond_pixel)
reload(cmbcr.utils)
reload(cmbcr.masked_solver)
reload(cmbcr)
from cmbcr.utils import *

from cmbcr import sharp
from healpy import mollzoom, mollview

import sys
from cmbcr.cg import cg_generator

config = cmbcr.load_config_file(sys.argv[1])

nside = int(sys.argv[2])
factor = 2048 // nside
rms_treshold = 1


full_res_system = cmbcr.CrSystem.from_config(config, udgrade=nside, mask_eps=0.8, rms_treshold=rms_treshold)

full_res_system.prepare_prior()

system = cmbcr.downgrade_system(full_res_system, 1. / factor)

lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (0, 0, 0)

system.set_params(
    lmax_ninv=lmax_ninv,
    rot_ang=rot_ang,
    flat_mixing=False,
    )

system.prepare_prior(scale_unity=False)
system.prepare(use_healpix=True, use_healpix_mixing=True, mixing_nside=nside)

rng = np.random.RandomState(1)

x0 = [
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)


class Benchmark(object):
    def __init__(self, label, style, preconditioner, n=100):
        self.label = label
        self.style = style
        self.preconditioner = preconditioner
        self.benchmark(n=n)

    def benchmark(self, n):
        from cmbcr import sharp

        r0 = None
        self.err_vecs = []
        self.err_norms = []
        self.reslst = []
        self.sht_counts = []

        if hasattr(self.preconditioner, 'starting_vector'):
            start_vec = system.stack(self.preconditioner.starting_vector(b))
        else:
            start_vec = np.zeros_like(x0_stacked)

        solver = cg_generator(
            lambda x: system.stack(system.matvec(system.unstack(x))),
            system.stack(b),
            x0=start_vec,
            M=lambda x: system.stack(self.preconditioner.apply(system.unstack(x))),
            )

        self.err_vecs.append(x0)
        try:
            sharp.sht_count = 0
            for i, (x, r, delta_new) in enumerate(solver):
                self.x = x
                if r0 is None:
                    r0 = np.linalg.norm(r)

                x = system.unstack(x)

                self.err_vecs.append([x0c - xc for x0c, xc in zip(x0, x)])
                err = np.linalg.norm(system.stack(x) - x0_stacked) / np.linalg.norm(x0_stacked)
                self.err_norms.append(err)
                print 'it', i, err
                self.sht_counts.append(sharp.sht_count)
                self.reslst.append(np.linalg.norm(r) / r0)
                if err < 1e-10 or i >= n:
                    break
        except ValueError as e:
            raise
        except AssertionError as e:
            raise
            print str(e)

    def ploterr(self):
        fig.gca().semilogy(self.err_norms, self.style, label=self.label)

    def plotscale(self):
        clf()
        scale = [(1. / cmbcr.norm_by_l(comp)) for comp in self.err_vecs[0]]
        for err in self.err_vecs:
            for compscale, comp in zip(scale, err):
                semilogy(cmbcr.norm_by_l(comp) * compscale)
        draw()

def save_benchmarks(benchmarks, filename):
    import yaml
    doc = [
        {
            'label': b.label,
            'err': [float(x) for x in b.err_norms],
            'sht_counts': b.sht_counts
        }
        for b in benchmarks]
    with open(filename, 'w') as f:
        yaml.dump(doc, f)


benchmarks = [
    Benchmark(
     'Fullsky diagonal',
      '-o',
     cmbcr.DiagonalPreconditioner2(system),
    ),

    Benchmark(
      'Fullsky pseudo-inverse',
      '-o',
      cmbcr.PseudoInverseWithMaskPreconditioner(system, inner_its=0),
     ),
    ]


##save_benchmarks(benchmarks, sys.argv[2], nside, rms_treshold))


fig = gcf()

for bench in benchmarks:
    bench.ploterr()
fig.gca().set_ylim((1e-10, 1e4))

legend()
show()
