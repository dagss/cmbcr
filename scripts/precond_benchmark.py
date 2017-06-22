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
reload(cmbcr.precond_pixel)
reload(cmbcr.utils)
reload(cmbcr.multilevel)
reload(cmbcr)
from cmbcr.utils import *

#reload(cmbcr.main)

import sys
from cmbcr.cg import cg_generator

config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))

full_res_system = cmbcr.CrSystem.from_config(config)
full_res_system.prepare_prior()

system = cmbcr.downgrade_system(full_res_system, 0.05)
system.prepare_prior()

#full_res_system.plot(lmax=2000)
#system.plot(lmax=200)
#1/0

#print system.lmax_list
#1/0
lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (0, 0, 0)
#rot_ang = (-1.71526923, -0.97844199, -0.03666168)

system.set_params(
    lmax_ninv=lmax_ninv,
    rot_ang=rot_ang,
    flat_mixing=True)
system.prepare_prior()
system.prepare()


rng = np.random.RandomState(1)

x0 = [
    #scatter_l_to_lm(system.dl_list[k]) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)





class Benchmark(object):
    def __init__(self, label, style, preconditioner, n=50):
        self.label = label
        self.style = style
        self.preconditioner = preconditioner
        self.benchmark(n=n)

    def benchmark(self, n):
        r0 = None
        self.err_vecs = []
        self.err_norms = []
        self.reslst = []

        solver = cg_generator(
            lambda x: system.stack(system.matvec(system.unstack(x))),
            system.stack(b),
            M=lambda x: system.stack(self.preconditioner.apply(system.unstack(x))))

        self.err_vecs.append(x0)
        for i, (x, r, delta_new) in enumerate(solver):
            print 'it', i
            if r0 is None:
                r0 = np.linalg.norm(r)

            self.err_vecs.append([x0c - xc for x0c, xc in zip(x0, system.unstack(x))])
            err = np.linalg.norm(x - x0_stacked) / np.linalg.norm(x0_stacked)
            self.err_norms.append(err)
            self.reslst.append(np.linalg.norm(r) / r0)
            if err < 1e-13 or i >= n:
                break

        
        
    def ploterr(self):
        semilogy(self.err_norms, self.style, label=self.label)

    def plotscale(self):
        clf()
        scale = [(1. / cmbcr.norm_by_l(comp)) for comp in self.err_vecs[0]]
        for err in self.err_vecs:
            for compscale, comp in zip(scale, err):
                semilogy(cmbcr.norm_by_l(comp) * compscale)
        draw()
        

    

# TODO: Looks like there's a bug in harmonic_preconditioner.f90 or something? The diagonal should
# look the same regardless of whether we include off-diagonal... perhaps simplify harmonic_preconditioner.f90
# and the bug will disappear...
#
#
#

#clf()
#p = cmbcr.BandedHarmonicPreconditioner(system, diagonal=False, couplings=True)
#semilogy(p.data[0, :])

#p = cmbcr.BandedHarmonicPreconditioner(system, diagonal=False, couplings=False)
#semilogy(p.data[0, :])
 
#1/0
    


#precond = cmbcr.PixelPreconditioner(system)
#bench = Benchmark(
#    'Pixel',
#    '-o',
#    precond)
##bench.ploterr()


#bench.plotscale()
#1/0


if 0:
    A = hammer(lambda x: system.stack(system.matvec(system.unstack(x))), system.x_offsets[-1])
    M = hammer(lambda x: system.stack(diag_precond.apply(system.unstack(x))), system.x_offsets[-1])
    clf()
    semilogy(A.diagonal(), '-o', label='A')
    semilogy(1 / M.diagonal(), '-o', label='M')
    legend()
    draw()
    1/0

if 0:
    clf()
    for i in range(system.band_count):
        alpha = 1
        def op(x):
            u = system.plan_ninv.synthesis(x)
            u *= system.ninv_gauss_lst[i] * alpha
            u = system.plan_ninv.adjoint_synthesis(u)
            return u

        Ni = hammer(op, (31 + 1)**2)

        plot(Ni.diagonal())
    draw()

    1/0

diag_precond_nocouplings = cmbcr.BandedHarmonicPreconditioner(system, diagonal=True, couplings=False)

benchmarks = [
    Benchmark(
        'Diagonal',
        '-o',
        diag_precond_nocouplings),

#    Benchmark(
#        'Diagonal (couplings)',
#        '-o',
#        diag_precond_couplings,
#        ),

    Benchmark(
        'Banded',
        '-o',
        cmbcr.BandedHarmonicPreconditioner(system, diagonal=False, couplings=False),
        ),
        
#    Benchmark(
#        'Banded (couplings)',
#        '-o',
#        cmbcr.BandedHarmonicPreconditioner(system, diagonal=False, couplings=True)),
    
    Benchmark(
        'Psuedo-inverse',
        '-o',
        cmbcr.PsuedoInversePreconditioner(system),
        #cmbcr.BlockPreconditioner(
        #    system,
        #    cmbcr.PsuedoInversePreconditioner(system),
        #    diag_precond_nocouplings,
        #)
        ),
        
    ]



    
    
## if sys.argv[1] == 'single':
##     benchmarks.extend([
##         Benchmark(
##             'Pixel',
##             '-o',
##             cmbcr.PixelPreconditioner(system, prior=False),
##         ),
##     ])

#clf()
#P = benchmarks[-1].preconditioner.P
#for i in range(P.shape[0]):
#    plot(P[i, 0, :])
#draw()
#1/0
    
#clf()
for bench in benchmarks:
    bench.ploterr()
legend()
draw()
