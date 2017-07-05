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

from cmbcr import sharp
from cmbcr.beams import gaussian_beam_by_l

#mstart = 20
#mstop = 50

d = 0
#mstart_M = mstart + d
#mstop_M = mstop - d

nside = 128
nside_ninv = nside // 2


npix = 12*nside**2
#mask = np.ones(npix)
#mask[mstart_M*npix//100:mstop_M*npix//100] = 0
from healpy import mollzoom
#if 1:
#    L = 4 * nside
#    mlm = sharp.sh_analysis(L, mask)
#    mlm *= scatter_l_to_lm(gaussian_beam_by_l(L, '1 deg'))
#    mask = sharp.sh_synthesis(nside, mlm)
#    #mollzoom(mask)
#    #1/0

#    #mask[mask > 1] = 1
#    #mask[mask < 0] = 0
    
#    mask[mask > 0.1] = 1
#    mask[mask <= 0.1] = 0


#mask_hi = np.ones(12*nside_ninv**2)
#npix_hi = 12*nside_ninv**2
#mask_hi[mstart*npix_hi//100:mstop*npix_hi//100] = 0


full_res_system = cmbcr.CrSystem.from_config(config, mask_eps=0.05, udgrade=nside_ninv)
full_res_system.prepare_prior()



system = cmbcr.downgrade_system(full_res_system, 0.02)
print system.lmax_list, 'nside', nside, 'nside_ninv', nside_ninv



lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (0, 0, 0)
system.set_params(
    lmax_ninv=lmax_ninv,
    rot_ang=rot_ang,
    flat_mixing=False)
system.prepare_prior()
system.prepare(use_healpix=True)

from cmbcr import sharp
from healpy import mollzoom



#A = hammer(lambda x: system.stack(system.matvec(system.unstack(x))), system.x_offsets[-1])
#Ainv = np.linalg.inv(A)

lmax = system.lmax_list[0]


data_precond = cmbcr.PsuedoInversePreconditioner(system)

from cmbcr.multilevel import lstmul, lstadd, lstsub

def lstscale(alpha, lst):
    return [alpha * x for x in lst]

alpha = 1

    
from scipy.linalg import eigvalsh

def plotspec(m, q=None, label=None):
    semilogy(np.abs(eigvalsh(m, q)), label=label)

rng = np.random.RandomState(1)

x0 = [
    #scatter_l_to_lm(system.dl_list[k]) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)





class Benchmark(object):
    def __init__(self, label, style, preconditioner, n=70):
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
        try:
            for i, (x, r, delta_new) in enumerate(solver):
                print 'it', i
                if r0 is None:
                    r0 = np.linalg.norm(r)

                self.err_vecs.append([x0c - xc for x0c, xc in zip(x0, system.unstack(x))])
                err = np.linalg.norm(x - x0_stacked) / np.linalg.norm(x0_stacked)
                self.err_norms.append(err)
                self.reslst.append(np.linalg.norm(r) / r0)
                if err < 1e-8 or i >= n:
                    break
        except ValueError as e:
            if 'not positive-definite' not in str(e):
                raise
            pass # ignore positive-definite, just terminate iterations

        
        
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

    ## Benchmark(
    ##     'Banded',
    ##     '-o',
    ##     cmbcr.BandedHarmonicPreconditioner(system, diagonal=False, couplings=False),
    ##     ),
        
#    Benchmark(
#        'Banded (couplings)',
#        '-o',
#        cmbcr.BandedHarmonicPreconditioner(system, diagonal=False, couplings=True)),
    
    Benchmark(
        'Psuedo-inverse',
        '-o',
        PsuedoInvWithMaskPreconditioner(system),
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
gca().set_ylim((1e-8, 2))
legend()
draw()
