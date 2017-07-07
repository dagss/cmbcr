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

nside = 64
factor = 2048 // nside

full_res_system = cmbcr.CrSystem.from_config(config, udgrade=nside, mask_eps=0.8)

full_res_system.prepare_prior()

system = cmbcr.downgrade_system(full_res_system, 1. / factor)
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
    flat_mixing=False,
    )
system.prepare_prior()
system.prepare(use_healpix=True)


rng = np.random.RandomState(1)

def load_Cl_cmb(lmax, filename='camb_11229992_scalcls.dat'):
    #dat = np.loadtxt()
    dat = np.loadtxt(filename)
    assert dat[0,0] == 0 and dat[1,0] == 1 and dat[2,0] == 2
    Cl = dat[:, 1][:lmax + 1]
    ls = np.arange(2, lmax + 1)
    Cl[2:] /= ls * (ls + 1) / 2 / np.pi
    Cl[0] = Cl[1] = Cl[2]
    return Cl
Cl_cmb = load_Cl_cmb(10000)


x0 = [
    #scatter_l_to_lm(1. / system.dl_list[k]) *
    #scatter_l_to_lm(np.sqrt(Cl_cmb[:system.lmax_list[k] + 1])) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)





class Benchmark(object):
    def __init__(self, label, style, preconditioner, n=30):
        self.label = label
        self.style = style
        self.preconditioner = preconditioner
        self.benchmark(n=n)

    def benchmark(self, n):
        r0 = None
        self.err_vecs = []
        self.err_norms = []
        self.reslst = []

        if hasattr(self.preconditioner, 'starting_vector'):
            start_vec = system.stack(self.preconditioner.starting_vector(b))
        else:
            start_vec = np.zeros_like(x0_stacked)
        
        solver = cg_generator(
            lambda x: system.stack(system.matvec(system.unstack(x))),
            system.stack(b),
            x0=start_vec,
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
            if 'positive-definite' in str(e):
                print str(e)
            else:
                raise

        
        
    def ploterr(self):
        fig3.gca().semilogy(self.err_norms, self.style, label=self.label)

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
    
method = 'add'

if 'op' in sys.argv:
    #p = cmbcr.PsuedoInversePreconditioner(system)
    p = cmbcr.PsuedoInverseWithMaskPreconditioner(system, method=method)
    
    def op(i):
        u = np.zeros(12*nside**2)
        u[i] = 1
        #return u + 1e-1
        alm = sharp.sh_analysis(system.lmax_list[0], u)
        #alm = system.matvec([alm])[0]
        alm = p.apply([alm])[0]
        return sharp.sh_adjoint_analysis(nside, alm)


    def doit(i):
        clf()
        mollzoom(np.log10(np.abs(op(i))), fig=gcf().number)
        draw()
        
    #mollzoom(op(0))
    #draw()
    doit(6*nside**2 + 2 * nside)
    1/0
    
#diag_precond_nocouplings = cmbcr.BandedHarmonicPreconditioner(system, diagonal=True, couplings=False)

benchmarks = [
    #Benchmark(
    #    'Diagonal',
    #    '-o',
    #    diag_precond_nocouplings),
    Benchmark(
        'Diagonal',
        '-o',
        cmbcr.DiagonalPreconditioner(system)),

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
        'Psuedo-inverse (nomask)',
        '-o',
        cmbcr.PsuedoInversePreconditioner(system),
        ),
        
    Benchmark(
        'Psuedo-inverse (method)',
        '-o',
        cmbcr.PsuedoInverseWithMaskPreconditioner(system, method=method),
        ),
    ]

#clf()
#fig1.clear()
#fig2.clear()
#fig3.clear()

#if 'maps' in sys.argv:
#ma = 100
#mi = -ma
#mollview(sharp.sh_synthesis(nside, benchmarks[1].err_vecs[15][0]), fig=fig1.number, min=mi, max=ma)
#mollview(sharp.sh_synthesis(nside, benchmarks[2].err_vecs[15][0]), fig=fig2.number, min=mi, max=ma)
#draw()
#    1/0
    
    
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

fig3 = gcf()
clf()
    
for bench in benchmarks:
    bench.ploterr()
fig3.gca().set_ylim((1e-8, 1e4))


ion()
#fig3.legend()
#fig1.draw()
#fig2.draw()
#fig3.draw()
