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
reload(cmbcr.masked_solver)
reload(cmbcr)
from cmbcr.utils import *

from cmbcr import sharp
from healpy import mollzoom, mollview

#reload(cmbcr.main)

import sys
from cmbcr.cg import cg_generator

config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))


w = 1

nside = 128 * w
factor = 2048 // nside * w
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

if 'plot' in sys.argv:
    clf()
    system.plot()
    1/0

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
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)





class Benchmark(object):
    def __init__(self, label, style, preconditioner, n=80):
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
                ##if hasattr(self.preconditioner, 'v_end'):
                ##    x = self.preconditioner.v_end(b, x)

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
            #if 'positive-definite' in str(e):
            print str(e)
        except AssertionError as e:
            raise
            print str(e)

            #else:
            #    raise


    def ploterr(self):
        fig3.gca().semilogy(self.err_norms, self.style, label=self.label)
        #fig3.gca().semilogy(self.sht_counts, self.err_norms, self.style, label=self.label)

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


if 'plot' in sys.argv:
    clf()
    system.plot()
    1/0

if 'pixmat' in sys.argv:
    pdia = cmbcr.DiagonalPreconditioner(system)
    #1/0
    s = np.sqrt(pdia.M_lst[0])
    def op(u):
        alm = sharp.sh_analysis(system.lmax_list[0], u) * s
        alm = system.matvec([alm])[0]
        return sharp.sh_adjoint_analysis(nside, alm * s)
    A = hammer(op, 12 * nside**2)
    s = 1 / np.sqrt(A.diagonal())
    A *= s[:, None]
    A *= s[None, :]
    clf()
    imshow(A, interpolation='none')
    draw()
    1/0

if 'eig' in sys.argv:
    p = cmbcr.PsuedoInversePreconditioner(system)
    A = hammer(lambda x: system.stack(system.matvec(system.unstack(x))), system.x_offsets[-1])
    M = hammer(lambda x: system.stack(p.apply(system.unstack(x))), system.x_offsets[-1])
    clf()
    from scipy.linalg import eigvalsh, eigvals
    semilogy(np.abs(eigvalsh(A)), '-', label='A')
    semilogy(np.abs(eigvalsh(M))[::-1], '-', label='M')
    semilogy(sorted(np.abs(eigvals(np.dot(M, A)))), '-', label='MA^-1')
    gca().set_ylim((1e-5, 1e8))
    legend()
    draw()
    1/0

if 'op' in sys.argv:
    #p = cmbcr.PsuedoInversePreconditioner(system)

    method = 'v2'

    p = cmbcr.PsuedoInverseWithMaskPreconditioner(system, method=method)

    from cmbcr import beams
    lmax = system.lmax_list[0]
    #sl = 1 / beams.gaussian_beam_by_l(lmax, '30 deg')
    sl = np.ones(lmax + 1)
    slm = scatter_l_to_lm(sl)

    def op(f, i):
        u = np.zeros(12*nside**2)
        u[i] = 1
        #return u + 1e-1
        alm = sharp.sh_analysis(system.lmax_list[0], u) * slm
        #alm = system.matvec([alm])[0]
        alm = f([alm])[0]
        return sharp.sh_adjoint_analysis(nside, alm * slm)

    #clf()
    #plot(op(system.matvec, 8*nside**2))
    #draw()
    #1/0

    def doit(i):
        clf()
        u = op(p.apply, i)
        #u = np.log10(np.abs(u))
        mollview(u, fig=gcf().number, sub=211)

        def q(x):
            if unity:
                x = [x[0] * scatter_l_to_lm(1 / system.wl_list[0])]
                x = system.matvec(x)
                x = [x[0] * scatter_l_to_lm(1 / system.wl_list[0])]
                return x
            else:
                x = system.matvec(x)
            return x

        u = op(q, i)
        #u = np.log10(np.abs(u))
        mollview(u, fig=gcf().number, sub=212)
        draw()

    t = 10
    if 'up' in sys.argv:
        doit(6*nside**2 + 2 * nside - t * 4 * nside)
    else:
        doit(6*nside**2 + 2 * nside + t * 4 * nside)
    1/0

#diag_precond_nocouplings = cmbcr.BandedHarmonicPreconditioner(system, diagonal=True, couplings=False)

benchmarks = [
    Benchmark(
     'Diagonal',
      '-o',
     cmbcr.DiagonalPreconditioner2(system),
    ),

    ## Benchmark(
    ##     'Psuedo-inverse (5 inner its)',
    ##     '-o',
    ##     cmbcr.PsuedoInverseWithMaskPreconditioner(system, inner_its=5),
    ##     ),
    Benchmark(
        'Psuedo-inverse (inner V-cycle)',
        '-o',
        cmbcr.PsuedoInverseWithMaskPreconditioner(system, inner_its=0),
        ),
    ]


save_benchmarks(benchmarks, '/home/dagss/writing/pseudoinv/results/{}_{}_{}_v4.yaml'.format(sys.argv[1], nside, rms_treshold))


fig3 = gcf()
#clf()

for bench in benchmarks:
    bench.ploterr()
fig3.gca().set_ylim((1e-10, 1e4))
#fig3.gca().set_xlim((0, 800))

legend()
ion()


#fig3.legend()
#fig1.draw()
#fig2.draw()
#fig3.draw()
