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
rms_treshold = 0


resultfile = sys.argv[3]


full_res_system = cmbcr.CrSystem.from_config(config, udgrade=nside, mask_eps=0.8, rms_treshold=rms_treshold)

full_res_system.prepare_prior()

system = cmbcr.downgrade_system(full_res_system, 1. / factor)

lmax_ninv = 2 * max(system.lmax_list)
rot_ang = (0, 0, 0)

system.set_params(
    lmax_ninv=lmax_ninv,
    rot_ang=rot_ang,

# setup1 : flat mixing without mixing mask
    flat_mixing=True,
    use_mixing_mask=False,##DONOTCHANGE

# setup2: 
##    flat_mixing=False,
##    use_mixing_mask=True,
    )

system.prepare_prior(scale_unity=False)
system.prepare(use_healpix=True, use_healpix_mixing=True, mixing_nside=nside)



# from cmbcr.mblocks import gauss_ring_map_to_phase_map

# from commander.compute.cr.sh_integrals import compute_approximate_Yt_D_Y_diagonal_mblock

# def compute_Yh_D_Y_diagonal(lmax, phase_map, thetas):
#     result = np.zeros((lmax + 1)**2)
#     idx = 0
#     for m in range(lmax + 1):
#         block = compute_real_Yh_D_Y_block(m, m, lmax, lmax, thetas, phase_map)
#         result[idx:idx + block.shape[0]] = block.diagonal()
#         idx += block.shape[0]
#     return result


# nu = 0
# from cmbcr.mblocks import compute_real_Yh_D_Y_block
# from cmbcr.healpix import nside_of
# from cmbcr import sharp

# nside = nside_of(system.ninv_maps[nu])
# lmax = 3 * nside
# alm = sharp.sh_analysis(2 * lmax, system.ninv_maps[nu])
# Ni_diag = np.zeros((lmax + 1)**2, np.double)
# #with timed('precond-diag-compute compute_Yh_D_Y_diagonal(drc3jj)'):
# #    compute_approximate_Yt_D_Y_diagonal_mblock(12*nside**2, 0, lmax, 0, lmax, alm, out=Ni_diag)
# #Ni_diag *= scatter_l_to_lm(system.bl_list[nu][:lmax + 1])**2

# with timed('precond-diag-compute compute_Yh_D_Y_diagonal'):
#     ninv_phase, thetas = gauss_ring_map_to_phase_map(system.ninv_gauss_lst[nu], system.lmax_ninv, lmax)
#     Ni_diag_p = compute_Yh_D_Y_diagonal(lmax, ninv_phase, thetas) * scatter_l_to_lm(system.bl_list[nu][:lmax + 1])**2
    

rng = np.random.RandomState(1)

x0 = [
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    * scatter_l_to_lm(np.sqrt(system.Cl_list[k]))
    for k in range(system.comp_count)
    ]

##x0[0][:] = 0
##x0[0][4] = 1
    
b = system.matvec([x0[0].copy()])
x0_stacked = system.stack(x0)

##bx = sharp.sh_synthesis(nside, b[0].copy())
##bpx = sharp.sh_synthesis(nside, system.matvec_scalar_mixing([x0[0].copy()])[0])

##1/0


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


        def mv(x):
            with timed('matvec'):
                return system.stack(system.matvec(system.unstack(x)))

        def pc(x):
            with timed('precond'):
                return system.stack(self.preconditioner.apply(system.unstack(x)))
            
        solver = cg_generator(
            mv,
            system.stack(b),
            x0=start_vec,
            M=pc
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

                with open(resultfile, 'a') as f:
                    f.write('{},{},{}\n'.format(self.label, i, err))
                
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
#    Benchmark(
#     'Fullsky diagonal',
#      '-o',
#     cmbcr.DiagonalPreconditioner2(system),
#    ),
#    Benchmark(
#     'diag-1',
#   '-o',
#     cmbcr.DiagonalPreconditioner2(system),
#    ),
#    Benchmark(
#     'Fullsky diagonal 2',
#      '-o',
#     cmbcr.DiagonalPreconditioner2(system),
#    ),

    Benchmark(
      'pseudo-inverse',
      '-o',
      cmbcr.PseudoInverseWithMaskPreconditioner(system, inner_its=0),
     ),
    ]

###save_benchmarks(benchmarks, resultfile, nside, rms_treshold)


fig = gcf()

for bench in benchmarks:
    bench.ploterr()
fig.gca().set_ylim((1e-10, 1e4))

legend()
show()
