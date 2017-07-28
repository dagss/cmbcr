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

nside = 16 * w
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


rng = np.random.RandomState(1)

x0 = [
    scatter_l_to_lm(1. / system.dl_list[k]) *
    rng.normal(size=(system.lmax_list[k] + 1)**2).astype(np.float64)
    for k in range(system.comp_count)
    ]
b = system.matvec(x0)
x0_stacked = system.stack(x0)


from cmbcr.precond_psuedoinv import *

x = lstscale(0, b)


mask_p = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='RING', power=0)
mask_p[mask_p != 1] = 0
pick = (mask_p == 0)
n = int(pick.sum())

z = np.zeros_like(mask_p)
z[0] = 1
z = sharp.sh_adjoint_synthesis(system.lmax_list[0], z)
z *= scatter_l_to_lm(system.dl_list[0])
z = sharp.sh_synthesis(nside, z)
estimated_max = z[0]

ridge = 5e-2 * estimated_max


call_count = 0

def YZ_Si_YZ(u_in):
    global call_count
    call_count += 1
    #for k in range(system.comp_count):
    if 1:
        u_pad = np.zeros_like(mask_p)
        u_pad[pick] = u_in
        u = sharp.sh_adjoint_synthesis(system.lmax_list[0], u_pad)
        u *= scatter_l_to_lm(system.dl_list[0])
        u_pad = sharp.sh_synthesis(nside, u)
        return u_pad[pick] + u_in * ridge

if 0:

    Q = hammer(YZ_Si_YZ, int(pick.sum()))
    Q += np.eye(Q.shape[0]) * Q.max() * 5e-2
    Qinv = np.linalg.inv(Q)   #, rcond=1e-3) ##, rcond=1e-3)

    def mask_inv(x):
        return np.dot(Qinv, x)
    
elif 1:
    from scipy.sparse.linalg import cg, LinearOperator
    def mask_inv(b):
        global call_count
        call_count = 0
        x, info = cg(LinearOperator((n, n), YZ_Si_YZ), b=b, tol=1e-300, maxiter=40)
        print call_count
        return x
    
else:
    def Sinv(u):
        #for k in range(system.comp_count):
        if 1:
            u_pad = np.zeros_like(mask_p)
            u_pad[pick] = u
            u = sharp.sh_analysis(system.lmax_list[0], u_pad)
            u *= scatter_l_to_lm(1. / system.dl_list[0])
            u_pad = sharp.sh_adjoint_analysis(nside, u)
            return u_pad[pick]
    Qinv = hammer(Sinv, int(pick.sum()))

precond_1 = cmbcr.PsuedoInversePreconditioner(system)


def precond_mask(u_lst):
    u_pad = np.zeros_like(mask_p)
    u_pad[pick] = mask_inv(sharp.sh_synthesis(nside, u_lst[0])[pick])
    return [sharp.sh_adjoint_synthesis(system.lmax_list[0], u_pad)]

def precond_both(b):
    x = precond_1.apply(b)

    if 1:
        x = lstadd(x, precond_mask(b))
        return x
    else:
        r = lstsub(b, system.matvec(x))
        x = lstadd(x, precond_mask(r))

        r = lstsub(b, system.matvec(x))
        x = lstadd(x, precond_1.apply(r))

        return x

    

precond = cmbcr.PsuedoInverseWithMaskPreconditioner(system, method='add1')

if 0:
    M = hammer(lambda x: system.stack(precond_both(system.unstack(x))), system.x_offsets[-1])
    #M = hammer(lambda x: system.stack(precond.apply(system.unstack(x))), system.x_offsets[-1])
    A = hammer(lambda x: system.stack(system.matvec(system.unstack(x))), system.x_offsets[-1])
    
    from scipy.linalg import eigvalsh, eigvals, eig

    
    #semilogy(np.abs(eigvalsh(A)), '-', label='A')
    #semilogy(np.abs(eigvalsh(M))[::-1], '-', label='M')

    w, vr = eig(np.dot(M, A))
    i = w.argsort()
    w = w[i]
    vr = vr[:,i]
    semilogy(w, '-o')
    1/0
    
    semilogy(sorted(np.abs(eigvals(np.dot(M, A)))), '-', label='MA^-1')
    #gca().set_ylim((1e-5, 1e8))
    legend()
    draw()
    1/0
#A = hammer(


errlst = []
x_its = []
err_its = []
norm0 = np.linalg.norm(x0_stacked)

err_its.append(x0)

solver = cg_generator(
    lambda x: system.stack(system.matvec(system.unstack(x))),
    system.stack(b),
    #x0=start_vec,
    M=lambda x: system.stack(precond_both(system.unstack(x))),
    )

for i, (x, r, delta_new) in enumerate(solver):
    x = system.unstack(x)

#for i in range(10):
#    r = lstsub(b, system.matvec(x))
#    x = lstadd(x, lstscale(1, precond_both(r)))

    errvec = lstsub(x0, x)
    err_its.append(errvec)
    x_its.append(x)
    
    errlst.append(np.linalg.norm(system.stack(errvec)) / norm0)

    print 'it', i
    if i > 20:
        break

#clf()
semilogy(errlst, '-o')
draw()
