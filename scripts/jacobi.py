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
from scipy.sparse import csc_matrix
#reload(cmbcr.main)

import sys
from cmbcr.cg import cg_generator

config = cmbcr.load_config_file('input/{}.yaml'.format(sys.argv[1]))


def csc_neighbours(nside, pick):
    # The returned matrix will be npix-times-npix, but have zero entries outside the mask
    pixels = pick.nonzero()[0]
    pixels_reverse = np.zeros(pick.shape, dtype=int)
    pixels_reverse[pixels] = np.arange(pixels.shape[0])

    length = pixels.shape[0]
    indices = np.zeros(9 * length, dtype=np.int)
    indptr = np.zeros(12 * nside**2 + 1, dtype=np.int)
    neighbours = healpy.get_all_neighbours(nside, pixels, nest=True)

    npix = 12 * nside**2
    
    idx = 0

    for j in range(npix):
        indptr[j] = idx

        # if columns is outside mask it is just entirely empty, no elements stored
        if pick[j]:
            # column is inside mask
            k = pixels_reverse[j]
            neighlst = neighbours[:, k]
            neighlst = neighlst[(neighlst != -1) & pick[neighlst]]
            n = neighlst.shape[0]
            indices[idx] = j
            #i_arr = pixels_reverse[neighlst]
            indices[idx + 1:idx + 1 + n] = neighlst
            idx += n + 1

    indptr[-1] = idx
    indices = indices[:idx]
    data = np.ones(idx)
    return csc_matrix((data, indices, indptr), shape=(npix, npix))


def make_Si_sparse_matrix(Si_pattern, dl, ridge):

    data = np.zeros_like(Si_pattern.data)
    for j in range(Si_pattern.shape[1]):
        i_arr = Si_pattern.indices[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]]

        if len(i_arr):
            # k is the offset of the diagonal entry
            k = (i_arr == j).nonzero()[0][0]
            
            x, y, z = healpy.pix2vec(nside, i_arr, nest=True)
            data[Si_pattern.indptr[j]:Si_pattern.indptr[j + 1]] = cmbcr.beam_by_cos_theta(
                dl,
                (x[k] * x + y[k] * y + z[k] * z))
            diag_ind = Si_pattern.indptr[j] + k
            data[diag_ind] += ridge

    return csc_matrix((data, Si_pattern.indices, Si_pattern.indptr), shape=Si_pattern.shape)



w = 1

nside = 64 * w
factor = 2048 // nside * w


def padvec(u):
    x = np.zeros(12 * nside**2)
    x[pick] = u
    return x

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


ridge_factor = 5e-3 #5e-3

dl = system.dl_list[0]

#
# SHOULD INTRODUCE V-CYCLE TO SEE IF THAT HELPS>...
#


# WORKS BEST:
nl = cmbcr.standard_needlet_by_l(1.5, 2 * dl.shape[0] - 1)
i = nl.argmax()
dl = np.concatenate([dl, nl[i:] * dl[-1] / nl[i]])

lmax = dl.shape[0] - 1


from cmbcr.precond_psuedoinv import *

x = lstscale(0, b)


mask_p = healpy.ud_grade(system.mask, nside, order_in='RING', order_out='NESTED', power=0)
mask_p[mask_p != 1] = 0
pick = (mask_p == 0)
n = int(pick.sum())


one_minus_mask = pick

## 1/0

z = np.zeros_like(mask_p)
z[0] = 1
z = sharp.sh_adjoint_synthesis(lmax, z)
z *= scatter_l_to_lm(dl)
z = sharp.sh_synthesis(nside, z)
estimated_max = z[0]

ridge = ridge_factor * estimated_max #5e-3 * estimated_max
#ridge = 0

Si_pattern = csc_neighbours(nside, pick)
Si_pattern = Si_pattern * Si_pattern * Si_pattern * Si_pattern
Si_pattern.sum_duplicates()
Si_sparse = make_Si_sparse_matrix(Si_pattern, dl, ridge)

#Si_sparse.data[:] = 0
#Si_sparse.data[Si_sparse.indptr[:-1]] = 1e10 #+= ridge

call_count = 0


def Si(u):
    return u_pad


def YZ_Si_YZ(u_in):
    global call_count
    call_count += 1

    u = padvec(u_in)
    u = healpy.reorder(u, n2r=True)
    u = sharp.sh_adjoint_synthesis(lmax, u)
    u *= scatter_l_to_lm(dl)
    u = sharp.sh_synthesis(nside, u)
    u = healpy.reorder(u, r2n=True)
    return u[pick] + u_in * ridge


#Si_dense = Si_sparse.toarray()


if 0:

    i = 200

    u = np.zeros(n)
    u[i] = 1

    clf()
    mollview(padvec(YZ_Si_YZ(u)), sub=311, nest=True)
    sp = Si_sparse.toarray()[:, pick.nonzero()[0][i]]
    mollview(sp, sub=312, nest=True)
    mollview(padvec(YZ_Si_YZ(u)) - sp, sub=313, nest=True)
    draw()
    1/0




if 1:
    Si_dense = Si_sparse.toarray()

    Si_dense = Si_dense[pick, :][:, pick]

    print 'inverting'
    Si_inv = np.linalg.inv(Si_dense)
    print 'done inverting'
    
    def mask_inv(x):
        return padvec(np.dot(Si_inv, x[pick]))
    

elif 1:

    Q = hammer(YZ_Si_YZ, int(pick.sum()))
    #Q += np.eye(Q.shape[0]) * Q.max() * 5e-2
    Qinv = np.linalg.inv(Q)   #, rcond=1e-3) ##, rcond=1e-3)

    def mask_inv(x):
        return padvec(np.dot(Qinv, x[pick]))
    
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
            u = sharp.sh_analysis(lmax, u_pad)
            u *= scatter_l_to_lm(1. / dl)
            u_pad = sharp.sh_adjoint_analysis(nside, u)
            return u_pad[pick]
    Qinv = hammer(Sinv, int(pick.sum()))

precond_1 = cmbcr.PsuedoInversePreconditioner(system)


def precond_mask(u_lst):
    x = sharp.sh_synthesis(nside, u_lst[0])
    x = healpy.reorder(x, r2n=True)

    x = mask_inv(x)
    #u_pad[pick] = mask_inv(x[pick])

    x = healpy.reorder(x, n2r=True)
    x = sharp.sh_adjoint_synthesis(system.lmax_list[0], x)
    
    return [x]

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
    if i > 40:
        break

#clf()
semilogy(errlst, '-o')
draw()
