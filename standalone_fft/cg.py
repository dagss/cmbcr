"""Preconditioned conjugate gradients
"""
import numpy as np
import logging

class ConvergenceError(RuntimeError):
    pass


def ndnorm(arr, axis, ord=None):
    if ord is None:
        # 2-norm
        x = arr * arr
        return np.sum(x, axis=axis)
    elif np.isinf(ord):
        # max(abs(s))
        return np.max(np.abs(arr), axis=axis)
        raise NotImplementedError()
    else:
        raise NotImplementedError('unsupported norm order')

def cg(A, b, preconditioner=None, x0=None, logger=None,
       eps=1e-8, stop_rule='residual', maxit=1000):
    it = stopping_cg_generator(A, b, preconditioner, x0, logger, eps, stop_rule, maxit)

    for x, r, delta, info in it:
        pass

    return x, info

def stopping_cg_generator(A, b, preconditioner=None, x0=None, logger=None,
                          eps=1e-8, stop_rule='residual', maxit=1000):

    it = cg_generator(A, b, preconditioner, x0, logger)
    x, r, delta, info = it.next()
    
    if stop_rule == 'preconditioned_residual':
        stop_treshold = eps**2 * delta
        stop_msg = u'(stop at delta < %.2e)' % np.sqrt(stop_treshold)
    elif stop_rule == 'residual':
        stop_treshold = eps**2 * np.dot(r.T, r)[0, 0]
        stop_msg = u'(stop at norm(r) < %.2e)' % np.sqrt(stop_treshold)
    
    for k in range(maxit):
        x, r, delta, info = it.next()
        if stop_rule == 'preconditioned_residual':
            stop_measure = delta
        elif stop_rule == 'residual':
            stop_measure = np.dot(r.T, r)[0, 0]
        
        logger.info('%5d: %.2e %s', k,
                    np.sqrt(stop_measure), stop_msg)

        yield tuple((x, r, delta, info))
        
        if stop_measure < stop_treshold:
            return

    raise ConvergenceError("Did not converge in %d iterations" % maxit)

def cg_generator(A, b, M=lambda x: x, M2=lambda x: x, M3=lambda x: x, x0=None):
    """
   

    """

    if x0 is None:
        x0 = np.zeros(b.shape, dtype=b.dtype, order='F')

    info = {}

    # Terminology/variable names follow Shewchuk, ch. B3
    #  r - residual
    #  d - preconditioned residual, "P r"
    #  
    # P = inv(M)
    r = b - A(x0)

    d = M(r)
    delta_0 = delta_new = np.dot(r, d)

    x = x0
    k = 0
    while True: # continue forever; caller is responsible for stopping to use generator
        yield x, r, delta_new

        q = M3(A(d))
        dAd = np.dot(d, q)
        if not np.isfinite(dAd):
            raise AssertionError("conjugate_gradients: A * d yielded inf values")
        if dAd == 0:
            raise AssertionError("conjugate_gradients: A is singular")
        alpha = delta_new / dAd
        x += alpha * d
        r -= alpha * q
        #if k > 0 and k % 50 == 0:
        #    r_est = r
        #    r = b - A(x)
        #    logging.info('Recomputing residual, relative error in estimate: %e',
        #                np.linalg.norm(r - r_est) / np.linalg.norm(r))

        s = M(r)
        delta_old = delta_new
        delta_new = np.dot(r, s)
        if delta_new < 0:
            raise ValueError('Preconditioner is not positive-definite: delta_new={}'.format(delta_new))
        beta = delta_new / delta_old
        d = M2(s) + beta * d
        k += 1
