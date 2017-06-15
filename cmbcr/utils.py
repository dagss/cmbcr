import numpy as np
import contextlib
from matplotlib import pyplot as plt
from time import time as wall_time


def format_duration(dt):
    if dt >= 1:
        unit = 's'
    elif dt >= 1e-3:
        unit = 'ms'
        dt *= 1e3
    elif dt >= 1e-6:
        unit = 'us'
        dt *= 1e6
    else:
        unit = 'ns'
        dt *= 1e9
    return '%.1f %s' % (dt, unit)


class TimingResults(object):
    pass


@contextlib.contextmanager
def timed(msg, stream=None): # for interactive use
    if msg is not None:
        if stream is None:
            from sys import stderr as stream
        stream.write(msg + '... ')
        stream.flush()
    t0 = wall_time()
    result = TimingResults()
    yield result
    dt = wall_time() - t0
    result.dt = dt
    if msg is not None:
        stream.write('done in %s\n' % format_duration(dt))
        stream.flush()


def hammer(matvec_func, n, m=None):
    m = n if m is None else m
    u = np.zeros(m)
    out = np.zeros((n, m))
    for i in range(m):
        if i < 100 or i % 100 == 0:
            print i, m
        u[i] = 1
        out[:, i] = matvec_func(u)
        u[i] = 0
    return out


def unitvec(n, i):
    r = np.zeros(n)
    r[i] = 1
    return r


def mshow(x):
    plt.clf()
    plt.imshow(x, interpolation='nearest')
    plt.colorbar()
    plt.draw()


def pad_or_trunc(x, n):
    if x.shape[0] >= n:
        return x[:n]
    else:
        r = np.zeros(n)
        r[:x.shape[0]] = x
        return r
