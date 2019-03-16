from . import sharp
import numpy as np

# Wrappers around SHTs in order to simulate approximate SHTs
eps = 0

def numericnoise(x):
    return x * np.random.uniform(1 - eps, 1 + eps, size=x.shape)

def sh_adjoint_synthesis(*args):
    return numericnoise(sharp.sh_adjoint_synthesis(*args))

def sh_adjoint_analysis(*args):
    return numericnoise(sharp.sh_adjoint_analysis(*args))

def sh_synthesis(*args):
    return numericnoise(sharp.sh_synthesis(*args))

def sh_analysis(*args):
    return numericnoise(sharp.sh_analysis(*args))
