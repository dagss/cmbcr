import numpy as np
from .utils import pad_or_truncate_alm
from .mmajor import scatter_l_to_lm
from . import sharp


def lstmul(a, b):
    return [ax * bx for ax, bx in zip(a, b)]

def lstadd(a, b):
    return [ax + bx for ax, bx in zip(a, b)]

def lstsub(a, b):
    return [ax - bx for ax, bx in zip(a, b)]


class BlockPreconditioner(object):
    def __init__(self, system, precond_lo, precond_hi):
        self.system = system
        self.precond_lo = precond_lo
        self.precond_hi = precond_hi

        def make_filter(k):
            lfilter = np.ones(system.lmax_list[k] + 1)
            lfilter[system.prior_list[k].lcross:] = 0
            return scatter_l_to_lm(lfilter)

        self.lo_filters = [make_filter(k) for k in range(system.comp_count)]
        self.hi_filters = [1 - f for f in self.lo_filters]

    def apply(self, b_lst):
        ## if self.mg:
        ##     x_lst = self.hi_l_precond.apply(b_lst)
            
        ##     r_lst = lstsub(b_lst, self.system.matvec(x_lst))
        ##     x_lst = lstadd(x_lst, self.apply_psuedo_inverse(r_lst))
            
        ##     r_lst = lstsub(b_lst, self.system.matvec(x_lst))
        ##     x_lst = lstadd(x_lst, self.hi_l_precond.apply(r_lst))
            
        ##     return x_lst
        ## else:
        M_lo_b = lstmul(self.lo_filters, self.precond_lo.apply(lstmul(self.lo_filters, b_lst)))
        M_hi_b = lstmul(self.hi_filters, self.precond_hi.apply(lstmul(self.hi_filters, b_lst)))
        return lstadd(M_lo_b, M_hi_b)
