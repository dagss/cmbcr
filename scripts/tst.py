
alpha = 1/600.

def op(x):
    u = system.plan_ninv.synthesis(x)
    u *= system.ninv_gauss_lst[0] * alpha
    u = system.plan_ninv.adjoint_synthesis(u)
    return u

Ni = hammer(op, (31 + 1)**2)

clf(); plot(Ni.diagonal()); draw()
