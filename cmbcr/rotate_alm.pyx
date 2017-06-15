from libc.stdint cimport int32_t, int8_t, int64_t

cdef extern:
    void rotate_alm_ "rotate_alm_cwrapper"(int32_t lmax, double *alm, double psi, double theta, double phi)


def rotate_alm(int32_t lmax, double[:] alm, double psi, double theta, double phi):
    rotate_alm_(lmax, &alm[0], psi, theta, phi)
