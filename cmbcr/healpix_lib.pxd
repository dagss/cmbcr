from libc.stdint cimport int32_t, int8_t

cdef extern:
    void proj_mollw "proj_mollw_cwrapper"(float *pmap, int32_t nside, int32_t ordering,
                                          int32_t xsize, double lon0, double lat0, float *pimage)

    void pix2ang_ring(int32_t nside, int32_t ipix, double *theta, double *phi)
    void pix2vec_ring(int32_t nside, int32_t ipix, double vec[3])

    void convert_nest2ring_double_1d(int32_t nside, double *map)
