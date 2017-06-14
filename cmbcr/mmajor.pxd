from libc.stdint cimport int64_t as index_t

cpdef index_t lm_to_idx(index_t lmin,
                        index_t lmax,
                        index_t mmin,
                        index_t l,
                        index_t m) nogil

cdef void idx_to_lm_fast(index_t lmin,
                         index_t lmax,
                         index_t mmin,
                         index_t idx,
                         index_t *out_l,
                         index_t *out_m) nogil
