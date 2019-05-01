if_is_monoid
void GB_red_scalar
(
    GB_atype *result,
    const GrB_Matrix A,
    int nthreads
) ;
endif_is_monoid

void GB_bild
(
    GB_atype *restrict Tx,
    int64_t  *restrict Ti,
    const GB_atype *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict iwork,
    const int64_t *restrict kwork,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
) ;

