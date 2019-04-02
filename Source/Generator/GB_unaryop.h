void GB_unop
(
    GB_ctype *restrict Cx,
    GB_atype *restrict Ax,
    int64_t anz,
    int nthreads
) ;

void GB_tran
(
    int64_t *restrict Cp,
    int64_t *restrict Ci,
    GB_ctype *restrict Cx,
    const GrB_Matrix A
) ;

