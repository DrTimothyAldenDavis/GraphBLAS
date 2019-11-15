GrB_Info GB_unop
(
    GB_ctype *GB_RESTRICT Cx,
    const GB_atype *GB_RESTRICT Ax,
    int64_t anz,
    int nthreads
) ;

GrB_Info GB_tran
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *GB_RESTRICT *Rowcounts,
    GBI_single_iterator Iter,
    const int64_t *GB_RESTRICT A_slice,
    int naslice
) ;

