GrB_Info GB_unop_apply
(
    GB_ctype *Cx,
    const GB_atype *Ax,
    const int8_t *GB_RESTRICT Ab,
    int64_t anz,
    int nthreads
) ;

GrB_Info GB_unop_tran
(
    GrB_Matrix C,
    const GrB_Matrix A,
    int64_t *GB_RESTRICT *Rowcounts,
    const int64_t *GB_RESTRICT A_slice,
    int naslice,
    int nthreads
) ;

