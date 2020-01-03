//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Config/user_def2.m4: code to call user semirings
//------------------------------------------------------------------------------

GrB_Info GB_AxB_user
(
    const GrB_Desc_Value GB_AxB_method,
    const GrB_Semiring GB_s,

    GrB_Matrix *GB_Chandle,
    const GrB_Matrix GB_M,          // not yet used for saxpy3 method
    const GrB_Matrix GB_A,          // not used for dot2 method
    const GrB_Matrix GB_B,
    bool GB_flipxy,

    // for dot method only:
    const GrB_Matrix *GB_Aslice,    // for dot2 only
    int64_t *GB_RESTRICT GB_B_slice,   // for dot2 only
    const int GB_dot_nthreads,      // for dot2, dot3, and saxpy3
    const int GB_naslice,           // for dot2 only
    const int GB_nbslice,           // for dot2 only
    int64_t **GB_C_counts,          // for dot2 only

    // for dot3 and saxpy3 methods only:
    GB_void *GB_RESTRICT GB_TaskList,
    const int GB_ntasks
)
{
    GrB_Info GB_info = GrB_SUCCESS ;
    GB_semirings()
    return (GB_info) ;
}

