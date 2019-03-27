//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Config/user_def2.m4: code to call user semirings
//------------------------------------------------------------------------------

GrB_Info GB_AxB_user
(
    const GrB_Desc_Value AxB_method,
    const GrB_Semiring s,

    GrB_Matrix *Chandle,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const GrB_Matrix B,
    bool flipxy,

    // for dot and dot2 methods only:
    const bool GB_mask_comp,

    // for heap method only:
    int64_t *restrict List,
    GB_pointer_pair *restrict pA_pair,
    GB_Element *restrict Heap,
    const int64_t bjnz_max,

    // for Gustavson's method only:
    GB_Sauna Sauna,

    // for dot2 method only:
    const int64_t *restrict C_count_start,
    const int64_t *restrict C_count_end
)
{
    GrB_Info GB_info = GrB_SUCCESS ;
    GB_semirings()
    return (GB_info) ;
}

