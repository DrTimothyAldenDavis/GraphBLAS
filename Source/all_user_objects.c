//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Source/all_user_objects.c
//------------------------------------------------------------------------------

// This file is constructed automatically by cmake and m4 when GraphBLAS is
// compiled, from the Config/user_def*.m4 and *.m4 files in User/.  Do not edit
// this file directly.  It contains references to internally-defined functions
// and objects inside GraphBLAS, which are not user-callable.

#include "GB.h"
#include "GB_mxm.h"
#include "GB_user.h"

//------------------------------------------------------------------------------
// SuiteSparse/GraphBLAS/Config/user_def1.m4: define user-defined objects
//------------------------------------------------------------------------------


















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
    if (0)
    {
        ;
    }
    return (GB_info) ;
}

