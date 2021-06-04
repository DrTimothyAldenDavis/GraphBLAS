//------------------------------------------------------------------------------
// GB_Adot4B:  hard-coded dot4 method for a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If this file is in the Generated/ folder, do not edit it (auto-generated).

if_not_any_pair_semiring
GB_AxB_defs_include
ifndef_compact

//------------------------------------------------------------------------------
// C+=A'*B: dense dot product
//------------------------------------------------------------------------------

GrB_Info GB (_Adot4B)
(
    GrB_Matrix C,
    const GrB_Matrix A, bool A_is_pattern,
    int64_t *restrict A_slice, int naslice,
    const GrB_Matrix B, bool B_is_pattern,
    int64_t *restrict B_slice, int nbslice,
    const int nthreads
)
{ 
    if_disabled
    return (GrB_NO_VALUE) ;
    #else
    #include "GB_AxB_dot4_meta.c"
    return (GrB_SUCCESS) ;
    #endif
}

#endif
#endif

