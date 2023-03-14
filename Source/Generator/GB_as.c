//------------------------------------------------------------------------------
// GB_as:  assign/subassign kernels with no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C(I,J)<M> = A

#include "GB.h"
#include "GB_control.h"
#include "GB_ek_slice.h"
#include "GB_as__include.h"

// A and C matrices
GB_atype
GB_ctype
GB_declarec
GB_copy_aij_to_cwork
GB_copy_aij_to_c
GB_copy_scalar_to_c
GB_ax_mask

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    GB_disable

#include "GB_subassign_shared_definitions.h"

#undef  GB_FREE_ALL 
#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// C<M> = scalar, when C is dense
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_05d)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GB_void *p_cwork,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_C_TYPE cwork = (*((GB_C_TYPE *) p_cwork)) ;
    #include "GB_subassign_05d_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<A> = A, when C is dense
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_06d)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const bool Mask_struct,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (C->type == A->type) ;
    #include "GB_subassign_06d_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<M> = A, when C is empty and A is dense
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_25)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    const int64_t *M_ek_slicing,
    const int M_ntasks,
    const int M_nthreads
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    ASSERT (C->type == A->type) ;
    #include "GB_subassign_25_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

