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
#define GB_A_TYPE int64_t
#define GB_C_TYPE int64_t
#define GB_DECLAREC(cwork) int64_t cwork
#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso) cwork = Ax [A_iso ? 0 : (pA)]
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork) Cx [pC] = (A_iso) ? cwork : Ax [pA]
#define GB_COPY_scalar_to_C(Cx,pC,cwork) Cx [pC] = cwork
#define GB_AX_MASK(Ax,pA,asize) (Ax [pA] != 0)

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_INT64)

#include "GB_subassign_shared_definitions.h"

#undef  GB_FREE_ALL 
#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// C<M> = scalar, when C is dense
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_05d__int64)
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

GrB_Info GB (_subassign_06d__int64)
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

GrB_Info GB (_subassign_25__int64)
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

