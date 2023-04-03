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
#define GB_A_TYPE GxB_FC32_t
#define GB_C_TYPE GxB_FC32_t
#define GB_DECLAREC(cwork) GxB_FC32_t cwork
#define GB_COPY_aij_to_cwork(cwork,Ax,pA,A_iso) cwork = Ax [A_iso ? 0 : (pA)]
#define GB_COPY_aij_to_C(Cx,pC,Ax,pA,A_iso,cwork) Cx [pC] = (A_iso) ? cwork : Ax [pA]
#define GB_COPY_scalar_to_C(Cx,pC,cwork) Cx [pC] = cwork
#define GB_AX_MASK(Ax,pA,asize) GB_MCAST (Ax, pA, sizeof (GxB_FC32_t))

// disable this operator and use the generic case if these conditions hold
#define GB_DISABLE \
    (GxB_NO_FC32)

#include "GB_assign_shared_definitions.h"

//------------------------------------------------------------------------------
// C<M> = scalar, when C is dense
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_05d__fc32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_struct,
    const GB_void *scalar,      // of type C->type
    GB_Werk Werk
)
{ 
    #if GB_DISABLE
    return (GrB_NO_VALUE) ;
    #else
    GB_C_TYPE cwork = (*((GB_C_TYPE *) scalar)) ;
    #include "GB_subassign_05d_template.c"
    return (GrB_SUCCESS) ;
    #endif
}

//------------------------------------------------------------------------------
// C<A> = A, when C is dense
//------------------------------------------------------------------------------

GrB_Info GB (_subassign_06d__fc32)
(
    GrB_Matrix C,
    const GrB_Matrix A,
    const bool Mask_struct,
    GB_Werk Werk
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

GrB_Info GB (_subassign_25__fc32)
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const GrB_Matrix A,
    GB_Werk Werk
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
