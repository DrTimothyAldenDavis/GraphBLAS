//------------------------------------------------------------------------------
// GB_select_generic_phase2.c: count entries for C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse

#define GB_DEBUG

#include "GB_select.h"
#include "GB_ek_slice.h"

GrB_Info GB_select_generic_phase2
(
    int64_t *restrict Ci,
    GB_void *restrict Cx,
    const int64_t *restrict Cp,
    const int64_t *restrict Cp_kfirst,
    const GrB_Matrix A,
    const bool flipij,
    const GB_void *restrict ythunk,
    const GrB_IndexUnaryOp op,
    const int64_t *A_ek_slicing,
    const int A_ntasks,
    const int A_nthreads
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_OPCODE_IS_POSITIONAL (opcode)) ;
    ASSERT (!A_iso) ;
    ASSERT ((opcode >= GB_VALUENE_idxunop_code &&
             opcode <= GB_VALUELE_idxunop_code)
         || (opcode == GB_NONZOMBIE_idxunop && !A_iso)) ;

    //--------------------------------------------------------------------------
    // phase2: generic entry selector
    //--------------------------------------------------------------------------

    GB_Type_code zcode = op->ztype->code ;
    GB_Type_code xcode = op->xtype->code ;
    GB_Type_code acode = A->type->code ;
    size_t zsize = op->ztype->size ;
    size_t xsize = op->xtype->size ;
    GxB_index_unary_function fkeep = op->idxunop_function ;
    GB_cast_function cast_Z_to_bool, cast_A_to_X ;

    #define GB_ENTRY_SELECTOR
    #define GB_A_TYPE GB_void

    // Cx [pC] = Ax [pA], no typecast
    #undef  GB_SELECT_ENTRY
    #define GB_SELECT_ENTRY(Cx,pC,Ax,pA)                                \
        memcpy (Cx +((pC)*asize), Ax +((pA)*asize), asize)

    if (opcode == GB_NONZOMBIE_idxunop_code)
    {

        //----------------------------------------------------------------------
        // nonzombie selector when A is not iso
        //----------------------------------------------------------------------

        #define GB_TEST_VALUE_OF_ENTRY(keep,p) bool keep = (Ai [p] >= 0)
        #include "GB_select_phase2.c"

    }
    else if (op->ztype == GrB_BOOL && op->xtype == A->type)
    {

        //----------------------------------------------------------------------
        // no typecasting is required
        //----------------------------------------------------------------------

        #undef  GB_TEST_VALUE_OF_ENTRY
        #define GB_TEST_VALUE_OF_ENTRY(keep,p)                          \
            bool keep ;                                                 \
            fkeep (&keep, Ax +(p)*asize,                                \
                flipij ? j : i, flipij ? i : j, ythunk) ;               \
        #include "GB_select_phase2.c"

    }
    else
    {

        //----------------------------------------------------------------------
        // A is non-iso and typecasting is required
        //----------------------------------------------------------------------

        cast_A_to_X = GB_cast_factory (xcode, acode) ;
        cast_Z_to_bool = GB_cast_factory (GB_BOOL_code, zcode) ; 

        #undef  GB_TEST_VALUE_OF_ENTRY
        #define GB_TEST_VALUE_OF_ENTRY(keep,p)                          \
            bool keep ;                                                 \
            GB_void z [GB_VLA(zsize)] ;                                 \
            GB_void x [GB_VLA(xsize)] ;                                 \
            cast_A_to_X (x, Ax +(p)*asize, asize) ;                     \
            fkeep (z, x, flipij ? j : i, flipij ? i : j, ythunk) ;      \
            cast_Z_to_bool (&keep, z, zsize) ;
        #include "GB_select_phase2.c"

    }

    return (GrB_SUCCESS) ;
}

