//------------------------------------------------------------------------------
// GB_subassign_22: C += scalar where C is dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C += scalar where C is a dense or full matrix.
// C can have any sparsity format, as long as all entries are present;
// GB_is_dense (C)) must hold.

#include "GB_subassign_dense.h"
#include "GB_binop.h"
#include "GB_unused.h"
#ifndef GBCUDA_DEV
#include "GB_aop__include.h"
#endif

#define GB_FREE_ALL ;

GrB_Info GB_subassign_22      // C += scalar where C is dense
(
    GrB_Matrix C,                   // input/output matrix
    const void *scalar,             // input scalar
    const GrB_Type atype,           // type of the input scalar
    const GrB_BinaryOp accum,       // operator to apply
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for C+=scalar", GB0) ;
    ASSERT (GB_as_if_full (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;

    ASSERT (scalar != NULL) ;
    ASSERT_TYPE_OK (atype, "atype for C+=scalar", GB0) ;
    ASSERT_BINARYOP_OK (accum, "accum for C+=scalar", GB0) ;
    ASSERT (!GB_OP_IS_POSITIONAL (accum)) ;

    GB_ENSURE_FULL (C) ;    // convert C to full, if sparsity control allows it

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    if (accum->opcode == GB_FIRST_binop_code || C->iso)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    // C = accum (C,scalar) will be computed
    ASSERT (C->type == accum->ztype) ;
    ASSERT (C->type == accum->xtype) ;
    ASSERT (GB_Type_compatible (atype, accum->ytype)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // typecast the scalar into the same type as the y input of the binary op
    //--------------------------------------------------------------------------

    int64_t csize = C->type->size ;
    size_t ysize = accum->ytype->size ;
    GB_cast_function 
        cast_A_to_Y = GB_cast_factory (accum->ytype->code, atype->code) ;
    GB_void ywork [GB_VLA(ysize)] ;
    cast_A_to_Y (ywork, scalar, atype->size) ;

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    #ifndef GBCUDA_DEV

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_sub22(accum,xname) GB (_subassign_22_ ## accum ## xname)

        #define GB_BINOP_WORKER(accum,xname)                                \
        {                                                                   \
            info = GB_sub22 (accum,xname) (C, ywork, nthreads) ;            \
        }                                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Opcode opcode ;
        GB_Type_code xcode, ycode, zcode ;
        // C = C + scalar where the scalar is cast to the Y input of accum
        if (GB_binop_builtin (C->type, false, atype, false,
            accum, false, &opcode, &xcode, &ycode, &zcode))
        { 
            // accumulate sparse matrix into dense matrix with built-in operator
            #include "GB_binop_factory.c"
        }

    #endif

    //--------------------------------------------------------------------------
    // via the JIT kernel
    //--------------------------------------------------------------------------

    #if GB_JIT_ENABLED
    // JIT TODO: aop: subassign 22
    #endif

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 

        #include "GB_generic.h"
        GB_BURBLE_MATRIX (C, "(generic C(:,:)+=x assign) ") ;

        GxB_binary_function faccum = accum->binop_function ;

        // C(i,j) = C(i,j) + y
        #define GB_ACCUMULATE_scalar(Cx,pC,ywork)           \
            faccum (Cx +((pC)*csize), Cx +((pC)*csize), ywork)

        #include "GB_subassign_22_template.c"
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "C+=scalar output", GB0) ;
    return (GrB_SUCCESS) ;
}

