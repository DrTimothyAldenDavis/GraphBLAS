//------------------------------------------------------------------------------
// GB_subassign_23: C += A where C is dense and A is sparse or dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C and A must have the same vector dimension and vector length.
// FUTURE::: the transposed case, C+=A' could easily be done.
// The parallelism used is identical to GB_colscale.

// The type of C must match the type of x and z for the accum function, since
// C(i,j) = accum (C(i,j), A(i,j)) is handled.  The generic case here can
// typecast A(i,j) but not C(i,j).  The case for typecasting of C is handled by
// Method 04.

// C and A can have any sparsity structure, but C must be as-if-full.

#define GB_DEBUG

#include "GB_subassign_dense.h"
#include "GB_binop.h"
#ifndef GBCUDA_DEV
#include "GB_aop__include.h"
#endif
#include "GB_unused.h"

#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
}

GrB_Info GB_subassign_23      // C += A; C is dense, A is sparse or dense
(
    GrB_Matrix C,                   // input/output matrix
    const GrB_Matrix A,             // input matrix
    const GrB_BinaryOp accum,       // operator to apply
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (!GB_aliased (C, A)) ;   // NO ALIAS of C==A

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for C+=A", GB0) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (GB_is_dense (C)) ;

    ASSERT_MATRIX_OK (A, "A for C+=A", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_BINARYOP_OK (accum, "accum for C+=A", GB0) ;
    ASSERT (!GB_OP_IS_POSITIONAL (accum)) ;
    ASSERT (A->vlen == C->vlen) ;
    ASSERT (A->vdim == C->vdim) ;

    GB_ENSURE_FULL (C) ;    // convert C to full, if sparsity control allows it

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    if (accum->opcode == GB_FIRST_binop_code || C->iso)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    // C = accum (C,A) will be computed
    ASSERT (!C->iso) ;
    ASSERT (C->type == accum->ztype) ;
    ASSERT (C->type == accum->xtype) ;
    ASSERT (GB_Type_compatible (A->type, accum->ytype)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // slice the entries for each task
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int A_ntasks, A_nthreads ;

    if (GB_IS_BITMAP (A) || GB_as_if_full (A))
    { 
        // C is dense and A is bitmap or as-if-full
        GBURBLE ("(Z bitmap/as-if-full) ") ;
        int64_t anvec = A->nvec ;
        int64_t anz = GB_nnz_held (A) ;
        A_nthreads = GB_nthreads (anz + anvec, chunk, nthreads_max) ;
        A_ntasks = 0 ;   // unused
        ASSERT (A_ek_slicing == NULL) ;
    }
    else
    { 
        // create tasks to compute over the matrix A
        GB_SLICE_MATRIX (A, 32, chunk) ;
        ASSERT (A_ek_slicing != NULL) ;
    }

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;

    #ifndef GBCUDA_DEV

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_sub23(accum,xname) GB (_subassign_23_ ## accum ## xname)
        #define GB_BINOP_WORKER(accum,xname)                    \
        {                                                       \
            info = GB_sub23 (accum,xname) (C, A,                \
                A_ek_slicing, A_ntasks, A_nthreads) ;           \
        }                                                       \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Opcode opcode ;
        GB_Type_code xcode, ycode, zcode ;
        // C = C + A so A must cast to the Y input of the accum operator
        if (GB_binop_builtin (C->type, false, A->type, false,
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
    // JIT TODO: aop: subassign 23
    #endif

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 

        //----------------------------------------------------------------------
        // get operators, functions, workspace, contents of A and C
        //----------------------------------------------------------------------

        #include "GB_generic.h"
        GB_BURBLE_MATRIX (A, "(generic C+=A) ") ;

        GxB_binary_function faccum = accum->binop_function ;

        size_t csize = C->type->size ;
        size_t asize = A->type->size ;
        size_t ysize = accum->ytype->size ;

        GB_cast_function cast_A_to_Y ;

        // A is typecasted to y
        cast_A_to_Y = GB_cast_factory (accum->ytype->code, A->type->code) ;

        // get the iso value of A
        GB_void ywork [GB_VLA(ysize)] ;
        if (A->iso)
        {
            // ywork = (ytype) Ax [0]
            cast_A_to_Y (ywork, A->x, asize) ;
        }

        #define C_iso false

        #ifndef GB_ACCUMULATE_aij
        #define GB_ACCUMULATE_aij(Cx,pC,Ax,pA,A_iso,ywork)              \
        {                                                               \
            /* Cx [pC] += (ytype) Ax [A_iso ? 0 : pA] */                \
            if (A_iso)                                                  \
            {                                                           \
                faccum (Cx +((pC)*csize), Cx +((pC)*csize), ywork) ;    \
            }                                                           \
            else                                                        \
            {                                                           \
                GB_void ywork [GB_VLA(ysize)] ;                         \
                cast_A_to_Y (ywork, Ax +((pA)*asize), asize) ;          \
                faccum (Cx +((pC)*csize), Cx +((pC)*csize), ywork) ;    \
            }                                                           \
        }
        #endif

        #include "GB_subassign_23_template.c"
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "C+=A output", GB0) ;
    return (GrB_SUCCESS) ;
}

