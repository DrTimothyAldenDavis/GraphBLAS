//------------------------------------------------------------------------------
// GB_AxB_dot2: compute C=A'*B or C<!M>=A'*B in parallel, in-place
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: rename to GB_bitmap_AxB_dot.c

//------------------------------------------------------------------------------

// The C<M>=A'*B dot product when C is sparse is computed by GB_AxB_dot3.
// This method always constructs C as bitmap.

#include "GB_mxm.h"
#include "GB_binop.h"
#ifndef GBCOMPACT
#include "GB_AxB__include.h"
#endif

#define GB_FREE_WORK                                            \
{                                                               \
    GB_FREE (A_slice) ;                                         \
    GB_FREE (B_slice) ;                                         \
}

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_AxB_dot2                // C=A'*B or C<!M>=A'*B, dot product method
(
    GrB_Matrix *Chandle,            // output matrix
    const GrB_Matrix M,             // mask matrix for C<!M>=A'*B, may be NULL
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Context Context
)
{
double ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    ASSERT (Chandle != NULL) ;
    ASSERT (*Chandle == NULL) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for dot A'*B", GB0) ;
    ASSERT_MATRIX_OK (A, "A for dot A'*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for dot A'*B", GB0) ;

    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;    // C is jumbled if M is jumbled
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for numeric A'*B", GB0) ;
    ASSERT (A->vlen == B->vlen) ;

    int64_t *GB_RESTRICT A_slice = NULL ;
    int64_t *GB_RESTRICT B_slice = NULL ;
    int64_t cnvec = B->nvec ;
    int64_t cvlen = A->vdim ;
    int64_t cvdim = B->vdim ;

    int64_t cnz ;
    if (!GB_Index_multiply ((GrB_Index *) (&cnz), cvlen, cvdim))
    {
        // problem too large
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t naslice = 0 ;
    int64_t nbslice = 0 ;

    int64_t anvec = A->nvec ;
    int64_t anz   = GB_NNZ_HELD (A) ;

    int64_t bnvec = B->nvec ;
    int64_t bnz   = GB_NNZ_HELD (B) ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz + bnz, chunk, nthreads_max) ;

    #define GB_NTASKS_PER_THREAD 32

    if (nthreads == 1)
    { 
        // do the entire computation with a single thread
        naslice = 1 ;
        nbslice = 1 ;
    }
    else
    {
        // determine number of slices for A' and B
        if (bnvec == 1)
        { 
            // C and B are single vectors
            naslice = GB_NTASKS_PER_THREAD * nthreads ;
            nbslice = 1 ;
        }
        else if (anvec == 1 || bnvec == 0
            || bnvec > GB_NTASKS_PER_THREAD * nthreads)
        { 
            // A is a single vector, or B is empty, or B is large: just slice B
            naslice = 1 ;
            nbslice = GB_NTASKS_PER_THREAD * nthreads ;
        }
        else
        { 
            // slice B into individual vectors
            nbslice = bnvec ;

            // slice A' to get a total of about 16*nthreads tasks
            naslice = (GB_NTASKS_PER_THREAD * nthreads) / nbslice ;

            // but do not slice A too finely
            naslice = GB_IMIN (naslice, anvec/4) ;
            naslice = GB_IMAX (naslice, nthreads) ;
        }
    }

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    bool A_is_pattern, B_is_pattern ;
    GB_AxB_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult->opcode) ;

    (*Chandle) = NULL ;

    //--------------------------------------------------------------------------
    // allocate workspace and slice A and B
    //--------------------------------------------------------------------------

    // A and B can have any sparsity: full, bitmap, sparse, or hypersparse.
    // C is always created as bitmap

    if (!GB_pslice (&A_slice, A->p, A->nvec, naslice, false))
    { 
        // out of memory
        GB_FREE_WORK ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    if (!GB_pslice (&B_slice, B->p, B->nvec, nbslice, false))
    { 
        // out of memory
        GB_FREE_WORK ;
        return (GrB_OUT_OF_MEMORY) ;
    }

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (17, ttt) ;
ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    GrB_Type ctype = add->op->ztype ;
    info = GB_new_bix (Chandle, // bitmap, new header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        GxB_BITMAP, B->hyper_switch, cnvec, cnz, true, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_WORK ;
        return (info) ;
    }

    GrB_Matrix C = (*Chandle) ;

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (18, ttt) ;
ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // TODO: if M is sparse, scatter it into the C bitmap
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // C<#>=A'*B, computing each entry with a dot product, via builtin semiring
    //--------------------------------------------------------------------------

    bool done = false ;

    #ifndef GBCOMPACT

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_Adot2B(add,mult,xname) GB_Adot2B_ ## add ## mult ## xname

        #define GB_AxB_WORKER(add,mult,xname)                                \
        {                                                                    \
            info = GB_Adot2B (add,mult,xname) (C, M, Mask_comp, Mask_struct, \
                A, A_is_pattern, A_slice, B, B_is_pattern, B_slice,          \
                nthreads, naslice, nbslice) ;                                \
            done = (info != GrB_NO_VALUE) ;                                  \
        }                                                                    \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        GB_Opcode mult_opcode, add_opcode ;
        GB_Type_code xcode, ycode, zcode ;

        if (GB_AxB_semiring_builtin (A, A_is_pattern, B, B_is_pattern, semiring,
            flipxy, &mult_opcode, &add_opcode, &xcode, &ycode, &zcode))
        { 
            #include "GB_AxB_factory.c"
        }
        ASSERT (info == GrB_SUCCESS || info == GrB_NO_VALUE) ;

    #endif

    //--------------------------------------------------------------------------
    // C = A'*B, computing each entry with a dot product, with typecasting
    //--------------------------------------------------------------------------

    if (!done)
    { 
        #define GB_DOT2_GENERIC
        GB_BURBLE_MATRIX (C, "(generic C%s=A'*B) ", (M == NULL) ? "" :
            (Mask_comp ? "<!M>" : "<M>")) ;
        #include "GB_AxB_dot_generic.c"
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "dot2: C = A'*B output", GB0) ;
    ASSERT (*Chandle == C) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (19, ttt) ;
ttt = omp_get_wtime ( ) ;

    return (GrB_SUCCESS) ;
}

