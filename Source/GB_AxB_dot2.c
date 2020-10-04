//------------------------------------------------------------------------------
// GB_AxB_dot2: compute C=A'*B or C<!M>=A'*B in parallel, in-place
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_AxB_dot2 does its computation in two phases.  The first phase counts the
// number of entries in each column of C.  The second phase can then construct
// the result C in-place, and thus this method can be done in parallel, for the
// single matrix computation C=A'*B.

// Two variants are handled: C=A'*B and C<!M>=A'*B.
// The C<M>=A'*B computation is computed by GB_AxB_dot3.

#include "GB_mxm.h"
#include "GB_binop.h"
#ifndef GBCOMPACT
#include "GB_AxB__include.h"
#endif

#define GB_FREE_WORK                                            \
{                                                               \
    GB_FREE (A_slice) ;                                         \
    GB_FREE (B_slice) ;                                         \
    if (C_counts != NULL)                                       \
    {                                                           \
        for (int tid = 0 ; tid < naslice ; tid++)               \
        {                                                       \
            GB_FREE (C_counts [tid]) ;                          \
        }                                                       \
    }                                                           \
    GB_FREE (C_counts) ;                                        \
}

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_AxB_dot2                // C=A'*B or C<!M>=A'*B, dot product method
(
    GrB_Matrix *Chandle,            // output matrix
    const GrB_Matrix M,             // mask matrix for C<!M>=A'*B
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Context Context
)
{

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
    int64_t **C_counts = NULL ;
    int64_t cnvec = B->nvec ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    if (B->nvec_nonempty < 0)
    { 
        B->nvec_nonempty = GB_nvec_nonempty (B, NULL) ;
    }

    if (A->nvec_nonempty < 0)
    { 
        A->nvec_nonempty = GB_nvec_nonempty (A, NULL) ;
    }

    int64_t naslice = 0 ;
    int64_t nbslice = 0 ;

    int64_t anvec = A->nvec ;
    int64_t anz   = GB_NNZ_HELD (A) ;

    int64_t bnvec = B->nvec ;
    int64_t bnz   = GB_NNZ_HELD (B) ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz + bnz, chunk, nthreads_max) ;

    if (nthreads == 1)
    {
        // do the entire computation with a single thread
        naslice = 1 ;
        nbslice = 1 ;
    }
    else
    {
        // determine number of slices for A' and B
        if (bnvec > 32 * nthreads || bnvec == 0)
        { 
            // just slice B
            nbslice = 32 * nthreads ;
            naslice = 1 ;
        }
        else
        { 
            // slice B into individual vectors
            nbslice = bnvec ;

            // slice A' to get a total of about 32*nthreads tasks
            naslice = (32 * nthreads) / nbslice ;

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
    // C is always created as sparse or hypersparse.

    if (!GB_pslice (&A_slice, A->p, A->nvec, naslice))
    { 
        // out of memory
        GB_FREE_WORK ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    if (!GB_pslice (&B_slice, B->p, B->nvec, nbslice))
    { 
        // out of memory
        GB_FREE_WORK ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // determine the sparsity structure of C
    //--------------------------------------------------------------------------

    int64_t cvlen = A->vdim ;
    int64_t cvdim = B->vdim ;
    GrB_Type ctype = add->op->ztype ;

    double C_bitmap_size = ((double) cvlen) * ((double) cvdim) ;
    double A_size = (double) GB_NNZ_HELD (A) ;
    double B_size = (double) GB_NNZ_HELD (B) ;
    int C_sparsity ;
    bool C_is_bitmap ;
    if (C_bitmap_size < 8 * (A_size + B_size))
    {
        // C is not too large: use a bitmap
        C_sparsity = GxB_BITMAP ;
        C_is_bitmap = true ;
    }
    else
    {
        // C is very large: construct it as sparse or hypersparse
        C_sparsity = GB_IS_HYPERSPARSE (B) ? GxB_HYPERSPARSE : GxB_SPARSE ;
        C_is_bitmap = false ;
    }

    //--------------------------------------------------------------------------
    // compute # of entries in each vector of C
    //--------------------------------------------------------------------------

    int64_t cnz ;

    if (C_is_bitmap)
    {
        cnz = cvlen * cvdim ;
    }
    else
    {
        C_counts = GB_CALLOC (naslice, int64_t *) ;
        if (C_counts == NULL)
        { 
            // out of memory
            GB_FREE_WORK ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        for (int a_tid = 0 ; a_tid < naslice ; a_tid++)
        {
            int64_t *GB_RESTRICT C_count = GB_CALLOC (B->nvec, int64_t) ;
            if (C_count == NULL)
            { 
                // out of memory
                GB_FREE_WORK ;
                return (GrB_OUT_OF_MEMORY) ;
            }
            C_counts [a_tid] = C_count ;
        }

        // phase1 parallel region: each thread computes C_counts [tid]
        // for its slice.
        #define GB_PHASE_1_OF_2
        #include "GB_AxB_dot2_meta.c"
        #undef  GB_PHASE_1_OF_2
    }

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    info = GB_new (Chandle, // sparse, hyper or bitmap; new header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        C_sparsity, B->hyper_switch, cnvec, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_WORK ;
        return (info) ;
    }

    GrB_Matrix C = (*Chandle) ;

    //--------------------------------------------------------------------------
    // cumulative sum of counts in each vector of C
    //--------------------------------------------------------------------------

    if (!C_is_bitmap)
    {
        int64_t *GB_RESTRICT Cp = C->p ;
        int64_t k ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (k = 0 ; k < cnvec ; k++)
        {
            int64_t s = 0 ;
            for (int tid = 0 ; tid < naslice ; tid++)
            { 
                int64_t *GB_RESTRICT C_count = C_counts [tid] ;
                int64_t c = C_count [k] ;
                C_count [k] = s ;
                s += c ;
            }
            Cp [k] = s ;    // ok: C is sparse
        }
        Cp [cnvec] = 0 ;    // ok: C is sparse
        C->nvec = cnvec ;
        // Cp = cumulative sum of Cp
        GB_cumsum (Cp, cnvec, &(C->nvec_nonempty), nthreads) ;
        cnz = Cp [cnvec] ;  // ok: C is sparse

        // C->h = B->h
        if (B->h != NULL)
        { 
            GB_memcpy (C->h, B->h, cnvec * sizeof (int64_t), nthreads) ;
        }

        // free C_count for the first thread; it is no longer needed
        GB_FREE (C_counts [0]) ;
    }

    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // allocate C->b, C->i, and C->x
    //--------------------------------------------------------------------------

    info = GB_bix_alloc (C, cnz, C_is_bitmap, !C_is_bitmap, true, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_Matrix_free (Chandle) ;
        GB_FREE_WORK ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // C = A'*B, computing each entry with a dot product, via builtin semiring
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
                C_counts, nthreads, naslice, nbslice) ;                      \
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
        GB_BURBLE_MATRIX (C, "(generic C%s=A'*B) ", (M == NULL) ? "" : "<!M>") ;
        #include "GB_AxB_dot_generic.c"
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    C->jumbled = GB_JUMBLED (M) ;
    ASSERT_MATRIX_OK (C, "dot2: C = A'*B output", GB0) ;
    ASSERT (*Chandle == C) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;        // C is jumbled if M is jumbled
    ASSERT (!GB_PENDING (C)) ;
    return (GrB_SUCCESS) ;
}

