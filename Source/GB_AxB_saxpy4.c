//------------------------------------------------------------------------------
// GB_AxB_saxpy4: compute C=A*B, C<M>=A*B, or C<!M>=A*B in parallel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A must be sparse (not hypersparse, bitmap, or full).  B must be sparse or
// hypersparse.  C has the same sparsity structure as B.  Any case for the mask
// M can be handled: M can have any sparsity structure, or it can be NULL.  M
// can be complemented or not, and structural or valued.

#include "GB_mxm.h"
#include "GB_ek_slice.h"
#ifndef GBCOMPACT
#include "GB_AxB__include.h"
#endif

#define GB_FREE_WORK                                            \
{                                                               \
    GB_WERK_POP (M_ek_slicing, int64_t) ;                       \
    GB_WERK_POP (B_ek_slicing, int64_t) ;                       \
    GB_FREE_WERK_UNLIMITED_FROM_CALLOC (&Wf, Wf_size) ;         \
    GB_FREE_WERK_UNLIMITED_FROM_MALLOC (&Wi, Wi_size) ;         \
    GB_FREE_WERK_UNLIMITED_FROM_MALLOC (&Wx, Wx_size) ;         \
    GB_FREE_WERK_UNLIMITED_FROM_MALLOC (&Bflops, Bflops_size) ; \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORK ;              \
    GB_Matrix_free (&C) ;       \
}

//------------------------------------------------------------------------------
// GB_AxB_saxpy4: compute C=A*B, C<M>=A*B, or C<!M>=A*B in parallel
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy4              // C=A*B using Gustavson + large workspace
(
    GrB_Matrix C,                   // output matrix (not done in-place)
    const GrB_Matrix M_input,       // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    const int do_sort,              // if nonzero, to sort in saxpy4
    GB_Context Context
)
{

double ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    (*mask_applied) = false ;
    bool apply_mask = false ;

    ASSERT (C != NULL && C->static_header) ;

    ASSERT_MATRIX_OK_OR_NULL (M_input, "M for saxpy4 A*B", GB0) ;
    ASSERT (!GB_PENDING (M_input)) ;
    ASSERT (GB_JUMBLED_OK (M_input)) ;
    ASSERT (!GB_ZOMBIES (M_input)) ;

    ASSERT_MATRIX_OK (A, "A for saxpy4 A*B", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_IS_SPARSE (A)) ;

    ASSERT_MATRIX_OK (B, "B for saxpy4 A*B", GB0) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for saxpy4 A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    //--------------------------------------------------------------------------
    // determine the # of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // define workspace
    //--------------------------------------------------------------------------

    int M_nthreads, M_ntasks ;
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;
    GB_WERK_DECLARE (B_ek_slicing, int64_t) ;

    int64_t *Wi = NULL ; size_t Wi_size = 0 ;
    int8_t  *Wf = NULL ; size_t Wf_size = 0 ;
    GB_void *Wx = NULL ; size_t Wx_size = 0 ;
    int64_t *GB_RESTRICT Bflops = NULL ; size_t Bflops_size = 0 ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    bool A_is_pattern, B_is_pattern ;
    GB_AxB_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult->opcode) ;

    GB_Opcode mult_opcode, add_opcode ;
    GB_Type_code xcode, ycode, zcode ;
    bool builtin_semiring = GB_AxB_semiring_builtin (A, A_is_pattern, B,
        B_is_pattern, semiring, flipxy, &mult_opcode, &add_opcode, &xcode,
        &ycode, &zcode) ;

    #ifdef GBCOMPACT
    bool is_any_pair_semiring = false ;
    #else
    bool is_any_pair_semiring = builtin_semiring
        && (add_opcode == GB_ANY_opcode)
        && (mult_opcode == GB_PAIR_opcode) ;
    #endif

    //--------------------------------------------------------------------------
    // get A and B
    //--------------------------------------------------------------------------

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t avlen = A->vlen ;

    const int64_t *GB_RESTRICT Bp = B->p ;
    const int64_t *GB_RESTRICT Bh = B->h ;
    const int64_t *GB_RESTRICT Bi = B->i ;
    const int64_t bvdim = B->vdim ;
    const int64_t bnz = GB_NNZ (B) ;
    const int64_t bnvec = B->nvec ;
    const int64_t bvlen = B->vlen ;
    const int B_sparsity = GB_sparsity (B) ;

    //--------------------------------------------------------------------------
    // allocate C (just C->p and C->h, but not C->i or C->x)
    //--------------------------------------------------------------------------

    GrB_Type ctype = add->op->ztype ;
    size_t csize = ctype->size ;
    int64_t cvlen = avlen ;
    int64_t cvdim = bvdim ;
    int64_t cnvec = bnvec ;

    GB_OK (GB_new (&C, true, // sparse or hyper, static header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        B_sparsity, B->hyper_switch, cnvec, Context)) ;
    bool C_and_B_are_hyper = (B_sparsity == GxB_HYPERSPARSE) ;

    int64_t *GB_RESTRICT Cp = C->p ;
    int64_t *GB_RESTRICT Ch = C->h ;
    if (C_and_B_are_hyper)
    { 
        // B and C are both hypersparse
        GB_memcpy (Ch, Bh, cnvec * sizeof (int64_t), nthreads_max) ;
        C->nvec = bnvec ;
    }

    //==========================================================================
    // phase0: flop count analysis, construct tasks, and scatter the mask
    //==========================================================================

    int nthreads ;
    int64_t total_flops ;

    if (nthreads_max == 1)
    {
        // use a single thread; no need for flop count analysis
        nthreads = 1 ;
        total_flops = GB_NNZ (A) + bnz ;
    }
    else
    {
        // determine # of threads to use for compute the flop count
        int B_nthreads = GB_nthreads (bnz/16, chunk, nthreads_max) ;
        Bflops = GB_MALLOC_WERK_UNLIMITED (bnz+1, int64_t, &Bflops_size) ;
        if (Bflops == NULL)
        {
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

ttt = omp_get_wtime ( ) ;

        if (B_nthreads == 1)
        {
            int64_t pB ;
            GB_PRAGMA_SIMD
            for (pB = 0 ; pB < bnz ; pB++)
            {
                int64_t k = Bi [pB] ;
                Bflops [pB] = (Ap [k+1] - Ap [k]) ;
            }
        }
        else
        {
            int taskid ;
            #pragma omp parallel for num_threads(B_nthreads) schedule(static)
            for (int taskid = 0 ; taskid < B_nthreads ; taskid++)
            {
                int64_t pB_first, pB_last ;
                GB_PARTITION (pB_first, pB_last, bnz, taskid, B_nthreads) ;
                int64_t pB ;
                GB_PRAGMA_SIMD
                for (pB = pB_first ; pB < pB_last ; pB++)
                {
                    int64_t k = Bi [pB] ;
                    Bflops [pB] = (Ap [k+1] - Ap [k]) ;
                }
            }
        }

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (0, ttt) ;
ttt = omp_get_wtime ( ) ;

        // cumulative sum to determine flop count
        GB_cumsum (Bflops, bnz, &total_flops, B_nthreads, Context) ;

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (1, ttt) ;
ttt = omp_get_wtime ( ) ;

        // compute # of threads to use for C=A*B and slice B according to flops
        nthreads = GB_nthreads (128*total_flops, chunk, nthreads_max) ;
        nthreads = GB_IMIN (nthreads, bnz) ;
        nthreads = GB_IMAX (nthreads, 1) ;
    }

    GB_WERK_PUSH (B_ek_slicing, 3*nthreads+1, int64_t) ;
    int64_t *GB_RESTRICT kfirst_Bslice = B_ek_slicing ;
    int64_t *GB_RESTRICT klast_Bslice  = B_ek_slicing + nthreads ;
    int64_t *GB_RESTRICT pstart_Bslice = B_ek_slicing + nthreads * 2 ;

    GBURBLE ("(Gustavson:werk threads:%d%s) ", nthreads, do_sort ? " sort":"") ;

    // slice the flops equally between the tasks (one thread per task)
    GB_pslice (pstart_Bslice, Bflops, bnz, nthreads, true) ;

    // free Bflops workspace
    GB_FREE_WERK_UNLIMITED_FROM_MALLOC (&Bflops, Bflops_size) ;

    // allocate Wf, Wi, and Wx workspace
    int64_t cnzmax = cvlen*cnvec ;
    cnzmax = GB_IMAX (cnzmax, 1) ;
    Wf = GB_CALLOC_WERK_UNLIMITED (cnzmax, int8_t, &Wf_size) ;
    Wi = GB_MALLOC_WERK_UNLIMITED (cnzmax, int64_t, &Wi_size) ;
    bool ok = (Wf != NULL && Wi != NULL) ;
    if (!is_any_pair_semiring)
    {
        // Wx is not used for the ANY_PAIR semiring
        Wx = GB_MALLOC_WERK_UNLIMITED (cnzmax*csize, GB_void, &Wx_size) ;
        ok = ok && (Wx != NULL) ;
    }

    if (!ok)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // initialize Cp
    if (nthreads > 1)
    {
        for (int64_t kk = 0 ; kk < cnvec ; kk++)
        {
            Cp [kk] = kk * cvlen ;
        }
    }

    //--------------------------------------------------------------------------
    // scatter the mask into Wf, if present and sparse
    //--------------------------------------------------------------------------

    // if M is present and sparse:
    // scatter into Wf, with Wf [i,k] = M(i,j) = 0 or 1
    // The mask is not exploited if it has too many entries
    // If M is bitmap/full, it will be used in-place

    bool M_is_sparse_or_hyper = false ;

    // determine if the mask should be applied
    if (M_input == NULL)
    {
        // no mask to apply
        apply_mask = false ;
    }
    else if (GB_IS_SPARSE (M_input) || GB_IS_HYPERSPARSE (M_input))
    {
        // discard the mask if it has too many entries
        apply_mask = GB_NNZ (M_input) < total_flops ;
        M_is_sparse_or_hyper = apply_mask ;
    }
    else
    {
        // always apply M if it is bitmap or full
        apply_mask = true ;
    }

    GrB_Matrix M = (apply_mask) ? M_input : NULL ;
    bool M_dense_in_place = (M != NULL &&
        (GB_IS_BITMAP (M) || GB_IS_FULL (M))) ;

    // if M is full and structural, GB_mxm has already handled it, by
    // passing in M as NULL if not complemented, or by a quick return if
    // complemented.
    ASSERT (!(GB_IS_FULL (M) && Mask_struct)) ;

    const int64_t *GB_RESTRICT Mp = NULL ;
    const int64_t *GB_RESTRICT Mh = NULL ;
    const int8_t  *GB_RESTRICT Mb = NULL ;
    const int64_t *GB_RESTRICT Mi = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        msize = (Mask_struct) ? 0 : M->type->size ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;
    }

    if (M_is_sparse_or_hyper)
    {

        GB_SLICE_MATRIX (M, 1, chunk) ;

        switch (msize)
        {

            default:
            case 0 :    // M is structural
                #undef  GB_MTYPE
                #define GB_MASK_ij(pM) 1
                #include "GB_AxB_saxpy4_mask_template.c"
                break ;

            case 1 :    // M is bool, int8_t, or uint8_t
                #define GB_MTYPE uint8_t
                #define GB_MASK_ij(pM) (Mx [pM] != 0)
                #include "GB_AxB_saxpy4_mask_template.c"
                break ;

            case 2 :    // M is int16 or uint16
                #define GB_MTYPE uint16_t
                #define GB_MASK_ij(pM) (Mx [pM] != 0)
                #include "GB_AxB_saxpy4_mask_template.c"
                break ;

            case 4 :    // M is int32, uint32, or float
                #define GB_MTYPE uint32_t
                #define GB_MASK_ij(pM) (Mx [pM] != 0)
                #include "GB_AxB_saxpy4_mask_template.c"
                break ;

            case 8 :    // M is int64, uint64, double, or complex float
                #define GB_MTYPE uint64_t
                #define GB_MASK_ij(pM) (Mx [pM] != 0)
                #include "GB_AxB_saxpy4_mask_template.c"
                break ;

            case 16 :    // M is complex double
                #define GB_MTYPE uint64_t
                #define GB_MASK_ij(pM) (Mx [2*pM] != 0) || (Mx [2*pM+1] != 0)
                #include "GB_AxB_saxpy4_mask_template.c"
                break ;
        }
    }

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (2, ttt) ;
ttt = omp_get_wtime ( ) ;

    //==========================================================================
    // C = A*B, via saxpy4 method and built-in semiring
    //==========================================================================

    bool done = false ;

    #ifndef GBCOMPACT

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_AsaxpyB(add,mult,xname) GB_AsaxpyB_ ## add ## mult ## xname

        #define GB_AxB_WORKER(add,mult,xname)                               \
        {                                                                   \
            info = GB_AsaxpyB (add,mult,xname) (C, M,                       \
                Mask_comp, Mask_struct, M_dense_in_place, A, A_is_pattern,  \
                B, B_is_pattern, GB_SAXPY_METHOD_4,                         \
                NULL, 0, 0, nthreads, do_sort,                              \
                Wf, &Wi, Wi_size, Wx, kfirst_Bslice, klast_Bslice,          \
                pstart_Bslice, Context) ;                                   \
            done = (info != GrB_NO_VALUE) ;                                 \
        }                                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        if (builtin_semiring)
        { 
            #include "GB_AxB_factory.c"
        }

    #endif

    //--------------------------------------------------------------------------
    // generic saxpy4 method
    //--------------------------------------------------------------------------

    if (!done)
    { 
        info = GB_AxB_saxpy_generic (C, M, Mask_comp,
            Mask_struct, M_dense_in_place, A, A_is_pattern, B, B_is_pattern,
            semiring, flipxy, GB_SAXPY_METHOD_4,
            NULL, 0, 0, nthreads, do_sort,
            Wf, &Wi, Wi_size, Wx, kfirst_Bslice, klast_Bslice, pstart_Bslice,
            Context) ;
    }

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (7, ttt) ;
ttt = omp_get_wtime ( ) ;

    //--------------------------------------------------------------------------
    // scan the mask to clear Wf
    //--------------------------------------------------------------------------

    // this must be done even if saxpy4 fails

    if (M_is_sparse_or_hyper)
    {
        const int64_t *kfirst_Mslice = M_ek_slicing ;
        const int64_t *klast_Mslice  = M_ek_slicing + M_ntasks ;
        const int64_t *pstart_Mslice = M_ek_slicing + M_ntasks*2 ;
        #undef  GB_MTYPE
        #define GB_MASK_ij(pM) 0
        #include "GB_AxB_saxpy4_mask_template.c"
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // prune empty vectors, free workspace, and return result
    //--------------------------------------------------------------------------

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (8, ttt) ;
ttt = omp_get_wtime ( ) ;

    C->magic = GB_MAGIC ;
    C->jumbled = (!do_sort) ;
//  ASSERT_MATRIX_OK (C, "saxpy4: output before prune", GB0) ;
    GB_FREE_WORK ;
    GB_OK (GB_hypermatrix_prune (C, Context)) ;
    ASSERT_MATRIX_OK (C, "saxpy4: output", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    (*mask_applied) = apply_mask ;

ttt = omp_get_wtime ( ) - ttt ;
GB_Global_timing_add (9, ttt) ;

    return (info) ;
}

