//------------------------------------------------------------------------------
// GB_add_phase0: find vectors of C to compute for C<M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The eWise add of two matrices, C=A+B, C<M>=A+B, or C<!M>=A+B starts with
// this phase, which determines which vectors of C need to be computed.

// On input, A and B are the two matrices being added, and M is the optional
// mask matrix, possibly complemented.

// The A matrix can be sparse, hypersparse, slice, or hyperslice.  The B matrix
// can only be sparse or hypersparse.  See GB_wait, which can pass in A as any
// of the four formats.  In this case, no mask is present.

// On output, two integers (max_Cnvec and Cnvec) a boolean (Ch_to_Mh) and up to
// 3 arrays are returned, either NULL or of size max_Cnvec.  If not NULL, only
// the first Cnvec entries in each array is used.  Let n = A->vdim be the
// vector dimension of A, B, M and C.

//      Ch:  the list of vectors to compute.  If not NULL, Ch [k] = j is the
//      kth vector in C to compute, which will become the hyperlist C->h of C.
//      Note that some of these vectors may turn out to be empty, because of
//      the mask, or because the vector j appeared in A or B, but is empty.
//      It is pruned at the end of GB_add_phase2.  If Ch is NULL then it is an
//      implicit list of size n, and Ch [k] == k for all k = 0:n-1.  In this
//      case, C will be a standard matrix, not hypersparse.  Thus, the kth
//      vector is j = (Ch == NULL) ? k : Ch [k].

//      Ch_is_Mh:  true if the mask M is present, hypersparse, and not
//      complemented, false otherwise.  In this case Ch is a copy of M->h.

//      C_to_A:  if A is hypersparse, then C_to_A [k] = kA if the kth vector, j
//      = (Ch == NULL) ? k : Ch [k] appears in A, as j = Ah [kA].  If j does
//      not appear in A, then C_to_A [k] = -1.  If A is not hypersparse, then
//      C_to_A is returned as NULL.

//      C_to_B:  if B is hypersparse, then C_to_B [k] = kB if the kth vector, j
//      = (Ch == NULL) ? k : Ch [k] appears in B, as j = Bh [kB].  If j does
//      not appear in B, then C_to_B [k] = -1.  If B is not hypersparse, then
//      C_to_B is returned as NULL.

// PARALLEL: done, except in one condition: A and B are hypersparse and
// Ch_is_Mh is false.  takes O(A->nvec + B->nvec) time.

#include "GB.h"

//------------------------------------------------------------------------------
// GB_allocate_result
//------------------------------------------------------------------------------

static inline bool GB_allocate_result
(
    int64_t max_Cnvec,
    int64_t **Ch_handle,
    int64_t **C_to_A_handle,
    int64_t **C_to_B_handle
)
{
    bool ok = true ;
    if (Ch_handle != NULL)
    { 
        GB_MALLOC_MEMORY (*Ch_handle, max_Cnvec, sizeof (int64_t)) ;
        ok = (*Ch_handle != NULL) ;
    }
    if (C_to_A_handle != NULL)
    { 
        GB_MALLOC_MEMORY (*C_to_A_handle, max_Cnvec, sizeof (int64_t)) ;
        ok = ok && (*C_to_A_handle != NULL) ;
    }
    if (C_to_B_handle != NULL)
    { 
        GB_MALLOC_MEMORY (*C_to_B_handle, max_Cnvec, sizeof (int64_t)) ;
        ok = ok && (*C_to_B_handle != NULL) ;
    }
    if (!ok)
    {
        // out of memory
        if (Ch_handle != NULL)
        { 
            GB_FREE_MEMORY (*Ch_handle,     max_Cnvec, sizeof (int64_t)) ;
        }
        if (C_to_A_handle != NULL)
        { 
            GB_FREE_MEMORY (*C_to_A_handle, max_Cnvec, sizeof (int64_t)) ;
        }
        if (C_to_B_handle != NULL)
        { 
            GB_FREE_MEMORY (*C_to_B_handle, max_Cnvec, sizeof (int64_t)) ;
        }
    }
    return (ok) ;
}

//------------------------------------------------------------------------------
// GB_add_phase0:  find the vectors of C for C<M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB_add_phase0      // find vectors in C for C=A+B, C<M>=A+B, C<!M>=A+B
(
    int64_t *p_Cnvec,           // # of vectors to compute in C
    int64_t *p_max_Cnvec,       // size of the 3 output arrays:
    int64_t **Ch_handle,        // Ch: output of size max_Cnvec, or NULL
    int64_t **C_to_A_handle,    // C_to_A: output of size max_Cnvec, or NULL
    int64_t **C_to_B_handle,    // C_to_B: output of size max_Cnvec, or NULL
    bool *p_Ch_is_Mh,           // if true, then Ch == M->h

    const GrB_Matrix M,         // optional mask, may be NULL
    const bool Mask_comp,       // if true, then M is complemented
    const GrB_Matrix A,         // standard, hypersparse, slice, or hyperslice
    const GrB_Matrix B,         // standard or hypersparse; never a slice
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_Cnvec != NULL) ;
    ASSERT (p_max_Cnvec != NULL) ;
    ASSERT (p_Ch_is_Mh != NULL) ;
    ASSERT (Ch_handle != NULL) ;
    ASSERT (C_to_A_handle != NULL) ;
    ASSERT (C_to_B_handle != NULL) ;
    ASSERT_OK (GB_check (A, "A for add phase0", GB0)) ;
    ASSERT_OK (GB_check (B, "B for add phase0", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for add phase0", GB0)) ;
    ASSERT (A->vdim == B->vdim) ;
    if (M != NULL) ASSERT (A->vdim == M->vdim) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t *restrict Ch = NULL ;
    int64_t *restrict C_to_A = NULL ;
    int64_t *restrict C_to_B = NULL ;

    (*Ch_handle) = NULL ;
    (*C_to_A_handle) = NULL ;
    (*C_to_B_handle) = NULL ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // get content of M, A, and B
    //--------------------------------------------------------------------------

    int64_t max_Cnvec, Cnvec ;

    int64_t n = A->vdim ;
    int64_t Anvec = A->nvec ;
    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    bool A_is_hyper = A->is_hyper ;
    bool A_is_slice = A->is_slice ;

    int64_t Bnvec = B->nvec ;
    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    bool B_is_hyper = B->is_hyper ;
    ASSERT (!B->is_slice) ;

    // if M is present, hypersparse, and not complemented, then C will be
    // hypersparse, and it will have set of vectors as M (Ch == M->h).
    bool Ch_is_Mh = (M != NULL && M->is_hyper && !Mask_comp) ;

    //--------------------------------------------------------------------------
    // find the set union of the non-empty vectors of A and B
    //--------------------------------------------------------------------------

    if (Ch_is_Mh)
    {

        //----------------------------------------------------------------------
        // C is hypersparse, with the same vectors as the hypersparse M
        //----------------------------------------------------------------------

        // GB_wait is the only place where A may be a slice, and it does not
        // use a mask.  So this phase can ignore the case where A is a slice.
        ASSERT (!A_is_slice) ;
        int64_t Mnvec = M->nvec ;
        Cnvec = Mnvec ;
        max_Cnvec = Mnvec ;
        if (!GB_allocate_result (max_Cnvec, &Ch,
            (A_is_hyper) ? (&C_to_A) : NULL,
            (B_is_hyper) ? (&C_to_B) : NULL))
        { 
            return (GB_OUT_OF_MEMORY) ;
        }

        // copy M->h into Ch
        GB_memcpy (Ch, M->h, Mnvec * sizeof (int64_t), nthreads) ;

        // TODO: if A and M are aliased (M->h == A->h) then C_to_A [k] == k
        // TODO: if B and M are aliased (M->h == B->h) then C_to_B [k] == k

        // construct the mapping from C to A and B, if they are hypersparse
        if (A_is_hyper || B_is_hyper)
        {
            #pragma omp parallel for num_threads(nthreads)
            for (int64_t k = 0 ; k < Cnvec ; k++)
            {
                int64_t j = Ch [k] ;
                if (A_is_hyper)
                { 
                    // C_to_A [k] = kA if Ah [kA] == j and A(:,j) is non-empty
                    int64_t kA = 0, pA, pA_end ;
                    C_to_A [k] = (GB_lookup (true, Ah, Ap, &kA, Anvec-1, j,
                        &pA, &pA_end) && (pA < pA_end)) ? kA : -1 ;
                }
                if (B_is_hyper)
                { 
                    // C_to_B [k] = kB if Bh [kB] == j and B(:,j) is non-empty
                    int64_t kB = 0, pB, pB_end ;
                    C_to_B [k] = (GB_lookup (true, Bh, Bp, &kB, Bnvec-1, j,
                        &pB, &pB_end) && (pB < pB_end)) ? kB : -1 ;
                }
            }
        }

    }
    else if ((A_is_hyper || A_is_slice) && B_is_hyper)
    {

        //----------------------------------------------------------------------
        // both A and B are hypersparse
        //----------------------------------------------------------------------

        // C will be hypersparse, so Ch is allocated.  The mask M is ignored.
        // Ch is the set union of Ah and Bh.

        max_Cnvec = GB_IMIN (Anvec + Bnvec, n) ;
        if (!GB_allocate_result (max_Cnvec, &Ch, &C_to_A, &C_to_B))
        { 
            return (GB_OUT_OF_MEMORY) ;
        }

        // TODO this is sequential.  Use a parallel merge?  Ah and Bh are
        // sorted, and the result Ch must be sorted too.  Or use qsort and
        // then a merge of duplicates?

        // merge Ah and Bh into Ch
        int64_t kA = 0 ;
        int64_t kB = 0 ;
        for (Cnvec = 0 ; kA < Anvec && kB < Bnvec ; Cnvec++)
        {
            int64_t jA = (A_is_hyper) ? Ah [kA] : (A->hfirst + kA) ;
            int64_t jB = Bh [kB] ;
            // printf ("jA "GBd" jB "GBd"\n", jA, jB) ;
            if (jA < jB)
            { 
                // append jA to Ch
                Ch     [Cnvec] = jA ;
                C_to_A [Cnvec] = kA++ ;
                C_to_B [Cnvec] = -1 ;       // jA does not appear in B
            }
            else if (jB < jA)
            { 
                // append jB to Ch
                Ch     [Cnvec] = jB ;
                C_to_A [Cnvec] = -1 ;       // jB does not appear in A
                C_to_B [Cnvec] = kB++ ;
            }
            else
            { 
                // j appears in both A and B; append it to Ch
                Ch     [Cnvec] = jA ;
                C_to_A [Cnvec] = kA++ ;
                C_to_B [Cnvec] = kB++ ;
            }
        }
        if (kA < Anvec)
        {
            // B is exhausted but A is not
            for ( ; kA < Anvec ; kA++, Cnvec++)
            { 
                // append jA to Ch
                int64_t jA = (A_is_hyper) ? Ah [kA] : (A->hfirst + kA) ;
                Ch     [Cnvec] = jA ;
                C_to_A [Cnvec] = kA ;
                C_to_B [Cnvec] = -1 ;
            }
        }
        else if (kB < Bnvec)
        {
            // A is exhausted but B is not
            for ( ; kB < Bnvec ; kB++, Cnvec++)
            { 
                // append jB to Ch
                int64_t jB = Bh [kB] ;
                Ch     [Cnvec] = jB ;
                C_to_A [Cnvec] = -1 ;
                C_to_B [Cnvec] = kB ;
            }
        }

    }
    else if ((A_is_hyper || A_is_slice) && !B_is_hyper)
    {

        //----------------------------------------------------------------------
        // A is hypersparse, B is standard
        //----------------------------------------------------------------------

        // C will be standard.  Construct the C_to_A mapping.

        Cnvec = n ;
        max_Cnvec = n ;
        if (!GB_allocate_result (max_Cnvec, NULL, &C_to_A, NULL))
        { 
            return (GB_OUT_OF_MEMORY) ;
        }

        #pragma omp parallel for num_threads(nthreads)
        for (int64_t j = 0 ; j < n ; j++)
        { 
            C_to_A [j] = -1 ;
        }

        // scatter Ah into C_to_A
        #pragma omp parallel for num_threads(nthreads)
        for (int64_t kA = 0 ; kA < Anvec ; kA++)
        { 
            int64_t jA = (A_is_hyper) ? Ah [kA] : (A->hfirst + kA) ;
            C_to_A [jA] = kA ;
        }

    }
    else if (!(A_is_hyper || A_is_slice) && B_is_hyper)
    {

        //----------------------------------------------------------------------
        // A is standard, B is hypersparse
        //----------------------------------------------------------------------

        // C will be standard.  Construct the C_to_B mapping.

        Cnvec = n ;
        max_Cnvec = n ;
        if (!GB_allocate_result (max_Cnvec, NULL, NULL, &C_to_B))
        { 
            return (GB_OUT_OF_MEMORY) ;
        }

        #pragma omp parallel for num_threads(nthreads)
        for (int64_t j = 0 ; j < n ; j++)
        { 
            C_to_B [j] = -1 ;
        }

        // scatter Bh into C_to_B
        #pragma omp parallel for num_threads(nthreads)
        for (int64_t kB = 0 ; kB < Bnvec ; kB++)
        { 
            int64_t jB = Bh [kB] ;
            C_to_B [jB] = kB ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // A and B are both standard
        //----------------------------------------------------------------------

        // nothing to do
        Cnvec = n ;
        max_Cnvec = n ;

    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT (Cnvec <= max_Cnvec) ;
    (*p_Cnvec      ) = Cnvec ;
    (*p_max_Cnvec  ) = max_Cnvec ;
    (*p_Ch_is_Mh   ) = Ch_is_Mh ;
    (*Ch_handle    ) = Ch ;
    (*C_to_A_handle) = C_to_A ;
    (*C_to_B_handle) = C_to_B ;

    //--------------------------------------------------------------------------
    // The code below describes what the output contains:
    //--------------------------------------------------------------------------

    #ifndef NDEBUG
    // printf ("Cnvec: " GBd"\n", Cnvec) ;
    ASSERT (A != NULL) ;        // A and B are always present
    ASSERT (B != NULL) ;
    int64_t jlast = -1 ;
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {

        // C(:,j) is in the list, as the kth vector
        int64_t j ;
        if (Ch == NULL)
        {
            // C will be constructed as standard sparse
            j = k ;
        }
        else
        {
            // C will be constructed as hypersparse
            j = Ch [k] ;
        }

        // printf ("phase0: k "GBd" j "GBd"\n", k, j) ;

        // vectors j in Ch are sorted, and in the range 0:n-1
        ASSERT (j >= 0 && j < n) ;
        ASSERT (j > jlast) ;
        jlast = j ;

        // see if A (:,j) exists
        if (C_to_A != NULL)
        {
            // A is hypersparse, or a slice
            ASSERT (A->is_hyper || A->is_slice) ;
            int64_t kA = C_to_A [k] ;
            ASSERT (kA >= -1 && kA < A->nvec) ;
            if (kA >= 0)
            {
                int64_t jA = (A->is_hyper) ? A->h [kA] : (A->hfirst + kA) ;
                ASSERT (j == jA) ;
            }
        }
        else
        {
            // A is in standard sparse form
            // C_to_A exists only if A is hypersparse
            ASSERT (!(A->is_hyper || A->is_slice)) ;
        }

        // see if B (:,j) exists
        if (C_to_B != NULL)
        {
            // B is hypersparse
            ASSERT (B->is_hyper) ;
            int64_t kB = C_to_B [k] ;
            ASSERT (kB >= -1 && kB < B->nvec) ;
            if (kB >= 0)
            {
                int64_t jB = B->h [kB] ;
                ASSERT (j == jB) ;
            }
        }
        else
        {
            // B is in standard sparse form
            // C_to_B exists only if B is hypersparse
            ASSERT (!B->is_hyper) ;
        }

    }
    #endif

    return (GrB_SUCCESS) ;
}

