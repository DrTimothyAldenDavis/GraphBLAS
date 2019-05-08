//------------------------------------------------------------------------------
// GB_emult_phase0: find vectors of C to compute for C<M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The eWise multiply of two matrices, C=A.*B, C<M>=A.*B, or C<!M>=A.*B starts
// with this phase, which determines which vectors of C need to be computed.

// On input, A and B are the two matrices being ewise multiplied, and M is the
// optional mask matrix, possibly complemented.

// The M, A, and B matrices are sparse or hypersparse (not a slice or
// hyperslice).  C will be standard (if Ch is returned NULL) or hypersparse
// (if Ch is returned non-NULL).

//      Ch: the vectors to compute in C.  Not allocated, but equal to either
//      A->h, B->h, or M->h, or NULL if C is not hypersparse.

//      C_to_A:  if A is hypersparse, and Ch is not A->h, then C_to_A [k] = kA
//      if the kth vector j = Ch [k] is equal to Ah [kA].  If j does not appear
//      in A, then C_to_A [k] = -1.  Otherwise, C_to_A is returned as NULL.
//      C is always hypersparse in this case.

//      C_to_B:  if B is hypersparse, and Ch is not B->h, then C_to_B [k] = kB
//      if the kth vector j = Ch [k] is equal to Bh [kB].  If j does not appear
//      in B, then C_to_B [k] = -1.  Otherwise, C_to_B is returned as NULL.
//      C is always hypersparse in this case.

//      C_to_M:  if M is hypersparse, and Ch is not M->h, then C_to_M [k] = kM
//      if the kth vector j = (Ch == NULL) ? k : Ch [k] is equal to Mh [kM].
//      If j does not appear in M, then C_to_M [k] = -1.  Otherwise, C_to_M is
//      returned as NULL.  C is hypersparse, except for one case.  If both A
//      and B are standard, and M is hypersparse and complemented, then C is
//      standard.  In this case, C_to_M must be computed.

// PARALLEL: done

#include "GB.h"

GrB_Info GB_emult_phase0 // find vectors in C for C=A.*B, C<M>=A.*B, C<!M>=A.*B
(
    int64_t *p_Cnvec,           // # of vectors to compute in C
    int64_t **Ch_handle,        // Ch is M->h, A->h, B->h, or NULL
    int64_t **C_to_M_handle,    // C_to_M: output of size Cnvec, or NULL
    int64_t **C_to_A_handle,    // C_to_A: output of size Cnvec, or NULL
    int64_t **C_to_B_handle,    // C_to_B: output of size Cnvec, or NULL

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
    ASSERT (Ch_handle != NULL) ;
    ASSERT (C_to_M_handle != NULL) ;
    ASSERT (C_to_A_handle != NULL) ;
    ASSERT (C_to_B_handle != NULL) ;
    ASSERT_OK (GB_check (A, "A for emult phase0", GB0)) ;
    ASSERT_OK (GB_check (B, "B for emult phase0", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for emult phase0", GB0)) ;
    ASSERT (A->vdim == B->vdim) ;
    if (M != NULL) ASSERT (A->vdim == M->vdim) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    const int64_t *restrict Ch = NULL ;
    int64_t *restrict C_to_M = NULL ;
    int64_t *restrict C_to_A = NULL ;
    int64_t *restrict C_to_B = NULL ;

    (*Ch_handle    ) = NULL ;
    (*C_to_M_handle) = NULL ;
    (*C_to_A_handle) = NULL ;
    (*C_to_B_handle) = NULL ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
    // TODO reduce nthreads for small problem (work: about O(Cnvec))

    //--------------------------------------------------------------------------
    // get content of M, A, and B
    //--------------------------------------------------------------------------

    int64_t n = A->vdim ;

    int64_t Anvec = A->nvec ;
    const int64_t *restrict Ah = A->h ;
    bool A_is_hyper = A->is_hyper ;
    ASSERT (!A->is_slice) ;

    int64_t Bnvec = B->nvec ;
    const int64_t *restrict Bh = B->h ;
    bool B_is_hyper = B->is_hyper ;
    ASSERT (!B->is_slice) ;

    int64_t Mnvec = 0 ;
    const int64_t *restrict Mh = NULL ;
    bool M_is_hyper = false ;

    if (M != NULL)
    {
        Mnvec = M->nvec ;
        Mh = M->h ;
        M_is_hyper = M->is_hyper ;
        ASSERT (!M->is_slice) ;
    }

    //--------------------------------------------------------------------------
    // determine how to construct the vectors of C
    //--------------------------------------------------------------------------

    if (M != NULL && !Mask_comp)
    {

        //----------------------------------------------------------------------
        // 8 cases to consider: A, B, M can each be hyper or standard
        //----------------------------------------------------------------------

        // Mask is present and not complemented

        if (A_is_hyper)
        {
            if (B_is_hyper)
            {
                if (M_is_hyper)
                {

                    //----------------------------------------------------------
                    // (1) A hyper, B hyper, M hyper
                    //----------------------------------------------------------

                    // Ch = smaller of Mh, Bh, Ah

                    int64_t nvec = GB_IMIN (Anvec, Bnvec) ;
                    nvec = GB_IMIN (nvec, Mnvec) ;
                    if (nvec == Anvec)
                    { 
                        Ch = Ah ;
                    }
                    else if (nvec == Bnvec)
                    { 
                        Ch = Bh ;
                    }
                    else // (nvec == Mnvec)
                    { 
                        Ch = Mh ;
                    }

                }
                else
                {

                    //----------------------------------------------------------
                    // (2) A hyper, B hyper, M standard
                    //----------------------------------------------------------

                    // Ch = smaller of Ah, Bh
                    if (Anvec <= Bnvec)
                    { 
                        Ch = Ah ;
                    }
                    else
                    { 
                        Ch = Bh ;
                    }
                }

            }
            else
            {

                if (M_is_hyper)
                {

                    //----------------------------------------------------------
                    // (3) A hyper, B standard, M hyper
                    //----------------------------------------------------------

                    // Ch = smaller of Mh, Ah
                    if (Anvec <= Mnvec)
                    { 
                        Ch = Ah ;
                    }
                    else
                    { 
                        Ch = Mh ;
                    }

                }
                else
                { 

                    //----------------------------------------------------------
                    // (4) A hyper, B standard, M standard
                    //----------------------------------------------------------

                    Ch = Ah ;
                }
            }

        }
        else
        {

            if (B_is_hyper)
            {
                if (M_is_hyper)
                {

                    //----------------------------------------------------------
                    // (5) A standard, B hyper, M hyper
                    //----------------------------------------------------------

                    // Ch = smaller of Mh, Bh

                    if (Bnvec <= Mnvec)
                    { 
                        Ch = Bh ;
                    }
                    else
                    { 
                        Ch = Mh ;
                    }

                }
                else
                { 

                    //----------------------------------------------------------
                    // (6) A standard, B hyper, M standard
                    //----------------------------------------------------------

                    Ch = Bh ;

                }
            }
            else
            {

                if (M_is_hyper)
                { 

                    //----------------------------------------------------------
                    // (7) A standard, B standard, M hyper
                    //----------------------------------------------------------

                    Ch = Mh ;

                }
                else
                { 

                    //----------------------------------------------------------
                    // (8) A standard, B standard, M standard
                    //----------------------------------------------------------

                    // C is standard
                    ;

                }
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // 4 cases to consider:  A, B can be hyper or standard
        //----------------------------------------------------------------------

        // Mask is not present, or present and complemented.

        if (A_is_hyper)
        {
            if (B_is_hyper)
            {

                //--------------------------------------------------------------
                // (1) A hyper, B hyper   
                //--------------------------------------------------------------

                // Ch = smaller of Ah, Bh
                if (Anvec <= Bnvec)
                { 
                    Ch = Ah ;
                }
                else
                { 
                    Ch = Bh ;
                }
            }
            else
            { 

                //--------------------------------------------------------------
                // (2) A hyper, B standard
                //--------------------------------------------------------------

                Ch = Ah ;

            }

        }
        else
        {

            if (B_is_hyper)
            { 

                //--------------------------------------------------------------
                // (3) A standard, B hyper   
                //--------------------------------------------------------------

                Ch = Bh ;

            }
            else
            { 

                //--------------------------------------------------------------
                // (4) A standard, B standard
                //--------------------------------------------------------------

                // C is standard
                ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // find Cnvec
    //--------------------------------------------------------------------------

    bool C_is_hyper = (Ch != NULL) ;

    int64_t Cnvec ;

    if (Ch == Ah)
    { 
        Cnvec = Anvec ;
    }
    else if (Ch == Bh)
    { 
        Cnvec = Bnvec ;
    }
    else if (Ch == Mh)
    { 
        Cnvec = Mnvec ;
    }
    else // Ch == NULL
    { 
        Cnvec = n ;
    }

    //--------------------------------------------------------------------------
    // construct C_to_M mapping
    //--------------------------------------------------------------------------

    if (M_is_hyper && Ch != Mh)
    {
        // allocate C_to_M
        GB_MALLOC_MEMORY (C_to_M, Cnvec, sizeof (int64_t)) ;
        if (C_to_M == NULL)
        { 
            // out of memory
            return (GB_OUT_OF_MEMORY) ;
        }

        // compute C_to_M
        if (C_is_hyper)
        {

            const int64_t *restrict Mp = M->p ;
            #pragma omp parallel for num_threads(nthreads)
            for (int64_t k = 0 ; k < Cnvec ; k++)
            { 
                int64_t pM, pM_end, kM = 0 ;
                int64_t j = Ch [k] ;
                GB_lookup (true, Mh, Mp, &kM, Mnvec-1, j, &pM, &pM_end) ;
                C_to_M [k] = (pM < pM_end) ? kM : -1 ;
            }

        }
        else
        {

            // this case can occur only if M is present, complemented, and
            // hypersparse, and C is standard (A standard and B standard)
            ASSERT (M != NULL && Mask_comp && !A_is_hyper && !B_is_hyper) ;

            #pragma omp parallel for num_threads(nthreads)
            for (int64_t j = 0 ; j < n ; j++)
            { 
                C_to_M [j] = -1 ;
            }

            // scatter Mh into C_to_M
            #pragma omp parallel for num_threads(nthreads)
            for (int64_t kM = 0 ; kM < Mnvec ; kM++)
            { 
                int64_t jM = Mh [kM] ;
                C_to_M [jM] = kM ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // construct C_to_A mapping
    //--------------------------------------------------------------------------

    if (A_is_hyper && Ch != Ah)
    {
        // allocate C_to_A
        GB_MALLOC_MEMORY (C_to_A, Cnvec, sizeof (int64_t)) ;
        if (C_to_A == NULL)
        { 
            // out of memory
            GB_FREE_MEMORY (C_to_M, Cnvec, sizeof (int64_t)) ;
            return (GB_OUT_OF_MEMORY) ;
        }

        // compute C_to_A
        ASSERT (C_is_hyper) ;
        const int64_t *restrict Ap = A->p ;
        #pragma omp parallel for num_threads(nthreads)
        for (int64_t k = 0 ; k < Cnvec ; k++)
        { 
            int64_t pA, pA_end, kA = 0 ;
            int64_t j = Ch [k] ;
            GB_lookup (true, Ah, Ap, &kA, Anvec-1, j, &pA, &pA_end) ;
            C_to_A [k] = (pA < pA_end) ? kA : -1 ;
        }
    }

    //--------------------------------------------------------------------------
    // construct C_to_B mapping
    //--------------------------------------------------------------------------

    if (B_is_hyper && Ch != Bh)
    {
        // allocate C_to_B
        GB_MALLOC_MEMORY (C_to_B, Cnvec, sizeof (int64_t)) ;
        if (C_to_B == NULL)
        { 
            // out of memory
            GB_FREE_MEMORY (C_to_M, Cnvec, sizeof (int64_t)) ;
            GB_FREE_MEMORY (C_to_A, Cnvec, sizeof (int64_t)) ;
            return (GB_OUT_OF_MEMORY) ;
        }

        // compute C_to_B
        ASSERT (C_is_hyper) ;
        const int64_t *restrict Bp = B->p ;
        #pragma omp parallel for num_threads(nthreads)
        for (int64_t k = 0 ; k < Cnvec ; k++)
        { 
            int64_t pB, pB_end, kB = 0 ;
            int64_t j = Ch [k] ;
            GB_lookup (true, Bh, Bp, &kB, Bnvec-1, j, &pB, &pB_end) ;
            C_to_B [k] = (pB < pB_end) ? kB : -1 ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*p_Cnvec      ) = Cnvec ;
    (*Ch_handle    ) = Ch ;
    (*C_to_M_handle) = C_to_M ;
    (*C_to_A_handle) = C_to_A ;
    (*C_to_B_handle) = C_to_B ;

    //--------------------------------------------------------------------------
    // The code below describes what the output contains:
    //--------------------------------------------------------------------------

    #ifdef GB_DEBUG
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

        // vectors j in Ch are sorted, and in the range 0:n-1
        ASSERT (j >= 0 && j < n) ;
        ASSERT (j > jlast) ;
        jlast = j ;

        // see if A (:,j) exists
        if (C_to_A != NULL)
        {
            // A is hypersparse
            ASSERT (A->is_hyper)
            int64_t kA = C_to_A [k] ;
            ASSERT (kA >= -1 && kA < A->nvec) ;
            if (kA >= 0)
            {
                int64_t jA = A->h [kA] ;
                ASSERT (j == jA) ;
            }
        }
        else if (A->is_hyper)
        {
            ASSERT (Ch == A->h) ;
        }

        // see if B (:,j) exists
        if (C_to_B != NULL)
        {
            // B is hypersparse
            ASSERT (B->is_hyper)
            int64_t kB = C_to_B [k] ;
            ASSERT (kB >= -1 && kB < B->nvec) ;
            if (kB >= 0)
            {
                int64_t jB = B->h [kB] ;
                ASSERT (j == jB) ;
            }
        }
        else if (B->is_hyper)
        {
            ASSERT (Ch == B->h) ;
        }

        // see if M (:,j) exists
        if (C_to_M != NULL)
        {
            // M is hypersparse
            ASSERT (M->is_hyper)
            int64_t kM = C_to_M [k] ;
            ASSERT (kM >= -1 && kM < M->nvec) ;
            if (kM >= 0)
            {
                int64_t jM = M->h [kM] ;
                ASSERT (j == jM) ;
            }
        }
    }
    #endif

    return (GrB_SUCCESS) ;
}

