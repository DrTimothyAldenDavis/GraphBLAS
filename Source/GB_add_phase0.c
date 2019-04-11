//------------------------------------------------------------------------------
// GB_add_phase0: find vectors of C to compute for C<M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The eWise add of two matrices, C=A+B, C<M>=A+B, or C<!M>=A+B starts with
// this phase, which determines which vectors of C need to be computed.

// On input, A and B are the two matrices being added, and M is the optional
// mask matrix.  If present (not NULL) then it is not complemented.  This
// function does not consider the complemented-Mask case (use M=NULL for that
// case).

// On output, two integers (max_Cnvec and Cnvec) and up to 4 arrays are
// returned, either NULL or of size max_Cnvec.  If not NULL, only the first
// Cnvec entries in each array is used.  Let n = A->vdim be the vector
// dimension of A, B, M and C.

//      Ch:  the list of vectors to compute.  If not NULL, Ch [k] = j is the
//      kth vector in C to compute, which will become the hyperlist C->h of C.
//      Note that some of these vectors may turn out to be empty, because of
//      the mask, or because the column j appeared in A or B, but is empty.  If
//      Ch is NULL then it is an implicit list of size n, and Ch [k] == k for
//      all k = 0:n-1.  In this case, C will be a standard matrix, not
//      hypersparse.

//      C_to_M:  if not NULL, then C_to_M [k] = kM if the kth column j = Ch [k]
//      in the Ch appears in the Mask, as j = Mh [kM].  If j does not appear in
//      M, then C_to_M [k] = -1.  If M is not present, or not hypersparse, then
//      C_to_M is returned as NULL.

//      C_to_A:  if not NULL, then C_to_A [k] = kA if the kth column j = Ch [k]
//      in the Ch appears in A, as j = Ah [kA].  If j does not appear in A,
//      then C_to_A [k] = -1.  If A is not hypersparse, then C_to_A is returned
//      as NULL.

//      C_to_B:  if not NULL, then C_to_B [k] = kB if the kth column j = Ch [k]
//      in the Ch appears in B, as j = Bh [kB].  If j does not appear in B,
//      then C_to_B [k] = -1.  If B is not hypersparse, then C_to_B is returned
//      as NULL.

#include "GB.h"

//------------------------------------------------------------------------------
// GB_allocate_result
//------------------------------------------------------------------------------

static inline bool GB_allocate_result
(
    int64_t max_Cnvec,
    int64_t **Ch_handle,
    int64_t **C_to_M_handle,
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
    if (C_to_M_handle != NULL)
    {
        GB_MALLOC_MEMORY (*C_to_M_handle, max_Cnvec, sizeof (int64_t)) ;
        ok = ok && (*C_to_M_handle != NULL) ;
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
        GB_FREE_MEMORY (*Ch_handle,     max_Cnvec, sizeof (int64_t)) ;
        GB_FREE_MEMORY (*C_to_M_handle, max_Cnvec, sizeof (int64_t)) ;
        GB_FREE_MEMORY (*C_to_A_handle, max_Cnvec, sizeof (int64_t)) ;
        GB_FREE_MEMORY (*C_to_B_handle, max_Cnvec, sizeof (int64_t)) ;
    }
    return (ok) ;
}

//------------------------------------------------------------------------------
// GB_add_phase0:  find the vectors of C for C<M>=A+B
//------------------------------------------------------------------------------

GrB_Info GB_add_phase0
(
    int64_t *p_Cnvec,           // # of vectors to compute in C
    int64_t *p_max_Cnvec,       // size of the 4 following arrays:
    int64_t **Ch_handle,    // output of size max_Cnvec, or NULL
    int64_t **C_to_M_handle,    // output of size max_Cnvec, or NULL
    int64_t **C_to_A_handle,    // output of size max_Cnvec, or NULL
    int64_t **C_to_B_handle,    // output of size max_Cnvec, or NULL
    const GrB_Matrix M,         // optional mask, may be NULL
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (p_Cnvec != NULL) ;
    ASSERT (p_max_Cnvec != NULL) ;
    ASSERT (Ch_handle != NULL) ;
    ASSERT (C_to_M_handle != NULL) ;
    ASSERT (C_to_A_handle != NULL) ;
    ASSERT (C_to_B_handle != NULL) ;
    ASSERT_OK (GB_check (A, "A for add phase0", GB0)) ;
    ASSERT_OK (GB_check (B, "B for add phase0", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for add phase0", GB0)) ;
    ASSERT (A->vdim == B->vdim) ;
    if (M != NULL) ASSERT (A->vdim == M->vdim) ;

    //--------------------------------------------------------------------------
    // swap A and B if A is standard and B is hypersparse
    //--------------------------------------------------------------------------

    if (!A->is_hyper && B->is_hyper)
    {
        return (GB_add_phase0 (p_Cnvec, p_max_Cnvec, Ch_handle,
            C_to_M_handle, C_to_B_handle, C_to_A_handle, M, B, A, Context)) ;
    }

    // Now if the hypersparsity of A and B differ, A will be hypersparse
    // and B will be standard.

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t *restrict Ch = NULL ;
    int64_t *restrict C_to_M = NULL ;
    int64_t *restrict C_to_A = NULL ;
    int64_t *restrict C_to_B = NULL ;

    (*Ch_handle) = NULL ;
    (*C_to_M_handle) = NULL ;
    (*C_to_A_handle) = NULL ;
    (*C_to_B_handle) = NULL ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // find # of non-empty vectors of M, A, and B
    //--------------------------------------------------------------------------

    if (M != NULL && M->nvec_nonempty < 0)
    {
        M->nvec_nonempty = GB_nvec_nonempty (M, Context) ;
    }

    if (A->nvec_nonempty < 0)
    {
        A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    }

    if (B->nvec_nonempty < 0)
    {
        B->nvec_nonempty = GB_nvec_nonempty (B, Context) ;
    }

    //--------------------------------------------------------------------------
    // get content of M, A, and B
    //--------------------------------------------------------------------------

    int64_t n = A->vdim ;
    int64_t max_Cnvec ;
    int64_t Cnvec = 0 ;

    int64_t Anvec = A->nvec ;
    int64_t Bnvec = B->nvec ;
    int64_t Mnvec = (M == NULL) ? n : M->nvec ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Mp = (M == NULL) ? NULL : M->p ;
    const int64_t *restrict Mh = (M == NULL) ? NULL : M->h ;

    int64_t Mnvec_nonempty = (M == NULL) ? n : M->nvec_nonempty ;

    //--------------------------------------------------------------------------
    // helper macros
    //--------------------------------------------------------------------------

    // helper macro to search for a non-empty vector j in a hypersparse A->h.
    // found is true if j == A->h [kA], and if vector j has at least one entry.
    #define GB_LOOKUP(j,A,found)                                        \
        int64_t p ## A, p ## A ## _end ;                                \
        bool found = (GB_lookup (true, A ## h, A ## p,                  \
            &k ## A, A ## nvec-1, j, &p ## A, &p ## A ## _end)          \
            && (p ## A < p ## A ## _end)) ;

    // # of nonzeros in the kth vector of A
    #define GB_JNZ(k,A) (A ## p [k+1] - A ## p [k])

    //--------------------------------------------------------------------------
    // find the set union of the non-empty vectors of A and B
    //--------------------------------------------------------------------------

    if (A->is_hyper && B->is_hyper)
    {

        //----------------------------------------------------------------------
        // both A and B are hypersparse
        //----------------------------------------------------------------------

        // C will be hypersparse, so Ch is allocated.

        max_Cnvec = A->nvec_nonempty + B->nvec_nonempty ;
        max_Cnvec = GB_IMIN (max_Cnvec, Mnvec_nonempty) ;
        max_Cnvec = GB_IMIN (max_Cnvec, n) ;

        if (!GB_allocate_result (max_Cnvec, &Ch,
            (M != NULL && M->is_hyper) ? (&C_to_M) : NULL, &C_to_A, &C_to_B))
        {
            return (GB_OUT_OF_MEMORY) ;
        }

        if (M == NULL)
        {

            //------------------------------------------------------------------
            // A and B hypersparse, M not present
            //------------------------------------------------------------------

            // merge Ah and Bh into Ch
            int64_t kA = 0 ;
            int64_t kB = 0 ;
            for ( ; kA < Anvec && kB < Bnvec ; Cnvec++)
            {
                int64_t jA = Ah [kA] ;
                int64_t jB = Bh [kB] ;
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
                    // j appears in both A and B
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
                    int64_t jA = Ah [kA] ;
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
        else if (!M->is_hyper)
        {

            //------------------------------------------------------------------
            // A and B hypersparse, M standard
            //------------------------------------------------------------------

            // merge Ah and Bh into Ch
            int64_t kA = 0 ;
            int64_t kB = 0 ;
            while (kA < Anvec && kB < Bnvec)
            {
                int64_t jA = Ah [kA] ;
                int64_t jB = Bh [kB] ;
                if (jA < jB)
                {
                    // append jA to Ch
                    if (GB_JNZ (jA, M) > 0)
                    {
                        Ch     [Cnvec] = jA ;
                        C_to_A [Cnvec] = kA ;
                        C_to_B [Cnvec] = -1 ;
                        Cnvec++ ;
                    }
                    kA++ ;
                }
                else if (jB < jA)
                {
                    // append jB to Ch
                    if (GB_JNZ (jB, M) > 0)
                    {
                        Ch     [Cnvec] = jB ;
                        C_to_A [Cnvec] = -1 ;
                        C_to_B [Cnvec] = kB ;
                        Cnvec++ ;
                    }
                    kB++ ;
                }
                else
                {
                    // j appears in both A and B
                    if (GB_JNZ (jA, M) > 0)
                    {
                        Ch     [Cnvec] = jA ;
                        C_to_A [Cnvec] = kA ;
                        C_to_B [Cnvec] = kB ;
                        Cnvec++ ;
                    }
                    kA++ ;
                    kB++ ;
                }
            }
            if (kA < Anvec)
            {
                // B is exhausted but A is not
                for ( ; kA < Anvec ; kA++)
                {
                    // append jA to Ch
                    int64_t jA = Ah [kA] ;
                    if (GB_JNZ (jA, M) > 0)
                    {
                        Ch     [Cnvec] = jA ;
                        C_to_A [Cnvec] = kA ;
                        C_to_B [Cnvec] = -1 ;
                        Cnvec++ ;
                    }
                }
            }
            else if (kB < Bnvec)
            {
                // A is exhausted but B is not
                for ( ; kB < Bnvec ; kB++)
                {
                    // append jB to Ch
                    int64_t jB = Bh [kB] ;
                    if (GB_JNZ (jB, M) > 0)
                    {
                        Ch     [Cnvec] = jB ;
                        C_to_A [Cnvec] = -1 ;
                        C_to_B [Cnvec] = kB ;
                    }
                }
            }

        }
        else if (Mnvec_nonempty < 16 * (Anvec + Bnvec))
        {

            //------------------------------------------------------------------
            // M, A, and B hypersparse; M is very sparse
            //------------------------------------------------------------------

            // iterate through all non-empty vectors of M

            int64_t kA = 0 ;
            int64_t kB = 0 ;
            GBI_for_each_vector (M)
            {
                // get M(:,j)
                GBI_jth_iteration (j, pstart, pend) ;
                int64_t mjnz = (pend - pstart) ;
                if (mjnz > 0)
                {
                    // j is in M
                    GB_LOOKUP (j, A, afound) ;
                    GB_LOOKUP (j, B, bfound) ;
                    if (afound || bfound)
                    {
                        // j is in A or B
                        Ch     [Cnvec] = j ;
                        C_to_M [Cnvec] = Iter_k ;
                        C_to_A [Cnvec] = afound ? kA : -1 ;
                        C_to_B [Cnvec] = bfound ? kB : -1 ;
                        Cnvec++ ;
                    }
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // M, A, and B hypersparse
            //------------------------------------------------------------------

            // merge Ah and Bh into Ch
            int64_t kM = 0 ;
            int64_t kA = 0 ;
            int64_t kB = 0 ;
            while (kA < Anvec && kB < Bnvec)
            {
                int64_t jA = Ah [kA] ;
                int64_t jB = Bh [kB] ;
                if (jA < jB)
                {
                    // append jA to Ch     if jA is in M
                    GB_LOOKUP (jA, M, mfound) ;
                    if (mfound)
                    {
                        Ch     [Cnvec] = jA ;
                        C_to_M [Cnvec] = kM ;
                        C_to_A [Cnvec] = kA ;
                        C_to_B [Cnvec] = -1 ;
                        Cnvec++ ;
                    }
                    kA++ ;
                }
                else if (jB < jA)
                {
                    // append jB to Ch     if jB is in M
                    GB_LOOKUP (jB, M, mfound) ;
                    if (mfound)
                    {
                        Ch     [Cnvec] = jB ;
                        C_to_M [Cnvec] = kM ;
                        C_to_A [Cnvec] = -1 ;
                        C_to_B [Cnvec] = kB ;
                        Cnvec++ ;
                    }
                    kB++ ;
                }
                else
                {
                    // j appears in both A and B, add to C if found in M
                    GB_LOOKUP (jA, M, mfound) ;
                    if (mfound)
                    {
                        Ch     [Cnvec] = jA ;
                        C_to_M [Cnvec] = kM ;
                        C_to_A [Cnvec] = kA ;
                        C_to_B [Cnvec] = kB ;
                        Cnvec++ ;
                    }
                    kA++ ;
                    kB++ ;
                }
            }
            if (kA < Anvec)
            {
                // B is exhausted but A is not
                for ( ; kA < Anvec ; kA++)
                {
                    // append jA to Ch     if nonempty in M
                    int64_t jA = Ah [kA] ;
                    GB_LOOKUP (jA, M, mfound) ;
                    if (mfound)
                    {
                        Ch     [Cnvec] = jA ;
                        C_to_M [Cnvec] = kM ;
                        C_to_A [Cnvec] = kA ;
                        C_to_B [Cnvec] = -1 ;
                        Cnvec++ ;
                    }
                }
            }
            else if (kB < Bnvec)
            {
                // A is exhausted but B is not
                for ( ; kB < Bnvec ; kB++)
                {
                    // append jB to Ch
                    int64_t jB = Bh [kB] ;
                    GB_LOOKUP (jB, M, mfound) ;
                    if (mfound)
                    {
                        Ch     [Cnvec] = jB ;
                        C_to_M [Cnvec] = kM ;
                        C_to_A [Cnvec] = -1 ;
                        C_to_B [Cnvec] = kB ;
                        Cnvec++ ;
                    }
                }
            }
        }

    }
    else if (A->is_hyper && !(B->is_hyper))
    {

        //----------------------------------------------------------------------
        // A is hypersparse, B is standard
        //----------------------------------------------------------------------

        if (M == NULL)
        {

            //------------------------------------------------------------------
            // A is hypersparse, B is standard, M is not present
            //------------------------------------------------------------------

            // C will be standard

            max_Cnvec = n ;
            if (!GB_allocate_result (max_Cnvec, NULL, NULL, &C_to_A, NULL))
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
                int64_t jA = Ah [kA] ;
                C_to_A [jA] = kA ;
            }

        }
        else if (!M->is_hyper)
        {

            //------------------------------------------------------------------
            // A is hypersparse, B is standard, M is standard
            //------------------------------------------------------------------

            // C will be standard

            max_Cnvec = n ;
            if (!GB_allocate_result (max_Cnvec, NULL, NULL, &C_to_A, NULL))
            {
                return (GB_OUT_OF_MEMORY) ;
            }

            #pragma omp parallel for num_threads(nthreads)
            for (int64_t j = 0 ; j < n ; j++)
            {
                C_to_A [j] = -1 ;
            }

            // scatter Ah into C_to_A, but only if M(:,j) has at least one entry
            #pragma omp parallel for num_threads(nthreads)
            for (int64_t kA = 0 ; kA < Anvec ; kA++)
            {
                int64_t jA = Ah [kA] ;
                if (GB_JNZ (jA, M) > 0)
                {
                    C_to_A [jA] = kA ;
                }
            }

        }
        else
        {

            //------------------------------------------------------------------
            // A is hypersparse, B is standard, M is hypersparse
            //------------------------------------------------------------------

            // C will be hypersparse, with Ch == Mh

            max_Cnvec = Mnvec ;
            if (!GB_allocate_result (max_Cnvec, &Ch, NULL, &C_to_A, NULL))
            {
                return (GB_OUT_OF_MEMORY) ;
            }

            // copy Mh into Ch
            GB_memcpy (Ch, Mh, Mnvec * sizeof (int64_t), nthreads) ;

            // scatter Ah into C_to_A
            #pragma omp parallel for num_threads(nthreads)
            for (int64_t kM = 0 ; kM < Mnvec ; kM++)
            {
                int64_t jM = Mh [kM] ;
                int64_t kA = 0 ;
                GB_LOOKUP (jM, A, afound) ;
                C_to_A [kM] = (afound) ? kA : (-1) ;
            }
        }

        Cnvec = max_Cnvec ;

    }
    else
    {

        //----------------------------------------------------------------------
        // A and B are both standard
        //----------------------------------------------------------------------

        // If A is standard and B hypersparse, they have been swapped already
        ASSERT (!A->is_hyper && !B->is_hyper) ;

        if (M == NULL || !M->is_hyper || n == 1)
        {

            //------------------------------------------------------------------
            // A and B are standard; M is standard, not present, or a vector
            //------------------------------------------------------------------

            // nothing to do.  C is standard, not hypersparse.
            max_Cnvec = n ;

        }
        else
        {
            //------------------------------------------------------------------
            // A and B are standard; M is hypersparse
            //------------------------------------------------------------------

            // C will be hypersparse, with Ch == Mh

            max_Cnvec = Mnvec ;
            if (!GB_allocate_result (max_Cnvec, &Ch, NULL, NULL, NULL))
            {
                return (GB_OUT_OF_MEMORY) ;
            }

            // copy Mh into Ch
            GB_memcpy (Ch, Mh, Mnvec * sizeof (int64_t), nthreads) ;

        }

        Cnvec = max_Cnvec ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // The code below describes what the output contains:

    #ifndef NDEBUG
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

        // printf ("k "GBd" j "GBd"\n", k, j) ;

        // columns j in Ch are sorted, and in the range 0:n-1
        ASSERT (j >= 0 && j < n) ;
        ASSERT (j > jlast) ;
        jlast = j ;

        // see if M (:,j) exists
        if (C_to_M != NULL)
        {
            // M is hypersparse, and present on input
            ASSERT (M != NULL && M->is_hyper) ;
            int64_t kM = C_to_M [k] ;
            ASSERT (kM >= -1 && kM < M->nvec) ;
            if (kM >= 0)
            {
                int64_t jM = M->h [kM] ;
                ASSERT (j == jM) ;
            }
        }
        else
        {
            // M is not present, or M is in standard sparse form
            ASSERT (M == NULL || !M->is_hyper) ;
        }

        // see if A (:,j) exists
        if (C_to_A != NULL)
        {
            // A is hypersparse
            ASSERT (A->is_hyper) ;
            int64_t kA = C_to_A [k] ;
            ASSERT (kA >= -1 && kA < A->nvec) ;
            if (kA >= 0)
            {
                int64_t jA = A->h [kA] ;
                ASSERT (j == jA) ;
            }
        }
        else
        {
            // A is in standard sparse form
            ASSERT (!A->is_hyper) ;
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
            ASSERT (!B->is_hyper) ;
        }

    }
    #endif

    ASSERT (Cnvec <= max_Cnvec) ;
    (*p_Cnvec) = Cnvec ;
    (*p_max_Cnvec) = max_Cnvec ;
    (*Ch_handle    ) = Ch ;
    (*C_to_M_handle) = C_to_M ;
    (*C_to_A_handle) = C_to_A ;
    (*C_to_B_handle) = C_to_B ;
    return (GrB_SUCCESS) ;
}

