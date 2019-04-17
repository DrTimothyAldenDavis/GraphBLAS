//------------------------------------------------------------------------------
// GB_add_phase1: find # of entries in C=A+B, C<M>=A+B, or C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_add_phase1 counts the number of entries in each vector of C, for C=A+B,
// C<M>=A+B, or C<!M>=A+B, and then does a cumulative sum to find Cp.
// GB_add_phase1 is preceded by GB_add_phase0, which finds the non-empty
// vectors of C.  This phase is done entirely in parallel.

// C, M, A, and B can be standard sparse or hypersparse, as determined by
// GB_add_phase0.  All cases of the mask M are handled: not present, present
// and not complemented, and present and complemented.

// GB_wait computes A=A+T where T is the matrix of the assembled pending
// tuples.  A and T are disjoint, so this function does not need to examine
// the pattern of A and T at all.  No mask is used in this case.

// PARALLEL: done

#include "GB.h"

GrB_Info GB_add_phase1                  // count nnz in each C(:,j)
(
    int64_t **Cp_handle,                // output of size Cnvec+1
    int64_t *Cnvec_nonempty,            // # of non-empty vectors in C
    const bool A_and_B_are_disjoint,    // if true, then A and B are disjoint

    // analysis from GB_add_phase0
    const int64_t Cnvec,
    const int64_t *restrict Ch,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Ch_is_Mh,                // if true, then Ch == M->h

    const GrB_Matrix M,                 // optional mask, may be NULL
    const bool Mask_comp,               // if true, then M is complemented
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp_handle != NULL) ;
    ASSERT_OK (GB_check (A, "A for add phase1", GB0)) ;
    ASSERT_OK (GB_check (B, "B for add phase1", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for add phase1", GB0)) ;
    ASSERT (A->vdim == B->vdim) ;

    int64_t *restrict Cp = NULL ;
    (*Cp_handle) = NULL ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // get content of M, A, and B
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ai = A->i ;
    int64_t vlen = A->vlen ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bi = B->i ;

    const int64_t *restrict Mp = NULL ;
    const int64_t *restrict Mh = NULL ;
    const int64_t *restrict Mi = NULL ;
    const GB_void *restrict Mx = NULL ;
    GB_cast_function cast_M = NULL ;
    size_t msize = 0 ;
    int64_t Mnvec = 0 ;
    bool M_is_hyper = false ;
    if (M != NULL)
    { 
        Mp = M->p ;
        Mh = M->h ;
        Mi = M->i ;
        Mx = M->x ;
        cast_M = GB_cast_factory (GB_BOOL_code, M->type->code) ;
        msize = M->type->size ;
        Mnvec = M->nvec ;
        M_is_hyper = M->is_hyper ;
    }

    //--------------------------------------------------------------------------
    // allocate the result
    //--------------------------------------------------------------------------

    GB_CALLOC_MEMORY (Cp, GB_IMAX (2, Cnvec+1), sizeof (int64_t)) ;
    if (Cp == NULL)
    { 
        return (GB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector of C
    //--------------------------------------------------------------------------

    int64_t cnvec_nonempty = 0 ;

    #pragma omp parallel for num_threads(nthreads) reduction(+:cnvec_nonempty)
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {

        //----------------------------------------------------------------------
        // get j, the kth vector of C
        //----------------------------------------------------------------------

        int64_t j = (Ch == NULL) ? k : Ch [k] ;
        // printf ("phase1 j : "GBd"\n", j) ;

        //----------------------------------------------------------------------
        // get A(:,j)
        //----------------------------------------------------------------------

        int64_t pA = -1 ;
        int64_t pA_end = -1 ;
        int64_t kA = (C_to_A == NULL) ? j : C_to_A [k] ;
        if (kA >= 0)
        { 
            pA     = Ap [kA] ;
            pA_end = Ap [kA+1] ;
        }
        int64_t ajnz = pA_end - pA ;    // nnz (A (:,j))
        // printf ("   ["GBd":"GBd"] ajnz  : "GBd"\n", pA, pA_end, ajnz) ;

        //----------------------------------------------------------------------
        // get B(:,j)
        //----------------------------------------------------------------------

        int64_t pB = -1 ;
        int64_t pB_end = -1 ;
        int64_t kB = (C_to_B == NULL) ? j : C_to_B [k] ;
        if (kB >= 0)
        { 
            pB     = Bp [kB] ;
            pB_end = Bp [kB+1] ;
        }
        int64_t bjnz = pB_end - pB ;    // nnz (B (:,j))
        // printf ("   ["GBd":"GBd"] bjnz  : "GBd"\n", pB, pB_end, bjnz) ;

        //----------------------------------------------------------------------
        // get M(:,j)
        //----------------------------------------------------------------------

        // TODO: if A==M or B==M is aliased, then no need to do GB_lookup

        int64_t pM = -1 ;
        int64_t pM_end = -1 ;
        if (Ch_is_Mh)
        { 
            // Ch is the same as M->h, so binary search is not needed
            ASSERT (Ch != NULL && Mh != NULL && Ch [k] == Mh [k]) ;
            pM     = Mp [k] ;
            pM_end = Mp [k+1] ;
        }
        else if (M != NULL)
        { 
            int64_t kM = 0 ;
            GB_lookup (M_is_hyper, Mh, Mp, &kM, Mnvec-1, j, &pM, &pM_end) ;
        }
        int64_t mjnz = pM_end - pM ;    // nnz (M (:,j))
        // printf ("   ["GBd":"GBd"] mjnz  : "GBd"\n", pM, pM_end, mjnz) ;

        //----------------------------------------------------------------------
        // count nnz (C (:,j))
        //----------------------------------------------------------------------

        int64_t cjnz = 0 ;

        if (M != NULL && mjnz == 0 && !Mask_comp)
        { 

            //------------------------------------------------------------------
            // M(:,j) is empty and not complemented
            //------------------------------------------------------------------

            // C(:,j) is empty, regardless of A(:,j) and B(:,j)
            ;

        }
        else if (M == NULL || (M != NULL && mjnz == 0 && Mask_comp))
        {

            //------------------------------------------------------------------
            // No mask, or M(:,j) is empty and complemented
            //------------------------------------------------------------------

            // if present, M(:,j) is ignored since !M(:,j) is all true

            if (A_and_B_are_disjoint)
            {

                // only used by GB_wait, which computes A+T where T is the
                // matrix of pending tuples for A.  The pattern of pending
                // tuples is always disjoint with the pattern of A.
                cjnz = ajnz + bjnz ;
                // printf ("disjoint cjnz "GBd"\n", cjnz) ;

            }
            else if (ajnz == vlen || bjnz == vlen)
            {
                // if A(:,j) or B(:,j) are dense, then C(:,j) is dense
                // and then cjnz = A->vlen

                cjnz = vlen ;

            }
            else if (ajnz == 0)
            { 

                //--------------------------------------------------------------
                // A(:,j) is empty
                //--------------------------------------------------------------

                cjnz = bjnz ;

            }
            else if (bjnz == 0)
            { 

                //--------------------------------------------------------------
                // B(:,j) is empty
                //--------------------------------------------------------------

                cjnz = ajnz ;

            }
            else if (Ai [pA_end-1] < Bi [pB] || Bi [pB_end-1] < Ai [pA])
            { 

                //--------------------------------------------------------------
                // intersection of A(:,j) and B(:,j) is empty
                //--------------------------------------------------------------

                // the last entry of A(:,j) comes before the first entry
                // of B(:,j), or visa versa
                cjnz = ajnz + bjnz ;

            }
            else if (ajnz > 32 * bjnz)
            {

                //--------------------------------------------------------------
                // A(:,j) is much denser than B(:,j)
                //--------------------------------------------------------------

                // cjnz = ajnz + bjnz - nnz in the intersection

                cjnz = ajnz + bjnz ;
                for ( ; pB < pB_end ; pB++)
                { 
                    int64_t i = Bi [pB] ;
                    // find i in A(:,j)
                    int64_t pright = pA_end ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Ai, pA, pright, found) ;
                    if (found) cjnz-- ;
                }

            }
            else if (bjnz > 32 * ajnz)
            {

                //--------------------------------------------------------------
                // B(:,j) is must denser than A(:,j)
                //--------------------------------------------------------------

                // cjnz = ajnz + bjnz - nnz in the intersection

                cjnz = ajnz + bjnz ;
                for ( ; pA < pA_end ; pA++)
                { 
                    int64_t i = Ai [pA] ;
                    // find i in B(:,j)
                    int64_t pright = pB_end ;
                    bool found ;
                    GB_BINARY_SEARCH (i, Bi, pB, pright, found) ;
                    if (found) cjnz-- ;
                }

            }
            else
            {

                //--------------------------------------------------------------
                // A(:,j) and B(:,j) have about the same # of entries
                //--------------------------------------------------------------

                // linear-time scan of A(:,j) and B(:,j)

                for ( ; pA < pA_end && pB < pB_end ; cjnz++)
                {
                    int64_t iA = Ai [pA] ;
                    int64_t iB = Bi [pB] ;
                    if (iA < iB)
                    { 
                        // A(i,j) exists but not B(i,j)
                        pA++ ;
                    }
                    else if (iB < iA)
                    { 
                        // B(i,j) exists but not A(i,j)
                        pB++ ;
                    }
                    else
                    { 
                        // both A(i,j) and B(i,j) exist
                        pA++ ;
                        pB++ ;
                    }
                }
                cjnz += (pA_end - pA) + (pB_end - pB) ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // M is present
            //------------------------------------------------------------------

            while (pA < pA_end || pB < pB_end)
            {

                //--------------------------------------------------------------
                // get the next i for A(:,j) + B (:,j)
                //--------------------------------------------------------------

                int64_t iA = (pA < pA_end) ? Ai [pA] : INT64_MAX ;
                int64_t iB = (pB < pB_end) ? Bi [pB] : INT64_MAX ;
                int64_t i = GB_IMIN (iA, iB) ;

                // printf ("i : "GBd"\n", i) ;

                //--------------------------------------------------------------
                // get M(i,j)
                //--------------------------------------------------------------

                bool mij = false ;  // M(i,j) false if not present
                int64_t pright = pM_end - 1 ;
                bool found ;
                GB_BINARY_SEARCH (i, Mi, pM, pright, found) ;
                if (found)
                { 
                    cast_M (&mij, Mx +(pM*msize), 0) ;
                }
                if (Mask_comp)
                { 
                    mij = !mij ;
                }

                cjnz += mij ;

                if (i == iA) pA++ ;
                if (i == iB) pB++ ;
            }
        }

        //----------------------------------------------------------------------
        // final count of nnz (C (:,j))
        //----------------------------------------------------------------------

        // printf ("here Cp ["GBd"] = "GBd"\n", k, cjnz) ;
        Cp [k] = cjnz ;
        if (cjnz > 0) cnvec_nonempty++ ;
    }

    //--------------------------------------------------------------------------
    // replace Cp with its cumulative sum and return result
    //--------------------------------------------------------------------------

    GB_cumsum (Cp, Cnvec, Cnvec_nonempty, nthreads) ;
    // printf ("Cnvec_nonempty "GBd"\n", *Cnvec_nonempty) ;
    (*Cp_handle) = Cp ;
    return (GrB_SUCCESS) ;
}

