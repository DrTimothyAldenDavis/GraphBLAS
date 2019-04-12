//------------------------------------------------------------------------------
// GB_add_phase2: C=A+B, C<M>=A+B, or C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_add_phase2 computes C=A+B, C<M>=A+B, or C<!M>=A+B.  It is preceded first
// by GB_add_phase0, which computes the list of vectors of C to compute (Ch)
// and their location in A and B (C_to_[AB]).  Next, GB_add_phase1 counts the
// entries in each vector C(:,j) and computes Cp.

// GB_add_phase2 computes the pattern and values of each vector of C(:,j),
// fully in parallel.

// C, M, A, and B can be standard sparse or hypersparse, as determined by
// GB_add_phase0.  All cases of the mask M are handled: not present, present
// and not complemented, and present and complemented.

// PARALLEL: fully parallel except for the last phase, to prune empty
// vectors from C, if it is hypersparse

#include "GB.h"

GrB_Info GB_add_phase2      // C=A+B, C<M>=A+B, or C<!M>=A+B
(
    GrB_Matrix *Chandle,    // output matrix (unallocated on input)
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_BinaryOp op,  // op to perform C = op (A,B)

    // from GB_add_phase1
    const int64_t *restrict Cp,         // vector pointers for C
    const int64_t Cnvec_nonempty,       // # of non-empty vectors in C

    // analysis from GB_add_phase0:
    const int64_t Cnvec,
    const int64_t max_Cnvec,
    const int64_t *restrict Ch,
    const int64_t *restrict C_to_A,
    const int64_t *restrict C_to_B,
    const bool Ch_is_Mh,        // if true, then Ch == M->h

    // original input to GB_add_phased
    const GrB_Matrix M,         // optional mask, may be NULL
    const bool Mask_comp,
    const GrB_Matrix A,
    const GrB_Matrix B,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Cp != NULL) ;
    ASSERT_OK (GB_check (op, "op for add phase2", GB0)) ;
    ASSERT_OK (GB_check (A, "A for add phase2", GB0)) ;
    ASSERT_OK (GB_check (B, "B for add phase2", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (M, "M for add phase2", GB0)) ;
    ASSERT (A->vdim == B->vdim) ;

    ASSERT (GB_Type_compatible (ctype,   op->ztype)) ;
    ASSERT (GB_Type_compatible (ctype,   A->type)) ;
    ASSERT (GB_Type_compatible (ctype,   B->type)) ;
    ASSERT (GB_Type_compatible (A->type, op->xtype)) ;
    ASSERT (GB_Type_compatible (B->type, op->ytype)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    int64_t cnz = Cp [Cnvec] ;
    (*Chandle) = NULL ;

    // C is hypersparse if both A and B are (contrast with GrB_Matrix_emult),
    // or if M is present, not complemented, and hypersparse.
    // C acquires the same hyperatio as A.

    bool C_is_hyper = (Ch != NULL) ;

    // allocate the result C (but do not allocate C->p or C->h)
    GrB_Info info ;
    GrB_Matrix C = NULL ;           // allocate a new header for C
    GB_CREATE (&C, ctype, A->vlen, A->vdim, GB_Ap_null, C_is_csc,
        GB_SAME_HYPER_AS (C_is_hyper), A->hyper_ratio, Cnvec, cnz, true,
        Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory; caller must free Cp, Ch, C_to_A, C_to_B
        return (info) ;
    }

    // add Cp as the vector pointers for C, from GB_add_phase1
    C->p = Cp ;

    // add Ch as the the hypersparse list for C, from GB_add_phase0
    if (C_is_hyper)
    { 
        C->h = Ch ;
        C->nvec = Cnvec ;
    }

    C->nvec_nonempty = Cnvec_nonempty ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // get content of C, M, A, and B
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ai = A->i ;
    const GB_void *restrict Ax = A->x ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bi = B->i ;
    const GB_void *restrict Bx = B->x ;

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

    int64_t *restrict Ci = C->i ;
    GB_void *restrict Cx = C->x ;

    GxB_binary_function fadd = op->function ;

    size_t csize = ctype->size ;
    size_t asize = A->type->size ;
    size_t bsize = B->type->size ;

    size_t xsize = op->xtype->size ;
    size_t ysize = op->ytype->size ;
    size_t zsize = op->ztype->size ;

    GB_cast_function
        cast_A_to_X, cast_B_to_Y, cast_A_to_C, cast_B_to_C, cast_Z_to_C ;
    cast_A_to_X = GB_cast_factory (op->xtype->code, A->type->code) ;
    cast_B_to_Y = GB_cast_factory (op->ytype->code, B->type->code) ;
    cast_A_to_C = GB_cast_factory (ctype->code,     A->type->code) ;
    cast_B_to_C = GB_cast_factory (ctype->code,     B->type->code) ;
    cast_Z_to_C = GB_cast_factory (ctype->code,     op->ztype->code) ;

    //--------------------------------------------------------------------------
    // compute each vector of C
    //--------------------------------------------------------------------------

    // TODO make this Template/GB_add_template.c:

    #pragma omp parallel for num_threads(nthreads)
    for (int64_t k = 0 ; k < Cnvec ; k++)
    {

        //----------------------------------------------------------------------
        // scalar workspace
        //----------------------------------------------------------------------

        GB_void xwork [xsize] ;
        GB_void ywork [ysize] ;
        GB_void zwork [zsize] ;

        //----------------------------------------------------------------------
        // get j, the kth vector of C
        //----------------------------------------------------------------------

        int64_t j = (Ch == NULL) ? k : Ch [k] ;
        int64_t pC     = Cp [k] ;
        int64_t pC_end = Cp [k+1] ;
        int64_t cjnz = pC_end - pC ;
        // printf ("phase2 j : "GBd" cjnz "GBd"\n", j, cjnz) ;
        if (cjnz == 0) continue ;

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
        // compute C(:,j)
        //----------------------------------------------------------------------

        if (M == NULL || (M != NULL && mjnz == 0 && Mask_comp))
        {

            //------------------------------------------------------------------
            // No mask, or M(:,j) is empty and complemented
            //------------------------------------------------------------------

            for ( ; pA < pA_end && pB < pB_end ; pC++)
            {
                // both A(iA,j) and B (iB,j) are at head of lists to merge
                int64_t iA = Ai [pA] ;
                int64_t iB = Bi [pB] ;
                if (iA < iB)
                { 
                    // C (iA,j) = A (iA,j)
                    Ci [pC] = iA ;
                    // Cx [pC] = Ax [pA]
                    cast_A_to_C (Cx +(pC*csize), Ax +(pA*asize), csize) ;
                    pA++ ;
                }
                else if (iA > iB)
                { 
                    // C (iB,j) = B (iB,j)
                    Ci [pC] = iB ;
                    // Cx [pC] = Bx [pB]
                    cast_B_to_C (Cx +(pC*csize), Bx +(pB*bsize), csize) ;
                    pB++ ;
                }
                else
                { 
                    // C (i,j) = A (i,j) + B (i,j)
                    Ci [pC] = iB ;
                    // xwork = (xtype) Ax [pA]
                    cast_A_to_X (xwork, Ax +(pA*asize), asize) ;
                    // ywork = (ytype) Bx [pA]
                    cast_B_to_Y (ywork, Bx +(pB*bsize), bsize) ;
                    // zwork = fadd (xwork, ywork), result is ztype
                    fadd (zwork, xwork, ywork) ;
                    // Cx [pC] = (ctype) zwork
                    cast_Z_to_C (Cx +(pC*csize), zwork, csize) ;
                    pA++ ;
                    pB++ ;
                }
            }

            //------------------------------------------------------------------
            // A (:,j) or B (:,j) have entries left; not both
            //------------------------------------------------------------------

            for ( ; pA < pA_end ; pA++, pC++)
            { 
                // C (i,j) = A (i,j)
                Ci [pC] = Ai [pA] ;
                // Cx [pC] = (ctype) Ax [pA]
                cast_A_to_C (Cx +(pC*csize), Ax +(pA*asize), csize) ;
            }
            for ( ; pB < pB_end ; pB++, pC++)
            { 
                // C (i,j) = B (i,j)
                Ci [pC] = Bi [pB] ;
                // Cx [pC] = (ctype) Bx [pB]
                cast_B_to_C (Cx +(pC*csize), Bx +(pB*bsize), csize) ;
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

                // both A(iA,j) and B (iB,j) are at head of lists to merge
                if (iA < iB)
                {
                    if (mij)
                    { 
                        // C (i,j) = A (i,j)
                        // Cx [pC] = Ax [pA]
                        Ci [pC] = i ;
                        cast_A_to_C (Cx +(pC*csize), Ax +(pA*asize), csize) ;
                        pC++ ;
                    }
                    pA++ ;
                }
                else if (iA > iB)
                {
                    if (mij)
                    { 
                        // C (i,j) = B (i,j)
                        // Cx [pC] = Bx [pB]
                        Ci [pC] = i ;
                        cast_B_to_C (Cx +(pC*csize), Bx +(pB*bsize), csize) ;
                        pC++ ;
                    }
                    pB++ ;
                }
                else
                {
                    if (mij)
                    { 
                        // C (i,j) = A (i,j) + B (i,j)
                        Ci [pC] = i ;
                        // xwork = (xtype) Ax [pA]
                        cast_A_to_X (xwork, Ax +(pA*asize), asize) ;
                        // ywork = (ytype) Bx [pA]
                        cast_B_to_Y (ywork, Bx +(pB*bsize), bsize) ;
                        // zwork = fadd (xwork, ywork), result is ztype
                        fadd (zwork, xwork, ywork) ;
                        // Cx [pC] = (ctype) zwork
                        cast_Z_to_C (Cx +(pC*csize), zwork, csize) ;
                        pC++ ;
                    }
                    pA++ ;
                    pB++ ;
                }
            }
        }

        // printf ("pC "GBd" pC_end "GBd"\n", pC, pC_end) ;
        ASSERT (pC == pC_end) ;
    }

    //--------------------------------------------------------------------------
    // prune empty vectors from Ch
    //--------------------------------------------------------------------------

    // TODO this is sequential.  Could use a parallel cumulative sum of the
    // Cp > 0 condition, and then an out-of-place copy to new Ch and Cp arrays.

    // printf ("Cnvec_nonempty "GBd" Cnvec "GBd"\n", Cnvec_nonempty, Cnvec) ;
    if (C->is_hyper && Cnvec_nonempty < Cnvec)
    {
        int64_t *restrict Cp = C->p ;
        int64_t *restrict Ch = C->h ;
        int64_t cnvec_new = 0 ;
        for (int64_t k = 0 ; k < Cnvec ; k++)
        {
            int64_t cjnz = Cp [k+1] - Cp [k] ;
            // printf ("consider "GBd" = "GBd"\n", k, cjnz) ;
            if (cjnz > 0)
            { 
                // printf ("keep k: "GBd" j: "GBd"\n", k, Ch [k]) ;
                Cp [cnvec_new] = Cp [k] ;
                Ch [cnvec_new] = Ch [k] ;
                cnvec_new++ ;
            }
        }
        Cp [cnvec_new] = Cp [Cnvec] ;
        C->nvec = cnvec_new ;
        // printf ("cnvec_new "GBd"\n", cnvec_new) ;
        ASSERT (cnvec_new == Cnvec_nonempty) ;
        // reduce the size of Cp and Ch (this cannot fail)
        bool ok ;
        GB_REALLOC_MEMORY (C->p, cnvec_new, GB_IMAX (2, Cnvec+1),
            sizeof (int64_t), &ok) ;
        ASSERT (ok) ;
        GB_REALLOC_MEMORY (C->h, cnvec_new, max_Cnvec, sizeof (int64_t), &ok) ;
        ASSERT (ok) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // caller must free C_to_A, and C_to_B, but not Cp or Ch
    ASSERT_OK (GB_check (C, "C output for add phase2", GB0)) ;
    (*Chandle) = C ;
    return (GrB_SUCCESS) ;
}

