//------------------------------------------------------------------------------
// GB_subassign_select: determine if S should be constructed for GB_subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_subassign.h"

bool GB_subassign_select            // true if S should be constructed
(
    const GrB_Matrix C,             // output of GB_subassigner
    const int64_t nzMask,           // nnz (M)
    const int64_t anz,              // nnz (A)
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Context Context
)
{

    bool S_Extraction = true ;

    int64_t cnz_to_scan = nzMask ;

    // get C
    const int64_t *restrict Cp = C->p ;
    const int64_t *restrict Ch = C->h ;
    const int64_t Cnvec = C->nvec ;
    const bool C_is_hyper = C->is_hyper ;
    const int64_t cvdim = C->vdim ;

    // TODO do this in parallel
    for (int64_t j = 0 ; S_Extraction && j < nJ ; j++)
    {
        // jC = J [j] ; or J is a colon expression
        int64_t jC = GB_ijlist (J, j, Jkind, Jcolon) ;
        if (jC < 0 || jC >= cvdim)
        { 
            // invalid vector; check them all in GB_subref
            break ;
        }
        // get the C(:,jC) vector where jC = J [j]
        GB_jC_LOOKUP ;
        cnz_to_scan += pC_end - pC_start ;
        if (cnz_to_scan/8 > anz)
        { 
            // nnz(C) + nnz(M) is much larger than nnz(A).  Do not
            // construct S=C(I,J).  Instead, scan through all of A and
            // use binary search to find the corresponding positions in
            // C and M.  Do not use S; use Methods 3, 4, 5, or 6
            // instead.
            S_Extraction = false ;
        }
    }
    if (cnz_to_scan == 0)
    { 
        // C(:,J) and M(:,J) are empty; no need to compute S
        S_Extraction = false ;
    }

    return (S_Extraction) ;
}

