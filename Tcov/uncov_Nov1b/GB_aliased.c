//------------------------------------------------------------------------------
// GB_aliased: determine if two matrices are aliased
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Returns true if A == B (and not NULL), or if any component A and B are
// aliased to each other.  In the latter case, that component of A and B will
// always be shallow, in either A or B, or both.  NULL pointers are not
// aliased.

#include "GB.h"

// true if pointers p1 and p2 are aliased and not NULL
#define GB_POINTER_ALIASED(p1,p2) ((p1) == (p2) && (p1) != NULL)

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
bool GB_aliased             // determine if A and B are aliased
(
    GrB_Matrix A,           // input A matrix
    GrB_Matrix B            // input B matrix
)
{

    //--------------------------------------------------------------------------
    // check the matrices themselves
    //--------------------------------------------------------------------------

    if (A == NULL || B == NULL)
    {   GB_cov[2569]++ ;
// covered (2569): 5001902
        // this is not an error condition; NULL pointers are not an alias
        return (false) ;
    }

    if (A == B)
    {   GB_cov[2570]++ ;
// covered (2570): 324994
        return (true) ;
    }

    //--------------------------------------------------------------------------
    // check their content
    //--------------------------------------------------------------------------

    bool aliased = false ;

    if (GB_POINTER_ALIASED (A->h, B->h))
    {   GB_cov[2571]++ ;
// covered (2571): 2
        ASSERT (A->h_shallow || B->h_shallow) ;
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->p, B->p))
    {   GB_cov[2572]++ ;
// NOT COVERED (2572):
GB_GOTCHA ;
        ASSERT (A->p_shallow || B->p_shallow) ;
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->b, B->b))
    {   GB_cov[2573]++ ;
// covered (2573): 31036
        ASSERT (A->b_shallow || B->b_shallow) ;
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->i, B->i))
    {   GB_cov[2574]++ ;
// NOT COVERED (2574):
GB_GOTCHA ;
        ASSERT (A->i_shallow || B->i_shallow) ;
        aliased = true ;
    }

    if (GB_POINTER_ALIASED (A->x, B->x))
    {   GB_cov[2575]++ ;
// covered (2575): 67680
        ASSERT (A->x_shallow || B->x_shallow) ;
        aliased = true ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (aliased) ;
}

