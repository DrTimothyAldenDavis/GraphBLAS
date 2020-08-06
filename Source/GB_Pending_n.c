//------------------------------------------------------------------------------
// GB_Pending_n: return the # of pending tuples in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// OK: no change for BITMAP (except for comments and explanatory assertions)

#include "GB_Pending.h"

int64_t GB_Pending_n        // return # of pending tuples in A
(
    GrB_Matrix A
)
{

    int64_t n = 0 ;
    if (A != NULL)
    {
        // Any matrix can be checked for this condition ...
        ASSERT (GB_IS_ANY_SPARSITY (A)) ;
        if (A->Pending != NULL)
        { 
            // but only sparse and hypersparse matries can have pending tuples
            ASSERT (!GB_IS_FULL (A)) ;
            ASSERT (!GB_IS_BITMAP (A)) ;
            ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
            n = A->Pending->n ;
        }
    }
    return (n) ;
}

