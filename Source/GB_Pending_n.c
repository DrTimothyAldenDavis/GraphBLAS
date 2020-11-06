//------------------------------------------------------------------------------
// GB_Pending_n: return the # of pending tuples in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_Pending.h"

int64_t GB_Pending_n        // return # of pending tuples in A
(
    GrB_Matrix A
)
{

    int64_t n = 0 ;
    if (A != NULL && A->Pending != NULL)
    { 
        // only sparse and hypersparse matries can have pending tuples
        n = A->Pending->n ;
    }
    return (n) ;
}

