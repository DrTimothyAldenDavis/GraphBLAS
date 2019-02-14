//------------------------------------------------------------------------------
// GB_free_slice: free a set of slices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

#include "GB.h"

GrB_Info GB_free_slice
(
    GrB_Matrix **BsliceHandle,      // Bslice [t] is a slice of B
    int nthreads                    // number of slices
)
{

    if (BsliceHandle != NULL)
    {
        GrB_Matrix *Bslice = (*BsliceHandle) ;
        if (Bslice != NULL)
        {
            for (int t = 0 ; t < nthreads ; t++)
            {
                // can return GrB_PANIC if critical section fails
                GB_MATRIX_FREE (&(Bslice [t])) ;
            }
            GB_FREE_MEMORY (*Bslice, nthreads, sizeof (GrB_Matrix)) ;
        }
    }
    return (GrB_SUCCESS) ;
}

