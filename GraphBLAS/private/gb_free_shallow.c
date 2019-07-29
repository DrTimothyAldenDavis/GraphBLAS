//------------------------------------------------------------------------------
// gb_free_shallow: free a shallow GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gb_matlab.h"

void gb_free_shallow        // free a shallow GrB_Matrix
(
    GrB_Matrix *S_handle    // GrB_Matrix to free; set to NULL on output
)
{

    CHECK_ERROR (S_handle == NULL, "internal gb error") ;

    GrB_Matrix S = (*S_handle) ;

    if (S != NULL)
    {
        // The GraphBLAS matrix S is already a shallow copy of the MATLAB
        // matrix, so the pointers Sp, Si, Sx, and Sh must not be freed.
        S->p = NULL ;
        S->h = NULL ;
        S->i = NULL ;
        S->x = NULL ;
        OK (GrB_Matrix_free (S_handle)) ;
    }
}

