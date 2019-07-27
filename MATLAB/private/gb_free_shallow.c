//------------------------------------------------------------------------------
// gb_free_shallow: free a shallow GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gbmex.h"

void gb_free_shallow        // free a shallow GrB_Matrix
(
    GrB_Matrix *S_handle    // GrB_Matrix to free; set to NULL on output
)
{

    // Since the GraphBLAS matrix S is already a shallow copy of the MATLAB
    // matrix, the exported pointers Sp, Si, Sx, and Sh are discarded.

    GrB_Matrix S = (*S_handle) ;
    S->p = NULL ;
    S->h = NULL ;
    S->i = NULL ;
    S->x = NULL ;
    OK (GrB_Matrix_free (S_handle)) ;
}

