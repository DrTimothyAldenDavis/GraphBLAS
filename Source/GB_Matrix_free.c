//------------------------------------------------------------------------------
// GB_Matrix_free: free a GrB_Matrix or GrB_Vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Free all the content of a matrix.  After GB_Matrix_free (&A), the header A
// is freed and set to NULL if the header of A was originally dynamically
// allocated.  Otherwise, A is not freed.

#include "GB.h"

void GB_Matrix_free             // free a matrix
(
    GrB_Matrix *Ahandle         // handle of matrix to free
)
{

    if (Ahandle != NULL)
    {
        GrB_Matrix A = *Ahandle ;
        if (A != NULL && (A->magic == GB_MAGIC || A->magic == GB_MAGIC2))
        { 
            // free all content of A
            GB_phbix_free (A) ;
            // free the header of A
            if (A->static_header)
            { 
                // A is static, not a pointer from malloc/calloc, so it cannot
                // be freed.  Just mark it as not containing a valid matrix.
                A->magic = GB_MAGIC2 ;
            }
            else
            { 
                // free the header of A itself
                A->magic = GB_FREED ;       // to help detect dangling pointers
                GB_FREE (*Ahandle) ;
                (*Ahandle) = NULL ;
            }
        }
    }
}

