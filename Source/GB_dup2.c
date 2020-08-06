//------------------------------------------------------------------------------
// GB_dup2: make a deep copy of a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C = A, making a deep copy.  The header for C may already exist.

// if numeric is false, C->x is allocated but not initialized.

// OK: BITMAP

#include "GB.h"

GrB_Info GB_dup2            // make an exact copy of a matrix
(
    GrB_Matrix *Chandle,    // handle of output matrix to create 
    const GrB_Matrix A,     // input matrix to copy
    const bool numeric,     // if true, duplicate the numeric values
    const GrB_Type ctype,   // type of C, if numeric is false
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // C = A
    //--------------------------------------------------------------------------

    if (A->nvec_nonempty < 0)
    { 
        A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    }

    // create C; allocate C->p and do not initialize it
    // C has the exact same hypersparsity as A.
    int64_t anz = GB_NNZ (A) ;

    // allocate a new header for C if (*Chandle) is NULL, or reuse the
    // existing header if (*Chandle) is not NULL.
    GrB_Matrix C = (*Chandle) ;
    GrB_Info info = GB_create (&C, numeric ? A->type : ctype, A->vlen, A->vdim,
        GB_Ap_malloc, A->is_csc,
        GB_IS_FULL (A) ? GB_FULL : GB_SAME_HYPER_AS (A->h != NULL),
        A->hyper_switch, A->plen, anz, true, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    // copy the contents of A into C
    int64_t anvec = A->nvec ;
    C->nvec = anvec ;
    C->nvec_nonempty = A->nvec_nonempty ;

    C->jumbled = A->jumbled ;       // C is jumbled if A is jumbled

    int nthreads = GB_nthreads (anvec, chunk, nthreads_max) ;
    if (A->p != NULL)
    { 
        GB_memcpy (C->p, A->p, (anvec+1) * sizeof (int64_t), nthreads) ;
    }
    if (A->h != NULL)
    { 
        GB_memcpy (C->h, A->h, anvec * sizeof (int64_t), nthreads) ;
    }

    nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
    if (A->b != NULL)
    { 
        GB_memcpy (C->b, A->b, anz * sizeof (int8_t), nthreads) ;
    }
    if (A->i != NULL)
    {
        GB_memcpy (C->i, A->i, anz * sizeof (int64_t), nthreads) ;
    }
    if (numeric)
    { 
        GB_memcpy (C->x, A->x, anz * A->type->size, nthreads) ;
    }

    C->magic = GB_MAGIC ;      // C->p and C->h are now initialized
    #ifdef GB_DEBUG
    if (numeric) ASSERT_MATRIX_OK (C, "C duplicate of A", GB0) ;
    #endif

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    (*Chandle) = C ;
    return (GrB_SUCCESS) ;
}

