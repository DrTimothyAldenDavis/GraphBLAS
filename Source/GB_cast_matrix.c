//------------------------------------------------------------------------------
// GB_cast_matrix: copy or typecast the values from A into C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The values of C must already be allocated, of size large enough to hold
// the values of A.  The pattern of C must match the pattern of A, but the
// pattern is not accessed (except to compute GB_nnz_held (A)).

// Note that A may contain zombies, or entries not in the bitmap pattern of A
// if A is bitmap, and the values of these entries might be uninitialized
// values in A->x.  All entries are typecasted or memcpy'ed from A->x to C->x,
// including zombies, non-entries, and live entries alike.  valgrind may
// complain about typecasting these uninitialized values, but these warnings
// are false positives.

#include "GB.h"

void GB_cast_matrix         // copy or typecast the values from A into C
(
    GrB_Matrix C,
    GrB_Matrix A,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    const int64_t anz = GB_nnz_held (A) ;
    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (anz, chunk, nthreads_max) ;
    ASSERT (A->iso == C->iso) ;
    if (anz == 0)
    { 
        // nothing to do
        return ;
    }

    //--------------------------------------------------------------------------
    // copy or typecast from A into C
    //--------------------------------------------------------------------------

    if (C->type == A->type)
    {

        //----------------------------------------------------------------------
        // copy A->x into C->x
        //----------------------------------------------------------------------

        if (A->iso)
        { 
GB_GOTCHA ; // A iso, ctype == atype (just memcpy)
            memcpy (C->x, A->x, C->type->size) ;
        }
        else
        { 
            GB_memcpy (C->x, A->x, anz * C->type->size, nthreads) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // typecast A->x into C->x
        //----------------------------------------------------------------------

        if (A->iso)
        { 
            // C->x [0] = (ctype) A->x [0]
            GB_iso_unop (C->x, C->type, GB_ISO_A, NULL, NULL, A, NULL) ;
        }
        else
        { 
            ASSERT (GB_IMPLIES (anz > 0, C->x != NULL)) ;
            GB_cast_array ((GB_void *) C->x, C->type->code, (GB_void *) A->x,
                A->type->code, A->b, A->type->size, anz, nthreads) ;
        }
    }
}

