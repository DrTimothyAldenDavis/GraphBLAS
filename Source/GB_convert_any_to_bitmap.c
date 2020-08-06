//------------------------------------------------------------------------------
// GB_convert_any_to_bitmap: convert any matrix to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input matrix may be jumbled and have zombies, and can still be converted
// to a bitmap.

#include "GB.h"
#define GB_FREE_ALL GB_phbix_free (A) ;

GrB_Info GB_convert_any_to_bitmap   // convert to bitmap
(
    GrB_Matrix A,           // matrix to convert to bitmap
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A being converted to bitmap", GB0) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;    // A can have zombies
    ASSERT (GB_JUMBLED_OK (A)) ;    // A can be jumbled
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // convert A to bitmap
    //--------------------------------------------------------------------------

    if (A->h != NULL)
    { 
        // convert from hypersparse to bitmap
        GB_OK (GB_convert_sparse_to_bitmap (A, Context)) ;
    }
    else if (GB_IS_FULL (A))
    { 
        // convert from full to bitmap
        GB_OK (GB_convert_full_to_bitmap (A, Context)) ;
    }
    else if (GB_IS_BITMAP (A))
    { 
        // already bitmap; nothing to do
        ;
    }
    else
    { 
        // convert from sparse to bitmap
        GB_OK (GB_convert_sparse_to_bitmap (A, Context)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to bitmap", GB0) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

