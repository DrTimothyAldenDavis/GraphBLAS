//------------------------------------------------------------------------------
// GB_is_diagonal: check if A is a diagonal matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Returns true if A is a square diagonal matrix, with all diagonal entries
// present.  Pending tuples are ignored.  Zombies are treated as entries.

// PARALLEL: simple parallel reduction

#include "GB.h"

bool GB_is_diagonal             // true if A is diagonal
(
    const GrB_Matrix A,         // input matrix to examine
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (A != NULL) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // trivial cases
    //--------------------------------------------------------------------------

    int64_t n     = GB_NROWS (A) ;
    int64_t ncols = GB_NCOLS (A) ;

    if (n != ncols)
    { 
        // A is rectangular
        return (false) ;
    }

    int64_t nvals = GB_NNZ (A) ;

    if (n != nvals)
    { 
        // A must have exactly n entries
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // examine each column of A
    //--------------------------------------------------------------------------

    const int64_t *restrict Ai = A->i ;

    GBI_for_each_vector (A)
    { 
        GBI_jth_iteration (j, p, pend) ;
        int64_t ajnz = pend - p ;
        if (ajnz != 1)
        { 
            // A(:,j) must have exactly one entry
            return (false) ;
        }
        int64_t i = Ai [p] ;
        if (i != j)
        { 
            // the single entry must be A(i,i)
            return (false) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // A is a square diagonal matrix with all entries present on the diagonal
    A->nvec_nonempty = n ;
    return (true) ;
}

