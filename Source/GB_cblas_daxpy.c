//------------------------------------------------------------------------------
// GB_cblas_daxpy: Y += alpha*X where X and Y are dense double arrays
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Y += alpha*X where are X and Y are dense arrays of stride 1, of type double.

// Note that currently, alpha is always passed in as 1.0, but this could change
// in the future, so alpha is passed in as a parameter to this function.

// X and Y can have any size, and will often be larger than 2^31.

#include "GB_dense.h"

void GB_cblas_daxpy         // Y += alpha*X
(
    const int64_t n,        // length of X and Y (note the int64_t type)
    const double alpha,     // scale factor
    const double *X,        // the array X, always stride 1
    double *Y,              // the array Y, always stride 1
    int nthreads            // maximum # of threads to use
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Y != NULL) ;
    ASSERT (X != NULL) ;
    ASSERT (nthreads >= 1) ;

    #if GB_HAS_CBLAS

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    // TODO see GB_cblas_saxpy for details

    //--------------------------------------------------------------------------
    // Y += alpha*X
    //--------------------------------------------------------------------------

    GBBURBLE ("cblas ") ;

    for (int64_t p = 0 ; p < n ; p += INT_MAX)
    {  
        int chunk = (int) GB_IMIN (n - p, INT_MAX) ;
        cblas_daxpy     // y += alpha*x
        (
            chunk,      // length of x and y (this chunk)
            alpha,      // scale factor (typically 1.0)
            X + p,      // this chunk of x
            (int) 1,    // x is stride 1
            Y + p,      // this chunk of y
            (int) 1     // y is stride 1
        ) ;
    }

    #endif
}

