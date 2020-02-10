//------------------------------------------------------------------------------
// GB_cblas_daxpy: Y += alpha*X where X and Y are dense double arrays
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Y += alpha*X where are X and Y are dense arrays of stride 1, of type double.

// X and Y can have any size, and will often be larger than 2^31.

#define GB_BURBLE 1
#include "GB_dense.h"
#include "GB_cblas.h"

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

    // See GB_cblas_saxpy.c for a discussion.

    #ifdef MKL_ILP64
    int save_nthreads = mkl_set_num_threads_local (nthreads) ;
    #endif

    //--------------------------------------------------------------------------
    // Y += alpha*X
    //--------------------------------------------------------------------------

    GB_CBLAS_INT stride1 = (GB_CBLAS_INT) 1 ;

    // call *axpy in chunks of size GB_CBLAS_INT_MAX (2^31 or 2^63).
    // If GB_CBLAS_INT_MAX is INT64_MAX, then this will iterate just once.
    // for (int64_t p = 0 ; p < n ; p += GB_CBLAS_INT_MAX)
    {
//      GB_CBLAS_INT chunk = (GB_CBLAS_INT) GB_IMIN (n - p, GB_CBLAS_INT_MAX) ;
        #define chunk n
        #define p 0

        cblas_daxpy     // y += alpha*x
        (
            chunk,      // length of x and y (this chunk)
            alpha,      // scale factor (typically 1.0)
            X + p,      // this chunk of x
            stride1,    // x is stride 1
            Y + p,      // this chunk of y
            stride1     // y is stride 1
        ) ;
    }

    //--------------------------------------------------------------------------
    // restore the # of threads for the BLAS
    //--------------------------------------------------------------------------

    #ifdef MKL_ILP64
    mkl_set_num_threads_local (save_nthreads) ;
    #endif

    #endif
}

