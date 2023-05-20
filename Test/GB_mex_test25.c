//------------------------------------------------------------------------------
// GB_mex_test25: more simple tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_test25"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // GrB_Vector size
    //--------------------------------------------------------------------------

    GrB_Vector v ;
    GrB_Index nmax = GrB_INDEX_MAX + 1 ;

    for (GrB_Index n = nmax - 5 ; n <= nmax ; n++)
    {
        printf ("n %lu, nmax %lu, (n == nmax): %d\n", n, nmax, n == nmax) ;
        OK (GrB_Vector_new (&v, GrB_BOOL, n)) ;
        OK (GrB_assign (v, NULL, NULL, true, GrB_ALL, n, NULL)) ;
        OK (GxB_print (v, 3)) ;
        GrB_Index nvals ;
        OK (GrB_Vector_nvals (&nvals, v)) ;
        CHECK (nvals == n) ;
        OK (GrB_Vector_free (&v)) ;
    }

    //--------------------------------------------------------------------------
    // GxB_Context_error
    //--------------------------------------------------------------------------

    char *s = NULL ;
    OK (GrB_error (&s, GxB_CONTEXT_WORLD)) ;
    CHECK (s != NULL) ;
    CHECK (strlen (s) == 0) ;
    printf ("GxB_Context_error [%s] ok\n", s) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test25:  all tests passed\n\n") ;
}

