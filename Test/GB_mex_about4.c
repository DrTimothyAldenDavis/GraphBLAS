//------------------------------------------------------------------------------
// GB_mex_about4: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"

#define USAGE "GB_mex_about4"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix C = NULL, A = NULL, M = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Vector w = NULL ;
    char *err ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // pack/unpack
    //--------------------------------------------------------------------------

    GrB_Index m = 4, n = 5 ;
    OK (GrB_Matrix_new (&C, GrB_FP64, m, n)) ;
    OK (GxB_Matrix_Option_set (C, GxB_FORMAT, GxB_BY_ROW)) ;
    double x = 0 ;
    for (int i = 0 ; i < m ; i++)
    {
        for (int j = 0 ; j < m ; j++)
        {
            x++ ;
            OK (GxB_Matrix_setElement_FP64 (C, x,
                (GrB_Index) i, (GrB_Index) j)) ;
        }
    }
    OK (GrB_Matrix_wait (&C)) ;
    OK (GxB_Matrix_fprint (C, "initial C by row", GxB_COMPLETE, NULL)) ;

    double *Cx = NULL ;
    GrB_Index Cx_size = 0 ;
    bool C_iso = false ;
    OK (GxB_Matrix_unpack_FullR (C, &Cx, &Cx_size, &C_iso, NULL)) ;
    OK (GxB_Matrix_fprint (C, "unpacked C by row", GxB_COMPLETE, NULL)) ;

    for (int k = 0 ; k < m*n ; k++)
    {
        CHECK (Cx [k] == (double) k) ;
    }

    OK (GxB_Matrix_pack_FullC (C, &Cx, &Cx_size, C_iso, NULL)) ;
    OK (GxB_Matrix_fprint (C, "packed C by col", GxB_COMPLETE, NULL)) ;
    CHECK (Cx == NULL) ;

    GrB_Index nrows = 0, ncols = 0 ;
    OK (GrB_Matrix_nrows (&nrows, C)) ;
    OK (GrB_Matrix_ncols (&nrows, C)) ;
    CHECK (nrows == m) ;
    CHECK (ncols == m) ;

    OK (GrB_Matrix_free (&C)) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;   
    fclose (f) ;
    printf ("\nGB_mex_about4: all tests passed\n\n") ;
}

