//------------------------------------------------------------------------------
// gbargminmax: argmin or argmax of a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0-or-later

//------------------------------------------------------------------------------

// usage:

// [x,i] = gbargminmax (A, minmax, dim)

// where minmax is 0 for min or 1 for max, and where dim = 1 to compute the
// argmin/max of each column of A, dim = 2 to compute the argmin/max of each
// row of A, or dim = 0 to compute the argmin/max of all of A.  For dim = 1 or
// 2, x and i are vectors of the same size.  For dim = 0, x is a scalar and i
// has size two, containing the row and column index of the argmin/max of A.

#include "gb_matlab.h"

#define USAGE "usage: [x,i] = gbargminmax (A, minmax, dim)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin == 3 && nargout == 2, USAGE) ;

    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
    bool is_min = (bool) mxGetScalar (pargin [1]) ;
    int dim = (int) mxGetScalar (pargin [2]) ;
    CHECK_ERROR (dim < 0 || dim > 2, "invalid dim") ;

    //--------------------------------------------------------------------------
    // get the inputs
    //--------------------------------------------------------------------------

    GrB_index nrows, ncols ;
    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;

    //--------------------------------------------------------------------------
    // select the semirings
    //--------------------------------------------------------------------------

    GrB_Semiring comparator, any_equal ;

    if (is_min)
    {

        //----------------------------------------------------------------------
        // semirings for argmin
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        {
            comparator = GxB_LAND_FIRST_BOOL ;
            any_equal = GxB_ANY_EQ_BOOL ;
        }
        else if (type == GrB_INT8)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT8 ;
            any_equal = GxB_ANY_EQ_INT8 ;
        }
        else if (type == GrB_INT16)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT16 ;
            any_equal = GxB_ANY_EQ_INT16 ;
        }
        else if (type == GrB_INT32) 
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT32 ;
            any_equal = GxB_ANY_EQ_INT32 ;
        }
        else if (type == GrB_INT64) 
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT64 ;
            any_equal = GxB_ANY_EQ_INT64 ;
        }
        else if (type == GrB_UINT8) 
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT8 ;
            any_equal = GxB_ANY_EQ_UINT8 ;
        }
        else if (type == GrB_UINT16)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT16 ;
            any_equal = GxB_ANY_EQ_UINT16 ;
        }
        else if (type == GrB_UINT32)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT32 ;
            any_equal = GxB_ANY_EQ_UINT32 ;
        }
        else if (type == GrB_UINT64)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT64 ;
            any_equal = GxB_ANY_EQ_UINT64 ;
        }
        else if (type == GrB_FP32)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_FP32 ;
            any_equal = GxB_ANY_EQ_FP32 ;
        }
        else if (type == GrB_FP64)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_FP64 ;
            any_equal = GxB_ANY_EQ_FP64 ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // semirings for argmax
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        {
            comparator = GxB_LAND_FIRST_BOOL ;
            any_equal = GxB_ANY_EQ_BOOL ;
        }
        else if (type == GrB_INT8)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT8 ;
            any_equal = GxB_ANY_EQ_INT8 ;
        }
        else if (type == GrB_INT16)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT16 ;
            any_equal = GxB_ANY_EQ_INT16 ;
        }
        else if (type == GrB_INT32) 
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT32 ;
            any_equal = GxB_ANY_EQ_INT32 ;
        }
        else if (type == GrB_INT64) 
        {
            comparator = GrB_MIN_FIRST_SEMIRING_INT64 ;
            any_equal = GxB_ANY_EQ_INT64 ;
        }
        else if (type == GrB_UINT8) 
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT8 ;
            any_equal = GxB_ANY_EQ_UINT8 ;
        }
        else if (type == GrB_UINT16)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT16 ;
            any_equal = GxB_ANY_EQ_UINT16 ;
        }
        else if (type == GrB_UINT32)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT32 ;
            any_equal = GxB_ANY_EQ_UINT32 ;
        }
        else if (type == GrB_UINT64)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_UINT64 ;
            any_equal = GxB_ANY_EQ_UINT64 ;
        }
        else if (type == GrB_FP32)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_FP32 ;
            any_equal = GxB_ANY_EQ_FP32 ;
        }
        else if (type == GrB_FP64)
        {
            comparator = GrB_MIN_FIRST_SEMIRING_FP64 ;
            any_equal = GxB_ANY_EQ_FP64 ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }
    }


    // dim 2: argmin/max of each row of A
    // dim 1: argmin/max of each column of A

    GrB_index n = (dim == 2) ? ncols : nrows ;
    GrB_index m = (dim == 2) ? nrows : ncols ;

    GrB_Vector_new (&x, type, m) ;
    GrB_Vector_new (&y, type, n) ;
    GrB_Vector_new (&a, GrB_INT64, n) ;
    // y (:) = 1
    GrB_assign (y, NULL, NULL, 1, GrB_ALL, n, NULL) ;
    // x = min/max (A) where x(i) = min/max (A (i,:))
    GrB_mxv (x, NULL, NULL, comparator, A, y, NULL) ;
    // D = diag (x)
    GrB_Matrix_new (&D, type, n, n) ;
    GxB_Matrix_diag (D, x, 0, NULL) ;
    // G = A*D using the ANY_EQ_type semiring
    GrB_Matrix_new (&G, GrB_BOOL, m, n) ;
    GrB_mxm (G, NULL, NULL, any_equal, A, D, NULL) ;
    // drop explicit zeros from G
    GxB_select (G, NULL, NULL, GxB_NONZERO, G, NULL, NULL) ;
    // find the position of the max entry in each row: a = G*x,
    // so that a(i) = j if v(i) = A(i,j) = max (A (i,:))
    GrB_mxv (a, NULL, NULL, GxB_ANY_SECONDI_INT64, G, x, NULL) ;

