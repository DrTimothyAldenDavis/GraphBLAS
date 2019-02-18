//------------------------------------------------------------------------------
// GB_mex_export_import: export and then reimport a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// format:
//  0: standard CSR
//  1: standard CSC
//  3: hyper CSR
//  4: hyper CSC

#include "GB_mex.h"

#define USAGE "C = GB_mex_export_import (A, format_matrix, format_export)"

#define FREE_ALL                        \
{                                       \
    if (Ap) { mxFree (Ap) ; Ap = NULL ; } \
    if (Ah) { mxFree (Ah) ; Ah = NULL ; } \
    if (Ai) { mxFree (Ai) ; Ai = NULL ; } \
    if (Ax) { mxFree (Ax) ; Ax = NULL ; } \
    GB_MATRIX_FREE (&A) ;               \
    GB_MATRIX_FREE (&C) ;               \
    GB_mx_put_global (true, 0) ;        \
}

#define OK(method)                              \
{                                               \
    info = method ;                             \
    if (info != GrB_SUCCESS) return (info) ;    \
}

GrB_Matrix A = NULL ;
GrB_Matrix C = NULL ;
GrB_Index *Ap = NULL, *Ah = NULL, *Ai = NULL ;
void *Ax = NULL ;

GrB_Info export_import
(
    int format_matrix,
    int format_export
)
{

    GrB_Type type ;
    GrB_Index nrows, ncols, nvals, nvec ;
    GrB_Info info = GrB_SUCCESS ;

    OK (GrB_Matrix_dup (&C, A)) ;

    //--------------------------------------------------------------------------
    // convert C to the requested format
    //--------------------------------------------------------------------------

    switch (format_matrix)
    {

        //----------------------------------------------------------------------
        case 0 :    // standard CSR
        //----------------------------------------------------------------------

            OK (GxB_set (C, GxB_HYPER,  GxB_NEVER_HYPER)) ;
            OK (GxB_set (C, GxB_FORMAT, GxB_BY_ROW)) ;
            break ;

        //----------------------------------------------------------------------
        case 1 :    // standard CSC
        //----------------------------------------------------------------------

            OK (GxB_set (C, GxB_HYPER,  GxB_NEVER_HYPER)) ;
            OK (GxB_set (C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;

        //----------------------------------------------------------------------
        case 2 :    // hypersparse CSR
        //----------------------------------------------------------------------

            OK (GxB_set (C, GxB_HYPER,  GxB_ALWAYS_HYPER)) ;
            OK (GxB_set (C, GxB_FORMAT, GxB_BY_ROW)) ;
            break ;

        //----------------------------------------------------------------------
        case 3 :    // hypersparse CSC
        //----------------------------------------------------------------------

            OK (GxB_set (C, GxB_HYPER,  GxB_ALWAYS_HYPER)) ;
            OK (GxB_set (C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;
    }

    //--------------------------------------------------------------------------
    // export then import
    //--------------------------------------------------------------------------

    switch (format_export)
    {

        //----------------------------------------------------------------------
        case 0 :    // standard CSR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSR (&C, &type, &nrows, &ncols, &nvals,
                &Ap, &Ai, &Ax, NULL)) ;

            OK (GxB_Matrix_import_CSR (&C, type, nrows, ncols, nvals,
                &Ap, &Ai, &Ax, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 1 :    // standard CSC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols, &nvals,
                &Ap, &Ai, &Ax, NULL)) ;

            OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols, nvals,
                &Ap, &Ai, &Ax, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 2 :    // hypersparse CSR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSR (&C, &type, &nrows, &ncols, &nvals,
                &nvec, &Ah, &Ap, &Ai, &Ax, NULL)) ;

            OK (GxB_Matrix_import_HyperCSR (&C, type, nrows, ncols, nvals,
                nvec, &Ah, &Ap, &Ai, &Ax, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 3 :    // hypersparse CSC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSC (&C, &type, &nrows, &ncols, &nvals,
                &nvec, &Ah, &Ap, &Ai, &Ax, NULL)) ;

            OK (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols, nvals,
                nvec, &Ah, &Ap, &Ai, &Ax, NULL)) ;

            break ;

    }

    return (GrB_SUCCESS) ;
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;

    // check inputs
    GB_WHERE (USAGE) ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy)
    {
        A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }
    }

    // get matrix format (0 to 3)
    int GET_SCALAR (1, int, format_matrix, 0) ;
    if (format_matrix < 0 || format_matrix > 3) mexErrMsgTxt ("bad format") ;

    // get export/import format (0 to 3)
    int GET_SCALAR (2, int, format_export, 0) ;
    if (format_export < 0 || format_export > 3) mexErrMsgTxt ("bad format") ;

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    // convert matrix, export, then import
    METHOD (export_import (format_matrix, format_export)) ;

    // return C to MATLAB as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;

    FREE_ALL ;
}

