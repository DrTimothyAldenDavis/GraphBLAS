//------------------------------------------------------------------------------
// gb_export_to_mxarray: export a GrB_Matrix to a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input GrB_Matrix A is exported to a MATLAB sparse mxArray S, and freed.

// The input GrB_Matrix A may be shallow or deep.  The output is a standard
// MATLAB sparse matrix as an mxArray.

#include "gb_matlab.h"

mxArray *gb_export_to_mxarray   // return exported MATLAB sparse matrix S
(
    GrB_Matrix *A_handle,       // matrix to export; freed on output
    bool A_is_deep              // true if A is deep, false if shallow
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (A_handle == NULL || (*A_handle) == NULL, "internal error") ;

    //--------------------------------------------------------------------------
    // typecast to a native MATLAB sparse type and free A
    //--------------------------------------------------------------------------

    GrB_Matrix T ;              // T will always be deep
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, *A_handle)) ;

    if (type == GrB_BOOL || type == GrB_FP64
        #ifdef GB_COMPLEX_TYPE
        || type == gb_complex_type
        #endif
        )
    {

        //----------------------------------------------------------------------
        // A is already in a native MATLAB sparse matrix type
        //----------------------------------------------------------------------

        // TODO handle CSR and CSC

        if (A_is_deep)
        {
            // A is already deep; just transplant it into T
            T = (*A_handle) ;
            (*A_handle) = NULL ;
        }
        else
        {
            // A is shallow so make a deep copy
            OK (GrB_Matrix_dup (&T, *A_handle)) ;
            gb_free_shallow (A_handle) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // typecast A to double
        //----------------------------------------------------------------------

        // MATLAB supports only logical, double, and double complex sparse
        // matrices.  These correspond to GrB_BOOL, GrB_FP64, and
        // gb_complex_type, respectively.  A is typecasted to double.

        T = gb_typecast (GrB_FP64, *A_handle) ;

        if (A_is_deep)
        {
            OK (GrB_free (A_handle)) ;
        }
        else
        {
            gb_free_shallow (A_handle) ;
        }
    }

    //--------------------------------------------------------------------------
    // export the content of T
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols, nvals ;
    int64_t nonempty, *Tp, *Ti ;
    void *Tx ;

    // get the content of T and free T
    OK (GxB_Matrix_export_CSC (&T, &type, &nrows, &ncols, &nvals, &nonempty,
        &Tp, &Ti, &Tx, NULL)) ;

    //--------------------------------------------------------------------------
    // create the new MATLAB sparse matrix
    //--------------------------------------------------------------------------

    mxArray *S ;

    if (nvals == 0)
    {

        //----------------------------------------------------------------------
        // allocate an empty sparse matrix of the right type and size
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        {
            S = mxCreateSparseLogicalMatrix (nrows, ncols, 1) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            S = mxCreateSparse (nrows, ncols, 1, mxCOMPLEX) ;
        }
        #endif
        else
        {
            S = mxCreateSparse (nrows, ncols, 1, mxREAL) ;
        }

        if (Tp != NULL) mxFree (Tp) ;
        if (Ti != NULL) mxFree (Ti) ;
        if (Tx != NULL) mxFree (Tx) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // allocate an empty sparse matrix of the right type, then set content
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        {
            S = mxCreateSparseLogicalMatrix (0, 0, 1) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            S = mxCreateSparse (0, 0, 1, mxCOMPLEX) ;
        }
        #endif
        else
        {
            S = mxCreateSparse (0, 0, 1, mxREAL) ;
        }

        // set the size
        mxSetM (S, nrows) ;
        mxSetN (S, ncols) ;
        mxSetNzmax (S, nvals) ;

        // set the column pointers
        void *p = mxGetJc (S) ;
        if (p != NULL) mxFree (p) ;
        mxSetJc (S, Tp) ;

        // set the row indices
        p = mxGetIr (S) ;
        if (p != NULL) mxFree (p) ;
        mxSetIr (S, Ti) ;

        // set the values
        if (type == GrB_BOOL)
        {
            p = mxGetData (S) ;
            if (p != NULL) mxFree (p) ;
            mxSetData (S, Tx) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            p = mxGetComplexDoubles (S) ;
            if (p != NULL) mxFree (p) ;
            mxSetComplexDoubles (S, Tx) ;
        }
        #endif
        else
        {
            p = mxGetDoubles (S) ;
            if (p != NULL) mxFree (p) ;
            mxSetDoubles (S, Tx) ;
        }
    }

    //--------------------------------------------------------------------------
    // return the new MATLAB sparse matrix
    //--------------------------------------------------------------------------

    return (S) ;
}

