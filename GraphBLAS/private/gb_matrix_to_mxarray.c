//------------------------------------------------------------------------------
// gb_matrix_to_mxarray: convert a GraphBLAS struct to a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input GrB_Matrix X may be shallow or deep.  The output is a standard
// MATLAB sparse matrix as an mxArray.

#include "gb_matlab.h"

mxArray *gb_matrix_to_mxarray   // return MATLAB sparse matrix of a GrB_Matrix
(
    GrB_Matrix *X_handle,       // matrix to copy; freed on output
    bool X_is_deep              // true if X is deep, false if shallow
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_HERE ;
    CHECK_ERROR (X_handle == NULL || (*X_handle) == NULL, "internal error") ;

    //--------------------------------------------------------------------------
    // typecast to a native MATLAB sparse type and free X
    //--------------------------------------------------------------------------

    GrB_Matrix A ;              // A will always be deep
    GB_HERE ;

    GrB_Type type ;

    OK (GxB_print (*X_handle, 3)) ;

    GB_HERE ;
    
    OK (GxB_Matrix_type (&type, *X_handle)) ;

    GB_HERE ;

    if (type == GrB_BOOL || type == GrB_FP64
        #ifdef GB_COMPLEX_TYPE
        || type == gb_complex_type
        #endif
        )
    {

        //----------------------------------------------------------------------
        // X is already in a native MATLAB sparse matrix type
        //----------------------------------------------------------------------

        // TODO handle CSR and CSC

        if (X_is_deep)
        {
            // X is already deep; just transplant it into A
    GB_HERE ;
            A = (*X_handle) ;
    GB_HERE ;
            (*X_handle) = NULL ;
    GB_HERE ;
        }
        else
        {
            // X is shallow so make a deep copy
    GB_HERE ;
            OK (GrB_Matrix_dup (&A, *X_handle)) ;
    GB_HERE ;
            gb_free_shallow (X_handle) ;
    GB_HERE ;
        }

    GB_HERE ;
    }
    else
    {

        //----------------------------------------------------------------------
        // typecast X to double
        //----------------------------------------------------------------------

    GB_HERE ;
        // MATLAB supports only logical, double, and double complex sparse
        // matrices.  These correspond to GrB_BOOL, GrB_FP64, and
        // gb_complex_type, respectively.  X is typecasted to double.

        A = gb_typecast (GrB_FP64, *X_handle) ;
    GB_HERE ;

        if (X_is_deep)
        {
    GB_HERE ;
            OK (GrB_free (X_handle)) ;
    GB_HERE ;
        }
        else
        {
    GB_HERE ;
            gb_free_shallow (X_handle) ;
    GB_HERE ;
        }
    }

    GB_HERE ;

    //--------------------------------------------------------------------------
    // export the content of A
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols, nvals ;
    int64_t nonempty, *Ap, *Ai ;
    void *Ax ;

    // get the content of A and free A
    OK (GxB_Matrix_export_CSC (&A, &type, &nrows, &ncols, &nvals, &nonempty,
        &Ap, &Ai, &Ax, NULL)) ;

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

        if (Ap != NULL) mxFree (Ap) ;
        if (Ai != NULL) mxFree (Ai) ;
        if (Ax != NULL) mxFree (Ax) ;

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
        mxSetJc (S, Ap) ;

        // set the row indices
        p = mxGetIr (S) ;
        if (p != NULL) mxFree (p) ;
        mxSetIr (S, Ai) ;

        // set the values
        if (type == GrB_BOOL)
        {
            p = mxGetData (S) ;
            if (p != NULL) mxFree (p) ;
            mxSetData (S, Ax) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            p = mxGetComplexDoubles (S) ;
            if (p != NULL) mxFree (p) ;
            mxSetComplexDoubles (S, Ax) ;
        }
        #endif
        else
        {
            p = mxGetDoubles (S) ;
            if (p != NULL) mxFree (p) ;
            mxSetDoubles (S, Ax) ;
        }
    }

    //--------------------------------------------------------------------------
    // return the new MATLAB sparse matrix
    //--------------------------------------------------------------------------

    GB_HERE ;
    return (S) ;
}

