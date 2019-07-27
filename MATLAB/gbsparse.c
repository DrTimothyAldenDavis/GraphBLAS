//------------------------------------------------------------------------------
// gbsparse: convert a GraphBLAS matrix struct into a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gbmex.h"

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

    gb_usage (nargin == 1 && nargout <= 1, "usage: A = gbsparse (X)") ;

    //--------------------------------------------------------------------------
    // get the GraphBLAS matrix as a shallow copy
    //--------------------------------------------------------------------------

    GrB_Matrix X = gb_get_shallow (pargin [0]) ;

    //--------------------------------------------------------------------------
    // typecast to a native MATLAB sparse type, making a deep copy
    //--------------------------------------------------------------------------

    GrB_Type type = X->type ;
    if (!(type == GrB_BOOL
        #ifdef GB_COMPLEX_TYPE
        || type == gb_complex_type
        #endif
        ))
    {
        // MATLAB supports only logical, double, and double complex sparse
        // matrices.  These correspond to GrB_BOOL, GrB_FP64, and
        // gb_complex_type, respectively.  If the GrB_Matrix has a different
        // type, it is typecasted to double.
        type = GrB_FP64 ;
    }

    GrB_Matrix A = gb_typecast (type, X) ;

    //--------------------------------------------------------------------------
    // free the shallow copy
    //--------------------------------------------------------------------------

    gb_free_shallow (&X) ;

    //--------------------------------------------------------------------------
    // export the content of A
    //--------------------------------------------------------------------------

    GrB_Index nrows, ncols, nvals ;
    int64_t nonempty, *Ap, *Ai ;
    void *Ax ;

    // get the content of A
    OK (GxB_Matrix_export_CSC (&A, &type, &nrows, &ncols, &nvals, &nonempty,
        &Ap, &Ai, &Ax, NULL)) ;

    //--------------------------------------------------------------------------
    // create the new MATLAB sparse matrix
    //--------------------------------------------------------------------------

    if (nvals == 0)
    {

        //----------------------------------------------------------------------
        // allocate an empty sparse matrix of the right type and size
        //----------------------------------------------------------------------

        if (type == GrB_BOOL)
        {
            pargout [0] = mxCreateSparseLogicalMatrix (nrows, ncols, 1) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            pargout [0] = mxCreateSparse (nrows, ncols, 1, mxCOMPLEX) ;
        }
        #endif
        else
        {
            pargout [0] = mxCreateSparse (nrows, ncols, 1, mxREAL) ;
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
            pargout [0] = mxCreateSparseLogicalMatrix (0, 0, 1) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            pargout [0] = mxCreateSparse (0, 0, 1, mxCOMPLEX) ;
        }
        #endif
        else
        {
            pargout [0] = mxCreateSparse (0, 0, 1, mxREAL) ;
        }

        // set the size
        mxSetM (pargout [0], nrows) ;
        mxSetN (pargout [0], ncols) ;
        mxSetNzmax (pargout [0], nvals) ;

        // set the column pointers
        void *p = mxGetJc (pargout [0]) ;
        if (p != NULL) mxFree (p) ;
        mxSetJc (pargout [0], Ap) ;

        // set the row indices
        p = mxGetIr (pargout [0]) ;
        if (p != NULL) mxFree (p) ;
        mxSetIr (pargout [0], Ai) ;

        // set the values
        if (type == GrB_BOOL)
        {
            p = mxGetData (pargout [0]) ;
            if (p != NULL) mxFree (p) ;
            mxSetData (pargout [0], Ax) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            p = mxGetComplexDoubles (pargout [0]) ;
            if (p != NULL) mxFree (p) ;
            mxSetComplexDoubles (pargout [0], Ax) ;
        }
        #endif
        else
        {
            p = mxGetDoubles (pargout [0]) ;
            if (p != NULL) mxFree (p) ;
            mxSetDoubles (pargout [0], Ax) ;
        }
    }
}

