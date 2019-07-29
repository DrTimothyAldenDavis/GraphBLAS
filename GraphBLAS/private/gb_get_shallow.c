//------------------------------------------------------------------------------
// gb_get_shallow: create a shallow copy of a MATLAB sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// A = gb_get_shallow (X) constructs a shallow GrB_Matrix from a MATLAB
// mxArray, which can either be a MATLAB sparse matrix (double, complex, or
// logical) or a MATLAB struct that contains a GraphBLAS matrix.

// X must not be NULL, but it be an empty matrix, as X = [ ] or even X = ''
// (the empty string).  In this case, A is returned as NULL.  This is not an
// error here, since the caller might be getting an optional input matrix, such
// as Cin or the Mask.

#include "gb_matlab.h"

#define IF(error,message) \
    CHECK_ERROR (error, "invalid GraphBLAS struct (" message ")" ) ;

GrB_Matrix gb_get_shallow   // return a shallow copy of MATLAB sparse matrix
(
    const mxArray *X
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (X == NULL, "matrix missing") ;

    //--------------------------------------------------------------------------
    // construct the shallow GrB_Matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A ;

    if (mxIsEmpty (X))
    {
    GB_HERE ;

        //----------------------------------------------------------------------
        // matrix is empty
        //----------------------------------------------------------------------

        // This is not an error, since some input matrices are optional,
        // and can be specified as "[ ]".
        return (NULL) ;

    }
    else if (mxIsSparse (X))
    {
    GB_HERE ;

        //----------------------------------------------------------------------
        // construct a shallow GrB_Matrix copy of a MATLAB sparse matrix
        //----------------------------------------------------------------------

        GrB_Type type = gb_mxarray_type (X) ;
        GrB_Index nrows = (GrB_Index) mxGetM (X) ;
        GrB_Index ncols = (GrB_Index) mxGetN (X) ;
        GrB_Index *Xp = (GrB_Index *) mxGetJc (X) ;
        GrB_Index *Xi = (GrB_Index *) mxGetIr (X) ;
        GrB_Index nvals = (GrB_Index) (Xp [ncols]) ;

        void *Xx = NULL ;
        if (type == GrB_FP64)
        {
            // MATLAB sparse double matrix
            Xx = mxGetDoubles (X) ;
        }
        #ifdef GB_COMPLEX_TYPE
        else if (type == gb_complex_type)
        {
            // MATLAB sparse double complex matrix
            Xx = mxGetComplexDoubles (X) ;
        }
        #endif
        else if (type == GrB_BOOL)
        {
            // MATLAB sparse logical matrix
            Xx = mxGetPr (X) ;
        }
        else
        {
            // MATLAB does not support any other kinds of sparse matrices
            ERROR ("type not supported") ;
        }

        // import the matrix.  This sets Xp, Xi, and Xx to NULL, but it does
        // not change the MATLAB matrix they came from.
        OK (GxB_Matrix_import_CSC (&A, type, nrows, ncols, nvals, -1,
            &Xp, &Xi, &Xx, NULL)) ;

        // make sure the shallow copy remains standard, not hypersparse
        OK (GxB_set (A, GxB_HYPER, GxB_NEVER_HYPER)) ;

    }
    else if (mxIsStruct (X))
    {
    GB_HERE ;

        //----------------------------------------------------------------------
        // construct a shallow GrB_Matrix copy from a MATLAB struct
        //----------------------------------------------------------------------

        // get the type
        mxArray *mx_type = mxGetField (X, 0, "GraphBLAS") ;
        CHECK_ERROR (mx_type == NULL, "not a GraphBLAS struct") ;
        GrB_Type type = gb_mxstring_to_type (mx_type) ;

        // allocate the header, with no content
        OK (GrB_Matrix_new (&A, type, 0, 0)) ;
        if (A->p != NULL) mxFree (A->p) ;
        if (A->i != NULL) mxFree (A->i) ;
        if (A->x != NULL) mxFree (A->x) ;
        if (A->h != NULL) mxFree (A->h) ;

        // get the scalar info
        mxArray *opaque = mxGetField (X, 0, "s") ;
        IF (opaque == NULL, ".s missing") ;
        double *s = mxGetDoubles (opaque) ;
        A->hyper_ratio   = s [0] ;
        A->plen          = (int64_t) s [1] ;
        A->vlen          = (int64_t) s [2] ;
        A->vdim          = (int64_t) s [3] ;
        A->nvec          = (int64_t) s [4] ;
        A->nvec_nonempty = (int64_t) s [5] ;
        A->is_hyper      = (int64_t) s [6] ;
        A->is_csc        = (int64_t) s [7] ;
        A->nzmax         = (int64_t) s [8] ;

        // get the pointers
        mxArray *Ap = mxGetField (X, 0, "p") ;
        IF (Ap == NULL, ".p missing") ;
        IF (mxGetM (Ap) != 1, ".p wrong size") ;
        IF (mxGetN (Ap) != A->plen+1, ".p wrong size") ;
        A->p = mxGetInt64s (Ap) ;
        IF (A->p == NULL, ".p wrong type") ;

        // get the indices
        mxArray *Ai = mxGetField (X, 0, "i") ;
        IF (Ai == NULL, ".i missing") ;
        IF (mxGetM (Ai) != 1, ".i wrong size") ;
        IF (mxGetN (Ai) != MAX (A->nzmax, 1), ".i wrong size") ;
        A->i = (A->nzmax == 0) ? NULL : mxGetInt64s (Ai) ;
        IF (A->i == NULL && A->nzmax > 0, ".i wrong type") ;

        // get the values
        mxArray *Ax = mxGetField (X, 0, "x") ;
        IF (Ax == NULL, ".x missing") ;
        IF (mxGetM (Ax) != 1, ".x wrong size") ;
        IF (mxGetN (Ax) != MAX (A->type_size*A->nzmax, 1), ".x wrong size") ;
        A->x = (A->nzmax == 0) ? NULL : ((void *) mxGetUint8s (Ax)) ;
        IF (A->x == NULL && A->nzmax > 0, ".x wrong type") ;

        A->h = NULL ;
        if (A->is_hyper)
        {
            // get the hyperlist
            mxArray *Ah = mxGetField (X, 0, "h") ;
            IF (Ah == NULL, ".h missing") ;
            IF (mxGetM (Ah) != 1, ".h wrong size") ;
            IF (mxGetN (Ah) != MAX (A->plen, 1), ".h wrong size") ;
            A->h = (void *) mxGetInt64s (Ah) ;
            IF (A->h == NULL, ".h wrong type") ;
        }

        // matrix is now initialized
        A->magic = GB_MAGIC ;

    }
    else
    {

        //----------------------------------------------------------------------
        // dense matrices are not supported
        //----------------------------------------------------------------------

        // TODO: create a partially shallow GrB_Matrix copy of X, by
        // allocating the row indices Xi and pointers Xp.

        ERROR ("invalid GraphBLAS matrix") ;

    }

    //--------------------------------------------------------------------------
    // return the result
    //--------------------------------------------------------------------------

    GB_HERE ;
    OK (GxB_print (A, 3)) ;
    return (A) ;
}

