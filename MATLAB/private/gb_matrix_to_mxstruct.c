//------------------------------------------------------------------------------
// gb_matrix_to_mxstruct: create a MATLAB struct for a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "gbmex.h"

#define NFIELDS 6

static const char *MatrixFields [NFIELDS] =
{
    "GraphBLAS",        // 0: "logical", "int8", ... "double", "complex"
    "s",                // 1: all scalar info goes here
    "p",                // 2: array of int64_t, size plen+1
    "i",                // 3: array of int64_t, size nzmax
    "x",                // 4: array of uint8, size (sizeof(type) * nzmax)
    "h"                 // 5: array of int64_t, size plen if hypersparse
} ;

// These components do not need to be exported: Pending, nzombies, queue_next,
// queue_head, enqueued, *_shallow.

mxArray *gb_matrix_to_mxstruct  // return a MATLAB struct
(
    GrB_Matrix *A_Handle        // GrB_Matrix to convert to MATLAB struct
)
{

    CHECK_ERROR (A_Handle == NULL, "matrix missing") ;

    GrB_Matrix A = (*A_Handle) ;

    CHECK_ERROR (A->p_shallow || A->i_shallow || A->x_shallow || A->h_shallow,
        "invalid shallow matrix") ;

    // make sure the matrix is finished
    GrB_Index nvals ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;

    // construct the output struct
    mxArray *X = mxCreateStructMatrix (1, 1,
        A->is_hyper ? NFIELDS : (NFIELDS-1), MatrixFields) ;

    // export the GraphBLAS type
    mxSetFieldByNumber (X, 0, 0, gb_type_to_mxstring (A->type)) ;

    // export the scalar content
    mxArray *opaque = mxCreateNumericMatrix (1, 9, mxDOUBLE_CLASS, mxREAL) ;
    double *s = mxGetDoubles (opaque) ;
    s [0] = A->hyper_ratio ; 
    s [1] = (double) (A->plen) ; 
    s [2] = (double) (A->vlen) ; 
    s [3] = (double) (A->vdim) ; 
    s [4] = (double) (A->nvec) ; 
    s [5] = (double) (A->nvec_nonempty) ; 
    s [6] = (double) (A->is_hyper) ; 
    s [7] = (double) (A->is_csc) ; 
    s [8] = (double) (A->nzmax) ; 
    mxSetFieldByNumber (X, 0, 1, opaque) ;

    // export the pointers
    mxArray *Ap = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
    mxSetN (Ap, A->plen+1) ;
    void *p = mxGetInt64s (Ap) ;
    if (p != NULL) mxFree (p) ;
    mxSetInt64s (Ap, A->p) ;
    A->p = NULL ;
    mxSetFieldByNumber (X, 0, 2, Ap) ;

    // export the indices
    mxArray *Ai = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
    if (A->nzmax > 0)
    {
        mxSetN (Ai, A->nzmax) ;
        p = mxGetInt64s (Ai) ;
        if (p != NULL) mxFree (p) ;
        mxSetInt64s (Ai, A->i) ;
    }
    A->i = NULL ;
    mxSetFieldByNumber (X, 0, 3, Ai) ;

    // export the values as uint8
    mxArray *Ax = mxCreateNumericMatrix (1, 1, mxUINT8_CLASS, mxREAL) ;
    if (A->nzmax > 0)
    {
        mxSetN (Ax, A->nzmax * A->type_size) ;
        p = mxGetUint8s (Ax) ;
        if (p != NULL) mxFree (p) ;
        mxSetUint8s (Ax, A->x) ;
    }
    A->x = NULL ;
    mxSetFieldByNumber (X, 0, 4, Ax) ;

    if (A->is_hyper)
    {
        // export the hyperlist
        if (A->nvec < A->plen)
        {
            // This is unused space in A->h; it might contain garbage, so
            // set it to zero to simplify the view of the MATLAB struct.
            A->h [A->nvec] = 0 ;
        }
        mxArray *Ah = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        if (A->plen > 0)
        {
            mxSetN (Ah, A->plen) ;
            p = mxGetInt64s (Ah) ;
            if (p != NULL) mxFree (p) ;
            mxSetInt64s (Ah, A->h) ;
        }
        mxSetFieldByNumber (X, 0, 5, Ah) ;
    }
    A->h = NULL ;

    // free the header of A
    GrB_Matrix_free (A_Handle) ;

    // return the MATLAB struct
    return (X) ;
}

