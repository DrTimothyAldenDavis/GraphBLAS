//------------------------------------------------------------------------------
// gb_export_to_mxstruct: export a GrB_Matrix to a MATLAB struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// The input GrB_Matrix A is exported to a GraphBLAS matrix struct G, and freed.

// The input GrB_Matrix A must be deep.  The output is a standard MATLAB sparse
// matrix as an mxArray.

#include "gb_matlab.h"

#define NFIELDS 6

static const char *MatrixFields [NFIELDS] =
{
    "GraphBLAS",        // 0: "logical", "int8", ... "double",
                        //    "single complex", or "double complex"
    "s",                // 1: all scalar info goes here
    "x",                // 2: array of uint8, size (sizeof(type) * nzmax)
    "p",                // 3: array of int64_t, size plen+1
    "i",                // 4: array of int64_t, size nzmax
    "h"                 // 5: array of int64_t, size plen if hypersparse
} ;

//------------------------------------------------------------------------------

mxArray *gb_export_to_mxstruct  // return exported MATLAB struct G
(
    GrB_Matrix *A_handle        // matrix to export; freed on output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (A_handle == NULL, "matrix missing") ;
    GrB_Matrix A = (*A_handle) ;
    CHECK_ERROR (gb_is_shallow (A), "internal error 4") ;

    //--------------------------------------------------------------------------
    // make sure the matrix is finished
    //--------------------------------------------------------------------------

    OK1 (A, GrB_Matrix_wait (&A)) ;

    //--------------------------------------------------------------------------
    // construct the output struct
    //--------------------------------------------------------------------------

    int nfields, sparsity ;
    if (GB_is_dense (A) && !GB_IS_FULL (A))
    {
        // convert A to full
        GB_sparse_to_full (A) ;
    }

    if (GB_IS_FULL (A))
    {
        // A is full
        sparsity = GB_FULL ;
        nfields = 3 ;
    }
    else if (A->h == NULL)
    {
        // A is sparse
        sparsity = GB_SPARSE ;
        nfields = 5 ;
    }
    else
    {
        // A is hypersparse
        sparsity = GB_HYPER ;
        nfields = 6 ;
    }

    mxArray *G = mxCreateStructMatrix (1, 1, nfields, MatrixFields) ;

    //--------------------------------------------------------------------------
    // export content into the output struct
    //--------------------------------------------------------------------------

    // export the GraphBLAS type as a string
    mxSetFieldByNumber (G, 0, 0, gb_type_to_mxstring (A->type)) ;

    // export the scalar content
    mxArray *opaque = mxCreateNumericMatrix (1, 8, mxINT64_CLASS, mxREAL) ;
    int64_t *s = mxGetInt64s (opaque) ;
    s [0] = A->plen ;
    s [1] = A->vlen ;
    s [2] = A->vdim ;
    s [3] = A->nvec ;
    s [4] = A->nvec_nonempty ;
    s [5] = (int64_t) sparsity ;  // 0:sparse, 1:hyper, 2:full
    s [6] = (int64_t) (A->is_csc) ;
    s [7] = A->nzmax ;
    mxSetFieldByNumber (G, 0, 1, opaque) ;

    // These components do not need to be exported: Pending, nzombies,
    // queue_next, queue_head, enqueued, *_shallow.

    if (sparsity != GB_FULL)
    {
        // export the pointers
        mxArray *Ap = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        mxSetN (Ap, A->plen+1) ;
        void *p = mxGetInt64s (Ap) ;
        gb_mxfree (&p) ;
        mxSetInt64s (Ap, A->p) ;
        A->p = NULL ;
        mxSetFieldByNumber (G, 0, 3, Ap) ;

        // export the indices
        mxArray *Ai = mxCreateNumericMatrix (1, 1, mxINT64_CLASS, mxREAL) ;
        if (A->nzmax > 0)
        { 
            mxSetN (Ai, A->nzmax) ;
            p = mxGetInt64s (Ai) ;
            gb_mxfree (&p) ;
            mxSetInt64s (Ai, A->i) ;
        }
        A->i = NULL ;
        mxSetFieldByNumber (G, 0, 4, Ai) ;
    }

    // export the values as uint8
    mxArray *Ax = mxCreateNumericMatrix (1, 1, mxUINT8_CLASS, mxREAL) ;
    if (A->nzmax > 0)
    { 
        mxSetN (Ax, A->nzmax * A->type->size) ;
        void *p = mxGetUint8s (Ax) ;
        gb_mxfree (&p) ;
        mxSetUint8s (Ax, A->x) ;
    }
    A->x = NULL ;
    mxSetFieldByNumber (G, 0, 2, Ax) ;

    if (sparsity == GB_HYPER)
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
            void *p = mxGetInt64s (Ah) ;
            gb_mxfree (&p) ;
            mxSetInt64s (Ah, A->h) ;
        }
        mxSetFieldByNumber (G, 0, 5, Ah) ;
    }
    A->h = NULL ;

    //--------------------------------------------------------------------------
    // free the header of A
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (A_handle)) ;

    //--------------------------------------------------------------------------
    // return the MATLAB struct
    //--------------------------------------------------------------------------

    return (G) ;
}

