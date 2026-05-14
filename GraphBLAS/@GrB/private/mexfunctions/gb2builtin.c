//------------------------------------------------------------------------------
// gb2builtin: convert a @GrB matrix to a MATLAB/Octave matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input is the opaque handle A.opaque of @GrB matrix A, but in a limited
// range of formats and sparsity structures, to be compatible with
// MATLAB/Octave built-in sparse/full matrices..  The format is by-column only.
// It is sparse or full, never bitmap or hypersparse.  It has no pending work.
// The integers for a sparse matrix are all 64-bit.  The matrix is not
// iso-valued.  If sparse, the matrix must be either GrB_BOOL, GrB_FP64, or
// GxB_FC64.

// This method does not malloc/free any content of a GraphBLAS matrix, so it
// is safe to use mxMalloc and mxCreate* throughout the mexFunction.  If the
// method fails, MATLAB will automatically destroy the output matrix C, and
// will leave the input @GrB matrix A unchanged.

// This strategy allows C to be safely created with no memory leaks.  The only
// downside is that this approach requires a copy to be made.  This method is
// used after another mexFunction has created a @GrB matrix with KIND_SPARSE,
// KIND_FULL, or KIND_BUILTIN (either sparse or full), in prepartion for
// this mexFunction which creates the final MATLAB/Octave sparse/full matrix.

// Usage:

// C = gb2builtin (A)

#include "gb_interface.h"

#define USAGE "usage: C = gb2builtin (A_opaque)"

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

    gbmx_usage (nargin == 1 && nargout == 1, USAGE) ;

    // input must be a uint8 matrix of size exactly 8
    CHECK_ERROR (mxGetClassID (pargin [0]) != mxUINT8_CLASS ||
        mxGetNumberOfElements (pargin [0]) != sizeof (GrB_Matrix *),
        "internal error 713") ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = (*((GrB_Matrix *) mxGetData (pargin [0]))) ;

    uint64_t nrows, ncols, nvals ;
    int sparsity_status, nthreads, fmt, bits, will_wait, iso ;

    OK (GrB_Matrix_nrows (&nrows, A)) ;
    OK (GrB_Matrix_ncols (&ncols, A)) ;
    OK (GrB_Matrix_get_INT32 (A, &sparsity_status, GxB_SPARSITY_STATUS)) ;
    GrB_Type type ;
    OK (GxB_Matrix_type (&type, A)) ;
    size_t typesize ;
    OK (GxB_Type_size (&typesize, type)) ;

    uint64_t *Ap = (uint64_t *) A->p ;
    uint64_t *Ai = (uint64_t *) A->i ;
    void *Ax = A->x ;

    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;

    //--------------------------------------------------------------------------
    // sanity checks
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
    CHECK_ERROR (fmt != GxB_BY_COL, "internal error 717") ;

    OK (GrB_Matrix_get_INT32 (A, &bits, GxB_OFFSET_INTEGER_BITS)) ;
    CHECK_ERROR (bits != 64, "internal error 718") ;

    OK (GrB_Matrix_get_INT32 (A, &bits, GxB_ROWINDEX_INTEGER_BITS)) ;
    CHECK_ERROR (bits != 64, "internal error 719") ;

    OK (GrB_Matrix_get_INT32 (A, &will_wait, GxB_WILL_WAIT)) ;
    CHECK_ERROR (will_wait, "internal error 720") ;

    OK (GrB_Matrix_get_INT32 (A, &iso, GxB_ISO)) ;
    CHECK_ERROR (iso, "internal error 721") ;

    //--------------------------------------------------------------------------
    // construct the output MATLAB/Octave matrix
    //--------------------------------------------------------------------------

    mxArray *C ;

    if (sparsity_status == GxB_SPARSE)
    {
        if (type == GrB_BOOL)
        { 
            C = mxCreateSparseLogicalMatrix (nrows, ncols, nvals+1) ;
        }
        else if (type == GrB_FP64)
        { 
            C = mxCreateSparse (nrows, ncols, nvals+1, mxREAL) ;
        }
        else if (type == GxB_FC64)
        { 
            C = mxCreateSparse (nrows, ncols, nvals+1, mxCOMPLEX) ;
        }
        else
        { 
            // A must be bool, fp64, or fc64
            ERROR ("internal error 722", GrB_INVALID_VALUE) ;
        }
        uint64_t *Cp = mxGetJc (C) ;
        uint64_t *Ci = mxGetIr (C) ;
        GB_memcpy (Cp, Ap, (ncols+1) * sizeof (uint64_t), nthreads) ;
        GB_memcpy (Ci, Ai, nvals * sizeof (uint64_t), nthreads) ;
    }
    else if (sparsity_status == GxB_FULL)
    { 
        C = gbmx_new_matlab_matrix (nrows, ncols, type) ;
    }
    else
    { 
        // A must be sparse or full, never bitmap or hypersparse
        ERROR ("internal error 723", GrB_INVALID_VALUE) ;
    }

    void *Cx = mxGetData (C) ;
    GB_memcpy (Cx, Ax, nvals * typesize, nthreads) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    pargout [0] = C ;
    gb_wrapup ( ) ;
}

