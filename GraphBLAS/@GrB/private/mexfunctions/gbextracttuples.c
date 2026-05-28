//------------------------------------------------------------------------------
// gbextracttuples: extract all entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// [I J X] = GrB.extracttuples (A)
// [I J X] = GrB.extracttuples (A, desc)

// The desciptor is optional.  If present, it must be a struct.

// desc.base = 'zero-based':    I and J are returned as 0-based integer indices
// desc.base = 'one-based int': I and J are returned as 1-based integer indices
// desc.base = 'one-based':     I and J are returned as 1-based integer indices
// desc.base = 'one-based double' one-based double unless max(size(A)) >
//                              flintmax, in which case 'one-based int' is used.
// desc.base = 'default':       'one-based int'

// The input matrix must have no pending work.

// I, J, and X are returned as MATLAB matrices.

// FUTURE: add an option to return I,J,X as GrB matrices instead
// FUTURE: reduce # of copies made

#define FREE_WORK                   \
    GrB_Vector_free (&I) ;          \
    GrB_Vector_free (&J) ;          \
    GrB_Vector_free (&X) ;          \
    GrB_Vector_free (&T) ;          \
    GrB_Matrix_free (&A_shallow) ;  \
    gb_free (&x) ;

#include "gb_interface.h"

#define USAGE "usage: [I,J,X] = GrB.extracttuples (A, desc)"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, A_shallow = NULL ;
    GrB_Vector I = NULL, J = NULL, X = NULL, T = NULL ;
    void *x = NULL ;

    gbmx_usage (nargin >= 1 && nargin <= 2 && nargout <= 3, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;
    // printf ("base: %d\n", gbdesc.base) ;

    CHECK_ERROR (nmatrices != 1 || nstrings > 0 || ncells > 0, USAGE) ;

    CHECK_ERROR (Matrix [0].will_wait, "matrix must have no pending work") ;

    //--------------------------------------------------------------------------
    // construct I, J, X outputs
    //--------------------------------------------------------------------------

    int64_t nvals = Matrix [0].nvals ;
    int64_t nrows = Matrix [0].nrows ;
    int64_t ncols = Matrix [0].ncols ;
    GrB_Type X_type = Matrix [0].type ;
    size_t X_typesize = Matrix [0].typesize ;

    bool extract_I = true ;
    bool extract_J = (nargout > 1) ;
    bool extract_X = (nargout > 2) ;

    if (gbdesc.base == BASE_1_DOUBLE && MAX (nrows, ncols) > FLINTMAX)
    { 
        // printf ("switching to base 1 int\n") ;
        gbdesc.base = BASE_1_INT ;
    }

    GrB_Type I_type, J_type ;
    if (gbdesc.base == BASE_1_DOUBLE)
    { 
        I_type = GrB_FP64 ;
        J_type = GrB_FP64 ;
    }
    else
    { 
        bool I_is_32 = (nrows <= INT32_MAX) ;
        bool J_is_32 = (ncols <= INT32_MAX) ;
        I_type = (I_is_32) ? GrB_INT32 : GrB_INT64 ;
        J_type = (J_is_32) ? GrB_INT32 : GrB_INT64 ;
    }

    void *I_out = NULL, *J_out = NULL, *X_out = NULL ;
    size_t I_typesize, J_typesize ;
    OK (GxB_Type_size (&I_typesize, I_type)) ;
    OK (GxB_Type_size (&J_typesize, J_type)) ;

    if (extract_I)
    { 
        pargout [0] = gbmx_new_matlab_matrix (nvals, 1, I_type) ;
        I_out = mxGetData (pargout [0]) ;
    }
    if (extract_J)
    { 
        pargout [1] = gbmx_new_matlab_matrix (nvals, 1, J_type) ;
        J_out = mxGetData (pargout [1]) ;
    }
    if (extract_X)
    { 
        pargout [2] = gbmx_new_matlab_matrix (nvals, 1, X_type) ;
        X_out = mxGetData (pargout [2]) ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &nthreads, GxB_NTHREADS)) ;

    //--------------------------------------------------------------------------
    // get the matrix; disable burble for scalars
    //--------------------------------------------------------------------------

    OK (gb_get_matrix (&A, &A_shallow, &(Matrix [0]))) ;
    int burble ;
    bool disable_burble = (nrows <= 1 && ncols <= 1) ;
    if (disable_burble)
    { 
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &burble, GxB_BURBLE)) ;
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, GxB_BURBLE)) ;
    }

    //--------------------------------------------------------------------------
    // create empty GrB_Vectors for I, J, and X
    //--------------------------------------------------------------------------

    // type of I and J will be revised as needed by GxB_Matrix_extractTuples
    if (extract_I) OK (GrB_Vector_new (&I, GrB_UINT64, 0)) ;
    if (extract_J) OK (GrB_Vector_new (&J, GrB_UINT64, 0)) ;
    if (extract_X) OK (GrB_Vector_new (&X, X_type, 0)) ;

    //--------------------------------------------------------------------------
    // extract the tuples from A into I, J, and X
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_extractTuples_Vector (I, J, X, A, NULL)) ;

    //--------------------------------------------------------------------------
    // determine if 1 must be added to the indices
    //--------------------------------------------------------------------------

    int base_offset = (gbdesc.base == BASE_0_INT) ? 0 : 1 ;

    //--------------------------------------------------------------------------
    // return I to MATLAB
    //--------------------------------------------------------------------------

    uint64_t size = 0, nvals2 = 0 ;
    int ignore = 0 ;
    GrB_Type type = NULL ;

    if (extract_I)
    { 
        if (gbdesc.base == BASE_1_DOUBLE)
        { 
            // I = (double) (I + 1)
            OK (GrB_Vector_new (&T, GrB_FP64, nvals)) ;
            OK (GrB_Vector_apply_BinaryOp2nd_FP64 (T, NULL, NULL,
                GrB_PLUS_FP64, I, base_offset, NULL)) ;
            GrB_Vector_free (&I) ;
            I = T ;
            T = NULL ;
        }
        else if (base_offset != 0)
        { 
            // I = I+1, as a uint64 or uint32 vector
            OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (I, NULL, NULL,
                GrB_PLUS_UINT64, I, 1, NULL)) ;
        }
        uint64_t nvals2 ;
        OK (GxB_Vector_unload (I, &x, &type, &nvals2, &size, &ignore, NULL)) ;
        if (type == GrB_UINT32) type = GrB_INT32 ;
        if (type == GrB_UINT64) type = GrB_INT64 ;
        ASSERT (type == I_type) ;
        ASSERT (nvals == nvals2) ;
        GB_memcpy (I_out, x, nvals * I_typesize, nthreads) ;
        gb_free (&x) ;
        GrB_Vector_free (&I) ;
    }

    //--------------------------------------------------------------------------
    // return J to MATLAB
    //--------------------------------------------------------------------------

    if (extract_J)
    { 
        if (gbdesc.base == BASE_1_DOUBLE)
        { 
            // J = (double) (J + 1)
            OK (GrB_Vector_new (&T, GrB_FP64, nvals)) ;
            OK (GrB_Vector_apply_BinaryOp2nd_FP64 (T, NULL, NULL,
                GrB_PLUS_FP64, J, base_offset, NULL)) ;
            GrB_Vector_free (&J) ;
            J = T ;
            T = NULL ;
        }
        else if (base_offset != 0)
        { 
            // J = J+1, as a uint64 or uint32 vector
            OK (GrB_Vector_apply_BinaryOp2nd_UINT64 (J, NULL, NULL,
                GrB_PLUS_UINT64, J, 1, NULL)) ;
        }
        OK (GxB_Vector_unload (J, &x, &type, &nvals2, &size, &ignore, NULL)) ;
        if (type == GrB_UINT32) type = GrB_INT32 ;
        if (type == GrB_UINT64) type = GrB_INT64 ;
        ASSERT (type == J_type) ;
        ASSERT (nvals == nvals2) ;
        GB_memcpy (J_out, x, nvals * J_typesize, nthreads) ;
        gb_free (&x) ;
        GrB_Vector_free (&J) ;
    }

    //--------------------------------------------------------------------------
    // return X to MATLAB
    //--------------------------------------------------------------------------

    if (extract_X)
    { 
        OK (GxB_Vector_unload (X, &x, &type, &nvals2, &size, &ignore, NULL)) ;
        ASSERT (type == X_type) ;
        ASSERT (nvals == nvals2) ;
        GB_memcpy (X_out, x, nvals * X_typesize, nthreads) ;
        gb_free (&x) ;
        GrB_Vector_free (&X) ;
    }

    //--------------------------------------------------------------------------
    // restore burble and return result
    //--------------------------------------------------------------------------

    if (disable_burble)
    { 
        OK (GrB_Global_set_INT32 (GrB_GLOBAL, burble, GxB_BURBLE)) ;
    }

    FREE_WORK ;
    gb_wrapup ( ) ;
}

