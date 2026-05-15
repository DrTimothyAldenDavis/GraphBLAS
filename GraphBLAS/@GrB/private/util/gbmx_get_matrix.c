//------------------------------------------------------------------------------
// gbmx_get_matrix: get a matrix argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmx_get_matrix (matrix,X) gets the contents of a GraphBLAS @GrB matrix
// object, or the properties of a MATLAB matrix (type, dimensions, and pointers
// to p,i,x, etc), and saves them in the gb_matrix struct.

// X must not be NULL, but it can be an empty matrix, as X = [ ].  In this
// case, the gb_matrix will be 0-by-0.

// This method allocates no memory, and thus mx* and GrB* methods are
// intermingled.

#include "gb_interface.h"

void gbmx_get_matrix
(
    // output
    gb_matrix matrix,       // either a GraphBLAS or MATLAB matrix, statically
                            // allocated (but undefined) on input
    // input
    const mxArray *X        // @GrB object or MATLAB matrix
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (matrix != NULL) ;
    CHECK_ERROR (X == NULL, "matrix is missing") ;
    memset (matrix, 0, sizeof (struct gb_matrix_struct)) ;

    //--------------------------------------------------------------------------
    // construct the gb_matrix
    //--------------------------------------------------------------------------

    if (mxIsStruct (X) || mxIsClass (X, "GrB"))
    { 

        //----------------------------------------------------------------------
        // X is a @GrB object
        //----------------------------------------------------------------------

//      printf ("get GrB matrix: %d %d\n", mxIsStruct (X), mxIsClass (X, "GrB")) ;

        matrix->G = gbmx_get_grb_matrix (X) ;
        matrix->will_wait = GB_will_wait (matrix->G) ;
        matrix->nvals = GB_nnz (matrix->G) ;
        OK (GrB_Matrix_nrows (&matrix->nrows, matrix->G)) ;
        OK (GrB_Matrix_ncols (&matrix->ncols, matrix->G)) ;
        OK (GxB_Matrix_type (&matrix->type, matrix->G)) ;
        OK (GxB_Type_size (&(matrix->typesize), matrix->type)) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // X is a MATLAB matrix
        //----------------------------------------------------------------------

//      printf ("get MATLAB matrix: \n") ;

        // get the type and dimensions
        matrix->type = gbmx_mxarray_type (X) ;
        OK (GxB_Type_size (&(matrix->typesize), matrix->type)) ;
        matrix->nrows = (uint64_t) mxGetM (X) ;
        matrix->ncols = (uint64_t) mxGetN (X) ;

        if (matrix->nrows == 0 && matrix->ncols == 0)
        {

            //------------------------------------------------------------------
            // X is an empty 0-by-0 MATLAB matrix.  X->[pix] are NULL.
            //------------------------------------------------------------------

            matrix->is_empty = true ;

        }
        else
        {

            //------------------------------------------------------------------
            // X is a non-empty MATLAB matrix
            //------------------------------------------------------------------

            matrix->is_sparse = mxIsSparse (X) ;
            // get matrix->p, matrix->i, and matrix->nvals, which depend on
            // whether or not the MATLAB matrix is sparse or full
            if (matrix->is_sparse)
            { 
                // X is a sparse MATLAB matrix
                matrix->p = (uint64_t *) mxGetJc (X) ;
                matrix->i = (uint64_t *) mxGetIr (X) ;
                matrix->nvals = matrix->p [matrix->ncols] ;
            }
            else
            { 
                // X is a full MATLAB matrix
                matrix->p = NULL ;
                matrix->i = NULL ;
                matrix->nvals = matrix->nrows * matrix->ncols ;
            }
            // get the matrix values
            matrix->x = (void *) mxGetData (X) ;
        }
    }


//  printf ("got gb_matrix:\n") ;
//  printf ("nvals %ld\n", matrix->nvals) ;
//  GxB_Type_fprint (matrix->type, "matrix->type", 5, NULL) ;
//  printf ("nrows %ld\n", matrix->nrows) ;
//  printf ("ncols %ld\n", matrix->ncols) ;
//  printf ("typesize %d\n", (int) matrix->typesize) ;
//  printf ("G: %p\n", matrix->G) ;
//  printf ("p: %p\n", matrix->p) ;
//  printf ("i: %p\n", matrix->i) ;
//  printf ("x: %p\n", matrix->x) ;
//  printf ("is_sparse: %p\n", matrix->is_sparse) ;
//  printf ("is_empty:  %p\n", matrix->is_empty) ;
//  printf ("will_wait  %d\n", matrix->will_wait) ;

}
