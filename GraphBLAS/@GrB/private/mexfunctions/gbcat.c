//------------------------------------------------------------------------------
// gbcat: matrix concatenation
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbcat is an interface to GxB_Matrix_concat.

// Usage:

// C = gbcat (Tiles, desc)

// where Tiles is a 2D cell array of matrices.

#define FREE_WORK                                       \
    if (Tiles_shallow != NULL)                          \
    {                                                   \
        for (int64_t k = 0 ; k < mn ; k++)              \
        {                                               \
            GrB_Matrix_free (&(Tiles_shallow [k])) ;    \
        }                                               \
    }                                                   \
    GrB_Descriptor_free (&desc) ;                       \
    gbmx_free ((void **) &gb_Tiles) ;                   \
    gbmx_free ((void **) &Tiles) ;                      \
    gbmx_free ((void **) &Tiles_shallow) ;

#define FREE_ALL            \
    FREE_WORK ;             \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = GrB.cat (Tiles, desc)"

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

    GrB_Matrix *C_opaque = NULL, C = NULL, *Tiles = NULL,
        *Tiles_shallow = NULL ;
    GrB_Descriptor desc = NULL ;
    gb_matrix gb_Tiles = NULL ;
    int64_t m = 0, n = 0, mn = 0 ;

    gbmx_usage (nargin >= 1 && nargin <= 2 && nargout <= 2, USAGE) ;
    pargout [0] = gbmx_export_struct (&C_opaque) ;
    pargout [1] = mxCreateDoubleScalar (0) ;
    double *kind_output = (double *) mxGetData (pargout [1]) ;

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

    CHECK_ERROR (nmatrices > 0 || nstrings > 0 || ncells != 1, USAGE) ;

    //--------------------------------------------------------------------------
    // get the inputs
    //--------------------------------------------------------------------------

    mxArray *mxTiles = Cell [0] ;
    m = mxGetM (mxTiles) ;
    n = mxGetN (mxTiles) ;
    mn = m * n ;
    gb_Tiles = mxCalloc (mn, sizeof (struct gb_matrix_struct)) ;
    Tiles = mxCalloc (mn, sizeof (GrB_Matrix)) ;
    Tiles_shallow = mxCalloc (mn, sizeof (GrB_Matrix)) ;

    for (int64_t j = 0 ; j < n ; j++)
    {
        for (int64_t i = 0 ; i < m ; i++)
        { 
            // get the gb_Tiles {i,j} matrix.
            // gb_Tiles is row-major but mxTiles is column-major
            const mxArray *X = mxGetCell (mxTiles, i+j*m) ;
            gbmx_get_matrix (&(gb_Tiles [i*n+j]), X) ;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc)) ;

    //--------------------------------------------------------------------------
    // get the input matrices
    //--------------------------------------------------------------------------

    for (int64_t k = 0 ; k < mn ; k++)
    { 
        // get the kth Tiles matrix; all arrays are row-major
        OK (gb_get_matrix (&(Tiles [k]), &(Tiles_shallow [k]),
            &(gb_Tiles [k]))) ;
    }

    //--------------------------------------------------------------------------
    // determine the # of rows of C from Tiles {:,0}
    //--------------------------------------------------------------------------

    uint64_t cnrows = 0 ;
    for (int64_t i = 0 ; i < m ; i++)
    { 
        uint64_t anrows ;
        OK (GrB_Matrix_nrows (&anrows, Tiles [i*n])) ;
        cnrows += anrows ;
    }

    //--------------------------------------------------------------------------
    // determine the # of columms of C from Tiles {0,:}
    //--------------------------------------------------------------------------

    uint64_t cncols = 0 ;
    for (int64_t j = 0 ; j < n ; j++)
    { 
        uint64_t ancols ;
        OK (GrB_Matrix_ncols (&ancols, Tiles [j])) ;
        cncols += ancols ;
    }

    //--------------------------------------------------------------------------
    // determine the type of C
    //--------------------------------------------------------------------------

    GrB_Type ctype ;
    OK (GxB_Matrix_type (&ctype, Tiles [0])) ;
    for (int64_t k = 1 ; k < mn ; k++)
    { 
        GrB_Type atype ;
        OK (GxB_Matrix_type (&atype, Tiles [k])) ;
        ctype = gb_default_type (ctype, atype) ;
    }

    //--------------------------------------------------------------------------
    // create the matrix C and set its format and sparsity
    //--------------------------------------------------------------------------

    OK (gb_get_format (cnrows, cncols, NULL, NULL, &(gbdesc.fmt))) ;
    OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity)) ;

    //--------------------------------------------------------------------------
    // C = concatenate (Tiles)
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_concat (C, Tiles, m, n, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, gbdesc.kind)) ;
    (*kind_output) = (double) gbdesc.kind ;
    gb_wrapup ( ) ;
}

