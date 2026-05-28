//------------------------------------------------------------------------------
// gbextract: extract entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbextract is an interface to GrB_Matrix_extract and
// GrB_Matrix_extract_[TYPE], computing the GraphBLAS expression:

//      C<#M,replace> = accum (C, A (I,J)) or
//      C<#M,replace> = accum (C, AT (I,J))

// Usage:

//      C = gbextract (Cin, M, accum, A, I, J, desc)

// A is required.  See GrB.m for more details.
// If accum or M is used, then Cin must appear.

#define FREE_WORK                   \
    GrB_Matrix_free (&C_shallow) ;  \
    GrB_Matrix_free (&M_shallow) ;  \
    GrB_Matrix_free (&A_shallow) ;  \
    GrB_Vector_free (&I_to_free) ;  \
    GrB_Vector_free (&J_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = GrB.extract (Cin, M, accum, A, I, J, desc)"

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

    GrB_Type atype, ctype = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL,
        C_shallow = NULL, M_shallow = NULL, A_shallow = NULL ;
    GrB_Vector I = NULL, J = NULL, I_to_free = NULL, J_to_free = NULL ;
    GrB_Descriptor desc = NULL ;

    gbmx_usage (nargin >= 1 && nargin <= 7 && nargout <= 2, USAGE) ;
    pargout [0] = gbmx_export_struct (&C_opaque) ;
    pargout [1] = mxCreateDoubleScalar (0) ;
    double *kind_output = (double *) mxGetData (pargout [1]) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    struct gb_matrix_struct Cell0_Matrix [3] ;
    struct gb_matrix_struct Cell1_Matrix [3] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    CHECK_ERROR (nmatrices < 1 || nmatrices > 3 || nstrings > 1, USAGE) ;

    int Cell0_len = 0, Cell1_len = 0 ;
    if (ncells > 0)
    { 
        gbmx_mxcell_to_matrices (Cell0_Matrix, &Cell0_len, Cell [0]) ;
    }
    if (ncells > 1)
    { 
        gbmx_mxcell_to_matrices (Cell1_Matrix, &Cell1_len, Cell [1]) ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    gbdesc.nondefault = true ;      // ensure the GrB_Descriptor is allocated
    OK (gb_get_descriptor (&desc, &gbdesc)) ;
    ASSERT (desc != NULL) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 1)
    { 
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [0]))) ;
    }
    else if (nmatrices == 2)
    { 
        OK (gb_get_deep   (&C, &C_shallow, &(Matrix [0]))) ;
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [1]))) ;
    }
    else // if (nmatrices == 3)
    { 
        OK (gb_get_deep   (&C, &C_shallow, &(Matrix [0]))) ;
        OK (gb_get_matrix (&M, &M_shallow, &(Matrix [1]))) ;
        OK (gb_get_matrix (&A, &A_shallow, &(Matrix [2]))) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;

    if (nstrings == 1)
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype)) ;
    }

    //--------------------------------------------------------------------------
    // get the size of A
    //--------------------------------------------------------------------------

    int in0 ;
    OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
    uint64_t anrows, ancols ;
    bool A_transpose = (in0 == GrB_TRAN) ;
    if (A_transpose)
    { 
        // T = AT (I,J) is to be extracted where AT = A'
        OK (GrB_Matrix_nrows (&ancols, A)) ;
        OK (GrB_Matrix_ncols (&anrows, A)) ;
    }
    else
    { 
        // T = A (I,J) is to be extracted
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;
    }

    //--------------------------------------------------------------------------
    // get I and J
    //--------------------------------------------------------------------------

    uint64_t cnrows = anrows, cncols = ancols ;
    int icells = 0, jcells = 0 ;
    int base_offset = (gbdesc.base == BASE_0_INT) ? 0 : 1 ;

    if (anrows == 1 && ncells == 1)
    { 
        // only J is present
        OK (gb_cell_to_list (&J, &J_to_free, &cncols, NULL,
            Cell0_Matrix, Cell0_len, base_offset, ancols)) ;
        jcells = Cell0_len ;
    }
    else if (ncells == 1)
    { 
        // only I is present
        OK (gb_cell_to_list (&I, &I_to_free, &cnrows, NULL,
            Cell0_Matrix, Cell0_len, base_offset, anrows)) ;
        icells = Cell0_len ;
    }
    else if (ncells == 2)
    { 
        // both I and J are present
        OK (gb_cell_to_list (&I, &I_to_free, &cnrows, NULL,
            Cell0_Matrix, Cell0_len, base_offset, anrows)) ;
        OK (gb_cell_to_list (&J, &J_to_free, &cncols, NULL,
            Cell1_Matrix, Cell1_len, base_offset, ancols)) ;
        icells = Cell0_len ;
        jcells = Cell1_len ;
    }

    if (icells > 1)
    { 
        // I is a 3-element vector containing a stride
        OK (GrB_Descriptor_set_INT32 (desc, GxB_IS_STRIDE, GxB_ROWINDEX_LIST)) ;
    }

    if (jcells > 1)
    { 
        // J is a 3-element vector containing a stride
        OK (GrB_Descriptor_set_INT32 (desc, GxB_IS_STRIDE, GxB_COLINDEX_LIST)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    if (C == NULL)
    { 
        // Cin is not present: determine its size, same type as A.
        // T = A(I,J) or AT(I,J) will be extracted; accum must be null.
        ctype = atype ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, cncols, A, NULL, &(gbdesc.fmt))) ;
        OK (gb_get_sparsity (A, NULL, &(gbdesc.sparsity))) ;
        OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity)) ;
    }

// printf ("C input:\n") ; GxB_Matrix_fprint (C, "C input", 5, NULL) ;
// printf ("M input:\n") ; GxB_Matrix_fprint (M, "M input", 5, NULL) ;
// printf ("A input:\n") ; GxB_Matrix_fprint (A, "A input", 5, NULL) ;
// printf ("I input:\n") ; GxB_Vector_fprint (I, "I input", 5, NULL) ;
// printf ("J input:\n") ; GxB_Vector_fprint (J, "J input", 5, NULL) ;

    //--------------------------------------------------------------------------
    // C<M> += A(I,J) or AT(I,J)
    //--------------------------------------------------------------------------

    OK1 (C, GxB_Matrix_extract_Vector (C, M, accum, A, I, J, desc)) ;

// GxB_Matrix_fprint (C, "C output", 5, NULL) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    OK (gb_export (C_opaque, &C, gbdesc.kind)) ;
    (*kind_output) = (double) gbdesc.kind ;
    gb_wrapup ( ) ;
}

