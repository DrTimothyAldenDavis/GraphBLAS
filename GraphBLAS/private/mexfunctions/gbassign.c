//------------------------------------------------------------------------------
// gbassign: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// gbassign is an interface to GrB_Matrix_assign and GrB_Matrix_assign_[TYPE],
// computing the GraphBLAS expression:

//      C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')

// where A can be a matrix or a scalar.

// Usage:

//      Cout = gbassign (Cin, M, accum, A, I, J, desc)

// Cin, A, and desc are required.  See gb.m for more details.

#define ASSIGN_USAGE "usage: Cout = gb.assign (Cin, M, accum, A, I, J, desc)"

#include "gb_matlab.h"

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

    gb_usage (nargin >= 3 && nargin <= 7 && nargout <= 1, ASSIGN_USAGE) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    kind_enum_t kind ;
    GrB_Descriptor desc = gb_mxarray_to_descriptor (pargin [nargin-1], &kind) ;

    //--------------------------------------------------------------------------
    // find the remaining arguents
    //--------------------------------------------------------------------------

    int I_arg = -1, J_arg = -1 ;            // cell arrays I and J
    int matrix_arg [3] = {-1, -1, -1} ;     // C, M, and A matrices
    int nmatrix_args = 0 ;                  // # of matrix arguments (C, M, A)
    int accum_arg = -1 ;                    // accum string

    for (int k = 0 ; k < (nargin-1) ; k++)
    {
        if (mxIsCell (pargin [k]))
        {
            // a cell array is either I or J
            if (I_arg == -1)
            {
                I_arg = k ;
            }
            else if (J_arg == -1)
            {
                J_arg = k ;
            }
            else
            {
                ERROR ("only 2D indexing is supported") ;
            }
        }
        else if (mxIsChar (pargin [k]))
        {
            // a string array is only the accum operator
            if (accum_arg == -1)
            {
                accum_arg = k ;
            }
            else
            {
                ERROR ("only a single accum operator string allowed") ;
            }
        }
        else
        {
            // a matrix argument is C, M, or A
            if (nmatrix_args >= 3)
            {
                // at most 3 matrix inputs are allowed
                USAGE (ASSIGN_USAGE) ;
            }
            matrix_arg [nmatrix_args++] = k ;
        }
    }

    //--------------------------------------------------------------------------
    // get the matrix arguments
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, M = NULL, A = NULL ;

    if (nmatrix_args < 2)
    {
        // at least 2 matrix inputs are required
        USAGE (ASSIGN_USAGE) ;
    }
    else if (nmatrix_args == 2)
    {
        // with 2 matrix arguments: Cin and A, in that order
        C = gb_get_deep    (pargin [matrix_arg [0]], NULL) ;
        A = gb_get_shallow (pargin [matrix_arg [1]]) ;
    }
    else if (nmatrix_args == 3)
    {
        // with 3 matrix arguments: Cin, M, and A, in that order
        C = gb_get_deep    (pargin [matrix_arg [0]], NULL) ;
        M = gb_get_shallow (pargin [matrix_arg [1]]) ;
        A = gb_get_shallow (pargin [matrix_arg [2]]) ;
    }

    // get the size and type of Cin
    GrB_Type ctype ;
    GrB_Index cnrows, cncols ;
    OK (GxB_Matrix_type (&ctype, C)) ;
    OK (GrB_Matrix_nrows (&cnrows, A)) ;
    OK (GrB_Matrix_ncols (&cncols, A)) ;

    // determine if A is a scalar (ignore the transpose descriptor)
    GrB_Index anrows, ancols, anvals ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nvals (&anvals, A)) ;
    bool scalar_assignment = (anrows == 1) && (ancols == 1) && (anvals == 1) ;

    //--------------------------------------------------------------------------
    // get I and J
    //--------------------------------------------------------------------------

    GrB_Index *I = (GrB_Index *) GrB_ALL ;
    GrB_Index *J = (GrB_Index *) GrB_ALL ;
    GrB_Index ni = cnrows, nj = cncols ;
    bool I_allocated = false, J_allocated = false ;

    if (I_arg >= 0)
    {
        // I is present
        I = gb_mxcell_to_index (pargin [I_arg], cnrows, &I_allocated, &ni) ;
    }

    if (J_arg >= 0)
    {
        // both I and J are present
        J = gb_mxcell_to_index (pargin [J_arg], cncols, &J_allocated, &nj) ;
    }

    //--------------------------------------------------------------------------
    // get accum
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    if (accum_arg >= 0)
    {
        accum = gb_mxstring_to_binop (pargin [accum_arg], ctype) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M>(I,J) += A
    //--------------------------------------------------------------------------

    if (scalar_assignment)
    {
        gb_matrix_assign_scalar (C, M, accum, A, I, ni, J, nj, desc) ;
    }
    else
    {
        OK (GrB_assign (C, M, accum, A, I, ni, J, nj, desc)) ;
    }

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_free (&M)) ;
    OK (GrB_free (&A)) ;
    OK (GrB_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
}

