//------------------------------------------------------------------------------
// gb_assign: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// With do_subassign false, gb_assign is an interface to GrB_Matrix_assign and
// GrB_Matrix_assign_[TYPE], computing the GraphBLAS expression:

//      C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')

// With do_subassign true, gb_assign is an interface to GxB_Matrix_subassign
// and GxB_Matrix_subassign_[TYPE], computing the GraphBLAS expression:

//      C(I,J)<#M,replace> = accum (C(I,J), A) or accum(C(I,J), A')

// A can be a matrix or a scalar.  If it is a scalar with nnz (A) == 0,
// then it is first expanded to an empty matrix of size length(I)-by-length(J),
// and G*B_Matrix_*assign is used (not GraphBLAS scalar assignment).

// MATLAB Usage:

//      Cout = gbassign    (Cin, M, accum, A, I, J, desc)
//      Cout = gbsubassign (Cin, M, accum, A, I, J, desc)

// Cin, A, and desc are required.  See gb.m for more details.

#include "gb_matlab.h"

void gb_assign                  // gbassign or gbsubassign mexFunctions
(
    int nargout,                // # output arguments for mexFunction
    mxArray *pargout [ ],       // output arguments for mexFunction
    int nargin,                 // # inpu arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    bool do_subassign,          // true: do subassign, false: do assign
    const char *usage           // usage string to print if error
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 3 && nargin <= 7 && nargout <= 1, usage) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    kind_enum_t kind ;
    GxB_Format_Value fmt ;
    GrB_Descriptor desc = 
        gb_mxarray_to_descriptor (pargin [nargin-1], &kind, &fmt) ;

    //--------------------------------------------------------------------------
    // find the remaining arguments
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
                USAGE (usage) ;
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
        USAGE (usage) ;
    }
    else if (nmatrix_args == 2)
    {
        // with 2 matrix arguments: Cin and A, in that order
        C = gb_get_deep    (pargin [matrix_arg [0]]) ;
        A = gb_get_shallow (pargin [matrix_arg [1]]) ;
    }
    else if (nmatrix_args == 3)
    {
        // with 3 matrix arguments: Cin, M, and A, in that order
        C = gb_get_deep    (pargin [matrix_arg [0]]) ;
        M = gb_get_shallow (pargin [matrix_arg [1]]) ;
        A = gb_get_shallow (pargin [matrix_arg [2]]) ;
    }

    // get the size and type of Cin
    GrB_Type ctype ;
    GrB_Index cnrows, cncols ;
    OK (GxB_Matrix_type (&ctype, C)) ;
    OK (GrB_Matrix_nrows (&cnrows, C)) ;
    OK (GrB_Matrix_ncols (&cncols, C)) ;

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
    // determine if A is a scalar (ignore the transpose descriptor)
    //--------------------------------------------------------------------------

    GrB_Index anrows, ancols, anvals ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    OK (GrB_Matrix_nvals (&anvals, A)) ;
    bool scalar_assignment = (anrows == 1) && (ancols == 1) ;

    if (scalar_assignment && anvals == 0)
    {
        // A is a sparse scalar.  Expand it to an ni-by-nj sparse matrix with
        // the same type as C, with no entries, and use matrix assignment.
        OK (GrB_free (&A)) ;
        OK (GrB_Matrix_new (&A, ctype, ni, nj)) ;
        OK (GxB_get (C, GxB_FORMAT, &fmt)) ;
        OK (GxB_set (A, GxB_FORMAT, fmt)) ;
        scalar_assignment = false ;
    }

    //--------------------------------------------------------------------------
    // compute C(I,J)<M> += A or C<M>(I,J) += A
    //--------------------------------------------------------------------------

    if (scalar_assignment)
    {
        gb_matrix_assign_scalar (C, M, accum, A, I, ni, J, nj, desc,
            do_subassign) ;
    }
    else
    {
        if (do_subassign)
        {
            OK (GxB_subassign (C, M, accum, A, I, ni, J, nj, desc)) ;
        }
        else
        {
            OK (GrB_assign (C, M, accum, A, I, ni, J, nj, desc)) ;
        }
    }

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_free (&M)) ;
    OK (GrB_free (&A)) ;
    OK (GrB_free (&desc)) ;
    if (I_allocated) gb_mxfree (&I) ;
    if (J_allocated) gb_mxfree (&J) ;

    //--------------------------------------------------------------------------
    // export the output matrix C back to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
}

