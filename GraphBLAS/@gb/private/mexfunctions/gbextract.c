//------------------------------------------------------------------------------
// gbextract: extract entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// gbextract is an interface to GrB_Matrix_extract and GrB_Matrix_extract_[TYPE],
// computing the GraphBLAS expression:

//      C<#M,replace> = accum (C, A (I,J)) or
//      C<#M,replace> = accum (C, AT (I,J))

// Usage:

//      Cout = gbextract (Cin, M, accum, A, I, J, desc)

// A and desc are required.  See gb.m for more details.
// If accum or M is used, then Cin must appear.

#define EXTRACT_USAGE "usage: Cout = gb.extract (Cin, M, accum, A, I, J, desc)"

#include "gb_matlab.h"
#include "GB_ij.h"

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

    gb_usage (nargin >= 2 && nargin <= 7 && nargout <= 1, EXTRACT_USAGE) ;

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
                ERROR (EXTRACT_USAGE) ;
            }
            matrix_arg [nmatrix_args++] = k ;
        }
    }

    //--------------------------------------------------------------------------
    // get the matrix arguments
    //--------------------------------------------------------------------------

    GrB_Matrix C = NULL, M = NULL, A = NULL ;

    if (nmatrix_args < 1)
    { 
        // at least 1 matrix input is required
        ERROR (EXTRACT_USAGE) ;
    }
    else if (nmatrix_args == 1)
    {
        // with 1 matrix argument: A.  Cin does not appear so neither can accum
        if (accum_arg >= 0)
        { 
            // if both A and accum are present, then Cin must appear
            ERROR (EXTRACT_USAGE) ;
        }
        A = gb_get_shallow (pargin [matrix_arg [0]]) ;
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

    //--------------------------------------------------------------------------
    // get the size and type of A
    //--------------------------------------------------------------------------

    GrB_Type atype ;
    OK (GxB_Matrix_type (&atype, A)) ;
    GrB_Desc_Value in0 ;
    OK (GxB_get (desc, GrB_INP0, &in0)) ;
    GrB_Index anrows, ancols ;
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

    GrB_Index *I = (GrB_Index *) GrB_ALL ;
    GrB_Index *J = (GrB_Index *) GrB_ALL ;
    GrB_Index ni = anrows, nj = ancols ;
    bool I_allocated = false, J_allocated = false ;

    if (I_arg >= 0)
    { 
        // I is present
        I = gb_mxcell_to_index (pargin [I_arg], anrows, &I_allocated, &ni) ;
    }

    if (J_arg >= 0)
    { 
        // both I and J are present
        J = gb_mxcell_to_index (pargin [J_arg], ancols, &J_allocated, &nj) ;
    }

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    // get the type of Cin, and construct it if it does not appear
    GrB_Type ctype = NULL ;

    if (C == NULL)
    { 
        // Cin is not present: determine its size, same type as A.
        // T = A(I,J) or AT(I,J) will be extracted.
        // accum must be null
        int I_kind, J_kind ;
        int64_t I_colon [3], J_colon [3] ;
        GrB_Index cnrows, cncols ;
        GB_ijlength (I, ni, anrows, &cnrows, &I_kind, I_colon) ;
        GB_ijlength (J, nj, ancols, &cncols, &J_kind, J_colon) ;
        ctype = atype ;

        OK (GrB_Matrix_new (&C, ctype, cnrows, cncols)) ;
        fmt = gb_get_format (cnrows, cncols, A, NULL, fmt) ;
        OK (GxB_set (C, GxB_FORMAT, fmt)) ;
    }
    else
    { 
        // Cin appears; get its type
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get accum
    //--------------------------------------------------------------------------

    // if accum appears, Cin must be present
    GrB_BinaryOp accum = NULL ;
    if (accum_arg >= 0)
    { 
        accum = gb_mxstring_to_binop (pargin [accum_arg], ctype) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += A(I,J) or AT(I,J)
    //--------------------------------------------------------------------------

    OK (GrB_extract (C, M, accum, A, I, ni, J, nj, desc)) ;

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

