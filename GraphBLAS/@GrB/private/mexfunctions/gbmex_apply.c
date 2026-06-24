//------------------------------------------------------------------------------
// gbmex_apply: apply a unary operator to a sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_apply is an interface to GrB_Matrix_apply.

// Usage for @GrB and @GhB:

// C = gbapply (unop, A)                        C = unop (A)
// C = gbapply (Cin, unop, A)                   C = Cin ; C = unop (A) (***)
// C = gbapply (Cin, accum, unop, A)            C = Cin + unop (A)
// C = gbapply (Cin, M, unop, A)                C = Cin ; C<M> = unop (A)
// C = gbapply (Cin, M, accum, unop, A)         C = Cin ; C<M> += unop(A)

// C = gbapply (unop, A, desc)                  ditto, with desc
// C = gbapply (Cin, unop, A, desc)
// C = gbapply (Cin, accum, unop, A, desc)
// C = gbapply (Cin, M, unop, A, desc)
// C = gbapply (Cin, M, accum, unop, A, desc)

// Usage for @GhB only:

// gbapply (C, unop)                            C = unop (C)
// gbapply (C, accum, unop)                     C += unop (C)
// gbapply (C, unop, A)                         C = unop (A)
// gbapply (C, accum, unop, A)                  C += unop (A)
// gbapply (C, M, unop, A)                      C<M> = unop (A)
// gbapply (C, M, accum, unop, A)               C<M> += unop (A)

// gbapply (C, unop, desc)                      ditto, with desc
// gbapply (C, accum, unop, desc)
// gbapply (C, unop, A, desc)
// gbapply (C, accum, unop, A, desc)
// gbapply (C, M, unop, A, desc)
// gbapply (C, M, accum, unop, A, desc)

// C is Cin for the inplace usage, and must be present.  For the non-inplace
// usage, if Cin is not present then it is implicitly a matrix with no entries,
// of the right size (which depends on A, B, and the descriptor).

// (***) note: the usage C = gbapply (Cin, unop, A) works, but is not useful.
// It does the same thing as C = gbapply (unop, A).  The corresponding in-place
// syntax, gbapply (C, unop, A), is useful, and does C = unop (A).  It is
// similar to C = gbapply (unop, A) which creates a new C.  The in-place usage
// gbapply (C, unop, A) uses an existing matrix C.  Its contents are
// overwritten, but its type is preserved, so the assignment will do a
// typecast.  This cannot be done with the non-in-place syntax C = gbapply
// (unop, A), where C takes its type from the output type of unop.

#define FREE_WORK                   \
    GrB_Matrix_free (&M_to_free) ;  \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#include "gb_interface.h"

#define USAGE "usage: C = GrB.apply (Cin, M, accum, op, A, desc)"

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
        M_to_free = NULL, A_to_free = NULL ;
    GrB_Descriptor desc = NULL ;
    int arena = GrB_DEFAULT ;

    GBMX_USAGE (nargin >= 3 && nargin <= 7 && nargout <= 2, USAGE) ;
    bool ghb = false ; // HACK (bool) mxGetScalar (pargin [0]) ;
    arena = ghb ? GrB_DEFAULT : MXARENA ;

    bool inplace = false ; // ghb && (nargout == 0) ;   // FIXME
    double *kind_output = NULL ;
    if (!inplace)
    { 
        if (ghb) pargout [0] = gbmx_export_struct (&C_opaque) ;
        pargout [1] = mxCreateDoubleScalar (0) ;
        kind_output = (double *) mxGetData (pargout [1]) ;
    }

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

    CHECK_ERROR (nmatrices < 1 || nmatrices > 3 || nstrings < 1 || ncells > 0,
        USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 1)
    { 
        if (inplace)
        { 
            // C = unop (C), in place usage, C and A are aliased
            OK (gb_get_deep (&C, inplace, &(Matrix [0]), arena, err)) ;
            A = C ;
        }
        else
        { 
            // C = unop (A), not in place; C created below
            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
        }
    }
    else if (nmatrices == 2)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [1]), arena, err)) ;
    }
    else // if (nmatrices == 3)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&M, &M_to_free, &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), arena, err)) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    GrB_UnaryOp op = NULL ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_unop (&op, &(String [0][0]), atype, err)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, &(String [0][0]), ctype, ctype, err)) ;
        OK (gb_string_to_unop (&op, &(String [1][0]), atype, err)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        ASSERT (!inplace) ;

        // get the descriptor contents to determine if A is transposed
        bool A_transpose = (gbdesc.in0 == GrB_TRAN) ;

        // get the size of A
        uint64_t anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;
        uint64_t cncols = (A_transpose) ? anrows : ancols ;

        // use the ztype of the op as the type of C
        int code ;
        OK (GrB_UnaryOp_get_INT32 (op, &code, GrB_OUTP_TYPE_CODE)) ;
        ctype = gb_code_to_type (code) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, cncols, A, NULL, &(gbdesc.fmt), err)) ;
        OK (gb_get_sparsity (A, NULL, &(gbdesc.sparsity), err)) ;
        OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity,
            arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += f(A)
    //--------------------------------------------------------------------------

    OK1 (C, GrB_Matrix_apply (C, M, accum, op, A, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    if (!inplace)
    { 
        OK (gb_export (C_opaque, &C, gbdesc.kind, ghb, err)) ;
        (*kind_output) = (double) gbdesc.kind ;
    }
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_to_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

