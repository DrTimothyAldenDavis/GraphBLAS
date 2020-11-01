//------------------------------------------------------------------------------
// GB_apply: apply a unary operator; optionally transpose a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C<M> = accum (C, op(A)) or accum (C, op(A)')

// GB_apply does the work for GrB_*_apply, including the binary op variants.

#include "GB_apply.h"
#include "GB_transpose.h"
#include "GB_accum_mask.h"

#define GB_FREE_ALL ;

GrB_Info GB_apply                   // C<M> = accum (C, op(A)) or op(A')
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // M descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
        const GrB_UnaryOp op1_in,       // unary operator to apply
        const GrB_BinaryOp op2_in,      // binary operator to apply
        const GxB_Scalar scalar,        // scalar to bind to binary operator
        bool binop_bind1st,             // if true, binop(x,A) else binop(A,y)
    const GrB_Matrix A,             // first input:  matrix A
    bool A_transpose,               // A matrix descriptor
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M and/or A
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    ASSERT_MATRIX_OK (C, "C input for GB_apply", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_apply", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_apply", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GB_apply", GB0) ;

    GrB_UnaryOp  op1 = op1_in ;
    GrB_BinaryOp op2 = op2_in ;
    GB_Opcode opcode ;
    GrB_Type T_type ;
    if (op1 != NULL)
    {   GB_cov[2576]++ ;
// covered (2576): 770427
        // apply a unary operator
        GB_RETURN_IF_FAULTY (op1) ;
        ASSERT_UNARYOP_OK (op1, "op1 for GB_apply", GB0) ;
        T_type = op1->ztype ;
        opcode = op1->opcode ;
        if (!GB_OPCODE_IS_POSITIONAL (opcode))
        {
            // A must also be compatible with op1->xtype
            if (!GB_Type_compatible (A->type, op1->xtype))
            {   GB_cov[2577]++ ;
// covered (2577): 6
                GB_ERROR (GrB_DOMAIN_MISMATCH,
                    "Incompatible type for z=%s(x):\n"
                    "input A of type [%s]\n"
                    "cannot be typecast to x input of type [%s]",
                    op1->name, A->type->name, op1->xtype->name) ;
            }
        }
    }
    else if (op2 != NULL)
    {
        // apply a binary operator, with one input bound to a scalar
        GB_RETURN_IF_FAULTY (op2) ;
        ASSERT_BINARYOP_OK (op2, "op2 for GB_apply", GB0) ;
        ASSERT_SCALAR_OK (scalar, "scalar for GB_apply", GB0) ;
        T_type = op2->ztype ;
        opcode = op2->opcode ;
        if (!GB_OPCODE_IS_POSITIONAL (opcode))
        {
            bool op_is_first  = opcode == GB_FIRST_opcode ;
            bool op_is_second = opcode == GB_SECOND_opcode ;
            bool op_is_pair   = opcode == GB_PAIR_opcode ;
            if (binop_bind1st)
            {
                // C = op (scalar,A)
                // A must be compatible with op2->ytype
                if (!(op_is_first || op_is_pair ||
                      GB_Type_compatible (A->type, op2->ytype)))
                {   GB_cov[2578]++ ;
// covered (2578): 2
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input A of type [%s]\n"
                        "cannot be typecast to y input of type [%s]",
                        op2->name, A->type->name, op2->ytype->name) ;
                }
                // scalar must be compatible with op2->xtype
                if (!(op_is_second || op_is_pair ||
                      GB_Type_compatible (scalar->type, op2->xtype)))
                {   GB_cov[2579]++ ;
// covered (2579): 2
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input scalar of type [%s]\n"
                        "cannot be typecast to x input of type [%s]",
                        op2->name, scalar->type->name, op2->xtype->name) ;
                }
            }
            else
            {
                // C = op (A,scalar)
                // A must be compatible with op2->xtype
                if (!(op_is_first || op_is_pair ||
                      GB_Type_compatible (A->type, op2->xtype)))
                {   GB_cov[2580]++ ;
// covered (2580): 2
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input scalar of type [%s]\n"
                        "cannot be typecast to x input of type [%s]",
                        op2->name, A->type->name, op2->xtype->name) ;
                }
                // scalar must be compatible with op2->ytype
                if (!(op_is_second || op_is_pair
                      || GB_Type_compatible (scalar->type, op2->ytype)))
                {   GB_cov[2581]++ ;
// covered (2581): 2
                    GB_ERROR (GrB_DOMAIN_MISMATCH,
                        "Incompatible type for z=%s(x,y):\n"
                        "input A of type [%s]\n"
                        "cannot be typecast to y input of type [%s]",
                        op2->name, scalar->type->name, op2->ytype->name) ;
                }
            }
        }
    }
    else
    {   GB_cov[2582]++ ;
// covered (2582): 4
        GB_ERROR (GrB_NULL_POINTER,
            "Required argument is null: [%s]", "op") ;
    }

    // check domains and dimensions for C<M> = accum (C,T)
    GrB_Info info = GB_compatible (C->type, C, M, accum, T_type, Context) ;
    if (info != GrB_SUCCESS)
    {   GB_cov[2583]++ ;
// covered (2583): 6
        return (info) ;
    }

    // check the dimensions
    int64_t tnrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
    int64_t tncols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
    if (GB_NROWS (C) != tnrows || GB_NCOLS (C) != tncols)
    {   GB_cov[2584]++ ;
// covered (2584): 2
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "Dimensions not compatible:\n"
            "output is " GBd "-by-" GBd "\n"
            "input is " GBd "-by-" GBd "%s",
            GB_NROWS (C), GB_NCOLS (C),
            tnrows, tncols, A_transpose ? " (transposed)" : "") ;
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp) ;

    // delete any lingering zombies and assemble any pending tuples
    GB_MATRIX_WAIT (M) ;        // TODO: postpone until accum/mask phase
    GB_MATRIX_WAIT (A) ;        // TODO: allow A and C to be jumbled
    GB_MATRIX_WAIT (scalar) ;

    GB_BURBLE_DENSE (C, "(C %s) ") ;
    GB_BURBLE_DENSE (M, "(M %s) ") ;
    GB_BURBLE_DENSE (A, "(A %s) ") ;

    if (op2 != NULL && GB_NNZ (scalar) != 1)
    {   GB_cov[2585]++ ;
// NOT COVERED (2585):
GB_GOTCHA ;
        // the scalar entry must be present
        GB_ERROR (GrB_INVALID_VALUE, "%s", "Scalar must contain an entry") ;
    }

    //--------------------------------------------------------------------------
    // rename first, second, any, and pair operators
    //--------------------------------------------------------------------------

    if (op2 != NULL)
    {   GB_cov[2586]++ ;
// covered (2586): 879794
        // first(A,x), second(y,A), and any(...) become identity(A)
        if ((opcode == GB_ANY_opcode) ||
            (opcode == GB_FIRST_opcode  && !binop_bind1st) ||
            (opcode == GB_SECOND_opcode &&  binop_bind1st))
        {   GB_cov[2587]++ ;
// covered (2587): 51462
            switch (op2->xtype->code)
            {
                default              :
                case GB_BOOL_code     : GB_cov[2588]++ ;  op1 = GrB_IDENTITY_BOOL   ; break ;
// covered (2588): 4004
                case GB_INT8_code     : GB_cov[2589]++ ;  op1 = GrB_IDENTITY_INT8   ; break ;
// covered (2589): 3924
                case GB_INT16_code    : GB_cov[2590]++ ;  op1 = GrB_IDENTITY_INT16  ; break ;
// covered (2590): 3978
                case GB_INT32_code    : GB_cov[2591]++ ;  op1 = GrB_IDENTITY_INT32  ; break ;
// covered (2591): 3926
                case GB_INT64_code    : GB_cov[2592]++ ;  op1 = GrB_IDENTITY_INT64  ; break ;
// covered (2592): 3914
                case GB_UINT8_code    : GB_cov[2593]++ ;  op1 = GrB_IDENTITY_UINT8  ; break ;
// covered (2593): 3926
                case GB_UINT16_code   : GB_cov[2594]++ ;  op1 = GrB_IDENTITY_UINT16 ; break ;
// covered (2594): 4060
                case GB_UINT32_code   : GB_cov[2595]++ ;  op1 = GrB_IDENTITY_UINT32 ; break ;
// covered (2595): 4052
                case GB_UINT64_code   : GB_cov[2596]++ ;  op1 = GrB_IDENTITY_UINT64 ; break ;
// covered (2596): 3870
                case GB_FP32_code     : GB_cov[2597]++ ;  op1 = GrB_IDENTITY_FP32   ; break ;
// covered (2597): 3898
                case GB_FP64_code     : GB_cov[2598]++ ;  op1 = GrB_IDENTITY_FP64   ; break ;
// covered (2598): 3946
                case GB_FC32_code     : GB_cov[2599]++ ;  op1 = GxB_IDENTITY_FC32   ; break ;
// covered (2599): 3920
                case GB_FC64_code     : GB_cov[2600]++ ;  op1 = GxB_IDENTITY_FC64   ; break ;
// covered (2600): 4044
            }
            op2 = NULL ;
        }
        else if (opcode == GB_PAIR_opcode)
        {   GB_cov[2601]++ ;
// covered (2601): 26230
            // pair (...) becomes one(A)
            switch (op2->xtype->code)
            {
                default              :
                case GB_BOOL_code     : GB_cov[2602]++ ;  op1 = GxB_ONE_BOOL   ; break ;
// covered (2602): 2014
                case GB_INT8_code     : GB_cov[2603]++ ;  op1 = GxB_ONE_INT8   ; break ;
// covered (2603): 1938
                case GB_INT16_code    : GB_cov[2604]++ ;  op1 = GxB_ONE_INT16  ; break ;
// covered (2604): 2054
                case GB_INT32_code    : GB_cov[2605]++ ;  op1 = GxB_ONE_INT32  ; break ;
// covered (2605): 1960
                case GB_INT64_code    : GB_cov[2606]++ ;  op1 = GxB_ONE_INT64  ; break ;
// covered (2606): 1994
                case GB_UINT8_code    : GB_cov[2607]++ ;  op1 = GxB_ONE_UINT8  ; break ;
// covered (2607): 2062
                case GB_UINT16_code   : GB_cov[2608]++ ;  op1 = GxB_ONE_UINT16 ; break ;
// covered (2608): 2028
                case GB_UINT32_code   : GB_cov[2609]++ ;  op1 = GxB_ONE_UINT32 ; break ;
// covered (2609): 2020
                case GB_UINT64_code   : GB_cov[2610]++ ;  op1 = GxB_ONE_UINT64 ; break ;
// covered (2610): 2090
                case GB_FP32_code     : GB_cov[2611]++ ;  op1 = GxB_ONE_FP32   ; break ;
// covered (2611): 1980
                case GB_FP64_code     : GB_cov[2612]++ ;  op1 = GxB_ONE_FP64   ; break ;
// covered (2612): 2078
                case GB_FC32_code     : GB_cov[2613]++ ;  op1 = GxB_ONE_FC32   ; break ;
// covered (2613): 2006
                case GB_FC64_code     : GB_cov[2614]++ ;  op1 = GxB_ONE_FC64   ; break ;
// covered (2614): 2006
            }
            op2 = NULL ;
        }

#if 0
        else
        {
            switch (opcode)
            {
                // commutative operators, no need for bind1st workers:
                case PLUS_opcode      : 
                case TIMES_opcode     : 
                case PAIR_opcode      : 
                case ANY_opcode       : 
                case ISEQ_opcode      : 
                case ISNE_opcode      : 
                case EQ_opcode        : 
                case NE_opcode        : 
                case MIN_opcode       : 
                case MAX_opcode       : 
                case LOR_opcode       : 
                case LAND_opcode      : 
                case LXOR_opcode      : 
                case LXNOR_opcode     : 
                case HYPOT_opcode     : 
                case BOR_opcode       : 
                case BAND_opcode      : 
                case BXOR_opcode      : 
                case BXNOR_opcode     : binop_bind1st = false ;
                default : ;
            }
        }
#endif

    }

    //--------------------------------------------------------------------------
    // T = op(A) or op(A')
    //--------------------------------------------------------------------------

    bool T_is_csc = C->is_csc ;
    if (T_is_csc != A->is_csc)
    {   GB_cov[2615]++ ;
// covered (2615): 213596
        // Flip the sense of A_transpose
        A_transpose = !A_transpose ;
    }

    if (!T_is_csc)
    {
        // positional ops must be flipped, with i and j swapped
        if (op1 != NULL)
        {   GB_cov[2616]++ ;
// covered (2616): 211464
            op1 = GB_positional_unop_ijflip (op1) ;
            opcode = op1->opcode ;
        }
        else if (op2 != NULL)
        {   GB_cov[2617]++ ;
// NOT COVERED (2617):
            op2 = GB_positional_binop_ijflip (op2) ;
            opcode = op2->opcode ;
        }
    }

    GrB_Matrix T = NULL ;

    if (A_transpose)
    {   GB_cov[2618]++ ;
// covered (2618): 700146
        // T = op (A'), typecasting to op*->ztype
        // transpose: typecast, apply an op, not in-place.
        GBURBLE ("(transpose-op) ") ;
        info = GB_transpose (&T, T_type, T_is_csc, A,
            op1, op2, scalar, binop_bind1st, Context) ;
        // A positional op is applied to C after the transpose is computed,
        // using the T_is_csc format.  The ijflip is handled
        // above.
    }
    else if (M == NULL && accum == NULL && (C == A) && C->type == T_type)
    {
        GBURBLE ("(inplace-op) ") ;
        // C = op (C), operating on the values in-place, with no typecasting
        // of the output of the operator with the matrix C.
        // No work to do if the op is identity.
        // FUTURE::: also handle C += op(C), with accum.
        if (opcode != GB_IDENTITY_opcode)
        {   GB_cov[2619]++ ;
// covered (2619): 24187
            // the output Cx is aliased with C->x in GB_apply_op.
            GB_void *Cx = (GB_void *) C->x ;
            info = GB_apply_op (Cx, op1, op2, scalar, binop_bind1st, C,
                Context) ;
        }
        return (info) ;
    }
    else
    {   GB_cov[2620]++ ;
// covered (2620): 925870
        // T = op (A), pattern is a shallow copy of A, type is op*->ztype.
        GBURBLE ("(shallow-op) ") ;
        info = GB_shallow_op (&T, T_is_csc,
            op1, op2, scalar, binop_bind1st, A, Context) ;
    }

    if (info != GrB_SUCCESS)
    {   GB_cov[2621]++ ;
// covered (2621): 515318
        GB_Matrix_free (&T) ;
        return (info) ;
    }

    ASSERT (T->is_csc == C->is_csc) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the M
    //--------------------------------------------------------------------------

    return (GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp,
        Mask_struct, Context)) ;
}

