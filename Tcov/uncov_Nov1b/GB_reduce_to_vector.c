//------------------------------------------------------------------------------
// GB_reduce_to_vector: reduce a matrix to a vector using a binary op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// C<M> = accum (C,reduce(A)) where C is n-by-1.  Reduces a matrix A or A'
// to a vector.

#include "GB_reduce.h"
#include "GB_binop.h"
#include "GB_mxm.h"

#define GB_FREE_ALL                     \
{                                       \
    GB_Matrix_free (&B) ;               \
    /* cannot use GrB_BinaryOp_free: */ \
    GB_FREE (first_op) ;                \
    GrB_Monoid_free (&monoid2) ;        \
    GrB_Semiring_free (&semiring) ;     \
}

GrB_Info GB_reduce_to_vector        // C<M> = accum (C,reduce(A))
(
    GrB_Matrix C,                   // input/output for results, size n-by-1
    const GrB_Matrix M,             // optional M for C, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C,T)
    const GrB_BinaryOp reduce_op,   // reduce operator for T=reduce(A)
    const GrB_Monoid reduce_monoid, // reduce monoid for T=reduce(A)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Descriptor desc,      // descriptor for C, M, and A
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Matrix B = NULL ;
    GrB_BinaryOp first_op = NULL ;
    GrB_Monoid monoid2 = NULL ;
    GrB_Semiring semiring = NULL ;

    // get the reduce operator and monoid
    GrB_Monoid monoid = reduce_monoid ;
    GrB_BinaryOp reduce = (monoid != NULL) ? monoid->op : reduce_op ;

    // C may be aliased with M and/or A
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (reduce) ;
    GB_RETURN_IF_FAULTY (desc) ;

    if (GB_OP_IS_POSITIONAL (reduce))
    {   GB_cov[3686]++ ;
// NOT COVERED (3686):
GB_GOTCHA ;
        // reduce operator cannot be a positional op
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Positional op z=%s(x,y) not supported as reduce op\n",
                reduce->name) ;
    }

    ASSERT_MATRIX_OK (C, "C input for reduce_BinaryOp", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for reduce_BinaryOp", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for reduce_BinaryOp", GB0) ;
    ASSERT_BINARYOP_OK (reduce, "reduce for reduce_BinaryOp", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for reduce_BinaryOp", GB0) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for reduce_BinaryOp", GB0) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2) ;

    // C and M are n-by-1 GrB_Vector objects, typecasted to GrB_Matrix
    ASSERT (GB_VECTOR_OK (C)) ;
    ASSERT (GB_IMPLIES (M != NULL, GB_VECTOR_OK (M))) ;

    // check domains and dimensions for C<M> = accum (C,T)
    GrB_Type ztype = reduce->ztype ;
    GB_OK (GB_compatible (C->type, C, M, accum, ztype, Context)) ;

    // check types of reduce
    if (reduce->xtype != ztype || reduce->ytype != ztype)
    {   GB_cov[3687]++ ;
// covered (3687): 2
        // all 3 types of z = reduce (x,y) must be the same.  reduce must also
        // be associative but there is no way to check this in general.
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "All domains of reduction operator must be identical;\n"
            "operator is: [%s] = %s ([%s],[%s])", ztype->name,
            reduce->name, reduce->xtype->name, reduce->ytype->name) ;
    }

    // T = reduce (T,A) must be compatible
    if (!GB_Type_compatible (A->type, ztype))
    {   GB_cov[3688]++ ;
// covered (3688): 4
        GB_ERROR (GrB_DOMAIN_MISMATCH,
            "Incompatible type for reduction operator z=%s(x,y):\n"
            "input matrix A of type [%s]\n"
            "cannot be typecast to reduction operator of type [%s]",
            reduce->name, A->type->name, ztype->name) ;
    }

    // check the dimensions
    int64_t n = GB_NROWS (C) ;
    if (A_transpose)
    {
        if (n != GB_NCOLS (A))
        {   GB_cov[3689]++ ;
// covered (3689): 4
            GB_ERROR (GrB_DIMENSION_MISMATCH,
                "w=reduce(A'):  length of w is " GBd ";\n"
                "it must match the number of columns of A, which is " GBd ".",
                n, GB_NCOLS (A)) ;
        }
    }
    else
    {
        if (n != GB_NROWS(A))
        {   GB_cov[3690]++ ;
// covered (3690): 4
            GB_ERROR (GrB_DIMENSION_MISMATCH,
                "w=reduce(A):  length of w is " GBd ";\n"
                "it must match the number of rows of A, which is " GBd ".",
                n, GB_NROWS (A)) ;
        }
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp) ;

    //--------------------------------------------------------------------------
    // allocate B as full vector but with B->x of NULL
    //--------------------------------------------------------------------------

    // B is constructed in O(1) time and space, even though it is m-by-1
    int64_t m = A_transpose ? GB_NROWS (A) : GB_NCOLS (A) ;
    GB_OK (GB_new (&B,  // full, new header
        ztype, m, 1, GB_Ap_null, true, GxB_FULL, GB_NEVER_HYPER, 1, Context)) ;
    B->nzmax = m ;
    B->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (B, "temp vector for reduce-to-vector", GB0) ;

    //--------------------------------------------------------------------------
    // create the REDUCE_FIRST_ZTYPE semiring
    //--------------------------------------------------------------------------

    if (monoid == NULL)
    {   GB_cov[3691]++ ;
// covered (3691): 11025
        // GrB_Matrix_reduce_BinaryOp does not pass in a monoid; construct it
        GB_OK (GB_Monoid_new (&monoid2, reduce_op, NULL, NULL, ztype->code,
            Context)) ;
        monoid = monoid2 ;
    }
    ASSERT_MONOID_OK (monoid, "monoid for reduce-to-vector", GB0) ;

    // create the FIRST_ZTYPE operator; note the function pointer is NULL.
    // first_op must be freed with GB_FREE, not GrB_BinaryOp_free, because the
    // opcode is GB_FIRST_opcode.  GrB_BinaryOp_free uses the opcode to decide
    // if the operator is user-defined or built-in, and it only frees operators
    // with an opcode of GB_USER_opcode.
    GB_OK (GB_binop_new (&first_op, NULL, ztype, ztype, ztype, "first_ztype",
        GB_FIRST_opcode)) ;
    ASSERT_BINARYOP_OK (first_op, "op for reduce-to-vector", GB0) ;

    // create the REDUCE_FIRST_ZTYPE semiring
    GB_OK (GB_Semiring_new (&semiring, monoid, first_op)) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for reduce-to-vector", GB0) ;

    //--------------------------------------------------------------------------
    // reduce the matrix to a vector via C<M> = accum (C, A*B)
    //--------------------------------------------------------------------------

    GB_OK (GB_mxm (C, C_replace, M, Mask_comp, Mask_struct, accum,
        semiring, A, A_transpose, B, false, false, GxB_DEFAULT, Context)) ;
    ASSERT_MATRIX_OK (C, "C result for reduce-to-vector", GB0) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

