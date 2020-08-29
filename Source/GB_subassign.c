//------------------------------------------------------------------------------
// GB_subassign: C(Rows,Cols)<M> = accum (C(Rows,Cols),A) or A'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// submatrix assignment: C(Rows,Cols)<M> = accum (C(Rows,Cols),A)

// All GxB_*_subassign operations rely on this function.

// With scalar_expansion = false, this method does the work for the standard
// GxB_*_subassign operations (GxB_Matrix_subassign, GxB_Vector_subassign,
// GxB_Row_subassign, and GxB_Col_subassign).  If scalar_expansion is true, it
// performs scalar assignment (the GxB_*_subassign_TYPE functions) in which
// case the input matrix A is ignored (it is NULL), and the scalar is used
// instead.

// Compare with GB_assign, which uses M and C_replace differently

// OK: BITMAP (in progress)

#include "GB_subassign.h"
#include "GB_bitmap_assign.h"

#define GB_FREE_ALL                 \
{                                   \
    GB_Matrix_free (&C2) ;          \
    GB_Matrix_free (&M2) ;          \
    GB_Matrix_free (&A2) ;          \
    GB_FREE (I2) ;                  \
    GB_FREE (J2) ;                  \
}

#define HACK /* TODO */ \
    if (C_in_is_bitmap ) GB_OK (GB_convert_any_to_sparse (C_in, Context)) ; \
    if (C_in_is_full   ) \
    { \
        if (GB_is_dense (C_in) && GB_is_packed (C_in)) \
        { \
            GB_convert_any_to_full (C_in) ; \
        } \
        else \
        { \
            GB_OK (GB_convert_any_to_sparse (C_in, Context)) ; \
        } \
    } \
    if (C_in_is_sparse ) GB_OK (GB_convert_any_to_sparse (C_in, Context)) ; \
    if (C_in_is_hyper  ) GB_OK (GB_convert_any_to_hyper (C_in, Context)) ;

GrB_Info GB_subassign               // C(Rows,Cols)<M> += A or A'
(
    GrB_Matrix C_in,                // input/output matrix for results
    bool C_replace,                 // descriptor for C
    const GrB_Matrix M_in,          // optional mask for C(Rows,Cols)
    const bool Mask_comp,           // true if mask is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const bool M_transpose,         // true if the mask should be transposed
    const GrB_BinaryOp accum,       // optional accum for accum(C,T)
    const GrB_Matrix A_in,          // input matrix
    const bool A_transpose,         // true if A is transposed
    const GrB_Index *Rows,          // row indices
    const GrB_Index nRows_in,       // number of row indices
    const GrB_Index *Cols,          // column indices
    const GrB_Index nCols_in,       // number of column indices
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,             // scalar to be expanded
    const GB_Type_code scalar_code, // type code of scalar to expand
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check and prep inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix C = NULL ;           // C_in or C2
    GrB_Matrix M = NULL ;           // M_in or M2
    GrB_Matrix A = NULL ;           // A_in or A2
    GrB_Index *I = NULL ;           // Rows, Cols, or I2
    GrB_Index *J = NULL ;           // Rows, Cols, or J2

    // HACK: return C to non-bitmap state
    bool C_in_is_bitmap = GB_IS_BITMAP (C_in) ;
    bool C_in_is_full   = GB_IS_FULL (C_in) ;
    bool C_in_is_sparse = GB_IS_SPARSE (C_in) ;
    bool C_in_is_hyper  = GB_IS_HYPER (C_in) ;

    // temporary matrices and arrays
    GrB_Matrix C2 = NULL ;
    GrB_Matrix M2 = NULL ;
    GrB_Matrix A2 = NULL ;
    GrB_Index *I2  = NULL ;
    GrB_Index *J2  = NULL ;

    GrB_Type *atype = NULL ;
    bool done = false ;
    int64_t ni, nj, nI, nJ, Icolon [3], Jcolon [3] ;
    int Ikind, Jkind ;
    int assign_kind = GB_SUBASSIGN ;

    GB_OK (GB_assign_prep (&C, &M, &A, &C2, &M2, &A2,
        &I, &I2, &ni, &nI, &Ikind, Icolon,
        &J, &J2, &nj, &nJ, &Jkind, Jcolon,
        &done, &atype, C_in, &C_replace, &assign_kind,
        M_in, Mask_comp, Mask_struct, M_transpose, accum,
        A_in, A_transpose, Rows, nRows_in, Cols, nCols_in,
        scalar_expansion, scalar, scalar_code, Context)) ;

    // GxB_Row_subassign, GxB_Col_subassign, GxB_Matrix_subassign and
    // GxB_Vector_subassign all use GB_SUBASSIGN.
    ASSERT (assign_kind == GB_SUBASSIGN) ;

    if (done)
    { 
        // GB_assign_prep has handle the entire assignment itself
        HACK ; // TODO
        ASSERT_MATRIX_OK (C, "Final C for subassign", GB0) ;
        return (GB_block (C, Context)) ;
    }

    //--------------------------------------------------------------------------
    // determine method for GB_subassigner
    //--------------------------------------------------------------------------

    int subassign_method = GB_subassigner_method (C, C_replace,
        M, Mask_comp, Mask_struct, accum, A, Ikind, Jkind, scalar_expansion) ;

    //--------------------------------------------------------------------------
    // C(I,J)<M> = A or accum (C(I,J),A) via GB_subassigner
    //--------------------------------------------------------------------------

    GB_OK (GB_subassigner (C, subassign_method, C_replace,
        M, Mask_comp, Mask_struct, accum, A,
        I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
        scalar_expansion, scalar, atype, Context)) ;

    //--------------------------------------------------------------------------
    // transplant C back into C_in
    //--------------------------------------------------------------------------

    if (C == C2)
    {
        // zombies can be transplanted into C_in but pending tuples cannot
        GB_MATRIX_WAIT_IF_PENDING (C) ;
        // transplants the content of C into C_in and frees C
        GB_OK (GB_transplant (C_in, C_in->type, &C, Context)) ;
    }

    //--------------------------------------------------------------------------

    HACK // HACK: return C to non-bitmap state

    //--------------------------------------------------------------------------
    // free workspace, finalize C, and return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C_in, "Final C for subassign", GB0) ;
    GB_FREE_ALL ;
    return (GB_block (C_in, Context)) ;
}

