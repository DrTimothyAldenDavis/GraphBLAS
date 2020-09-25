//------------------------------------------------------------------------------
// GB_add: C = A+B, C<M>=A+B, and C<!M>=A+B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// GB_add computes C=A+B, C<M>=A+B, or C<!M>=A+B using the given operator
// element-wise on the matrices A and B.  The result is typecasted as needed.
// The pattern of C is the union of the pattern of A and B, intersection with
// the mask M, if present.

// Let the op be z=f(x,y) where x, y, and z have type xtype, ytype, and ztype.
// If both A(i,j) and B(i,j) are present, then:

//      C(i,j) = (ctype) op ((xtype) A(i,j), (ytype) B(i,j))

// If just A(i,j) is present but not B(i,j), then:

//      C(i,j) = (ctype) A (i,j)

// If just B(i,j) is present but not A(i,j), then:

//      C(i,j) = (ctype) B (i,j)

// ctype is the type of matrix C.  The pattern of C is the union of A and B.

// op may be NULL.  In this case, the intersection of A and B must be empty.
// This is used by GB_Matrix_wait only, for merging the pending tuple matrix T
// into A.  In this case, the result C is always sparse or hypersparse, not
// bitmap or full.  Any duplicate pending tuples have already been summed in T,
// so the intersection of T and A is always empty.

// TODO: some methods should not exploit the mask, but leave it for later.
// See GB_ewise and GB_accum_mask: the only places where this function is
// called with a non-null mask M.  Both of those callers can handle the
// mask being applied later.

#include "GB_add.h"

#define GB_FREE_ALL ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_add             // C=A+B, C<M>=A+B, or C<!M>=A+B
(
    GrB_Matrix *Chandle,    // output matrix (unallocated on input)
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M_in,  // optional mask for C, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A_in,  // input A matrix
    const GrB_Matrix B_in,  // input B matrix
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

// HACK to test bitmap
GrB_Matrix M = M_in ;
GrB_Matrix A = A_in ;
GrB_Matrix B = B_in ;

    GrB_Info info ;
    GBURBLE ((M == NULL) ? "add " : "masked_add ") ;

    ASSERT (Chandle != NULL) ;
    (*Chandle) = NULL ;
    GrB_Matrix C = NULL ;

    ASSERT (mask_applied != NULL) ;
    (*mask_applied) = false ;

    ASSERT_MATRIX_OK (A, "A for add", GB0) ;
    ASSERT_MATRIX_OK (B, "B for add", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (op, "op for add", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for add", GB0) ;
    ASSERT (A->vdim == B->vdim && A->vlen == B->vlen) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vdim == M->vdim && A->vlen == M->vlen)) ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (M) ;        // cannot be jumbled
    GB_MATRIX_WAIT (A) ;        // cannot be jumbled
    GB_MATRIX_WAIT (B) ;        // cannot be jumbled

// HACK to test bitmap
GrB_Matrix M_bitmap = NULL ;
GrB_Matrix A_bitmap = NULL ;
GrB_Matrix B_bitmap = NULL ;
bool A_was_full = GB_IS_FULL (A) ;
bool A_was_bitmap = GB_IS_BITMAP (A) ;
bool A_was_sparse = GB_IS_SPARSE (A) ;
bool A_was_hyper = GB_IS_HYPERSPARSE (A) ;
bool B_was_full = GB_IS_FULL (B) ;
bool B_was_bitmap = GB_IS_BITMAP (B) ;
bool B_was_sparse = GB_IS_SPARSE (B) ;
bool B_was_hyper = GB_IS_HYPERSPARSE (B) ;
bool M_was_full = GB_IS_FULL (M) ;
bool M_was_bitmap = GB_IS_BITMAP (M) ;
bool M_was_sparse = GB_IS_SPARSE (M) ;
bool M_was_hyper = GB_IS_HYPERSPARSE (M) ;
if (A->vlen <= 100 && A->vdim <= 100 && op != NULL)
{
    int64_t n = A->vlen ;
    bool hack = (n % 5 == 1) || (n % 4 == 1) || (n % 3 == 1 && M != NULL) ;
    if (hack) GBURBLE ("@(") ;
    if (n % 3 == 1 && M != NULL)
    {
        if (hack) GBURBLE ("M") ;
        GB_OK (GB_dup2 (&M_bitmap, M, true, M->type, Context)) ;
        GB_OK (GB_convert_any_to_bitmap (M_bitmap, Context)) ;
        M = M_bitmap ;
        ASSERT_MATRIX_OK (M, "M bitmap hacked for add", GB0) ;
    }
    if (n % 5 == 1)
    {
        if (hack) GBURBLE ("A") ;
        GB_OK (GB_dup2 (&A_bitmap, A, true, A->type, Context)) ;
        GB_OK (GB_convert_any_to_bitmap (A_bitmap, Context)) ;
        A = A_bitmap ;
        ASSERT_MATRIX_OK (A, "A bitmap hacked for add", GB0) ;
    }
    if (n % 4 == 1)
    {
        if (hack) GBURBLE ("B") ;
        GB_OK (GB_dup2 (&B_bitmap, B, true, B->type, Context)) ;
        GB_OK (GB_convert_any_to_bitmap (B_bitmap, Context)) ;
        B = B_bitmap ;
        ASSERT_MATRIX_OK (B, "B bitmap hacked for add", GB0) ;
    }
    if (hack) GBURBLE (")") ;
}

    //--------------------------------------------------------------------------
    // determine the sparsity of C
    //--------------------------------------------------------------------------

    bool apply_mask ;
    int C_sparsity = GB_add_sparsity (&apply_mask, M, Mask_comp, A, B) ;

    //--------------------------------------------------------------------------
    // C=A+B, C<M>=A+B, or C<!M>=A+B for each sparsity structure of C
    //--------------------------------------------------------------------------

    int64_t Cnvec, Cnvec_nonempty ;
    int64_t *Cp = NULL, *Ch = NULL ;
    int64_t *C_to_M = NULL, *C_to_A = NULL, *C_to_B = NULL ;
    bool Ch_is_Mh ;
    int ntasks, max_ntasks, nthreads ;
    GB_task_struct *TaskList = NULL ;

    //--------------------------------------------------------------------------
    // phase0: determine the sparsity structure of C and the vectors in C
    //--------------------------------------------------------------------------

    info = GB_add_phase0 (
        // computed by by phase0:
        &Cnvec, &Ch, &C_to_M, &C_to_A, &C_to_B, &Ch_is_Mh,
        // input/output to phase0:
        &C_sparsity,
        // original input:
        (apply_mask) ? M : NULL, A, B, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    GBURBLE ("add:(%s<%s>=%s+%s)",
        GB_sparsity_char (C_sparsity),
        GB_sparsity_char_matrix (M),
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // phase1: split C into tasks, and count entries in each vector of C
    //--------------------------------------------------------------------------

    if (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE)
    {

        // phase1a: split C into tasks
        info = GB_ewise_slice (
            // computed by phase1a
            &TaskList, &max_ntasks, &ntasks, &nthreads,
            // computed by phase0:
            Cnvec, Ch, C_to_M, C_to_A, C_to_B, Ch_is_Mh,
            // original input:
            (apply_mask) ? M : NULL, A, B, Context) ;
        if (info != GrB_SUCCESS)
        { 
            // out of memory; free everything allocated by GB_add_phase0
            GB_FREE (Ch) ;
            GB_FREE (C_to_M) ;
            GB_FREE (C_to_A) ;
            GB_FREE (C_to_B) ;
            return (info) ;
        }

        // count the number of entries in each vector of C
        info = GB_add_phase1 (
            // computed or used by phase1:
            &Cp, &Cnvec_nonempty, op == NULL,
            // from phase1a:
            TaskList, ntasks, nthreads,
            // from phase0:
            Cnvec, Ch, C_to_M, C_to_A, C_to_B, Ch_is_Mh,
            // original input:
            (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B, Context) ;
        if (info != GrB_SUCCESS)
        { 
            // out of memory; free everything allocated by GB_add_phase0
            GB_FREE (TaskList) ;
            GB_FREE (Ch) ;
            GB_FREE (C_to_M) ;
            GB_FREE (C_to_A) ;
            GB_FREE (C_to_B) ;
            return (info) ;
        }
    }

    //--------------------------------------------------------------------------
    // phase2: compute the entries (indices and values) in each vector of C
    //--------------------------------------------------------------------------

    // Cp and Ch are either freed by phase2, or transplanted into C.
    // Either way, they are not freed here.

    info = GB_add_phase2 (
        // computed or used by phase2:
        &C, ctype, C_is_csc, op,
        // from phase1 and phase1a: (C sparse/hypersparse only):
        Cp, Cnvec_nonempty, TaskList, ntasks, nthreads,
        // from phase0:
        Cnvec, Ch, C_to_M, C_to_A, C_to_B, Ch_is_Mh, C_sparsity,
        // original input:
        (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B, Context) ;

    // free workspace
    GB_FREE (TaskList) ;
    GB_FREE (C_to_M) ;
    GB_FREE (C_to_A) ;
    GB_FREE (C_to_B) ;

    // Ch and Cp must not be freed; they are now C->h and C->p.
    // If the method failed, Cp and Ch have already been freed.

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

GB_Matrix_free (&M_bitmap) ;
GB_Matrix_free (&A_bitmap) ;
GB_Matrix_free (&B_bitmap) ;

    if (info != GrB_SUCCESS)
    { 
        // out of memory
        return (info) ;
    }

    ASSERT_MATRIX_OK (C, "C output for add", GB0) ;

// HACK
if (GB_IS_BITMAP (C))
{
    GB_OK (GB_convert_any_to_sparse (C, Context)) ;
}

    (*Chandle) = C ;
    return (GrB_SUCCESS) ;
}

