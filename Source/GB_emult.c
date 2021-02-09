//------------------------------------------------------------------------------
// GB_emult: C = A.*B, C<M>=A.*B, or C<!M>=A.*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_emult, does C=A.*B, C<M>=A.*B, C<!M>=A.*B, using the given operator
// element-wise on the matrices A and B.  The result is typecasted as needed.
// The pattern of C is the intersection of the pattern of A and B, intersection
// with the mask M or !M, if present.

// Let the op be z=f(x,y) where x, y, and z have type xtype, ytype, and ztype.
// If both A(i,j) and B(i,j) are present, then:

//      C(i,j) = (ctype) op ((xtype) A(i,j), (ytype) B(i,j))

// If just A(i,j) is present but not B(i,j), or
// if just B(i,j) is present but not A(i,j), then C(i,j) is not present.

// ctype is the type of matrix C, and currently it is always op->ztype.

// The pattern of C is the intersection of A and B, and also intersection with
// M if present and not complemented.

// TODO: if C is bitmap on input and C_sparsity is GxB_BITMAP, then C=A.*B,
// C<M>=A.*B and C<M>+=A.*B can all be done in-place.  Also, if C is bitmap
// but T<M>=A.*B is sparse (M sparse, with A and B bitmap), then it too can
// be done in place.

#include "GB_emult.h"
#include "GB_add.h"

#define GB_FREE_WORK            \
{                               \
    GB_FREE (TaskList) ;        \
    GB_FREE (C_to_M) ;          \
    GB_FREE (C_to_A) ;          \
    GB_FREE (C_to_B) ;          \
    GB_FREE (M_ek_slicing) ;    \
    GB_FREE (A_ek_slicing) ;    \
    GB_FREE (B_ek_slicing) ;    \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORK ;              \
    GB_Matrix_free (Chandle) ;  \
}

GrB_Info GB_emult           // C=A.*B, C<M>=A.*B, or C<!M>=A.*B
(
    GrB_Matrix *Chandle,    // output matrix (unallocated on input)
    const GrB_Type ctype,   // type of output matrix C
    const bool C_is_csc,    // format of output matrix C
    const GrB_Matrix M,     // optional mask, unused if NULL
    const bool Mask_struct, // if true, use the only structure of M
    const bool Mask_comp,   // if true, use !M
    bool *mask_applied,     // if true, the mask was applied
    const GrB_Matrix A,     // input A matrix
    const GrB_Matrix B,     // input B matrix
    const GrB_BinaryOp op,  // op to perform C = op (A,B)
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (Chandle != NULL) ;
    (*Chandle) = NULL ;

    ASSERT_MATRIX_OK (A, "A for emult phased", GB0) ;
    ASSERT_MATRIX_OK (B, "B for emult phased", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for emult phased", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for emult phased", GB0) ;
    ASSERT (A->vdim == B->vdim && A->vlen == B->vlen) ;
    ASSERT (GB_IMPLIES (M != NULL, A->vdim == M->vdim && A->vlen == M->vlen)) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    int64_t *M_ek_slicing = NULL ; int M_ntasks = 0 ; int M_nthreads = 0 ;
    int64_t *A_ek_slicing = NULL ; int A_ntasks = 0 ; int A_nthreads = 0 ;
    int64_t *B_ek_slicing = NULL ; int B_ntasks = 0 ; int B_nthreads = 0 ;
    GB_task_struct *TaskList = NULL ;
    int64_t *GB_RESTRICT C_to_M = NULL ;
    int64_t *GB_RESTRICT C_to_A = NULL ;
    int64_t *GB_RESTRICT C_to_B = NULL ;

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    // some cases can allow M, A, and/or B to be jumbled
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (M) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (B) ;

    //--------------------------------------------------------------------------
    // determine the sparsity of C and the method to use
    //--------------------------------------------------------------------------

    bool apply_mask ;           // if true, mask is applied during emult
    int emult_method ;          // method to use

    int C_sparsity = GB_emult_sparsity (&apply_mask, &emult_method,
        M, Mask_comp, A, B) ;

    //--------------------------------------------------------------------------
    // C<M or !M> = A.*B
    //--------------------------------------------------------------------------

    switch (emult_method)
    {

        case GB_EMULT_METHOD_ADD :  // A and B both full (or as-if-full)

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      full    .           full            full    (method: GB_add)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      full            full    (method: GB_add)
            //      bitmap  bitmap      full            full    (method: GB_add)
            //      bitmap  full        full            full    (method: GB_add)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      bitmap  sparse      full            full    (method: GB_add)
            //      bitmap  bitmap      full            full    (method: GB_add)
            //      bitmap  full        full            full    (method: GB_add)

            // A and B are both full (or as-if-full).  The mask M may be
            // anything.  GB_add computes the same thing in this case, so it is
            // used instead, to reduce the code needed for GB_emult.

            return (GB_add (Chandle, ctype, C_is_csc, M, Mask_struct,
                Mask_comp, mask_applied, A, B, op, Context)) ;

        case GB_EMULT_METHOD_01A :  // A sparse/hyper, B bitmap/full

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      sparse  .           sparse          bitmap  (method: 01a)
            //      sparse  .           sparse          full    (method: 01a)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
// TODO     //      sparse  bitmap      sparse          bitmap  (method: 01a)
// TODO     //      sparse  bitmap      sparse          full    (method: 01a)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
// TODO     //      sparse  full        sparse          bitmap  (method: 01a)
// TODO     //      sparse  full        sparse          full    (method: 01a)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          bitmap  (01a: M later)
            //      sparse  sparse      sparse          full    (01a: M later)
            //      ------------------------------------------
            //      C       <!M> =       A       .*      B
            //      ------------------------------------------
// TODO     //      sparse  bitmap      sparse          bitmap  (method: 01a)
// TODO     //      sparse  bitmap      sparse          full    (method: 01a)
            //      ------------------------------------------
            //      C       <!M> =       A       .*      B
            //      ------------------------------------------
// TODO     //      sparse  full        sparse          bitmap  (method: 01a)
// TODO     //      sparse  full        sparse          full    (method: 01a)

            // A is sparse/hyper and B is bitmap/full.  M is either not
            // present, not applied (!M when sparse/hyper), or
            // TODO:bitmap/full.
            // This method does not handle the case when M is sparse/hyper,
            // unless M is ignored and applied later.

            return (GB_emult_01 (Chandle, ctype, C_is_csc, A, B, op, false,
                    Context)) ;

        case GB_EMULT_METHOD_01B :  // A bitmap/full, B sparse/hyper

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      sparse  .           bitmap          sparse  (method: 01b)
            //      sparse  .           full            sparse  (method: 01b)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
//TODO      //      sparse  bitmap      bitmap          sparse  (method: 01b)
//TODO      //      sparse  bitmap      full            sparse  (method: 01b)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
//TODO      //      sparse  full        bitmap          sparse  (method: 01b)
//TODO      //      sparse  full        full            sparse  (method: 01b)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          sparse  (01b: M later)
            //      sparse  sparse      full            sparse  (01b: M later)
            //      ------------------------------------------
            //      C       <!M> =      A       .*      B
            //      ------------------------------------------
//TODO      //      sparse  bitmap      bitmap          sparse  (method: 01b)
//TODO      //      sparse  bitmap      full            sparse  (method: 01b)
            //      ------------------------------------------
            //      C       <!M> =      A       .*      B
            //      ------------------------------------------
//TODO      //      sparse  full        bitmap          sparse  (method: 01b)
//TODO      //      sparse  full        full            sparse  (method: 01b)

            // A is bitmap/full and B is sparse/hyper, with flipxy true.
            // M is not present, not applied, or TODO: bitmap/full

            return (GB_emult_01 (Chandle, ctype, C_is_csc, B, A, op, true,
                Context)) ;

        case GB_EMULT_METHOD_99 :   break ; // punt

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      sparse  .           sparse          sparse  (method: 99)
            //      ------------------------------------------
            //      C       <M> =       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (method: 99)
            //      sparse  bitmap      sparse          sparse  (method: 99)
            //      sparse  full        sparse          sparse  (method: 99)
            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          sparse  (99: M later)
            //      sparse  bitmap      sparse          sparse  (method: 99)
            //      sparse  full        sparse          sparse  (method: 99)

            // TODO: break this into different methods

        case GB_EMULT_METHOD_18 : 

            //      ------------------------------------------
            //      C       =           A       .*      B
            //      ------------------------------------------
            //      bitmap  .           bitmap          bitmap  (method: 18)
            //      bitmap  .           bitmap          full    (method: 18)
            //      bitmap  .           full            bitmap  (method: 18)

        case GB_EMULT_METHOD_19 : 

            //      ------------------------------------------
            //      C       <!M>=       A       .*      B
            //      ------------------------------------------
            //      bitmap  sparse      bitmap          bitmap  (method: 19)
            //      bitmap  sparse      bitmap          full    (method: 19)
            //      bitmap  sparse      full            bitmap  (method: 19)

        case GB_EMULT_METHOD_20 : 

            //      ------------------------------------------
            //      C      <M> =        A       .*      B
            //      ------------------------------------------
            //      bitmap  bitmap      bitmap          bitmap  (method: 20)
            //      bitmap  bitmap      bitmap          full    (method: 20)
            //      bitmap  bitmap      full            bitmap  (method: 20)
            //      bitmap  full        bitmap          bitmap  (method: 20)
            //      bitmap  full        bitmap          full    (method: 20)
            //      bitmap  full        full            bitmap  (method: 20)
            //      ------------------------------------------
            //      C      <!M> =       A       .*      B
            //      ------------------------------------------
            //      bitmap  bitmap      bitmap          bitmap  (method: 20)
            //      bitmap  bitmap      bitmap          full    (method: 20)
            //      bitmap  bitmap      full            bitmap  (method: 20)
            //      bitmap  full        bitmap          bitmap  (method: 20)
            //      bitmap  full        bitmap          full    (method: 20)
            //      bitmap  full        full            bitmap  (method: 20)

            // For methods 18, 19, and 20, C is constructed as bitmap.
            // Both A and B are bitmap/full.  M is either not present,
            // complemented, or not complemented and bitmap/full.  The
            // case when M is not complemented and sparse/hyper is handled
            // by method 100, which constructs C as sparse/hyper (the same
            // structure as M), not bitmap.

// TODO: if C is bitmap on input and C_sparsity is GxB_BITMAP, then C=A.*B,
// C<M>=A.*B and C<M>+=A.*B can all be done in-place.

//          return (GB_bitmap_emult (Chandle, ctype, C_is_csc,
//              emult_method, M, Mask_struct, Mask_comp, mask_applied, A, B,
//              op, Context)) ;

            // punt
            break ;

        case GB_EMULT_METHOD_100 : 

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          bitmap  (method: 100)
            //      sparse  sparse      bitmap          full    (method: 100)
            //      sparse  sparse      full            bitmap  (method: 100)

            return (GB_emult_100 (Chandle, ctype, C_is_csc, M, Mask_struct,
                mask_applied, A, B, op, Context)) ;

        case GB_EMULT_METHOD_101A : break ; // punt

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      sparse          bitmap  (method: 101a)
            //      sparse  sparse      sparse          full    (method: 101a)

            // TODO: this will use 101 (M,A,B, flipxy=false)

            // The method will compute the 2-way intersection of M and A,
            // using the same parallization as C=A.*B when both A and B are
            // both sparse.  It will then lookup B in O(1) time.

        case GB_EMULT_METHOD_101B : break ; // punt

            //      ------------------------------------------
            //      C       <M>=        A       .*      B
            //      ------------------------------------------
            //      sparse  sparse      bitmap          sparse  (method: 101b)
            //      sparse  sparse      full            sparse  (method: 101b)

            // TODO: this will use 101 (M,B,A, flipxy=true)

        default : ;
    }

    //--------------------------------------------------------------------------
    // punt
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (M) ;
    GB_MATRIX_WAIT (A) ;
    GB_MATRIX_WAIT (B) ;

    GBURBLE ("emult:(%s<%s>=%s.*%s) ",
        GB_sparsity_char (C_sparsity),
        GB_sparsity_char_matrix (M),
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    int64_t Cnvec, Cnvec_nonempty ;
    int64_t *GB_RESTRICT Cp = NULL ;
    const int64_t *GB_RESTRICT Ch = NULL ;  // shallow; must not be freed
    int C_ntasks = 0, TaskList_size = 0, C_nthreads ;

    //--------------------------------------------------------------------------
    // phase0: finalize the sparsity C and find the vectors in C
    //--------------------------------------------------------------------------

    GB_OK (GB_emult_phase0 (
        // computed by phase0:
        &Cnvec, &Ch, &C_to_M, &C_to_A, &C_to_B,
        // input/output to phase0:
        &C_sparsity,
        // original input:
        (apply_mask) ? M : NULL, A, B, Context)) ;

    //--------------------------------------------------------------------------
    // phase1: split C into tasks, and count entries in each vector of C
    //--------------------------------------------------------------------------

    if (C_sparsity == GxB_SPARSE || C_sparsity == GxB_HYPERSPARSE)
    {

        //----------------------------------------------------------------------
        // C is sparse or hypersparse: slice and analyze the C matrix
        //----------------------------------------------------------------------

        // phase1a: split C into tasks
        GB_OK (GB_ewise_slice (
            // computed by phase1a:
            &TaskList, &TaskList_size, &C_ntasks, &C_nthreads,
            // computed by phase0:
            Cnvec, Ch, C_to_M, C_to_A, C_to_B, false,
            // original input:
            (apply_mask) ? M : NULL, A, B, Context)) ;

        // count the number of entries in each vector of C
        GB_OK (GB_emult_phase1 (
            // computed by phase1:
            &Cp, &Cnvec_nonempty,
            // from phase1a:
            TaskList, C_ntasks, C_nthreads,
            // from phase0:
            Cnvec, Ch, C_to_M, C_to_A, C_to_B,
            // original input:
            (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B, Context)) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // C is bitmap or full
        //----------------------------------------------------------------------

        // determine how many threads to use
        GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
        C_nthreads = GB_nthreads (A->vlen * A->vdim, chunk, nthreads_max) ;

        // slice the M matrix for method 19
        if (emult_method == GB_EMULT_METHOD_19)
        {
//          printf ("slice M for method19\n") ;
            GB_SLICE_MATRIX (M, 8) ;
        }
    }

    //--------------------------------------------------------------------------
    // phase2: compute the entries (indices and values) in each vector of C
    //--------------------------------------------------------------------------

    // Cp is either freed by phase2, or transplanted into C.
    // Either way, it is not freed here.

    GB_OK (GB_emult_phase2 (
        // computed or used by phase2:
        Chandle, ctype, C_is_csc, op,
        // from phase1:
        Cp, Cnvec_nonempty,
        // from phase1a:
        TaskList, C_ntasks, C_nthreads,
        // from phase0:
        Cnvec, Ch, C_to_M, C_to_A, C_to_B, C_sparsity,
        // from GB_emult_sparsity:
        emult_method,
        // to slice M, A, and/or B:
        M_ek_slicing, M_ntasks, M_nthreads,
        A_ek_slicing, A_ntasks, A_nthreads,
        B_ek_slicing, B_ntasks, B_nthreads,
        // original input:
        (apply_mask) ? M : NULL, Mask_struct, Mask_comp, A, B, Context)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
    ASSERT_MATRIX_OK (*Chandle, "C output for emult phased", GB0) ;
    (*mask_applied) = apply_mask ;
    return (GrB_SUCCESS) ;
}

