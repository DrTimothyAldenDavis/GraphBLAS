//------------------------------------------------------------------------------
// GB_enumify_assign: enumerate a GrB_assign problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Enumify an assign operation: C(I,J)<M> += A.  No transpose is handled; this
// is done first in GB_assign_prep.

// The user-callable methods, GrB_assign and GxB_subassign and their variants,
// call GB_assign and GB_subassign, respectively.  Both of those call either
// GB_bitmap_assign or GB_subassigner to do the actual work, or related methods
// that do not need a JIT (GB_*assign_zombie, in particular).

// GB_bitmap_assign and GB_subassigner will not call the JIT directly.
// Instead, they call one of the many assign/subassign kernels, each of which
// will have a JIT variant (39 of them):
//
//      GB_subassign_01
//      GB_subassign_02
//      GB_subassign_03
//      GB_subassign_04
//      GB_subassign_05
//      GB_subassign_05d
//      GB_subassign_06d
//      GB_subassign_06n
//      GB_subassign_06s_and_14
//      GB_subassign_07
//      GB_subassign_08n
//      GB_subassign_08s_and_16
//      GB_subassign_09
//      GB_subassign_10_and_18
//      GB_subassign_11
//      GB_subassign_12_and_20
//      GB_subassign_13
//      GB_subassign_15
//      GB_subassign_17
//      GB_subassign_19
//      GB_subassign_22
//      GB_subassign_23
//      GB_subassign_25
//      GB_bitmap_assign_M_accum
//      GB_bitmap_assign_M_accum_whole
//      GB_bitmap_assign_M_noaccum
//      GB_bitmap_assign_M_noaccum_whole
//      GB_bitmap_assign_fullM_accum
//      GB_bitmap_assign_fullM_accum_whole
//      GB_bitmap_assign_fullM_noaccum
//      GB_bitmap_assign_fullM_noaccum_whole
//      GB_bitmap_assign_noM_accum
//      GB_bitmap_assign_noM_accum_whole
//      GB_bitmap_assign_noM_noaccum
//      GB_bitmap_assign_noM_noaccum_whole
//      GB_bitmap_assign_notM_accum
//      GB_bitmap_assign_notM_accum_whole
//      GB_bitmap_assign_notM_noaccum
//      GB_bitmap_assign_notM_noaccum_whole

#include "GB.h"
#include "GB_stringify.h"

// FIXME: change 4 to 3 for GB_LIST

bool GB_enumify_assign      // enumerate a GrB_assign problem
(
    // output:
    uint64_t *scode,        // unique encoding of the entire operation
    // input:
    // C matrix:
    GrB_Matrix C,
    bool C_replace,
    // index types:
    int Ikind,              // 0: all (no I), 1: range, 2: stride, 4: list
    int Jkind,              // ditto
    // M matrix:
    GrB_Matrix M,           // may be NULL
    bool Mask_struct,       // mask is structural
    bool Mask_comp,         // mask is complemented
    // operator:
    GrB_BinaryOp accum,     // the accum operator to enumify (may be NULL)
    // A matrix
    GrB_Matrix A,           // NULL for scalar assign
    int assign_kind         // 0: assign, 1: subassign, 2: row, 3: col
)
{

    //--------------------------------------------------------------------------
    // get the types of C, A, and M
    //--------------------------------------------------------------------------

    GrB_Type ctype = C->type ;
    GrB_Type atype = (A == NULL) ? NULL : A->type ;
    GrB_Type mtype = (M == NULL) ? NULL : M->type ;

    //--------------------------------------------------------------------------
    // get the types of X, Y, and Z
    //--------------------------------------------------------------------------

    GB_Opcode accum_opcode ;
    GB_Type_code xcode, ycode, zcode ;

    if (accum == NULL)
    {
        // accum is not present
        accum_opcode = GB_NOP_code ;
        xcode = 0 ;
        ycode = 0 ;
        zcode = 0 ;
    }
    else
    {
        accum_opcode = accum->opcode ;
        xcode = accum->xtype->code ;
        ycode = accum->ytype->code ;
        zcode = accum->ztype->code ;
    }

    //--------------------------------------------------------------------------
    // rename redundant boolean operators
    //--------------------------------------------------------------------------

    // consider z = op(x,y) where both x and y are boolean:
    // DIV becomes FIRST
    // RDIV becomes SECOND
    // MIN and TIMES become LAND
    // MAX and PLUS become LOR
    // NE, ISNE, RMINUS, and MINUS become LXOR
    // ISEQ becomes EQ
    // ISGT becomes GT
    // ISLT becomes LT
    // ISGE becomes GE
    // ISLE becomes LE

    if (xcode == GB_BOOL_code)  // && (ycode == GB_BOOL_code)
    {
        // rename the operator
        accum_opcode = GB_boolean_rename (accum_opcode) ;
    }

    //--------------------------------------------------------------------------
    // enumify the accum operator, if present
    //--------------------------------------------------------------------------

    // accum_ecode is 255 if no accum is present

    int accum_ecode ;
    GB_enumify_binop (&accum_ecode, accum_opcode, xcode, false) ;

    //--------------------------------------------------------------------------
    // enumify the types
    //--------------------------------------------------------------------------

    // if (acode == 15): scalar assignment.  The method must first cast the
    // scalar to ctype and accum->ytype (if present), and then call the JIT or
    // generic kernel with those typecasted scalars.  The JIT and generic
    // kernels assume the scalar is already typecasted, if needed, so they do
    // not need to be specialized based on the scalar type.

    int acode = (A == NULL) ? 15 : atype->code ;        // 0 to 15
    int A_iso_code = (A != NULL && A->iso) ? 1 : 0 ;

    // if (ccode == 0): C is iso and the kernel does not access its values
    int ccode = (C->iso) ? 0 : ctype->code ;            // 0 to 14

    //--------------------------------------------------------------------------
    // enumify the mask
    //--------------------------------------------------------------------------

    // mtype_code == 0: no mask present
    int mtype_code = (mtype == NULL) ? 0 : mtype->code ; // 0 to 14
    int mask_ecode ;
    GB_enumify_mask (&mask_ecode, mtype_code, Mask_struct, Mask_comp) ;

    //--------------------------------------------------------------------------
    // enumify the sparsity structures of C, M, A, and B
    //--------------------------------------------------------------------------

    int C_sparsity = GB_sparsity (C) ;
    int M_sparsity = (M == NULL) ? 0 : GB_sparsity (M) ;
    int A_sparsity = (A == NULL) ? 0 : GB_sparsity (A) ;

    int csparsity, msparsity, asparsity ;
    GB_enumify_sparsity (&csparsity, C_sparsity) ;
    GB_enumify_sparsity (&msparsity, M_sparsity) ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ;

    int C_repl = (C_replace) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // construct the ewise scode
    //--------------------------------------------------------------------------

    // total scode bits: 48 (12 hex digits)

    (*scode) =
                                               // range        bits

                // assign_kind, Ikind, and Jkind (2 hex digits)
                GB_LSHIFT (assign_kind, 46) |  // 0 to 3       2
                GB_LSHIFT (Ikind      , 43) |  // 0 to 4       3    FIXME
                GB_LSHIFT (Jkind      , 40) |  // 0 to 4       3    FIXME

                // accum, z = f(x,y) (5 hex digits)
                GB_LSHIFT (accum_ecode, 32) |  // 0 to 255     8
                GB_LSHIFT (zcode      , 28) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 24) |  // 0 to 14      4
                GB_LSHIFT (ycode      , 20) |  // 0 to 14      4

                // mask (one hex digit)
                GB_LSHIFT (mask_ecode , 16) |  // 0 to 13      4

                // types of C, A, and B (2 hex digits)
                GB_LSHIFT (ccode      , 12) |  // 0 to 14      4
                GB_LSHIFT (acode      ,  8) |  // 0 to 15      4

                // sparsity structures of C, M, and A (2 hex digits),
                // iso status of A and C_replace
                GB_LSHIFT (csparsity  ,  6) |  // 0 to 3       2
                GB_LSHIFT (msparsity  ,  4) |  // 0 to 3       2
                GB_LSHIFT (C_repl     ,  3) |  // 0 to 1       1
                GB_LSHIFT (A_iso_code ,  2) |  // 0 or 1       1
                GB_LSHIFT (asparsity  ,  0) ;  // 0 to 3       2

}

