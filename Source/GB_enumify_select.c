//------------------------------------------------------------------------------
// GB_enumify_select: construct the unique code for a select problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

bool GB_enumify_select
(
    // output:
    uint64_t *select_code,      // unique encoding of the selector
    // input:
    bool C_iso,                 // true if C is iso
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL for resize/nonzombie
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    bool in_place_A             // true if select is done in-place
)
{

    //--------------------------------------------------------------------------
    // determine xtype, ytype, and ztype
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype ;
    GB_typify_select (&xtype, &ytype, &ztype, opcode, op, A->type) ;

    //--------------------------------------------------------------------------
    // enumify the types (each 0 to 14, or 4 bits each)
    //--------------------------------------------------------------------------

    int acode = A->type->code ;
    int xcode = (xtype == NULL) ? 0 : xtype->code ;
    int ycode = (ytype == NULL) ? 0 : ytype->code ;
    int zcode = (ztype == NULL) ? 0 : ztype->code ;

    //--------------------------------------------------------------------------
    // enumify the sparsity of A
    //--------------------------------------------------------------------------

    int A_sparsity = GB_sparsity (A) ;
    int asparsity ;
    GB_enumify_sparsity (&asparsity, A_sparsity) ;
    int A_iso_code = (A->iso) ? 1 : 0 ;
    int C_iso_code = (C_iso) ? 1 : 0 ;
    int inplace = (in_place_A) ? 1 : 0 ;
    int flip_ij = (flipij) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // enumify the builtin property
    //--------------------------------------------------------------------------

    // Many built-in operators work on user-defined types, so the x,y,z and
    // acodes must all be checked.

    bool builtin =
        (opcode != GB_USER_idxunop_code) &&
        (opcode != GB_USER_selop_code) &&
        (xcode != GB_UDT_code) &&
        (ycode != GB_UDT_code) &&
        (zcode != GB_UDT_code) &&
        (acode != GB_UDT_code) ;

    //--------------------------------------------------------------------------
    // construct the select code
    //--------------------------------------------------------------------------

    // The entire opcode is encoded, but only a subset are needed (currently
    // 24 unique versions), to simplify the determination of types later on.

    // total select_code bits: 30

    (*select_code) =
                                               // range        bits
                // flipij, inplace (2 bits)
                GB_LSHIFT (flip_ij    , 29) |  // 0 to 1       1
                GB_LSHIFT (inplace    , 28) |  // 0 to 1       1

                // opcode (2 hex digits)
                GB_LSHIFT (opcode     , 20) |  // 0 to 140     8

                // type of x, y, z, and A (4 hex digits)
                GB_LSHIFT (zcode      , 16) |  // 0 to 14      4
                GB_LSHIFT (xcode      , 12) |  // 0 to 14      4
                GB_LSHIFT (ycode      ,  8) |  // 0 to 14      4
                GB_LSHIFT (acode      ,  4) |  // 0 to 14      4

                // A sparstiy, A and C iso properties (1 hex digit)
                GB_LSHIFT (C_iso_code ,  3) |  // 0 or 1       1
                GB_LSHIFT (A_iso_code ,  2) |  // 0 or 1       1
                GB_LSHIFT (asparsity  ,  0) ;  // 0 to 3       2

    return (builtin) ;
}

