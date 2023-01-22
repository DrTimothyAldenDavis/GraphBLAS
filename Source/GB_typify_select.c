//------------------------------------------------------------------------------
// GB_typify_select: determine the x,y,z types of a select operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_typify_select           // determine x,y,z types for select
(
    // outputs:
    GrB_Type *xtype,            // x,y,z types for select operator
    GrB_Type *ytype,
    GrB_Type *ztype,
    // inputs:
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL in some cases
    GrB_Type atype              // the type of the A matrix
)
{

    // types used for most operators
    (*xtype) = NULL ;
    (*ytype) = NULL ; 
    (*ztype) = GrB_BOOL ;

    switch (opcode)
    {

        // positional: depends on (i,j) and y
        case GB_TRIL_selop_code       : // C = tril (A,k)
        case GB_TRIU_selop_code       : // C = triu (A,k)
        case GB_DIAG_selop_code       : // C = diag (A,k)
        case GB_OFFDIAG_selop_code    : // C = offdiag (A,k)
        case GB_ROWINDEX_idxunop_code : // C = rowindex (A,k)
        case GB_ROWLE_idxunop_code    : // C = rowle (A,k)
        case GB_ROWGT_idxunop_code    : // C = rowgt (A,k)
        case GB_COLINDEX_idxunop_code : // C = colindex (A,k)
        case GB_COLLE_idxunop_code    : // C = colle (A,k)
        case GB_COLGT_idxunop_code    : // C = colgt (A,k)

        // depends on zombie status of A(i,j)
        case GB_NONZOMBIE_selop_code  : // C = nonzombies(A)
            (*xtype) = NULL ;
            (*ytype) = NULL ;
            (*ztype) = GrB_BOOL ;
            break ;

        // depends on A, OK for user-defined types
        case GB_NONZERO_selop_code    : // A(i,j) != 0
        case GB_EQ_ZERO_selop_code    : // A(i,j) == 0

        // depends on A
        case GB_GT_ZERO_selop_code    : // A(i,j) > 0
        case GB_GE_ZERO_selop_code    : // A(i,j) >= 0
        case GB_LT_ZERO_selop_code    : // A(i,j) < 0
        case GB_LE_ZERO_selop_code    : // A(i,j) <= 0

        // depends on A, OK for user-defined types
        case GB_NE_THUNK_selop_code   : // A(i,j) != thunk
        case GB_EQ_THUNK_selop_code   : // A(i,j) == thunk
        // depends on A and Thunk, not for user-defined types
        case GB_GT_THUNK_selop_code   : // A(i,j) > thunk
        case GB_GE_THUNK_selop_code   : // A(i,j) >= thunk
        case GB_LT_THUNK_selop_code   : // A(i,j) < thunk
        case GB_LE_THUNK_selop_code   : // A(i,j) <= thunk
            (*xtype) = atype ;         // no typecasting of A
            (*ytype) = atype ;         // thunk is typecasted to atype
            (*ztype) = GrB_BOOL ;
            break ;

        // depends on A and Thunk, not for user-defined types
        case GB_VALUEEQ_idxunop_code  : 
        case GB_VALUENE_idxunop_code  : 
        case GB_VALUEGT_idxunop_code  : 
        case GB_VALUEGE_idxunop_code  : 
        case GB_VALUELT_idxunop_code  : 
        case GB_VALUELE_idxunop_code  : 
            (*xtype) = op->xtype ;     // A is typecasted to op->xtype
            (*ytype) = op->ytype ;     // thunk is typecasted to op->ytype 
            (*ztype) = GrB_BOOL ;
            break ;

        // depends on A and Thunk (type and values); user-defined operators
        case GB_USER_idxunop_code     : 
        case GB_USER_selop_code       : 
            (*xtype) = op->xtype ;
            (*ytype) = op->ytype ;
            (*ztype) = op->ztype ;
            break ;

        default: ;
    }
}

