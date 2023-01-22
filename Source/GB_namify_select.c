//------------------------------------------------------------------------------
// GB_namify_select: determine the name of the select operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// In some cases the select op can be NULL, and in other cases the opcode can
// be renamed by GB_select.  The opcode captures the operator to perform in all
// cases, but it's name is not given by op->name.

#include "GB.h"
#include "GB_stringify.h"

char *GB_namify_select          // determine the select op name
(
    // inputs:
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op        // user operator, NULL in some cases
)
{

    switch (opcode)
    {

        // positional: depends on (i,j) and y
        case GB_TRIL_selop_code       : return ("tril") ;
        case GB_TRIU_selop_code       : return ("triu") ;
        case GB_DIAG_selop_code       : return ("diag") ;
        case GB_OFFDIAG_selop_code    : return ("offdiag") ;
        case GB_ROWINDEX_idxunop_code : return ("rowindex") ;
        case GB_ROWLE_idxunop_code    : return ("rowle") ;
        case GB_ROWGT_idxunop_code    : return ("rowgt") ;
        case GB_COLINDEX_idxunop_code : return ("colindex") ;
        case GB_COLLE_idxunop_code    : return ("colle") ;
        case GB_COLGT_idxunop_code    : return ("colgt") ;

        // depends on zombie status of A(i,j)
        case GB_NONZOMBIE_selop_code  : return ("nonzombie") ;

        // depends on A, OK for user-defined types
        case GB_NONZERO_selop_code    : return ("nonzero") ;
        case GB_EQ_ZERO_selop_code    : return ("eqzero") ;

        // depends on A
        case GB_GT_ZERO_selop_code    : return ("gtzero") ;
        case GB_GE_ZERO_selop_code    : return ("gezero") ;
        case GB_LT_ZERO_selop_code    : return ("ltzero") ;
        case GB_LE_ZERO_selop_code    : return ("lezero") ;

        // depends on A, OK for user-defined types
        case GB_NE_THUNK_selop_code   : return ("ne") ;
        case GB_EQ_THUNK_selop_code   : return ("eq") ;
        // depends on A and Thunk, not for user-defined types
        case GB_GT_THUNK_selop_code   : return ("gt") ;
        case GB_GE_THUNK_selop_code   : return ("ge") ;
        case GB_LT_THUNK_selop_code   : return ("lt") ;
        case GB_LE_THUNK_selop_code   : return ("le") ;

        // depends on A and Thunk, not for user-defined types
        case GB_VALUEEQ_idxunop_code  : return ("ne") ;
        case GB_VALUENE_idxunop_code  : return ("eq") ;
        case GB_VALUEGT_idxunop_code  : return ("gt") ;
        case GB_VALUEGE_idxunop_code  : return ("ge") ;
        case GB_VALUELT_idxunop_code  : return ("lt") ;
        case GB_VALUELE_idxunop_code  : return ("le") ;

        // depends on A and Thunk (type and values); user-defined operators
        case GB_USER_idxunop_code     : 
            return ((op->name == NULL) ? "useridx" : op->name) ;
        case GB_USER_selop_code       : 
            return ((op->name == NULL) ? "usersel" : op->name) ;

        default: return (NULL) ;
    }
}

