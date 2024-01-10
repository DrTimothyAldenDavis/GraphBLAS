//------------------------------------------------------------------------------
// GB_macrofy_decl: construct defn for an operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"
#include "GB_opaque.h"

void GB_macrofy_decl    // construct a decl for an operator
(
    FILE *fp,
    int kind,           // 0: built-in function
                        // 3: user-defined function
    GB_Operator op
)
{
    if (op->name != NULL && op->defn != NULL)
    { 
        // construct the guard to prevent duplicate definitions
        fprintf (fp,
            "#ifndef GB_GUARD_%s_DEFINED\n"
            "#define GB_GUARD_%s_DEFINED\n",
            op->name, op->name) ;
        if (op->opcode == GB_USER_unop_code) // user-defined GrB_UnaryOp
        {
            fprintf (fp,
                "extern void %s (%s *z, %s *x) ;\n",
                op->name, op->ztype->name, op->xtype->name) ;
        }
        else if (op->opcode == GB_USER_idxunop_code) // user-defined GrB_IndexUnaryOp
        {
            fprintf (fp,
                "extern void %s (%s *z, %s *x, int64_t i, int64_t j, %s *thunk) ;\n",
                op->name, op->ztype->name, op->xtype->name, op->ytype->name) ;
        }
        else if (op->opcode == GB_USER_binop_code)
        {
            fprintf (fp,
                "extern void %s (%s *z, %s *x, %s *y) ;\n",
                op->name, op->ztype->name, op->xtype->name, op->ytype->name) ;
        }
        
        if (kind == 3)
        { 
            // define the user-defined function as a string
            GB_macrofy_string (fp, op->name, op->defn) ;
        }
        // end the guard
        fprintf (fp, "#endif\n") ;
    }
}

