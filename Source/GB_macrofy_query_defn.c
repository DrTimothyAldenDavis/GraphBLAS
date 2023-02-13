//------------------------------------------------------------------------------
// GB_macrofy_query_defn: construct query_defn function for a kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_query_defn
(
    FILE *fp,
    GB_Operator op0,    // monoid op, select op, unary op, etc
    GB_Operator op1,    // binaryop for a semring
    GrB_Type type0,
    GrB_Type type1,
    GrB_Type type2
)
{

    //--------------------------------------------------------------------------
    // create a function to query the operator and type definitions
    //--------------------------------------------------------------------------

    fprintf (fp, 
        "\n// to query the kernel for its op and type definitions:\n"
        "const char *GB_jit_query_defn (int k)\n"
        "{\n"
        "    const char **defn [5] ;\n") ;

    // create the definition string for op0
    if (op0 == NULL || op0->defn == NULL)
    {
        // op0 does not appear, or is builtin
        fprintf (fp, "    defn [0] = NULL ;\n") ;
    }
    else
    {
        // op0 is user-defined
        fprintf (fp, "    defn [0] = GB_%s_USER_DEFN ;\n", op0->name) ;
    }

    // create the definition string for op1
    if (op1 == NULL || op1->defn == NULL)
    {
        // op1 does not appear, or is builtin
        fprintf (fp, "    defn [1] = NULL ;\n") ;
    }
    else if (op0 == op1)
    {
        // op1 is user-defined, but the same as op0
        fprintf (fp, "    defn [1] = defn [0] ;\n") ;
    }
    else
    {
        // op1 is user-defined, and differs from op0
        fprintf (fp, "    defn [1] = GB_%s_USER_DEFN ;\n", op1->name) ;
    }

    // create the definition string for the 3 types
    GrB_Type types [3] ;
    types [0] = type0 ;
    types [1] = type1 ;
    types [2] = type2 ;
    for (int k = 0 ; k <= 2 ; k++)
    {
        GrB_Type type = types [k] ;
        if (type == NULL || type->defn == NULL)
        {
            // types [k] does not appear, or is builtin
            fprintf (fp, "    defn [%d] = NULL ;\n", k+2) ;
        }
        else
        {
            // see if the type definition already appears
            bool is_unique = true ;
            for (int j = 0 ; j < k ; j++)
            {
                if (type == types [j])
                {
                    is_unique = false ;
                    fprintf (fp, "    defn [%d] = defn [%d] ;\n", k+2, j+2) ;
                }
            }
            if (is_unique)
            {
                // this type is unique, and user-defined
                fprintf (fp, "    defn [%d] = GB_%s_USER_DEFN ;\n", k+2,
                    type->name) ;
            }
        }
    }

    // return the requested defn
    fprintf (fp,
        "    return (defn [k]) ;\n"
        "}\n\n") ;
}

