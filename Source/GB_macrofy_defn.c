//------------------------------------------------------------------------------
// GB_macrofy_defn: construct defn for an operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

bool GB_macrofy_defn    // return true if user-defined operator is a macro
(
    FILE *fp,
    int kind,           // 0: built-in function
                        // 1: built-in macro
                        // 2: built-in macro needed for CUDA only
                        // 3: user-defined function or macro
    const char *name,
    const char *defn
)
{
    bool is_macro = false ;

    if (name != NULL && defn != NULL)
    {
        if (kind == 2)
        {
            // built-in macro only need for C++ or CUDA
            fprintf (fp, "#if defined ( __cplusplus ) || "
                "defined ( __NVCC__ )\n") ;
        }

        // construct the guard to prevent duplicate definitions
        fprintf (fp,
            "#ifndef GB_GUARD_%s_DEFINED\n"
            "#define GB_GUARD_%s_DEFINED\n", name, name) ;

        if (kind == 0)
        {
            // built-in operator defined by a function
            fprintf (fp, "GB_STATIC_INLINE\n%s\n", defn) ;
        }
        else if (kind == 3)
        {
            // built-in operator defined by a function or macro
            if (defn [0] == '#')
            {
                // user-defined operator defined as macro
                is_macro = true ;
                fprintf (fp, "%s\n", defn) ;
            }
            else
            {
                // user-defined operator defined as function
                fprintf (fp, "GB_STATIC_INLINE\n%s\n", defn) ;
            }
        }
        else // kind is 1 or 2
        {
            // built-in operator defined by a macro
            fprintf (fp, "#define %s\n", defn) ;
        }

        if (kind == 3)
        {
            // define the user-defined function as a string
            GB_macrofy_string (fp, name, defn) ;
        }

        // end the guard
        fprintf (fp, "#endif\n") ;

        if (kind == 2)
        {
            // end the C++/NVCC guard
            fprintf (fp, "#endif\n") ;
        }

    }

    return (is_macro) ;
}

