//------------------------------------------------------------------------------
// GB_macrofy_defn: construct defn for an operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_defn
(
    FILE *fp,
    bool is_macro,
    const char *name,
    const char *defn
)
{

    if (name != NULL && defn != NULL)
    {
        // construct the guard to prevent duplicate definitions
        fprintf (fp, "#ifndef GB_GUARD_%s_DEFINED\n", name) ;
        fprintf (fp, "#define GB_GUARD_%s_DEFINED\n", name) ;

        if (is_macro)
        {
            // built-in operator defined by a macro
            fprintf (fp, "#define %s\n", defn) ;
        }
        else
        {
            // built-in operator defined by a function,
            // or a user-defined operator
            fprintf (fp, "GB_STATIC_INLINE\n%s\n", defn) ;
        }

        // end the guard
        fprintf (fp, "#endif\n") ;
    }
}

