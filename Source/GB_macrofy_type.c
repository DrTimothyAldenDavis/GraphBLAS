//------------------------------------------------------------------------------
// GB_macrofy_type: construct macros for a type name
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_type
(
    FILE *fp,
    // input:
    const char *what,       // typically X, Y, Z, A, B, or C
    const char *name        // name of the type
)
{

    if (strcmp (name, "GB_void") == 0)
    {
        fprintf (fp, "#define GB_%s_TYPENAME GB_void  /* not used */\n", what) ;
    }
    else
    {
        fprintf (fp, "#define GB_%s_TYPENAME %s\n", what, name) ;
    }
}
