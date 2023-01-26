//------------------------------------------------------------------------------
// GB_macrofy_type: construct macros for a type name and size
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_type
(
    FILE *fp,
    const char *what,
    const char *name,
    size_t size
)
{

    if (size == 0)
    {
        fprintf (fp, "#define GB_%s_TYPENAME GB_void  /* not used */\n", what) ;
        fprintf (fp, "#define GB_%s_TYPESIZE 1        /* not used */\n", what) ;
    }
    else
    {
        fprintf (fp, "#define GB_%s_TYPENAME %s\n", what, name) ;
        fprintf (fp, "#define GB_%s_TYPESIZE %d\n", what, (int) size) ;
    }
}

