//------------------------------------------------------------------------------
// GB_stringify_load: return a string to load/save a value
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// TODO: delete me

#include "GB.h"
#include "GB_stringify.h"

void GB_stringify_load         // return a string to load/typecast macro
(
    // input:
    FILE *fp,                       // File to write macros, assumed open already
    const char *load_macro_name,    // name of macro to construct
    bool is_pattern                 // if true, load/cast does nothing
)
{

    if (is_pattern)
    {
        fprintf ( fp, "#define %s(blob)\n", load_macro_name) ;
    }
    else
    {
        fprintf ( fp, "#define %s(blob) blob\n", load_macro_name) ;
    }
}

