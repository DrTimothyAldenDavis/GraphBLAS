//------------------------------------------------------------------------------
// GB_macrofy_query_version: construct query_version function for a kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_query_version
(
    FILE *fp
)
{

    fprintf (fp,
        "void GB_jit_query_version (int *v) "
        "{ v [0] = %d ; v [1] = %d ; v [2] = %d ; }\n",
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB) ;
}

