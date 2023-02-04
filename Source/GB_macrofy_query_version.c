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
        "\n// This kernel was created by SuiteSparse:GraphBLAS v%d.%d.%d;\n"
        "void GB_jit_query_version\n"
        "(\n"
        "    int *version   /* array of size 3 */\n"
        ")\n"
        "{\n"
        "    version [0] = %d ;\n"
        "    version [1] = %d ;\n"
        "    version [2] = %d ;\n"
        "}\n",
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB,
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB) ;
}

