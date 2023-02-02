//------------------------------------------------------------------------------
// GB_macrofy_query_monoid: construct query_monoid function for a kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_query_monoid
(
    FILE *fp,
    const char *kernel_name,
    GrB_Monoid monoid
)
{

    if (!monoid->builtin)
    {
        bool has_terminal = (monoid->terminal != NULL) ;
        int zsize = (int) monoid->op->ztype->size ;
        int tsize = (has_terminal) ? zsize : 0 ;
        fprintf (fp,
            "// return true if identity and terminal match expected values\n"
            "bool %s__query_monoid\n"
            "(\n"
            "    void *id,          // input: expected identity value\n"
            "    void *term,        // input: expected terminal value, if any\n"
            "    size_t id_size,    // input: expected identity size\n"
            "    size_t term_size   // input: expected terminal size\n"
            ") ;\n\n"
            "bool %s__query_monoid\n"
            "(\n"
            "    void *id,\n"
            "    void *term,\n"
            "    size_t id_size,\n"
            "    size_t term_size\n"
            ")\n"
            "{\n"
            "    if (id_size != %d || term_size != %d) return (false) ;\n"
            "    GB_DECLARE_MONOID_IDENTITY (identity) ;\n"
            "    if (memcmp (id, &identity, %d) != 0) return (false) ;\n",
            kernel_name, kernel_name, zsize, tsize, zsize) ;
        if (has_terminal)
        {
            fprintf (fp,
            "    GB_DECLARE_MONOID_TERMINAL (terminal) ;\n"
            "    if (memcmp (term, &terminal, %d) != 0) return (false) ;\n",
            tsize) ;
        }
        fprintf (fp,
            "    return (true) ;\n"
            "}\n") ;
    }
}

