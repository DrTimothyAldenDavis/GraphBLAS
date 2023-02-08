//------------------------------------------------------------------------------
// GB_macrofy_query_monoid: construct query_monoid function for a kernel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// FIXME: const is a problem for user-defined-types (memcpy in decl)

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_query_monoid
(
    FILE *fp,
    GrB_Monoid monoid
)
{

    if (monoid->hash != 0)
    {
        // only create the query_monoid method if the monoid is not builtin
        bool has_terminal = (monoid->terminal != NULL) ;
        int zsize = (int) monoid->op->ztype->size ;
        int tsize = (has_terminal) ? zsize : 0 ;
        fprintf (fp,
            "// return true if identity and terminal match expected values\n"
            "bool GB_jit_query_monoid\n"
            "(\n"
            "    void *id,\n"
            "    void *term,\n"
            "    size_t id_size,\n"
            "    size_t term_size\n"
            ")\n"
            "{\n"
            "    if (id_size != %d || term_size != %d) return (false) ;\n"
            "    GB_DECLARE_MONOID_IDENTITY (/*const*/, zidentity) ;\n"
            "    if (memcmp (id, &zidentity, %d) != 0) return (false) ;\n",
            zsize, tsize, zsize) ;
        if (has_terminal)
        {
            fprintf (fp,
            "    GB_DECLARE_MONOID_TERMINAL (/*const*/, zterminal) ;\n"
            "    if (memcmp (term, &zterminal, %d) != 0) return (false) ;\n",
            tsize) ;
        }
        fprintf (fp,
            "    return (true) ;\n"
            "}\n") ;
    }
}

