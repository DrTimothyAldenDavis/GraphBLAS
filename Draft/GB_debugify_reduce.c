//------------------------------------------------------------------------------
// GB_debugify_reduce: create the header file for a reduction problem
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_debugify_reduce     // enumerate and macrofy a GrB_reduce problem
(
    // input:
    GrB_Monoid monoid,      // the monoid to enumify
    GrB_Matrix A
)
{

    uint64_t rcode ;        // unique encoding of the entire problem

    // enumify the reduce problem
    GB_enumify_reduce (&rcode, monoid, A) ;
    bool builtin = (monoid->hash == 0) ;

    // namify the reduce problem
    #define RLEN (256 + GxB_MAX_NAME_LEN)
    char reduce_name [RLEN]  ;
    if (builtin)
    {
        snprintf (reduce_name, RLEN-1, "GB_jit_reduce_%0*" PRIx64,
            7, rcode) ;
    }
    else
    {
        snprintf (reduce_name, RLEN-1, "GB_jit_reduce_%0*" PRIx64
            "__%s", 7, rcode, monoid->op->name) ;
    }

    // construct the filename and create the file
    char filename [512 + GxB_MAX_NAME_LEN] ;
    sprintf (filename, "/tmp/grb/%s.h", reduce_name) ;
    FILE *fp = fopen (filename, "w") ;
    if (fp == NULL) return ;

    // FIXME: pass this to GB_macrofy_reduce so it can create this:
    fprintf (fp,
        "//--------------------------------------"
        "----------------------------------------\n") ;
    fprintf (fp, "// %s.h\n", reduce_name) ;

    GB_macrofy_reduce (fp, rcode, monoid, A->type) ;
    fclose (fp) ;
} 

