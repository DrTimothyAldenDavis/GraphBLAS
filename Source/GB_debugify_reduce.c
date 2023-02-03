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
    bool builtin = GB_enumify_reduce (&rcode, monoid, A) ;

    // namify the reduce problem
    char reduce_name [256 + 2 * GxB_MAX_NAME_LEN] ;
    GB_namify_problem (reduce_name, "GB_jit_reduce_", 7, rcode, builtin,
        monoid->op->name,
        NULL,
        monoid->op->ztype->name,
        A->type->name,
        NULL,
        NULL,
        NULL,
        NULL) ;

    // construct the filename and create the file
    char filename [512 + 2 * GxB_MAX_NAME_LEN] ;
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

