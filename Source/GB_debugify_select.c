//------------------------------------------------------------------------------
// GB_debugify_select:  dump definitions for select to /tmp/grb/GB_select_*.h
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_debugify_select
(
    bool C_iso,                 // true if C is iso
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL in some cases
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    int64_t ithunk,             // (int64_t) Thunk, if Thunk is NULL
    const GrB_Scalar Thunk,     // optional input for select operator
    bool in_place_A             // true if select is done in-place
)
{

    uint64_t select_code ;

    GrB_Type atype = A->type ;

    // enumify the select problem
    GB_enumify_select (&select_code, C_iso, opcode, op, flipij, A,
        in_place_A) ;

    // get the operator name and type names
    GrB_Type xtype, ytype, ztype ;
    GB_typify_select (&xtype, &ytype, &ztype, opcode, op, atype) ;
    char *opname = GB_namify_select (opcode, op) ;
    char *xname = (xtype == NULL) ? NULL : xtype->name ;
    char *yname = (ytype == NULL) ? NULL : ytype->name ;
    char *zname = (ztype == NULL) ? NULL : ztype->name ;

    // namify the select problem
    bool builtin = (op == NULL || (op->header_size == 0)) &&
        ((xtype == NULL) || xtype->header_size == 0) &&
        ((ytype == NULL) || ytype->header_size == 0) &&
        ((ztype == NULL) || ztype->header_size == 0) &&
        (atype->header_size == 0) ;
    char select_name [256 + 8*GxB_MAX_NAME_LEN] ;
    GB_namify_problem (select_name, "GB_jit_select_", 8, select_code, builtin,
        opname,
        NULL,
        xname,
        yname,
        zname,
        atype->name,
        NULL,
        NULL) ;

    // construct the filename and create the file
    char filename [512 + 8*GxB_MAX_NAME_LEN] ;
    sprintf (filename, "/tmp/grb/%s.h", select_name) ;
    FILE *fp = fopen (filename, "w") ;
    if (fp == NULL) return ;

    // FIXME: pass this to GB_macrofy_select so it can create this
    fprintf (fp,
        "//--------------------------------------"
        "----------------------------------------\n") ;
    fprintf (fp, "// %s.h\n", select_name) ;

    // macrofy the select problem
    GB_macrofy_select (fp, select_code, opcode, op, atype) ;
    fclose (fp) ;
}

