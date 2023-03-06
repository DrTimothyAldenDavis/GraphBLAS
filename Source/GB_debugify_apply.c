//------------------------------------------------------------------------------
// GB_debugify_apply: dump definitions for apply to /tmp/grb/GB_ewise_*.h file
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_debugify_apply
(
    // C matrix:
    int C_sparsity,         // sparse, hyper, bitmap, or full
    bool C_is_matrix,       // true for C=op(A), false for Cx=op(A)
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
        bool flipij,                // if true, flip i,j for user idxunop
    // A matrix:
    GrB_Matrix A
)
{

    ASSERT (op != NULL) ;
    uint64_t scode ;

    GrB_Type atype = A->type ;

    // enumify the apply problem
    GB_enumify_apply (&scode, C_sparsity, C_is_matrix, ctype, op, flipij, A) ;
    bool builtin = (op->hash == 0) ;

    // namify the apply problem
    char apply_name [256 + 8*GxB_MAX_NAME_LEN] ;
    GB_namify_problem (apply_name, "GB_jit_apply_", 9, scode, builtin,
        op->name,
        NULL,
        op->ztype->name,
        (op->xtype == NULL) ? NULL : op->xtype->name,
        (op->ytype == NULL) ? NULL : op->ytype->name,
        ctype->name,
        atype->name,
        NULL) ;

    // construct the filename and create the file
    char filename [512 + 8*GxB_MAX_NAME_LEN] ;
    sprintf (filename, "/tmp/grb/%s.h", apply_name) ;
    FILE *fp = fopen (filename, "w") ;
    if (fp == NULL) return ;

    // FIXME: pass apply_name to GB_macrofy_apply so it can create this:
    fprintf (fp,
        "//--------------------------------------"
        "----------------------------------------\n") ;
    fprintf (fp, "// %s.h\n", apply_name) ;

    // macrofy the apply problem
    GB_macrofy_apply (fp, scode, op, ctype, atype) ;
    fclose (fp) ;
}

