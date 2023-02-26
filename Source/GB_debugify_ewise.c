//------------------------------------------------------------------------------
// GB_debugify_ewise: dump definitions for ewise to /tmp/grb/GB_ewise_*.h file
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#define GB_DEBUG

#include "GB.h"
#include "GB_stringify.h"

void GB_debugify_ewise
(
    // method:
    bool is_eWiseMult,      // if true, method is emult
    // C matrix:
    bool C_iso,             // if true, operator is ignored
    bool C_in_iso,          // if true, C is iso on input
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,
    bool Mask_struct,
    bool Mask_comp,
    // operator:
    GrB_BinaryOp binaryop,
    bool flipxy,
    // A and B matrices:
    GrB_Matrix A,
    GrB_Matrix B
)
{

    ASSERT (binaryop != NULL) ;
    uint64_t scode ;

    GrB_Type atype = A->type ;
    GrB_Type btype = B->type ;

    // enumify the ewise problem
    bool builtin = GB_enumify_ewise (&scode, is_eWiseMult, C_iso, C_in_iso,
        C_sparsity, ctype, M, Mask_struct, Mask_comp, binaryop, flipxy, A, B) ;

    // namify the ewise problem
    char ewise_name [256 + 8*GxB_MAX_NAME_LEN] ;
    GB_namify_problem (ewise_name, "GB_jit_ewise_", 12, scode, builtin,
        binaryop->name,
        NULL,
        binaryop->ztype->name,
        binaryop->xtype->name,
        binaryop->ytype->name,
        ctype->name,
        atype->name,
        btype->name) ;

    // construct the filename and create the file
    char filename [512 + 8*GxB_MAX_NAME_LEN] ;
    sprintf (filename, "/tmp/grb/%s.h", ewise_name) ;
    FILE *fp = fopen (filename, "w") ;
    if (fp == NULL) return ;

    // FIXME: pass ewise_name to GB_macrofy_ewise so it can create this:
    fprintf (fp,
        "//--------------------------------------"
        "----------------------------------------\n") ;
    fprintf (fp, "// %s.h\n", ewise_name) ;

    // macrofy the ewise problem
    GB_macrofy_ewise (fp, scode, binaryop, ctype, atype, btype) ;
    fclose (fp) ;
}

