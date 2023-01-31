//------------------------------------------------------------------------------
// GB_debugify_mxm: dump definitions for mxm to /tmp/grb/GB_mxm_*.h file
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_debugify_mxm
(
    // C matrix:
    bool C_iso,             // if true, operator is ignored
    int C_sparsity,         // sparse, hyper, bitmap, or full
    GrB_Type ctype,         // C=((ctype) T) is the final typecast
    // M matrix:
    GrB_Matrix M,
    bool Mask_struct,
    bool Mask_comp,
    // semiring:
    GrB_Semiring semiring,
    bool flipxy,
    // A and B matrices:
    GrB_Matrix A,
    GrB_Matrix B
)
{

    uint64_t scode ;

    GrB_Type atype = A->type ;
    GrB_Type btype = B->type ;

    if (C_iso)
    {
        // the kernel does not access any values of C, A, or B
        semiring = GxB_ANY_PAIR_BOOL ;
        flipxy = false ;
    }

    // enumify the mxm problem
    GB_enumify_mxm (&scode, C_iso, C_sparsity, ctype,
        M, Mask_struct, Mask_comp, semiring, flipxy, A, B) ;
    int zcode       = GB_RSHIFT (scode, 32, 4) ;    // if 0: C is iso
    int xcode       = GB_RSHIFT (scode, 28, 4) ;    // if 0: ignored
    int ycode       = GB_RSHIFT (scode, 24, 4) ;    // if 0: ignored
    int ccode       = GB_RSHIFT (scode, 16, 4) ;    // if 0: C is iso
    int acode       = GB_RSHIFT (scode, 12, 4) ;    // if 0: A is pattern
    int bcode       = GB_RSHIFT (scode,  8, 4) ;    // if 0: B is pattern

    // namify the mxm problem
    char mxm_name [256 + 8*GxB_MAX_NAME_LEN] ;
    bool builtin = (semiring->add->builtin) &&
        (semiring->multiply->header_size == 0) &&
        (atype->header_size == 0) &&
        (btype->header_size == 0) &&
        (ctype->header_size == 0) ;
    GB_namify_problem (mxm_name, "GB_jit_mxm_", 16, scode, builtin,
        semiring->add->op->name,
        semiring->multiply->name,
        (xcode == 0) ? "void" : semiring->multiply->xtype->name,
        (ycode == 0) ? "void" : semiring->multiply->ytype->name,
        (zcode == 0) ? "void" : semiring->multiply->ztype->name,
        (acode == 0) ? "void" : atype->name,
        (bcode == 0) ? "void" : btype->name,
        (ccode == 0) ? "void" : ctype->name) ;

    // construct the filename and create the file
    char filename [512 + 8*GxB_MAX_NAME_LEN] ;
    sprintf (filename, "/tmp/grb/%s.h", mxm_name) ;
    FILE *fp = fopen (filename, "w") ;
    if (fp == NULL) return ;

    // FIXME: pass this to GB_macrofy_mxm so it can create this:
    fprintf (fp,
        "//--------------------------------------"
        "----------------------------------------\n") ;
    fprintf (fp, "// %s.h\n", mxm_name) ;

    // macrofy the mxm problem
    GB_macrofy_mxm (fp, scode, semiring, ctype, atype, btype) ;
    fclose (fp) ;
}

