//------------------------------------------------------------------------------
// GB_debugify_mxm: dump definitions for mxm to /tmp/grb/GB_mxm_*.h file
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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
    bool builtin = (semiring->hash == 0) ;

    // namify the mxm problem
    char mxm_name [256 + 4*GxB_MAX_NAME_LEN] ;
    if (builtin)
    {
        // no need for the semiring->name or its types to appear in the name
        sprintf (mxm_name, "GB_jit_mxm_%0*" PRIx64, 16, scode) ;
//      // hack
//      sprintf (mxm_name, "GB_jit_mxm_%0*" PRIx64  "_%s_%s_%s", 16, scode,
//          semiring->add->op->name,
//          semiring->multiply->name,
//          semiring->multiply->ztype->name) ;
    }
    else
    {
        // either the monoid or the multiply op are user-defined
        sprintf (mxm_name, "GB_jit_mxm_%0*" PRIx64 "__%s", 16, scode,
            semiring->name) ;
    }

    // construct the filename and create the file
    char filename [512 + 2*GxB_MAX_NAME_LEN] ;
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

