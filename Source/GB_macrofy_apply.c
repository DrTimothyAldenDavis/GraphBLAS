//------------------------------------------------------------------------------
// GB_macrofy_apply: construct all macros for apply methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_apply           // construct all macros for GrB_apply
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    // operator:
        const GB_Operator op,       // unary/index-unary to apply; not binaryop
    GrB_Type ctype,
    GrB_Type atype
)
{

    //--------------------------------------------------------------------------
    // extract the apply scode
    //--------------------------------------------------------------------------

    // C kind, i/j dependency and flipij (4 bits)
    int C_mat       = GB_RSHIFT (scode, 35, 1) ;

    int i_dep       = GB_RSHIFT (scode, 34, 1) ;
    int j_dep       = GB_RSHIFT (scode, 33, 1) ;
    bool flipij     = GB_RSHIFT (scode, 32, 1) ;

    // op, z = f(x,i,j,y) (5 hex digits)
    int unop_ecode  = GB_RSHIFT (scode, 24, 8) ;
    int zcode       = GB_RSHIFT (scode, 20, 4) ;
    int xcode       = GB_RSHIFT (scode, 16, 4) ;
    int ycode       = GB_RSHIFT (scode, 12, 4) ;

    // types of C and A (2 hex digits)
    int ccode       = GB_RSHIFT (scode,  8, 4) ;
    int acode       = GB_RSHIFT (scode,  4, 4) ;

    // sparsity structures of C and A (1 hex digit)
    int csparsity   = GB_RSHIFT (scode,  2, 2) ;
    int asparsity   = GB_RSHIFT (scode,  0, 2) ;

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype ;
    const char *xtype_name, *ytype_name, *ztype_name ;

    // GB_macrofy_copyright (fp) ;

    xtype = (xcode == 0) ? NULL : op->xtype ;
    ytype = (ycode == 0) ? NULL : op->ytype ;
    ztype = op->ztype ;
    xtype_name = (xtype == NULL) ? "void" : xtype->name ;
    ytype_name = (ytype == NULL) ? "void" : ytype->name ;
    ztype_name = ztype->name ;
    if (op->hash == 0)
    {
        // builtin operator
        fprintf (fp, "// op: (%s%s, %s)\n\n",
            op->name, flipij ? " (flipped ij)" : "", xtype_name) ;
    }
    else
    {
        // user-defined operator
        fprintf (fp,
            "// op: %s%s, ztype: %s, xtype: %s, ytype: %s\n\n",
            op->name, flipij ? " (flipped ij)" : "",
            ztype_name, xtype_name, ytype_name) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, ctype, (acode == 0) ? NULL : atype, NULL,
        xtype, ytype, ztype) ;

    fprintf (fp, "// unary operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the unary operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// unary operator%s:\n", flipij ? " (flipped ij)" : "") ;
    GB_macrofy_unop (fp, "GB_UNARYOP", flipij, unop_ecode, op) ;

    fprintf (fp, "#define GB_DEPENDS_ON_X %d\n", (xtype != NULL) ? 1 : 0) ;
    fprintf (fp, "#define GB_DEPENDS_ON_Y %d\n", (ytype != NULL) ? 1 : 0) ;
    fprintf (fp, "#define GB_DEPENDS_ON_I %d\n", i_dep) ;
    fprintf (fp, "#define GB_DEPENDS_ON_J %d\n", j_dep) ;

    //--------------------------------------------------------------------------
    // macros for the C array or matrix
    //--------------------------------------------------------------------------

    if (C_mat)
    {
        // C = op(A'), for unary op with transpose
        GB_macrofy_output (fp, "c", "C", "C", ctype, ztype, csparsity, false,
            false) ;
    }
    else
    {
        // Cx = op(A) for unary or index unary op apply, no transpose
        fprintf (fp, "\n// C type:\n") ;
        GB_macrofy_type (fp, "C", "_", ctype->name) ;
    }

    //--------------------------------------------------------------------------
    // construct the macros for A
    //--------------------------------------------------------------------------

    GB_macrofy_input (fp, "a", "A", "A", true, xtype,
        atype, asparsity, acode, 0, -1) ;
}

