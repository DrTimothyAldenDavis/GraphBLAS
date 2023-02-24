//------------------------------------------------------------------------------
// GB_macrofy_ewise: construct all macros for GrB_eWise* (Add, Mult, Union)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_ewise           // construct all macros for GrB_eWise
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_BinaryOp binaryop,      // binaryop to macrofy (NULL for GB_wait)
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
)
{

    //--------------------------------------------------------------------------
    // extract the binaryop scode
    //--------------------------------------------------------------------------

    // C in, A, and B iso-valued and flipxy (one hex digit)
    bool C_in_iso   = GB_RSHIFT (scode, 47, 1) ;
    int A_iso_code  = GB_RSHIFT (scode, 46, 1) ;
    int B_iso_code  = GB_RSHIFT (scode, 45, 1) ;
    bool flipxy     = GB_RSHIFT (scode, 44, 1) ;

    // binary operator (5 hex digits)
    int binop_ecode = GB_RSHIFT (scode, 36, 8) ;
    int zcode       = GB_RSHIFT (scode, 32, 4) ;
    int xcode       = GB_RSHIFT (scode, 28, 4) ;
    int ycode       = GB_RSHIFT (scode, 24, 4) ;

    // mask (one hex digit)
    int mask_ecode  = GB_RSHIFT (scode, 20, 4) ;

    // types of C, A, and B (3 hex digits)
    int ccode       = GB_RSHIFT (scode, 16, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (scode, 12, 4) ;   // if 0: A is pattern
    int bcode       = GB_RSHIFT (scode,  8, 4) ;   // if 0: B is pattern

    bool C_iso = (ccode == 0) ;

    // formats of C, M, A, and B (2 hex digits)
    int csparsity   = GB_RSHIFT (scode,  6, 2) ;
    int msparsity   = GB_RSHIFT (scode,  4, 2) ;
    int asparsity   = GB_RSHIFT (scode,  2, 2) ;
    int bsparsity   = GB_RSHIFT (scode,  0, 2) ;

    //--------------------------------------------------------------------------
    // describe the operator
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype ;
    const char *xtype_name, *ytype_name, *ztype_name ;

    GB_macrofy_copyright (fp) ;

    if (C_iso)
    {
        // values of C are not computed by the kernel
        xtype_name = "GB_void" ;
        ytype_name = "GB_void" ;
        ztype_name = "GB_void" ;
        xtype = NULL ;
        ytype = NULL ;
        ztype = NULL ;
        fprintf (fp, "// op: symbolic only (C is iso)\n\n") ;
    }
    else if (binaryop == NULL)
    {  
        // GB_wait: A and B are disjoint and the operator is not applied
        xtype_name = atype->name ;
        ytype_name = atype->name ;
        ztype_name = atype->name ;
        xtype = atype ;
        ytype = atype ;
        ztype = atype ;
        if (atype->hash == 0)
        {
            // GrB_wait for a built-in type
            fprintf (fp, "// op: none (for GrB_wait), type: %s\n\n",
                xtype_name) ;
        }
        else
        {
            // GrB_wait for a user-defined type
            fprintf (fp, "// op: none (for GrB_wait), user type: %s\n\n",
                xtype_name) ;
        }
    }
    else
    { 
        // general case
        xtype = binaryop->xtype ;
        ytype = binaryop->ytype ;
        ztype = binaryop->ztype ;
        xtype_name = xtype->name ;
        ytype_name = ytype->name ;
        ztype_name = ztype->name ;
        if (binaryop->hash == 0)
        {
            // builtin operator
            fprintf (fp, "// op: (%s%s, %s)\n\n",
                binaryop->name, flipxy ? " (flipped)" : "", xtype_name) ;
        }
        else
        {
            // user-defined operator
            fprintf (fp,
                "// user op: %s%s, ztype: %s, xtype: %s, ytype: %s\n\n",
                binaryop->name, flipxy ? " (flipped)" : "",
                ztype_name, xtype_name, ytype_name) ;
        }
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    if (C_iso)
    {
        // no types to handle for the symbolic case
    }
    else if (binaryop == NULL)
    { 
        // GB_wait: all types must be the same
        GB_macrofy_typedefs (fp, atype, NULL, NULL, NULL, NULL, NULL) ;
    }
    else
    { 
        // general case
        GB_macrofy_typedefs (fp,
            (ccode == 0 ) ? NULL : ctype,
            (acode == 0 || acode == 15) ? NULL : atype,
            (bcode == 0 || bcode == 15) ? NULL : btype,
            xtype, ytype, ztype) ;
    }

    fprintf (fp, "// binary operator types:\n") ;
    GB_macrofy_type (fp, "Z", "_", ztype_name) ;
    GB_macrofy_type (fp, "X", "_", xtype_name) ;
    GB_macrofy_type (fp, "Y", "_", ytype_name) ;

    //--------------------------------------------------------------------------
    // construct macros for the multiply
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// binary operator%s:\n", flipxy ? " (flipped)" : "") ;
    GB_macrofy_binop (fp, "GB_BINOP", flipxy, false, binop_ecode,
        (C_iso) ? NULL : binaryop, NULL, NULL) ;

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    GB_macrofy_output (fp, "c", "C", "C", ctype, ztype, csparsity, C_iso,
        false) ;

    //--------------------------------------------------------------------------
    // construct the macros to access the mask (if any), and its name
    //--------------------------------------------------------------------------

    GB_macrofy_mask (fp, mask_ecode, "M", msparsity) ;

    //--------------------------------------------------------------------------
    // construct the macros for A and B
    //--------------------------------------------------------------------------

    // if flipxy false:  A is typecasted to x, and B is typecasted to y.
    // if flipxy true:   A is typecasted to y, and B is typecasted to x.

    GB_macrofy_input (fp, "a", "A", "A", true,
        (binaryop == NULL) ? ctype :
            (flipxy ? binaryop->ytype : binaryop->xtype),
        atype, asparsity, acode, A_iso_code, -1) ;

    GB_macrofy_input (fp, "b", "B", "B", true,
        (binaryop == NULL) ? ctype :
            (flipxy ? binaryop->xtype : binaryop->ytype),
        btype, bsparsity, bcode, B_iso_code, -1) ;

}

