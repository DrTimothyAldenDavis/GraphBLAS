//------------------------------------------------------------------------------
// GB_macrofy_mxm: construct all macros for a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_mxm        // construct all macros for GrB_mxm
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t scode,
    GrB_Semiring semiring,  // the semiring to macrofy
    GrB_Type ctype,
    GrB_Type atype,
    GrB_Type btype
)
{

    //--------------------------------------------------------------------------
    // extract the semiring scode
    //--------------------------------------------------------------------------

    // monoid (4 hex digits)
//  int unused      = GB_RSHIFT (scode, 63, 1) ;
    int add_ecode   = GB_RSHIFT (scode, 58, 5) ;
    int id_ecode    = GB_RSHIFT (scode, 53, 5) ;
    int term_ecode  = GB_RSHIFT (scode, 48, 5) ;
    bool is_term    = (term_ecode < 30) ;

    // A and B iso-valued and flipxy (one hex digit)
//  int unused      = GB_RSHIFT (scode, 47, 1) ;
    int A_iso_code  = GB_RSHIFT (scode, 46, 1) ;
    int B_iso_code  = GB_RSHIFT (scode, 45, 1) ;
    bool flipxy     = GB_RSHIFT (scode, 44, 1) ;

    // multiplier (5 hex digits)
    int mult_ecode  = GB_RSHIFT (scode, 36, 8) ;
    int zcode       = GB_RSHIFT (scode, 32, 4) ;    // if 0: C is iso
    int xcode       = GB_RSHIFT (scode, 28, 4) ;    // if 0: ignored
    int ycode       = GB_RSHIFT (scode, 24, 4) ;    // if 0: ignored

    // mask (one hex digit)
    int mask_ecode  = GB_RSHIFT (scode, 20, 4) ;

    // types of C, A, and B (3 hex digits)
    int ccode       = GB_RSHIFT (scode, 16, 4) ;   // if 0: C is iso
    int acode       = GB_RSHIFT (scode, 12, 4) ;   // if 0: A is pattern
    int bcode       = GB_RSHIFT (scode,  8, 4) ;   // if 0: B is pattern

    // formats of C, M, A, and B (2 hex digits)
    int csparsity   = GB_RSHIFT (scode,  6, 2) ;
    int msparsity   = GB_RSHIFT (scode,  4, 2) ;
    int asparsity   = GB_RSHIFT (scode,  2, 2) ;
    int bsparsity   = GB_RSHIFT (scode,  0, 2) ;

    //--------------------------------------------------------------------------
    // construct the semiring name
    //--------------------------------------------------------------------------

    GrB_Monoid monoid = semiring->add ;
    GrB_BinaryOp mult = semiring->multiply ;
    GrB_BinaryOp addop = monoid->op ;

    GB_macrofy_copyright (fp) ;

    bool C_iso = (ccode == 0) ;

    if (C_iso)
    {
        // C is iso; no operators are used
        fprintf (fp, "// semiring: symbolic only (C is iso)\n\n") ;
    }
    else
    {
        // general case
        fprintf (fp, "// semiring: (%s, %s%s, %s)\n\n",
            addop->name, mult->name, flipxy ? " (flipped)" : "",
            mult->xtype->name) ;
    }

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GrB_Type xtype = (xcode == 0) ? NULL : mult->xtype ;
    GrB_Type ytype = (ycode == 0) ? NULL : mult->ytype ;
    GrB_Type ztype = (zcode == 0) ? NULL : mult->ztype ;

    if (!C_iso)
    {
        GB_macrofy_typedefs (fp,
            (ccode == 0) ? NULL : ctype,
            (acode == 0) ? NULL : atype,
            (bcode == 0) ? NULL : btype,
            xtype, ytype, ztype) ;
    }

    //--------------------------------------------------------------------------
    // construct the macros for the type names
    //--------------------------------------------------------------------------

    fprintf (fp, "// semiring types:\n") ;
    GB_macrofy_type (fp, "X", (xcode == 0) ? "GB_void" : xtype->name) ;
    GB_macrofy_type (fp, "Y", (ycode == 0) ? "GB_void" : ytype->name) ;
    GB_macrofy_type (fp, "Z", (zcode == 0) ? "GB_void" : ztype->name) ;

    //--------------------------------------------------------------------------
    // construct the monoid macros
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// additive monoid:\n") ;
    const char *u_expr ;
    GB_macrofy_monoid (fp, add_ecode, id_ecode, term_ecode,
        (C_iso) ? NULL : monoid, &u_expr) ;

    //--------------------------------------------------------------------------
    // construct macros for the multiply operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// multiplicative operator%s:\n",
        flipxy ? " (flipped)" : "") ;
    const char *f_expr ;
    GB_macrofy_binop (fp, "GB_MULT", flipxy, false, mult_ecode,
        (C_iso) ? NULL : mult, &f_expr, NULL) ;

    //--------------------------------------------------------------------------
    // multiply-add operator
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// multiply-add operator:\n") ;
    GB_Opcode mult_opcode = mult->opcode ;
    bool is_bool   = (zcode == GB_BOOL_code) ;
    bool is_float  = (zcode == GB_FP32_code) ;
    bool is_double = (zcode == GB_FP64_code) ;
    bool is_first  = (mult_opcode == GB_FIRST_binop_code) ;
    bool is_second = (mult_opcode == GB_SECOND_binop_code) ;
    bool is_pair   = (mult_opcode == GB_PAIR_binop_code) ;
    bool is_positional = GB_IS_BINARYOP_CODE_POSITIONAL (mult_opcode) ;

    if (C_iso)
    {

        //----------------------------------------------------------------------
        // ANY_PAIR_BOOL semiring: nothing to do
        //----------------------------------------------------------------------

        fprintf (fp, "#define GB_MULTADD(z,x,y,i,k,j)\n") ;

    }
    else if (u_expr != NULL && f_expr != NULL &&
        (is_float || is_double || is_bool || is_first || is_second || is_pair
            || is_positional))
    {

        //----------------------------------------------------------------------
        // create a fused multiply-add operator
        //----------------------------------------------------------------------

        // Fusing operators can only be done if it avoids ANSI C integer
        // promotion rules.

        // float and double do not get promoted.
        // bool is OK since promotion of the result (0 or 1) to int is safe.
        // first and second are OK since no promotion occurs.
        // positional operators are OK too.

        if (flipxy)
        {
            fprintf (fp, "#define GB_MULTADD(z,y,x,j,k,i) ") ;
        }
        else
        {
            fprintf (fp, "#define GB_MULTADD(z,x,y,i,k,j) ") ;
        }
        for (const char *p = u_expr ; (*p) != '\0' ; p++)
        {
            // all update operators have a single 'y'
            if ((*p) == 'y')
            {
                // inject the multiply operator; all have the form "z = ..."
                fprintf (fp, "%s", f_expr + 4) ;
            }
            else
            {
                // otherwise, print the update operator character
                fprintf (fp, "%c", (*p)) ;
            }
        }
        fprintf (fp, "\n") ;

    }
    else
    {

        //----------------------------------------------------------------------
        // use a temporary variable for multiply-add
        //----------------------------------------------------------------------

        // All user-defined operators use this method. Built-in operators on
        // integers must use a temporary variable to avoid ANSI C integer
        // promotion.  Complex operators may use macros, so they use
        // temporaries as well.

        fprintf (fp,
            "#define GB_MULTADD(z,x,y,i,k,j)    \\\n"
            "{                                  \\\n"
            "   GB_ZTYPE x_op_y ;               \\\n"
            "   GB_MULT (x_op_y, x,y,i,k,j) ;   \\\n"
            "   GB_UPDATE (z, x_op_y) ;         \\\n"
            "}\n") ;
    }

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// special cases:\n") ;

    // semiring is plus_pair_real
    bool is_plus_pair_real =
        (add_ecode == 11 // plus monoid
        && mult_ecode == 133 // pair multiplicative operator
        && !(zcode == GB_FC32_code || zcode == GB_FC64_code)) ; // real
    fprintf (fp, "#define GB_IS_PLUS_PAIR_REAL_SEMIRING %d\n",
        is_plus_pair_real) ;

    // can ignore overflow in ztype when accumulating the result via the monoid
    bool ztype_ignore_overflow = (zcode == 0 ||
        zcode == GB_INT64_code || zcode == GB_UINT64_code ||
        zcode == GB_FP32_code  || zcode == GB_FP64_code ||
        zcode == GB_FC32_code  || zcode == GB_FC64_code) ;
    fprintf (fp, "#define GB_ZTYPE_IGNORE_OVERFLOW %d\n",
        ztype_ignore_overflow) ;

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    GB_macrofy_output (fp, "c", "C", "C", ctype, ztype, csparsity, C_iso) ;

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
        flipxy ? mult->ytype : mult->xtype,
        atype, asparsity, acode, A_iso_code, -1) ;

    GB_macrofy_input (fp, "b", "B", "B", true,
        flipxy ? mult->xtype : mult->ytype,
        btype, bsparsity, bcode, B_iso_code, -1) ;

    //--------------------------------------------------------------------------
    // include shared definitions
    //--------------------------------------------------------------------------

    fprintf (fp, "\n#include \"GB_AxB_shared_definitions.h\"\n\n") ;
}

