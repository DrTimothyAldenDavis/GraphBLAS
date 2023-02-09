//------------------------------------------------------------------------------
// GB_macrofy_mxm: construct all macros for a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_select     // construct all macros for GrB_select and GxB_select
(
    // output:
    FILE *fp,                   // target file to write, already open
    // input:
    uint64_t select_code,
    GB_Opcode opcode,           // selector opcode
    const GB_Operator op,       // user operator, NULL in some cases
    GrB_Type atype
)
{

    //--------------------------------------------------------------------------
    // extract the select codes
    //--------------------------------------------------------------------------

    // FIXME: add csparsity and ccode

    // flipij, inplace (2 bits)
    int flip_ij     = GB_RSHIFT (select_code, 29, 1) ;
    int inplace     = GB_RSHIFT (select_code, 28, 1) ;

    // opcde (8 bits; 2 hex digits)
    ASSERT (opcode == GB_RSHIFT (select_code, 20, 8)) ;

    // type of x, y, z, and A (4 hex digits)
    int zcode       = GB_RSHIFT (select_code, 16, 4) ;
    int xcode       = GB_RSHIFT (select_code, 12, 4) ;
    int ycode       = GB_RSHIFT (select_code,  8, 4) ;
    int acode       = GB_RSHIFT (select_code,  4, 4) ;

    // A sparstiy, A and C iso properties (1 hex digit)
    int C_iso_code  = GB_RSHIFT (select_code,  3, 1) ;
    int A_iso_code  = GB_RSHIFT (select_code,  2, 1) ;
    int asparsity   = GB_RSHIFT (select_code,  0, 2) ;

    //--------------------------------------------------------------------------
    // construct the select name
    //--------------------------------------------------------------------------

    GrB_Type xtype, ytype, ztype ;
    GB_typify_select (&xtype, &ytype, &ztype, opcode, op, atype) ;
    char *opname = GB_namify_select (opcode, op) ;
    char *xname = (xtype == NULL) ? "GB_void" : xtype->name ;
    char *yname = (ytype == NULL) ? "GB_void" : ytype->name ;
    char *zname = (ztype == NULL) ? "GB_void" : ztype->name ;

    GB_macrofy_copyright (fp) ;
    fprintf (fp, "// select: (%s", opname) ;
    if (xtype != NULL) fprintf (fp, ", xtype: %s", xname) ;
    if (ytype != NULL) fprintf (fp, ", ytype: %s", yname) ;
    if (ztype != NULL) fprintf (fp, ", ztype: %s", zname) ;
    int asize = (int) atype->size ;
    fprintf (fp, ", atype: %s, size: %d bytes)\n\n", atype->name, (int) asize) ;

    // true if asize is a multiply of sizeof (uint32_t)
    bool asize_multiple_of_uint32 = ((asize % sizeof (uint32_t)) == 0) ;
    int asize32 = asize / sizeof (uint32_t) ;

    //--------------------------------------------------------------------------
    // construct the typedefs
    //--------------------------------------------------------------------------

    GB_macrofy_typedefs (fp, NULL, atype, NULL, xtype, ytype, ztype) ;

    //--------------------------------------------------------------------------
    // construct the macros for the type names
    //--------------------------------------------------------------------------

    fprintf (fp, "// select types:\n") ;
    GB_macrofy_type (fp, "X", "_", xname) ;
    GB_macrofy_type (fp, "Y", "_", yname) ;
    GB_macrofy_type (fp, "Z", "_", zname) ;

    //--------------------------------------------------------------------------
    // construct the select macros
    //--------------------------------------------------------------------------

    fprintf (fp, "\n// select op:\n") ;

    char *kind ;
    switch (opcode)
    {
        case GB_TRIL_selop_code       : kind = "TRIL"       ; break ;
        case GB_TRIU_selop_code       : kind = "TRIU"       ; break ;
        case GB_DIAG_selop_code       : kind = "DIAG"       ; break ;
        case GB_OFFDIAG_selop_code    : kind = "OFFDIAG"    ; break ;
        case GB_ROWINDEX_idxunop_code : kind = "ROWINDEX"   ; break ;
        case GB_ROWLE_idxunop_code    : kind = "ROWLE"      ; break ;
        case GB_ROWGT_idxunop_code    : kind = "ROWGT"      ; break ;
        case GB_COLINDEX_idxunop_code : kind = "COLINDEX"   ; break ;
        case GB_COLLE_idxunop_code    : kind = "COLLE"      ; break ;
        case GB_COLGT_idxunop_code    : kind = "COLGT"      ; break ;
        default                       : kind = "ENTRY"      ; break ;
    }
    fprintf (fp, "#define GB_%s_SELECTOR\n", kind) ;

    // handle flip_ij (needed for user-defined operators only)
    char *i_user = (flip_ij) ? "j" : "i" ;
    char *j_user = (flip_ij) ? "i" : "j" ;

    char *keep = NULL ;
    switch (opcode)
    {

        // positional: depends on (i,j) and y
        case GB_TRIL_selop_code       : keep = "(j)-(i) <= (y)" ; break ;
        case GB_TRIU_selop_code       : keep = "(j)-(i) >= (y)" ; break ;
        case GB_DIAG_selop_code       : keep = "(j)-(i) == (y)" ; break ;
        case GB_OFFDIAG_selop_code    : keep = "(j)-(i) != (y)" ; break ;
        case GB_ROWINDEX_idxunop_code : keep = "(i) != -(y)"    ; break ;
        case GB_ROWLE_idxunop_code    : keep = "(i) <= (y)"     ; break ;
        case GB_ROWGT_idxunop_code    : keep = "(i) > (y)"      ; break ;
        case GB_COLINDEX_idxunop_code : keep = "(j) != -(y)"    ; break ;
        case GB_COLLE_idxunop_code    : keep = "(j) <= (y)"     ; break ;
        case GB_COLGT_idxunop_code    : keep = "(j) > (y)"      ; break ;

        // depends on zombie status of A(i,j)
        case GB_NONZOMBIE_selop_code  : keep = "(i) >= 0" ; break ;

        // depends on A, OK for user-defined types
        case GB_NONZERO_selop_code    : 

            switch (xcode)
            {
                case GB_BOOL_code     : 
                    keep = "(x)" ;
                    break ;
                case GB_INT8_code     : 
                case GB_INT16_code    : 
                case GB_INT32_code    : 
                case GB_INT64_code    : 
                case GB_UINT8_code    : 
                case GB_UINT16_code   : 
                case GB_UINT32_code   : 
                case GB_UINT64_code   : 
                case GB_FP32_code     : 
                case GB_FP64_code     : 
                    keep = "(x) != 0" ;
                    break ;
                case GB_FC32_code     : 
                    keep = "GB_FC32_ne0 (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC32_NE0", GB_FC32_NE0_DEFN) ;
                    break ;
                case GB_FC64_code     : 
                    keep = "GB_FC64_ne0 (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC64_NE0", GB_FC64_NE0_DEFN) ;
                    break ;

                case GB_UDT_code      : 
                    fprintf (fp,
                        "#ifndef GB_GUARD_UDT_NE0_%d_DEFINED\n"
                        "#define GB_GUARD_UDT_NE0_%d_DEFINED\n"
                        "GB_STATIC_INLINE bool GB_udt_ne0_%d ",
                        asize, asize, asize) ;
                    if (asize_multiple_of_uint32)
                    { 
                        fprintf (fp,
                        "(uint32_t *aij)                        \n"
                        "{                                      \n"
                        "    bool ne0 = false ;                 \n"
                        "    for (int k = 0 ; k < %d ; i++)     \n"
                        "    {                                  \n"
                        "        ne0 = ne0 || (aij [k] != 0) ;  \n"
                        "    }                                  \n"
                        "    return (ne0) ;                     \n"
                        "}                                      \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_ne0_%d ((uint32_t *) &(x))\n",
                        asize32, asize) ;
                    }
                    else
                    { 
                        fprintf (fp,
                        "(uint8_t *aij)                         \n"
                        "{                                      \n"
                        "    bool ne0 = false ;                 \n"
                        "    for (int k = 0 ; k < %d ; i++)     \n"
                        "    {                                  \n"
                        "        ne0 = ne0 || (aij [k] != 0) ;  \n"
                        "    }                                  \n"
                        "    return (ne0) ;                     \n"
                        "}                                      \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_ne0_%d ((uint8_t *) &(x))\n",
                        asize, asize) ;
                    }
                    break ;

                default: ;
            }
            break ;

        // depends on A, OK for user-defined types
        case GB_EQ_ZERO_selop_code    : 

            switch (xcode)
            {
                case GB_BOOL_code     : 
                    keep = "!(x)" ;
                    break ;
                case GB_INT8_code     : 
                case GB_INT16_code    : 
                case GB_INT32_code    : 
                case GB_INT64_code    : 
                case GB_UINT8_code    : 
                case GB_UINT16_code   : 
                case GB_UINT32_code   : 
                case GB_UINT64_code   : 
                case GB_FP32_code     : 
                case GB_FP64_code     : 
                    keep = "(x) == 0" ;
                    break ;
                case GB_FC32_code     : 
                    keep = "GB_FC32_eq0 (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC32_EQ0", GB_FC32_EQ0_DEFN) ;
                    break ;
                case GB_FC64_code     : 
                    keep = "GB_FC64_eq0 (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC64_EQ0", GB_FC64_EQ0_DEFN) ;
                    break ;

                case GB_UDT_code      : 
                    fprintf (fp,
                        "#ifndef GB_GUARD_UDT_EQ0_%d_DEFINED\n"
                        "#define GB_GUARD_UDT_EQ0_%d_DEFINED\n"
                        "GB_STATIC_INLINE bool GB_udt_eq0_%d ",
                        asize, asize, asize) ;
                    if (asize_multiple_of_uint32)
                    { 
                        fprintf (fp,
                        "(uint32_t *aij)                        \n"
                        "{                                      \n"
                        "    bool eq0 = true ;                  \n"
                        "    for (int k = 0 ; k < %d ; i++)     \n"
                        "    {                                  \n"
                        "        eq0 = eq0 && (aij [k] == 0) ;  \n"
                        "    }                                  \n"
                        "    return (eq0) ;                     \n"
                        "}                                      \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_eq0_%d ((uint32_t *) &(x))\n",
                        asize32, asize) ;
                    }
                    else
                    { 
                        fprintf (fp,
                        "(uint8_t *aij)                         \n"
                        "{                                      \n"
                        "    bool eq0 = true ;                  \n"
                        "    for (int k = 0 ; k < %d ; i++)     \n"
                        "    {                                  \n"
                        "        eq0 = eq0 && (aij [k] == 0) ;  \n"
                        "    }                                  \n"
                        "    return (eq0) ;                     \n"
                        "}                                      \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_eq0_%d ((uint8_t *) &(x))\n",
                        asize, asize) ;
                    }
                    break ;

                default: ;
            }
            break ;

        // depends on A, for real built-in types only
        case GB_GT_ZERO_selop_code    : keep = "(x) > 0"    ; break ; 
        case GB_GE_ZERO_selop_code    : keep = "(x) >= 0"   ; break ;
        case GB_LT_ZERO_selop_code    : keep = "(x) < 0"    ; break ;
        case GB_LE_ZERO_selop_code    : keep = "(x) <= 0"   ; break ;

        // depends on A, OK for user-defined types
        case GB_NE_THUNK_selop_code   : 

            switch (xcode)
            {
                case GB_BOOL_code     : 
                case GB_INT8_code     : 
                case GB_INT16_code    : 
                case GB_INT32_code    : 
                case GB_INT64_code    : 
                case GB_UINT8_code    : 
                case GB_UINT16_code   : 
                case GB_UINT32_code   : 
                case GB_UINT64_code   : 
                case GB_FP32_code     : 
                case GB_FP64_code     : 
                    keep = "(x) != (y)" ;
                    break ;
                case GB_FC32_code     : 
                    keep = "GB_FC32_ne (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC32_NE", GB_FC32_NE_DEFN) ;
                    break ;
                case GB_FC64_code     : 
                    keep = "GB_FC64_ne (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC64_NE", GB_FC64_NE_DEFN) ;
                    break ;

                case GB_UDT_code      : 
                    fprintf (fp,
                        "#ifndef GB_GUARD_UDT_NE_%d_DEFINED\n"
                        "#define GB_GUARD_UDT_NE_%d_DEFINED\n"
                        "GB_STATIC_INLINE bool GB_udt_ne_%d ",
                        asize, asize, asize) ;
                    if (asize_multiple_of_uint32)
                    { 
                        fprintf (fp,
                        "(uint32_t *aij, uint32_t *yy)              \n"
                        "{                                          \n"
                        "    bool ne = false ;                      \n"
                        "    for (int k = 0 ; k < %d ; i++)         \n"
                        "    {                                      \n"
                        "        ne = ne || (aij [k] != yy [k]) ;   \n"
                        "    }                                      \n"
                        "    return (ne) ;                          \n"
                        "}                                          \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_ne_%d "
                        "((uint32_t *) &(x), (uint32_t *) &(y))\n",
                        asize32, asize) ;
                    }
                    else
                    { 
                        fprintf (fp,
                        "(uint8_t *aij, uint8_t *yy)                \n"
                        "{                                          \n"
                        "    bool ne = false ;                      \n"
                        "    for (int k = 0 ; k < %d ; i++)         \n"
                        "    {                                      \n"
                        "        ne = ne || (aij [k] != yy [k]) ;   \n"
                        "    }                                      \n"
                        "    return (ne) ;                          \n"
                        "}                                          \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_ne_%d "
                        "((uint8_t *) &(x), (uint8_t *) &(y))\n",
                        asize, asize) ;
                    }
                    break ;

                default: ;
            }
            break ;

        // depends on A, OK for user-defined types
        case GB_EQ_THUNK_selop_code   : 

            switch (xcode)
            {
                case GB_BOOL_code     : 
                case GB_INT8_code     : 
                case GB_INT16_code    : 
                case GB_INT32_code    : 
                case GB_INT64_code    : 
                case GB_UINT8_code    : 
                case GB_UINT16_code   : 
                case GB_UINT32_code   : 
                case GB_UINT64_code   : 
                case GB_FP32_code     : 
                case GB_FP64_code     : 
                    keep = "(x) == (y)" ;
                    break ;
                case GB_FC32_code     : 
                    keep = "GB_FC32_eq (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC32_EQ", GB_FC32_EQ_DEFN) ;
                    break ;
                case GB_FC64_code     : 
                    keep = "GB_FC64_eq (x)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC64_EQ", GB_FC64_EQ_DEFN) ;
                    break ;

                case GB_UDT_code      : 
                    fprintf (fp,
                        "#ifndef GB_GUARD_UDT_EQ_%d_DEFINED\n"
                        "#define GB_GUARD_UDT_EQ_%d_DEFINED\n"
                        "GB_STATIC_INLINE bool GB_udt_eq_%d ",
                        asize, asize, asize) ;
                    if (asize_multiple_of_uint32)
                    { 
                        fprintf (fp,
                        "(uint32_t *aij, uint32_t *yy)              \n"
                        "{                                          \n"
                        "    bool eq = true ;                       \n"
                        "    for (int k = 0 ; k < %d ; i++)         \n"
                        "    {                                      \n"
                        "        eq = eq && (aij [k] == yy [k]) ;   \n"
                        "    }                                      \n"
                        "    return (eq) ;                          \n"
                        "}                                          \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_eq_%d "
                        "((uint32_t *) &(x), (uint32_t *) &(y))\n",
                        asize32, asize) ;
                    }
                    else
                    { 
                        fprintf (fp,
                        "(uint8_t *aij, uint8_t *yy)                \n"
                        "{                                          \n"
                        "    bool eq = true ;                       \n"
                        "    for (int k = 0 ; k < %d ; i++)         \n"
                        "    {                                      \n"
                        "        eq = eq && (aij [k] == yy [k]) ;   \n"
                        "    }                                      \n"
                        "    return (eq) ;                          \n"
                        "}                                          \n"
                        "#define GB_KEEP(keep,x,i,j,y) "
                        "keep = GB_udt_eq_%d "
                        "((uint8_t *) &(x), (uint8_t *) &(y))\n",
                        asize, asize) ;
                    }
                    break ;

                default: ;
            }
            break ;

        // depends on A and Thunk, for real built-in types only
        case GB_GT_THUNK_selop_code   : keep = "(x) > (y)"  ; break ;
        case GB_GE_THUNK_selop_code   : keep = "(x) >= (y)"  ; break ;
        case GB_LT_THUNK_selop_code   : keep = "(x) < (y)"  ; break ;
        case GB_LE_THUNK_selop_code   : keep = "(x) <= (y)"  ; break ;

        // depends on A and Thunk, for built-in types only
        case GB_VALUEEQ_idxunop_code  : 

            switch (xcode)
            {
                case GB_BOOL_code     : 
                case GB_INT8_code     : 
                case GB_INT16_code    : 
                case GB_INT32_code    : 
                case GB_INT64_code    : 
                case GB_UINT8_code    : 
                case GB_UINT16_code   : 
                case GB_UINT32_code   : 
                case GB_UINT64_code   : 
                case GB_FP32_code     : 
                case GB_FP64_code     : 
                    keep = "(x) == (y)" ;
                    break ;
                case GB_FC32_code     : 
                    keep = "GB_FC32_eq (x,y)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC32_EQ", GB_FC32_EQ_DEFN) ;
                    break ;
                case GB_FC64_code     : 
                    keep = "GB_FC64_eq (x,y)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC64_EQ", GB_FC64_EQ_DEFN) ;
                    break ;
                default: ;
            }
            break ;

        // depends on A and Thunk, for built-in types only
        case GB_VALUENE_idxunop_code  : 

            switch (xcode)
            {
                case GB_BOOL_code     : 
                case GB_INT8_code     : 
                case GB_INT16_code    : 
                case GB_INT32_code    : 
                case GB_INT64_code    : 
                case GB_UINT8_code    : 
                case GB_UINT16_code   : 
                case GB_UINT32_code   : 
                case GB_UINT64_code   : 
                case GB_FP32_code     : 
                case GB_FP64_code     : 
                    keep = "(x) != (y)" ;
                    break ;
                case GB_FC32_code     : 
                    keep = "GB_FC32_ne (x,y)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC32_NE", GB_FC32_NE_DEFN) ;
                    break ;
                case GB_FC64_code     : 
                    keep = "GB_FC64_ne (x,y)" ;
                    GB_macrofy_defn (fp, 1, "GB_FC64_NE", GB_FC64_NE_DEFN) ;
                    break ;
                default: ;
            }
            break ;

        // depends on A and Thunk, for real built-in types only
        case GB_VALUEGT_idxunop_code  : keep = "(x) > (y)"  ; break ;
        case GB_VALUEGE_idxunop_code  : keep = "(x) >= (y)" ; break ;
        case GB_VALUELT_idxunop_code  : keep = "(x) < (y)"  ; break ;
        case GB_VALUELE_idxunop_code  : keep = "(x) <= (y)" ; break ;

        // depends on A and Thunk (type and values); user-defined operators
        case GB_USER_idxunop_code     : 
            if (ztype == GrB_BOOL)
            { 
                // no need to typecast result of the user-defined operator
                fprintf (fp,
                "#define GB_KEEP(keep,x,i,j,y) "
                "%s (&(keep), &(x), %s, %s, &(y)) ;\n",
                opname, i_user, j_user) ;
            }
            else
            { 
                // need to typecast result to bool
                GB_macrofy_cast_input (fp, "CAST_Z_TO_KEEP", "keep", "z", "z",
                    GrB_BOOL, ztype) ;
                fprintf (fp,
                    "#define GB_KEEP(keep,x,i,j,y) { %s zkeep ; "
                    "%s (&(zkeep), &(x), %s, %s, &(y)) ; "
                    "CAST_Z_TO_KEEP (keep, zkeep) ; }\n",
                    ztype->name, opname, i_user, j_user) ;
            }
            GB_macrofy_defn (fp, 3, op->name, op->defn) ;
            break ;

        case GB_USER_selop_code       : 

            fprintf (fp,
                "#define GB_KEEP(keep,x,i,j,y) "
                "keep = %s (%s, %s, &(x), &(y)) ;\n",
                opname, i_user, j_user) ;
            GB_macrofy_defn (fp, 3, op->name, op->defn) ;
            break ;

        default: ;
    }

    if (keep != NULL)
    { 
        fprintf (fp, "#define GB_KEEP(keep,x,i,j,y) keep = (%s) ;\n", keep) ;
    }

    //--------------------------------------------------------------------------
    // macros for the C matrix
    //--------------------------------------------------------------------------

    // FIXME: need csparsity (is it always the same as asparsity?)
    // FIXME: this kernel could typecast from A to C

    GB_macrofy_output (fp, "c", "C", "C", atype, atype, asparsity, C_iso_code) ;

    //--------------------------------------------------------------------------
    // construct the macros for A
    //--------------------------------------------------------------------------

    // GB_GETA macro to get aij = A(i,j)
    // aij is not typecasted
    GB_macrofy_input (fp, "a", "A", "A", true, atype,
        atype, asparsity, acode, A_iso_code, -1) ;

    // GB_GETX macro to get x = (xtype) A(i,j)
    // x is a value A(i,j) typecasted to the op->xtype of the select operator
    GB_macrofy_input (fp, "x", "X", "A", false, xtype,
        atype, asparsity, acode, A_iso_code, -1) ;
}

