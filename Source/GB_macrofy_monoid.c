//------------------------------------------------------------------------------
// GB_macrofy_monoid: build macros for a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"

void GB_macrofy_monoid  // construct the macros for a monoid
(
    FILE *fp,           // File to write macros, assumed open already
    // inputs:
    int add_ecode,      // binary op as an enum
    int id_ecode,       // identity value as an enum
    int term_ecode,     // terminal value as an enum (<= 28 is terminal)
    GrB_Monoid monoid   // monoid to macrofy
)
{

    GrB_BinaryOp op = (monoid == NULL) ? NULL : monoid->op ;
    const char *ztype_name = (monoid == NULL) ? "void" : op->ztype->name ;
    int zcode = (monoid == NULL) ? 0 : op->ztype->code ;

    //--------------------------------------------------------------------------
    // create macros for the additive operator
    //--------------------------------------------------------------------------

    GB_macrofy_binop (fp, "GB_ADD", false, true, add_ecode, op) ;

    //--------------------------------------------------------------------------
    // create macros for the identity value
    //--------------------------------------------------------------------------

    if (monoid == NULL)
    {
        // no values computed
        fprintf (fp, "#define GB_DECLARE_MONOID_IDENTITY(z)\n") ;
    }
    else if (id_ecode <= 28)
    {
        // built-in monoid: a simple assignment
        const char *id_value = GB_charify_identity_or_terminal (id_ecode) ;
        fprintf (fp, "#define GB_DECLARE_MONOID_IDENTITY(z) "
            "%s z = (%s) (%s) ;\n", ztype_name, ztype_name, id_value) ;
    }
    else
    {
        // user-defined monoid:  all we have are the bytes
        GB_macrofy_bytes (fp, "MONOID_IDENTITY", ztype_name,
            (uint8_t *) (monoid->identity), op->ztype->size) ;
    }

    //--------------------------------------------------------------------------
    // create macros for the terminal value and terminal conditions
    //--------------------------------------------------------------------------

    if (term_ecode == 18)
    {
        // ANY monoid is terminal but with no specific terminal value
        fprintf (fp, "#define GB_DECLARE_MONOID_TERMINAL(z)\n") ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(cij,z) (true)\n") ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(cij,z) break\n") ;
    }
    else if (monoid == NULL || monoid->terminal == NULL)
    {
        // monoid is not terminal (either built-in or user-defined)
        fprintf (fp, "#define GB_DECLARE_MONOID_TERMINAL(z)\n") ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(cij,z) (false)\n") ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(cij,z)\n") ;
    }
    else if (term_ecode <= 28)
    {
        // built-in terminal monoid: terminal value is a simple assignment
        const char *term_value = GB_charify_identity_or_terminal (term_ecode) ;
        fprintf (fp, "#define GB_DECLARE_MONOID_TERMINAL(z) "
            "%s z = (%s) (%s) ;\n", ztype_name, ztype_name, term_value) ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(cij,z) ((cij) == (z))\n") ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(cij,z) "
            "if ((cij) == (z)) break\n") ;
    }
    else
    {
        // user-defined terminal monoid
        GB_macrofy_bytes (fp, "MONOID_TERMINAL", ztype_name,
            monoid->terminal, op->ztype->size) ;
        fprintf (fp, "#define GB_TERMINAL_CONDITION(cij,z)"
            " (memcmp (&(cij), &(z), %d) == 0)\n",
            (int) op->ztype->size) ;
        fprintf (fp, "#define GB_IF_TERMINAL_BREAK(cij,z) "
            " if (memcmp (&(cij), &(z), %d) == 0) break\n",
            (int) op->ztype->size) ;
    }

    //--------------------------------------------------------------------------
    // create macros for the atomic CUDA operator, if available
    //--------------------------------------------------------------------------

    // All built-in monoids are handled, except for the double complex cases of
    // ANY and TIMES.  Those need to be done the same way user-defined monoids
    // are computed.

    char *a = NULL ;
    fprintf (fp, "// add_ecode: %d, zcode: %d\n", add_ecode, zcode) ;

    switch (add_ecode)
    {

        // FIRST, ANY, SECOND: atomic write (not double complex)
        case  1 :
        case  2 :

            switch (zcode)
            {
                case GB_ignore_code  :      // any_pair semiring, C is iso
                case GB_BOOL_code    :
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  :
                case GB_FP32_code    :
                case GB_FP64_code    : 
                case GB_FC32_code    : a = "GB_atomic_write" ;
                default              : break ;
            }
            break ;

        // MIN (real only)
        case  3 :
        case  4 :
        case  5 :

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  : 
                case GB_FP32_code    :
                case GB_FP64_code    : a = "GB_atomic_min" ;
                default              : break ;
            }
            break ;

        // MAX (real only)
        case  6 :
        case  7 :
        case  8 :

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  : 
                case GB_FP32_code    :
                case GB_FP64_code    : a = "GB_atomic_max" ;
                default              : break ;
            }
            break ;

            switch (zcode)
            {
                case GB_BOOL_code    :a = "GB_atomic_bor" ;
                default              : break ;
            }
            break ;

        // PLUS:  all types
        case  9 :
        case 10 :
        case 11 :

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  :
                case GB_FP32_code    :
                case GB_FP64_code    :
                case GB_FC32_code    :
                case GB_FC64_code    : a = "GB_atomic_add" ;
                default              : break ;
            }
            break ;

        // TIMES: all real types, and float complex (but not double complex)
        case 12 : 
        case 14 : 

            switch (zcode)
            {
                case GB_INT8_code    :
                case GB_UINT8_code   :
                case GB_INT16_code   :
                case GB_UINT16_code  :
                case GB_INT32_code   :
                case GB_UINT32_code  :
                case GB_INT64_code   :
                case GB_UINT64_code  :
                case GB_FP32_code    :
                case GB_FP64_code    : 
                case GB_FC32_code    : a = "GB_atomic_times" ;
                default              : break ;
            }
            break ;

        // BOR: z = (x | y), bitwise or,
        // logical LOR (via upscale to uint32_t and BOR)
        case 17 :
        case 19 :

            switch (zcode)
            {
                case GB_BOOL_code    :
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_atomic_bor" ;
                default              : break ;
            }
            break ;

        // BAND: z = (x & y), bitwise and
        // logical LAND (via upscale to uint32_t and BAND)
        case 18 :
        case 20 :

            switch (zcode)
            {
                case GB_BOOL_code    :
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_atomic_band" ;
                default              : break ;
            }
            break ;

        // BXOR: z = (x ^ y), bitwise xor, and boolean LXOR
        case 16 :
        case 21 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_atomic_bxor" ;
                default              : break ;
            }
            break ;

        // BXNOR: z = ~(x ^ y), bitwise xnor, and boolean LXNOR
        case 15 :
        case 22 :

            switch (zcode)
            {
                case GB_BOOL_code    : 
                case GB_UINT8_code   :
                case GB_UINT16_code  :
                case GB_UINT32_code  :
                case GB_UINT64_code  : a = "GB_atomic_bxnor" ;
                default              : break ;
            }
            break ;

        // all other monoids
        default: break ;
    }

    if (a == NULL)
    {
        // no CUDA atomic available
        fprintf (fp, "#define GB_HAS_CUDA_ATOMIC 0\n") ;
    }
    else
    {
        // CUDA atomic available
        fprintf (fp, "#define GB_HAS_CUDA_ATOMIC 1\n") ;
        fprintf (fp, "#define GB_CUDA_ATOMIC %s\n", a) ;

        // upscale 8-bit types to 16-bits, all others use their native types
        char *t = "" ;
        switch (zcode)
        {
            case GB_INT8_code    : 
            case GB_INT16_code   : t = "int16_t"    ; break ;
            case GB_INT32_code   : t = "int32_t"    ; break ;
            case GB_INT64_code   : t = "int64_t"    ; break ;
            case GB_ignore_code  : 
            case GB_BOOL_code    : 
            case GB_UINT8_code   : 
            case GB_UINT16_code  : t = "uint16_t"   ; break ;
            case GB_UINT32_code  : t = "uint32_t"   ; break ;
            case GB_UINT64_code  : t = "uint64_t"   ; break ;
            case GB_FP32_code    : t = "float"      ; break ;
            case GB_FP64_code    : t = "double"     ; break ;
            case GB_FC32_code    : t = "float complex"  ; break ;
            case GB_FC64_code    : t = "double complex" ; break ;
            default :;
        }

        fprintf (fp, "#define GB_CUDA_ATOMIC_TYPE %s\n", t) ;
    }
}

