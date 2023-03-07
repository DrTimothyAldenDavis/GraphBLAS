//------------------------------------------------------------------------------
// GB_macrofy_binop: construct the macro and defn for a binary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"
#include <ctype.h>

void GB_macrofy_binop
(
    FILE *fp,
    // input:
    const char *macro_name,
    bool flipxy,                // if true: op is f(y,x) for a semiring
    bool is_monoid_or_build,    // if true: additive operator for monoid,
                                // or binary op for GrB_Matrix_build
    bool is_ewise,              // if true: binop for ewise methods
    int ecode,
    bool C_iso,                 // if true: C is iso
    GrB_BinaryOp op,            // NULL if C is iso
    // output:
    const char **f_handle,
    const char **u_handle
)
{

    const char *f = NULL, *u = NULL, *g = NULL ;
    const char *karg = is_ewise ? "" : ",k" ;

    if (C_iso)
    {

        //----------------------------------------------------------------------
        // C is iso: no operator
        //----------------------------------------------------------------------

        if (is_monoid_or_build)
        {
            fprintf (fp, "#define GB_UPDATE(z,y)\n") ;
            fprintf (fp, "#define %s(z,x,y)\n", macro_name) ;
        }
        else
        {
            fprintf (fp, "#define %s(z,x,y,i%s,j)\n", macro_name, karg) ;
        }

    }
    else if (ecode == 0)
    {

        //----------------------------------------------------------------------
        // user-defined operator
        //----------------------------------------------------------------------

        ASSERT (op != NULL) ;
        bool is_macro = GB_macrofy_defn (fp, 3, op->name, op->defn) ;
        if (is_macro)
        {
            fprintf (fp, "// binary operator %s defined as a macro:\n",
                op->name) ;
        }

        if (is_monoid_or_build)
        {
            // additive/build operator: no i,k,j parameters, never flipped
            fprintf (fp, "#define %s(z,x,y) ", macro_name) ;
        }
        else if (flipxy)
        {
            // flipped multiplicative or ewise operator
            // note: no positional operands for user-defined ops (yet)
            fprintf (fp, "#define %s(z,y,x,j%s,i) ", macro_name, karg) ;
        }
        else
        {
            // unflipped multiplicative or ewise operator
            fprintf (fp, "#define %s(z,x,y,i%s,j) ", macro_name, karg) ;
        }

        if (is_macro)
        {
            for (char *p = op->name ; (*p) != '\0' ; p++)
            {
                int c = (*p) ;
                fputc (toupper (c), fp) ;
            }
            fprintf (fp, " (z, x, y)\n") ;
        }
        else
        {
            fprintf (fp, " %s (&(z), &(x), &(y))\n", op->name) ;
        }

        if (is_monoid_or_build)
        {
            fprintf (fp, "#define GB_UPDATE(z,y) %s(z,z,y)\n", macro_name) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // built-in operator
        //----------------------------------------------------------------------

        switch (ecode)
        {

            //------------------------------------------------------------------
            // built-in ops, can be used in a monoid or build
            //------------------------------------------------------------------

            // first
            case   1 : 
                f = "z = x" ;
                u = "" ;
                break ;

            // any, second
            case   2 : 
                f = "z = y" ;
                u = "z = y" ;
                break ;

            // min (float)
            case   3 : 
                f = "z = fminf (x,y)" ;
                g = "z = fminf (z,y)" ;
                u = "if (!isnan (y) && !islessequal (z,y)) { z = y ; }" ;
                break ;

            // min (double)
            case   4 : 
                f = "z = fmin (x,y)" ;
                g = "z = fmin (z,y)" ;
                u = "if (!isnan (y) && !islessequal (z,y)) { z = y ; }" ;
                break ;

            // min (integer)
            case   5 : 
                f = "z = (((x) < (y)) ? (x) : (y))" ;
                g = "z = (((z) < (y)) ? (z) : (y))" ;
                u = "if ((z) > (y)) { z = y ; }" ;
                break ;

            // max (float)
            case   6 : 
                f = "z = fmaxf (x,y)" ;
                g = "z = fmaxf (z,y)" ;
                u = "if (!isnan (y) && !isgreaterequal (z,y)) { z = y ; }" ;
                break ;

            // max (double)
            case   7 : 
                f = "z = fmax (x,y)" ;
                g = "z = fmax (z,y)" ;
                u = "if (!isnan (y) && !isgreaterequal (z,y)) { z = y ; }" ;
                break ;

            // max (integer)
            case   8 : 
                f = "z = (((x) > (y)) ? (x) : (y))" ;
                g = "z = (((z) > (y)) ? (z) : (y))" ;
                u = "if ((z) < (y)) { z = y ; }" ;
                break ;

            // plus (complex)
            case   9 : 
                f = "z = GB_FC32_add (x,y)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC32_add", GB_FC32_add_DEFN) ;
                break ;

            case  10 : 
                f = "z = GB_FC64_add (x,y)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC64_add", GB_FC64_add_DEFN) ;
                break ;

            // plus (real)
            case  11 : 
                f = "z = (x) + (y)" ;
                u = "z += y" ;              // plus real update
                break ;

            // times (complex)
            case  12 : 
                f = "z = GB_FC32_mul (x,y)" ;
                GB_macrofy_defn (fp, 1, "GB_FC32_mul", GB_FC64_mul_DEFN) ;
                break ;
            case  13 : 
                f = "z = GB_FC64_mul (x,y)" ;
                GB_macrofy_defn (fp, 1, "GB_FC64_mul", GB_FC64_mul_DEFN) ;
                break ;

            // times (real)
            case  14 : 
                f = "z = (x) * (y)" ;
                u = "z *= y" ;              // times real update
                break ;

            // eq, iseq, lxnor
            case  15 : 
                f = "z = ((x) == (y))" ;
                u = "z = (z == (y))" ;
                break ;

            // ne, isne, lxor
            case  16 : 
                f = "z = ((x) != (y))" ;
                u = "z = (z != (y))" ;
                break ;

            // lor
            case  17 : 
                f = "z = ((x) || (y))" ;
                u = "z = (z || (y))" ;
                break ;

            // land
            case  18 : 
                f = "z = ((x) && (y))" ;
                u = "z = (z && (y))" ;
                break ;

            // bor
            case  19 : 
                f = "z = ((x) | (y))" ;
                u = "z |= y" ;              // bor update
                break ;

            // band
            case  20 : 
                f = "z = ((x) & (y))" ;
                u = "z &= y" ;              // band update
                break ;

            // bxor
            case  21 : 
                f = "z = ((x) ^ (y))" ;
                u = "z ^= y" ;              // bxor update
                break ;

            // bxnor
            case  22 : 
                f = "z = (~((x) ^ (y)))" ;
                u = "z = (~(z ^ (y)))" ;
                break ;

            // 23 to 31 are unused, but reserved for future monoids

            //------------------------------------------------------------------
            // built-in ops, cannot be used in a monoid
            //------------------------------------------------------------------

            // eq for complex
            case  32 : 
                f = "z = GB_FC32_eq (x,y)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC32_eq", GB_FC32_eq_DEFN) ;
                break ;
            case  33 : 
                f = "z = GB_FC64_eq (x,y)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC64_eq", GB_FC64_eq_DEFN) ;
                break ;

            // iseq for complex
            case  34 : 
                f = "z = GB_FC32_iseq (x,y)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC32_eq", GB_FC32_eq_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC32_iseq", GB_FC32_iseq_DEFN) ;
                break ;
            case  35 : 
                f = "z = GB_FC64_iseq (x,y)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC64_eq", GB_FC64_eq_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC64_iseq", GB_FC64_iseq_DEFN) ;
                break ;

            // ne for complex
            case  36 : 
                f = "z = GB_FC32_ne (x,y)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC32_ne", GB_FC32_ne_DEFN) ;
                break ;
            case  37 : 
                f = "z = GB_FC64_ne (x,y)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC64_ne", GB_FC64_ne_DEFN) ;
                break ;

            // isne for complex
            case  38 : 
                f = "z = GB_FC32_isne (x,y)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC32_ne", GB_FC32_ne_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC32_isne", GB_FC32_isne_DEFN) ;
                break ;
            case  39 : 
                f = "z = GB_FC64_isne (x,y)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC64_ne", GB_FC64_ne_DEFN) ;
                GB_macrofy_defn (fp, 1, "GB_FC64_isne", GB_FC64_isne_DEFN) ;
                break ;

            // lor for non-boolean
            case  40 : f = "z = (((x)!=0) || ((y)!=0))" ; break ;

            // land for non-boolean
            case  41 : f = "z = (((x)!=0) && ((y)!=0))" ; break ;

            // lxor for non-boolean
            case  42 : f = "z = (((x)!=0) != ((y)!=0))" ; break ;

            // minus
            case  43 : 
                f = "z = GB_FC32_minus (x,y)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC32_minus", GB_FC32_minus_DEFN) ;
                break ;
            case  44 : 
                f = "z = GB_FC64_minus (x,y)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC64_minus", GB_FC64_minus_DEFN) ;
                break ;
            case  45 : 
                f = "z = (x) - (y)" ;
                break ;

            // rminus
            case  46 : 
                f = "z = GB_FC32_minus (y,x)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC32_minus", GB_FC32_minus_DEFN) ;
                break ;
            case  47 : 
                f = "z = GB_FC64_minus (y,x)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC64_minus", GB_FC64_minus_DEFN) ;
                break ;
            case  48 : 
                f = "z = (y) - (x)" ;
                break ;

            // div (integer)
            case  49 : 
                f = "z = GB_idiv_int8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int8", GB_idiv_int8_DEFN) ;
                break ;
            case  50 : 
                f = "z = GB_idiv_int16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int16", GB_idiv_int16_DEFN) ;
                break ;
            case  51 : 
                f = "z = GB_idiv_int32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int32", GB_idiv_int32_DEFN) ;
                break ;
            case  52 : 
                f = "z = GB_idiv_int64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int64", GB_idiv_int64_DEFN) ;
                break ;
            case  53 : 
                f = "z = GB_idiv_uint8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint8", GB_idiv_uint8_DEFN) ;
                break ;
            case  54 : 
                f = "z = GB_idiv_uint16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint16", GB_idiv_uint16_DEFN) ;
                break ;
            case  55 : 
                f = "z = GB_idiv_uint32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint32", GB_idiv_uint32_DEFN) ;
                break ;
            case  56 : 
                f = "z = GB_idiv_uint64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint64", GB_idiv_uint64_DEFN) ;
                break ;

            // div (complex floating-point)
            case  57 : 
                f = "z = GB_FC32_div (x,y)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC32_div", GB_FC32_div_DEFN) ;
                break ;
            case  58 : 
                f = "z = GB_FC64_div (x,y)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                break ;

            // div (float and double)
            case  59 : f = "z = (x) / (y)" ; break ;

            // rdiv (integer)
            case  60 : 
                f = "z = GB_idiv_int8 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int8", GB_idiv_int8_DEFN) ;
                break ;
            case  61 : 
                f = "z = GB_idiv_int16 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int16", GB_idiv_int16_DEFN) ;
                break ;
            case  62 : 
                f = "z = GB_idiv_int32 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int32", GB_idiv_int32_DEFN) ;
                break ;
            case  63 : 
                f = "z = GB_idiv_int64 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int64", GB_idiv_int64_DEFN) ;
                break ;
            case  64 : 
                f = "z = GB_idiv_uint8 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint8", GB_idiv_uint8_DEFN) ;
                break ;
            case  65 : 
                f = "z = GB_idiv_uint16 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint16", GB_idiv_uint16_DEFN) ;
                break ;
            case  66 : 
                f = "z = GB_idiv_uint32 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint32", GB_idiv_uint32_DEFN) ;
                break ;
            case  67 : 
                f = "z = GB_idiv_uint64 (y,x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint64", GB_idiv_uint64_DEFN) ;
                break ;

            // rdiv (complex floating-point)
            case  68 : 
                f = "z = GB_FC32_div (y,x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC32_div", GB_FC32_div_DEFN) ;
                break ;
            case  69 : 
                f = "z = GB_FC64_div (y,x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                break ;

            // rdiv (real floating-point)
            case  70 : f = "z = (y) / (x)" ; break ;

            // gt, isgt
            case  71 : f = "z = ((x) > (y))" ; break ;

            // lt, islt
            case  72 : f = "z = ((x) < (y))" ; break ;

            // ge, isge
            case  73 : f = "z = ((x) >= (y))" ; break ;

            // le, isle
            case  74 : f = "z = ((x) <= (y))" ; break ;

            // bget
            case  75 : 
                f = "z = GB_bitget_int8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_int8", GB_bitget_int8_DEFN) ;
                break ;
            case  76 : 
                f = "z = GB_bitget_int16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_int16",
                    GB_bitget_int16_DEFN) ;
                break ;
            case  77 : 
                f = "z = GB_bitget_int32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_int32",
                    GB_bitget_int32_DEFN) ;
                break ;
            case  78 : 
                f = "z = GB_bitget_int64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_int64",
                    GB_bitget_int64_DEFN) ;
                break ;
            case  79 : 
                f = "z = GB_bitget_uint8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_uint8",
                    GB_bitget_uint8_DEFN) ;
                break ;
            case  80 : 
                f = "z = GB_bitget_uint16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_uint16",
                    GB_bitget_uint16_DEFN) ;
                break ;
            case  81 : 
                f = "z = GB_bitget_uint32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_uint32",
                    GB_bitget_uint32_DEFN) ;
                break ;
            case  82 : 
                f = "z = GB_bitget_uint64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitget_uint64",
                    GB_bitget_uint64_DEFN) ;
                break ;

            // bset
            case  83 : 
                f = "z = GB_bitset_int8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_int8", GB_bitset_int8_DEFN) ;
                break ;
            case  84 : 
                f = "z = GB_bitset_int16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_int16",
                    GB_bitset_int16_DEFN) ;
                break ;
            case  85 : 
                f = "z = GB_bitset_int32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_int32",
                    GB_bitset_int32_DEFN) ;
                break ;
            case  86 : 
                f = "z = GB_bitset_int64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_int64",
                    GB_bitset_int64_DEFN) ;
                break ;
            case  87 : 
                f = "z = GB_bitset_uint8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_uint8",
                    GB_bitset_uint8_DEFN) ;
                break ;
            case  88 : 
                f = "z = GB_bitset_uint16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_uint16",
                    GB_bitset_uint16_DEFN) ;
                break ;
            case  89 : 
                f = "z = GB_bitset_uint32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_uint32",
                    GB_bitset_uint32_DEFN) ;
                break ;
            case  90 : 
                f = "z = GB_bitset_uint64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitset_uint64",
                    GB_bitset_uint64_DEFN) ;
                break ;

            // bclr
            case  91 : 
                f = "z = GB_bitclr_int8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_int8", GB_bitclr_int8_DEFN) ;
                break ;
            case  92 : 
                f = "z = GB_bitclr_int16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_int16",
                    GB_bitclr_int16_DEFN) ;
                break ;
            case  93 : 
                f = "z = GB_bitclr_int32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_int32",
                    GB_bitclr_int32_DEFN) ;
                break ;
            case  94 : 
                f = "z = GB_bitclr_int64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_int64",
                    GB_bitclr_int64_DEFN) ;
                break ;
            case  95 : 
                f = "z = GB_bitclr_uint8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_uint8",
                    GB_bitclr_uint8_DEFN) ;
                break ;
            case  96 : 
                f = "z = GB_bitclr_uint16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_uint16",
                    GB_bitclr_uint16_DEFN) ;
                break ;
            case  97 : 
                f = "z = GB_bitclr_uint32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_uint32",
                    GB_bitclr_uint32_DEFN) ;
                break ;
            case  98 : 
                f = "z = GB_bitclr_uint64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitclr_uint64",
                    GB_bitclr_uint64_DEFN) ;
                break ;

            // bshift
            case  99 : 
                f = "z = GB_bitshift_int8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_int8",
                    GB_bitshift_int8_DEFN) ;
                break ;
            case 100 : 
                f = "z = GB_bitshift_int16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_int16",
                    GB_bitshift_int16_DEFN) ;
                break ;
            case 101 : 
                f = "z = GB_bitshift_int32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_int32",
                    GB_bitshift_int32_DEFN) ;
                break ;
            case 102 : 
                f = "z = GB_bitshift_int64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_int64",
                    GB_bitshift_int64_DEFN) ;
                break ;
            case 103 : 
                f = "z = GB_bitshift_uint8 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_uint8",
                    GB_bitshift_uint8_DEFN) ;
                break ;
            case 104 : 
                f = "z = GB_bitshift_uint16 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_uint16",
                    GB_bitshift_uint16_DEFN) ;
                break ;
            case 105 : 
                f = "z = GB_bitshift_uint32 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_uint32",
                    GB_bitshift_uint32_DEFN) ;
                break ;
            case 106 : 
                f = "z = GB_bitshift_uint64 (x,y)" ;
                GB_macrofy_defn (fp, 0, "GB_bitshift_uint64",
                    GB_bitshift_uint64_DEFN) ;
                break ;

            // pow (integer cases)
            case 107 : 
                f = "z = GB_pow_int8 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_int8", GB_pow_int8_DEFN) ;
                break ;
            case 108 : 
                f = "z = GB_pow_int16 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_int16", GB_pow_int16_DEFN) ;
                break ;
            case 109 : 
                f = "z = GB_pow_int32 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_int32", GB_pow_int32_DEFN) ;
                break ;
            case 110 : 
                f = "z = GB_pow_int64 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_int64", GB_pow_int64_DEFN) ;
                break ;
            case 111 : 
                f = "z = GB_pow_uint8 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_uint8", GB_pow_uint8_DEFN) ;
                break ;
            case 112 : 
                f = "z = GB_pow_uint16 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_uint16", GB_pow_uint16_DEFN) ;
                break ;
            case 113 : 
                f = "z = GB_pow_uint32 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_uint32", GB_pow_uint32_DEFN) ;
                break ;
            case 114 : 
                f = "z = GB_pow_uint64 (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow_uint64", GB_pow_uint64_DEFN) ;
                break ;

            // pow (float and double)
            case 115 : 
                f = "z = GB_powf (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_powf", GB_powf_DEFN) ;
                break ;
            case 116 : 
                f = "z = GB_pow (x, y)" ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                break ;

            // pow (complex float and double)
            case 117 : 
                f = "z = GB_FC32_pow (x, y)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cpowf", GB_cpowf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_powf", GB_powf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC32_pow", GB_FC32_pow_DEFN) ;
                break ;
            case 118 : 
                f = "z = GB_FC64_pow (x, y)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "cpow", GB_cpow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_pow", GB_FC64_pow_DEFN) ;
                break ;

            // atan2
            case 119 : f = "z = atan2f (x, y)" ; break ;
            case 120 : f = "z = atan2 (x, y)" ; break ;

            // hypot
            case 121 : f = "z = hypotf (x, y)" ; break ;
            case 122 : f = "z = hypot (x, y)" ; break ;

            // fmod
            case 123 : f = "z = fmodf (x, y)" ; break ;
            case 124 : f = "z = fmod (x, y)" ; break ;

            // remainder
            case 125 : f = "z = remainderf (x, y)" ; break ;
            case 126 : f = "z = remainder (x, y)" ; break ;

            // copysign
            case 127 : f = "z = copysignf (x, y)" ; break ;
            case 128 : f = "z = copysign (x, y)" ; break ;

            // ldexp
            case 129 : f = "z = ldexpf (x, y)" ; break ;
            case 130 : f = "z = ldexp (x, y)" ; break ;

            // cmplex
            case 131 : f = "z = GB_CMPLX32 (x, y)" ; break ;
            case 132 : f = "z = GB_CMPLX64 (x, y)" ; break ;

            // pair
            case 133 : f = "z = 1" ; break ;

            //------------------------------------------------------------------
            // positional ops
            //------------------------------------------------------------------

            // in a semiring:  cij += aik * bkj
            //      firsti is i, firstj is k, secondi k, secondj is j

            // in an ewise operation:  cij = aij + bij
            //      firsti is i, firstj is j, secondi i, secondj is j

            case 134 : f = "z = (i)" ; break ;
            case 135 : f = "z = (k)" ; break ;
            case 136 : f = "z = (j)" ; break ;
            case 137 : f = "z = (i) + 1" ; break ;
            case 138 : f = "z = (k) + 1" ; break ;
            case 139 : f = "z = (j) + 1" ; break ;

            //------------------------------------------------------------------
            // no-op: same as second operator
            //------------------------------------------------------------------

            case 140 : 
            default  : f = "z = y" ; break ;
        }

        //----------------------------------------------------------------------
        // create the macro
        //----------------------------------------------------------------------

        if (is_monoid_or_build)
        {
            // additive operator: no i,k,j parameters
            fprintf (fp, "#define %s(z,x,y) %s\n", macro_name, f) ;
            if (g != NULL)
            {
                // create an update expression of the form z += y,
                // but it differs for the CPU and CUDA JIT kernels
                fprintf (fp, "#ifdef  GB_CUDA_KERNEL\n"
                             "#define GB_UPDATE(z,y) %s\n"
                             "#else\n"
                             "#define GB_UPDATE(z,y) %s\n"
                             "#endif\n", g, u) ;
            }
            else if (u != NULL)
            {
                // create an update expression of the form z += y
                fprintf (fp, "#define GB_UPDATE(z,y) %s\n", u) ;
            }
            else
            {
                // create an update expression of the form z = z + y
                fprintf (fp, "#define GB_UPDATE(z,y) %s(z,z,y)\n", macro_name) ;
            }
        }
        else if (flipxy)
        {
            // flipped multiplicative or ewise operator
            fprintf (fp, "#define %s(z,y,x,j%s,i) %s\n", macro_name, karg, f) ;
        }
        else
        {
            // unflipped multiplicative or ewise operator
            fprintf (fp, "#define %s(z,x,y,i%s,j) %s\n", macro_name, karg, f) ;
        }
    }

    //--------------------------------------------------------------------------
    // return the u and f expressions
    //--------------------------------------------------------------------------

    if (u_handle != NULL) (*u_handle) = u ;
    if (f_handle != NULL) (*f_handle) = f ;
}

