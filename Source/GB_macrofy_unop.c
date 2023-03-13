//------------------------------------------------------------------------------
// GB_macrofy_unop: construct the macro and defn for a unary operator
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_stringify.h"
#include <ctype.h>

void GB_macrofy_unop
(
    FILE *fp,
    // input:
    const char *macro_name,
    bool flipij,                // if true: op is f(z,x,j,i,y) with ij flipped
    int ecode,
    GB_Operator op              // GrB_UnaryOp or GrB_IndexUnaryOp
)
{

    const char *f = "" ;
    const char *ij = (flipij) ? "j,i" : "i,j" ;

    if (ecode == 0)
    {

        //----------------------------------------------------------------------
        // user-defined GrB_UnaryOp
        //----------------------------------------------------------------------

        ASSERT (op != NULL) ;
        bool is_macro = GB_macrofy_defn (fp, 3, op->name, op->defn) ;
        if (is_macro)
        {
            fprintf (fp, "// unary operator %s defined as a macro:\n",
                op->name) ;
        }
        fprintf (fp, "#define %s(z,x,%s,y) ", macro_name, ij) ;
        if (is_macro)
        {
            for (char *p = op->name ; (*p) != '\0' ; p++)
            {
                int c = (*p) ;
                fputc (toupper (c), fp) ;
            }
            fprintf (fp, " (z, x)\n") ;
        }
        else
        {
            fprintf (fp, " %s (&(z), &(x))\n", op->name) ;
        }

    }
    else if (ecode == 255)
    {

        //----------------------------------------------------------------------
        // user-defined GrB_IndexUnaryOp
        //----------------------------------------------------------------------

        ASSERT (op != NULL) ;
        bool is_macro = GB_macrofy_defn (fp, 3, op->name, op->defn) ;
        if (is_macro)
        {
            fprintf (fp, "// index unary operator %s defined as a macro:\n",
                op->name) ;
        }
        fprintf (fp, "#define %s(z,x,%s,y) ", macro_name, ij) ;
        if (is_macro)
        {
            for (char *p = op->name ; (*p) != '\0' ; p++)
            {
                int c = (*p) ;
                fputc (toupper (c), fp) ;
            }
            fprintf (fp, " (z, x, i, j, y)\n") ;
        }
        else
        {
            fprintf (fp, " %s (&(z), &(x), i, j, &(y))\n", op->name) ;
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
            // primary unary operators x=f(x)
            //------------------------------------------------------------------

            case   1 : f = "z = 1" ;                            break ;

            case   2 : f = "z = x" ;                            break ;

            case   3 : f = "z = GB_FC32_ainv (x)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC32_ainv", GB_FC32_ainv_DEFN) ;
                break ;

            case   4 : f = "z = GB_FC64_ainv (x)" ;
                #if GB_COMPILER_MSC
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                #endif
                GB_macrofy_defn (fp, 1, "GB_FC64_ainv", GB_FC64_ainv_DEFN) ;
                break ;

            case   5 : f = "z = -(x)" ;                         break ;

            case   6 : f = "z = (((x) >= 0) ? (x) : (-(x)))" ;  break ;
            case   7 : f = "z = fabsf (x)" ;                    break ;
            case   8 : f = "z = fabs (x)" ;                     break ;
            case   9 : f = "z = cabsf (x)" ;                    break ;
            case  10 : f = "z = cabs (x)" ;                     break ;

            case  11 : f = "z = GB_idiv_int8 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int8", GB_idiv_int8_DEFN) ;
                break ;

            case  12 : f = "z = GB_idiv_int16 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int16", GB_idiv_int16_DEFN) ;
                break ;

            case  13 : f = "z = GB_idiv_int32 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int32", GB_idiv_int32_DEFN) ;
                break ;

            case  14 : f = "z = GB_idiv_int64 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_int64", GB_idiv_int64_DEFN) ;
                break ;

            case  15 : f = "z = GB_idiv_uint8 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint8", GB_idiv_uint8_DEFN) ;
                break ;

            case  16 : f = "z = GB_idiv_uint16 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint16", GB_idiv_uint16_DEFN) ;
                break ;

            case  17 : f = "z = GB_idiv_uint32 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint32", GB_idiv_uint32_DEFN) ;
                break ;

            case  18 : f = "z = GB_idiv_uint64 (1, x)" ;
                GB_macrofy_defn (fp, 0, "GB_idiv_uint64", GB_idiv_uint64_DEFN) ;
                break ;

            case  19 : f = "z = (1.0F)/(x)" ;               break ;
            case  20 : f = "z = 1./(x)" ;                   break ;

            case  21 : f = "z = GB_FC32_div (GxB_CMPLXF (1,0), x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC32_div", GB_FC32_div_DEFN) ;
                break ;

            case  22 : f = "z = GB_FC64_div (GxB_CMPLX  (1,0), x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                break ;

            case  23 : f = "z = !(x)" ;                     break ;
            case  24 : f = "z = !(x != 0)" ;                break ;

            case  25 : f = "z = ~(x)" ;                     break ;

            //------------------------------------------------------------------
            // unary operators for floating-point types (real and complex)" ;
            //------------------------------------------------------------------

            case  26 : f = "z = sqrtf (x)" ;        break ;
            case  27 : f = "z = sqrt (x)" ;         break ;

            case  28 : f = "z = csqrtf (x)" ;
                GB_macrofy_defn (fp, 2, "csqrtf", GB_csqrtf_DEFN) ;
                break ;

            case  29 : f = "z = csqrt (x)" ;
                GB_macrofy_defn (fp, 2, "csqrt", GB_csqrt_DEFN) ;
                break ;

            case  30 : f = "z = logf (x)" ;         break ;
            case  31 : f = "z = log (x)" ;          break ;

            case  32 : f = "z = clogf (x)" ;
                GB_macrofy_defn (fp, 2, "clogf", GB_clogf_DEFN) ;
                break ;

            case  33 : f = "z = clog (x)" ;
                GB_macrofy_defn (fp, 2, "clog", GB_clog_DEFN) ;
                break ;

            case  34 : f = "z = expf (x)" ;         break ;
            case  35 : f = "z = exp (x)" ;          break ;

            case  36 : f = "z = cexpf (x)" ;
                GB_macrofy_defn (fp, 2, "cexpf", GB_cexpf_DEFN) ;
                break ;

            case  37 : f = "z = cexp (x)" ; 
                GB_macrofy_defn (fp, 2, "cexp", GB_cexp_DEFN) ;
                break ;

            case  38 : f = "z = sinf (x)" ;         break ;
            case  39 : f = "z = sin (x)" ;          break ;

            case  40 : f = "z = csinf (x)" ;
                GB_macrofy_defn (fp, 2, "csinf", GB_csinf_DEFN) ;
                break ;

            case  41 : f = "z = csin (x)" ;
                GB_macrofy_defn (fp, 2, "csin", GB_csin_DEFN) ;
                break ;

            case  42 : f = "z = cosf (x)" ;         break ;
            case  43 : f = "z = cos (x)" ;          break ;

            case  44 : f = "z = ccosf (x)" ;
                GB_macrofy_defn (fp, 2, "ccosf", GB_ccosf_DEFN) ;
                break ;

            case  45 : f = "z = ccos (x)" ;
                GB_macrofy_defn (fp, 2, "ccos", GB_ccos_DEFN) ;
                break ;

            case  46 : f = "z = tanf (x)" ;         break ;
            case  47 : f = "z = tan (x)" ;          break ;

            case  48 : f = "z = ctanf (x)" ;
                GB_macrofy_defn (fp, 2, "ctanf", GB_ctanf_DEFN) ;
                break ;

            case  49 : f = "z = ctan (x)" ;
                GB_macrofy_defn (fp, 2, "ctan", GB_ctan_DEFN) ;
                break ;

            case  50 : f = "z = asinf (x)" ;        break ;
            case  51 : f = "z = asin (x)" ;         break ;

            case  52 : f = "z = casinf (x)" ;
                GB_macrofy_defn (fp, 2, "casinf", GB_casinf_DEFN) ;
                break ;

            case  53 : f = "z = casin (x)" ;
                GB_macrofy_defn (fp, 2, "casin", GB_casin_DEFN) ;
                break ;

            case  54 : f = "z = acosf (x)" ;        break ;
            case  55 : f = "z = acos (x)" ;         break ;

            case  56 : f = "z = cacosf (x)" ;
                GB_macrofy_defn (fp, 2, "cacosf", GB_cacosf_DEFN) ;
                break ;

            case  57 : f = "z = cacos (x)" ;
                GB_macrofy_defn (fp, 2, "cacos", GB_cacos_DEFN) ;
                break ;

            case  58 : f = "z = atanf (x)" ;        break ;
            case  59 : f = "z = atan (x)" ;         break ;

            case  60 : f = "z = catanf (x)" ;
                GB_macrofy_defn (fp, 2, "catanf", GB_catanf_DEFN) ;
                break ;

            case  61 : f = "z = catan (x)" ;
                GB_macrofy_defn (fp, 2, "catan", GB_catan_DEFN) ;
                break ;

            case  62 : f = "z = sinhf (x)" ;        break ;
            case  63 : f = "z = sinh (x)" ;         break ;

            case  64 : f = "z = csinhf (x)" ;
                GB_macrofy_defn (fp, 2, "csinhf", GB_csinhf_DEFN) ;
                break ;

            case  65 : f = "z = csinh (x)" ;
                GB_macrofy_defn (fp, 2, "csinh", GB_csinh_DEFN) ;
                break ;

            case  66 : f = "z = coshf (x)" ;        break ;
            case  67 : f = "z = cosh (x)" ;         break ;

            case  68 : f = "z = ccoshf (x)" ;
                GB_macrofy_defn (fp, 2, "ccoshf", GB_ccoshf_DEFN) ;
                break ;

            case  69 : f = "z = ccosh (x)" ;
                GB_macrofy_defn (fp, 2, "ccosh", GB_ccosh_DEFN) ;
                break ;

            case  70 : f = "z = tanhf (x)" ;        break ;
            case  71 : f = "z = tanh (x)" ;         break ;

            case  72 : f = "z = ctanhf (x)" ;
                GB_macrofy_defn (fp, 2, "ctanhf", GB_ctanhf_DEFN) ;
                break ;

            case  73 : f = "z = ctanh (x)" ;
                GB_macrofy_defn (fp, 2, "ctanh", GB_ctanh_DEFN) ;
                break ;

            case  74 : f = "z = asinhf (x)" ;       break ;
            case  75 : f = "z = asinh (x)" ;        break ;

            case  76 : f = "z = casinhf (x)" ;
                GB_macrofy_defn (fp, 2, "casinhf", GB_casinhf_DEFN) ;
                break ;

            case  77 : f = "z = casinh (x)" ;
                GB_macrofy_defn (fp, 2, "casinh", GB_casinh_DEFN) ;
                break ;

            case  78 : f = "z = acoshf (x)" ;       break ;
            case  79 : f = "z = acosh (x)" ;        break ;

            case  80 : f = "z = cacoshf (x)" ;
                GB_macrofy_defn (fp, 2, "cacoshf", GB_cacoshf_DEFN) ;
                break ;

            case  81 : f = "z = cacosh (x)" ;
                GB_macrofy_defn (fp, 2, "cacosh", GB_cacosh_DEFN) ;
                break ;

            case  82 : f = "z = atanhf (x)" ;       break ;
            case  83 : f = "z = atanh (x)" ;        break ;

            case  84 : f = "z = catanhf (x)" ;
                GB_macrofy_defn (fp, 2, "catanhf", GB_catanhf_DEFN) ;
                break ;

            case  85 : f = "z = catanh (x)" ;
                GB_macrofy_defn (fp, 2, "catanh", GB_catanh_DEFN) ;
                break ;

            case  86 : f = "z = GB_signumf (x)" ;
                GB_macrofy_defn (fp, 0, "GB_signumf", GB_signumf_DEFN) ;
                break ;

            case  87 : f = "z = GB_signum (x)" ;
                GB_macrofy_defn (fp, 0, "GB_signum", GB_signum_DEFN) ;
                break ;

            case  88 : f = "z = GB_csignumf (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_csignumf", GB_csignumf_DEFN) ;
                break ;

            case  89 : f = "z = GB_csignum (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_csignum", GB_csignum_DEFN) ;
                break ;

            case  90 : f = "z = ceilf (x)" ;        break ;
            case  91 : f = "z = ceil (x)" ;         break ;

            case  92 : f = "z = GB_cceilf (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cceilf", GB_cceilf_DEFN) ;
                break ;

            case  93 : f = "z = GB_cceil (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cceil", GB_cceil_DEFN) ;
                break ;

            case  94 : f = "z = floorf (x)" ;       break ;
            case  95 : f = "z = floor (x)" ;        break ;

            case  96 : f = "z = GB_cfloorf (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cfloorf", GB_cfloorf_DEFN) ;
                break ;

            case  97 : f = "z = GB_cfloor (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cfloor", GB_cfloor_DEFN) ;
                break ;

            case  98 : f = "z = roundf (x)" ;       break ;
            case  99 : f = "z = round (x)" ;        break ;

            case 100 : f = "z = GB_croundf (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_croundf", GB_croundf_DEFN) ;
                break ;

            case 101 : f = "z = GB_cround (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cround", GB_cround_DEFN) ;
                break ;

            case 102 : f = "z = truncf (x)" ;       break ;
            case 103 : f = "z = trunc (x)" ;        break ;

            case 104 : f = "z = GB_ctruncf (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_ctruncf", GB_ctruncf_DEFN) ;
                break ;

            case 105 : f = "z = GB_ctrunc (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_ctrunc", GB_ctrunc_DEFN) ;
                break ;

            case 106 : f = "z = exp2f (x)" ;        break ;
            case 107 : f = "z = exp2 (x)" ;         break ;

            case 108 : f = "z = GB_cexp2f (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cpowf", GB_cpowf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_powf", GB_powf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC32_pow", GB_FC32_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cexp2f", GB_cexp2f_DEFN) ;
                break ;

            case 109 : f = "z = GB_cexp2 (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "cpow", GB_cpow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_pow", GB_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_pow", GB_FC64_pow_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cexp2", GB_cexp2_DEFN) ;
                break ;

            case 110 : f = "z = expm1f (x)" ;       break ;
            case 111 : f = "z = expm1 (x)" ;        break ;

            case 112 : f = "z = GB_cexpm1f (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "cexp", GB_cexp_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cexpm1", GB_cexpm1_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cexpm1f", GB_cexpm1f_DEFN) ;
                break ;

            case 113 : f = "z = GB_cexpm1 (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "cexp", GB_cexp_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cexpm1", GB_cexpm1_DEFN) ;
                break ;

            case 114 : f = "z = log10f (x)" ;       break ;
            case 115 : f = "z = log10 (x)" ;        break ;

            case 116 : f = "z = GB_clog10f (x)" ;
                GB_macrofy_defn (fp, 2, "clogf", GB_clogf_DEFN) ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_clog10f", GB_clog10f_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC32_div", GB_FC32_div_DEFN) ;
                break ;


            case 117 : f = "z = GB_clog10 (x)" ;
                GB_macrofy_defn (fp, 2, "clog", GB_clog_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_clog10", GB_clog10_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                break ;

            case 118 : f = "z = log1pf (x)" ;       break ;

            case 119 : f = "z = log1p (x)" ;        break ;

            case 120 : f = "z = GB_clog1pf (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "clog", GB_clog_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_clog1p", GB_clog1p_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_clog1pf", GB_clog1pf_DEFN) ;
                break ;

            case 121 : f = "z = GB_clog1p (x)" ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "clog", GB_clog_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_clog1p", GB_clog1p_DEFN) ;
                break ;

            case 122 : f = "z = log2f (x)" ;        break ;
            case 123 : f = "z = log2 (x)" ;         break ;

            case 124 : f = "z = GB_clog2f (x)" ;
                GB_macrofy_defn (fp, 2, "clogf", GB_clogf_DEFN) ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC32_div", GB_FC32_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_clog2f", GB_clog2f_DEFN) ;
                break ;

            case 125 : f = "z = GB_clog2 (x)" ;
                GB_macrofy_defn (fp, 2, "clog", GB_clog_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_FC64_div", GB_FC64_div_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_clog2", GB_clog2_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // unary operators for real floating-point types
            //------------------------------------------------------------------

            case 126 : f = "z = lgammaf (x)" ;      break ;
            case 127 : f = "z = lgamma (x)" ;       break ;

            case 128 : f = "z = tgammaf (x)" ;      break ;
            case 129 : f = "z = tgamma (x)" ;       break ;

            case 130 : f = "z = erff (x)" ;         break ;
            case 131 : f = "z = erf (x)" ;          break ;

            case 132 : f = "z = erfcf (x)" ;        break ;
            case 133 : f = "z = erfc (x)" ;         break ;

            case 134 : f = "z = cbrtf (x)" ;        break ;
            case 135 : f = "z = cbrt (x)" ;         break ;

            case 136 : f = "z = GB_frexpxf (x)" ;
                GB_macrofy_defn (fp, 0, "GB_frexpxf", GB_frexpxf_DEFN) ;
                break ;

            case 137 : f = "z = GB_frexpx (x)" ;
                GB_macrofy_defn (fp, 0, "GB_frexpx", GB_frexpx_DEFN) ;
                break ;

            case 138 : f = "z = GB_frexpef (x)" ;
                GB_macrofy_defn (fp, 0, "GB_frexpef", GB_frexpef_DEFN) ;
                break ;

            case 139 : f = "z = GB_frexpe (x)" ;
                GB_macrofy_defn (fp, 0, "GB_frexpe", GB_frexpe_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // unary operators for complex types only
            //------------------------------------------------------------------

            case 140 : f = "z = conjf (x)" ;
                GB_macrofy_defn (fp, 2, "conjf", GB_conjf_DEFN) ;
                break ;

            case 141 : f = "z = conj (x)" ;
                GB_macrofy_defn (fp, 2, "conj", GB_conj_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // unary operators where z is real and x is complex
            //------------------------------------------------------------------

            case 142 : f = "z = crealf (x)" ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                break ;

            case 143 : f = "z = creal (x)" ; 
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                break ;

            case 144 : f = "z = cimagf (x)" ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                break ;

            case 145 : f = "z = cimag (x)" ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                break ;

            case 146 : f = "z = cargf (x)" ;
                GB_macrofy_defn (fp, 2, "cargf", GB_cargf_DEFN) ;
                break ;

            case 147 : f = "z = carg (x)" ;
                GB_macrofy_defn (fp, 2, "carg", GB_carg_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // unary operators where z is bool and x is any floating-point type
            //------------------------------------------------------------------

            case 148 : f = "z = isinf (x)" ;            break ;

            case 149 : f = "z = GB_cisinff (x)" ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cisinff", GB_cisinff_DEFN) ;
                break ;

            case 150 : f = "z = GB_cisinf (x)" ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cisinf", GB_cisinf_DEFN) ;
                break ;

            case 151 : f = "z = isnan (x)" ;            break ;

            case 152 : f = "z = GB_cisnanf (x)" ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cisnanf", GB_cisnanf_DEFN) ;
                break ;

            case 153 : f = "z = GB_cisnan (x)" ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cisnan", GB_cisnan_DEFN) ;
                break ;

            case 154 : f = "z = isfinite (x)" ;         break ;

            case 155 : f = "z = GB_cisfinitef (x)" ;
                GB_macrofy_defn (fp, 2, "cimagf", GB_cimagf_DEFN) ;
                GB_macrofy_defn (fp, 2, "crealf", GB_crealf_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cisfinitef", GB_cisfinitef_DEFN) ;
                break ;

            case 156 : f = "z = GB_cisfinite (x)" ;
                GB_macrofy_defn (fp, 2, "cimag", GB_cimag_DEFN) ;
                GB_macrofy_defn (fp, 2, "creal", GB_creal_DEFN) ;
                GB_macrofy_defn (fp, 0, "GB_cisfinite", GB_cisfinite_DEFN) ;
                break ;

            //------------------------------------------------------------------
            // positional unary operators: z is int32 or int64, x is ignored
            //------------------------------------------------------------------

            case 157 : f = "z = (i)" ;                  break ;
            case 158 : f = "z = (i) + 1" ;              break ;
            case 159 : f = "z = (j)" ;                  break ;
            case 160 : f = "z = (j) + 1" ;              break ;

            //------------------------------------------------------------------
            // IndexUnaryOps
            //------------------------------------------------------------------

            case 232 : f = "z = (i >= 0)" ;             break ;

            case 233 : f = "z = ((i) + (y))" ;          break ;
            case 234 : f = "z = ((i) <= (y))" ;         break ;
            case 235 : f = "z = ((i) > (y))" ;          break ;

            case 236 : f = "z = ((j) + (y))" ;          break ;
            case 237 : f = "z = ((j) <= (y))" ;         break ;
            case 238 : f = "z = ((j) > (y))" ;          break ;

            case 239 : f = "z = ((j) - ((i) + (y)))" ;  break ;
            case 240 : f = "z = ((i) - ((j) + (y)))" ;  break ;
            case 241 : f = "z = ((j) <= ((i) + (y)))" ; break ;
            case 242 : f = "z = ((j) >= ((i) + (y)))" ; break ;
            case 243 : f = "z = ((j) == ((i) + (y)))" ; break ;
            case 244 : f = "z = ((j) != ((i) + (y)))" ; break ;

            case 245 : f = "z = GB_FC32_ne (x,y)" ;     break ;
            case 246 : f = "z = GB_FC64_ne (x,y)" ;     break ;
            case 247 : f = "z = ((x) != (y))" ;         break ;

            case 248 : f = "z = GB_FC32_eq (x,y)" ;     break ;
            case 249 : f = "z = GB_FC64_eq (x,y)" ;     break ;
            case 250 : f = "z = ((x) == (y))" ;         break ;

            case 251 : f = "z = ((x) > (y))" ;          break ;
            case 252 : f = "z = ((x) >= (y))" ;         break ;
            case 253 : f = "z = ((x) < (y))" ;          break ;
            case 254 : f = "z = ((x) <= (y))" ;         break ;

            default: ;
        }

        //----------------------------------------------------------------------
        // create the macro
        //----------------------------------------------------------------------

        fprintf (fp, "#define %s(z,x,%s,y) %s\n", macro_name, ij, f) ;
    }
}

