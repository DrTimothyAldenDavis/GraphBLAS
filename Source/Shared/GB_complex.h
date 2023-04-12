//------------------------------------------------------------------------------
// GB_complex.h: definitions for complex types
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These macros allow GraphBLAS to be compiled with a C++ compiler. See:
// https://www.drdobbs.com/complex-arithmetic-in-the-intersection-o/184401628#

#ifndef GB_COMPLEX_H
#define GB_COMPLEX_H

//------------------------------------------------------------------------------
// macros for complex built-in functions
//------------------------------------------------------------------------------

#if defined ( __cplusplus ) || defined ( __NVCC__ )

    //--------------------------------------------------------------------------
    // ANSI C++ or NVCC
    //--------------------------------------------------------------------------

    #define GB_crealf(x)   std::real(x)
    #define GB_creal(x)    std::real(x)
    #define GB_cimagf(x)   std::imag(x)
    #define GB_cimag(x)    std::imag(x)
    #define GB_cpowf(x,y)  std::pow(x,y)
    #define GB_cpow(x,y)   std::pow(x,y)
    #define GB_cexpf(x)    std::exp(x)
    #define GB_cexp(x)     std::exp(x)
    #define GB_clogf(x)    std::log(x)
    #define GB_clog(x)     std::log(x)
    #define GB_cabsf(x)    std::abs(x)
    #define GB_cabs(x)     std::abs(x)
    #define GB_csqrtf(x)   std::sqrt(x)
    #define GB_csqrt(x)    std::sqrt(x)
    #define GB_conjf(x)    std::conj(x)
    #define GB_conj(x)     std::conj(x)
    #define GB_cargf(x)    std::arg(x)
    #define GB_carg(x)     std::arg(x)
    #define GB_csinf(x)    std::sin(x)
    #define GB_csin(x)     std::sin(x)
    #define GB_ccosf(x)    std::cos(x)
    #define GB_ccos(x)     std::cos(x)
    #define GB_ctanf(x)    std::tan(x)
    #define GB_ctan(x)     std::tan(x)
    #define GB_casinf(x)   std::asin(x)
    #define GB_casin(x)    std::asin(x)
    #define GB_cacosf(x)   std::acos(x)
    #define GB_cacos(x)    std::acos(x)
    #define GB_catanf(x)   std::atan(x)
    #define GB_catan(x)    std::atan(x)
    #define GB_csinhf(x)   std::sinh(x)
    #define GB_csinh(x)    std::sinh(x)
    #define GB_ccoshf(x)   std::cosh(x)
    #define GB_ccosh(x)    std::cosh(x)
    #define GB_ctanhf(x)   std::tanh(x)
    #define GB_ctanh(x)    std::tanh(x)
    #define GB_casinhf(x)  std::asinh(x)
    #define GB_casinh(x)   std::asinh(x)
    #define GB_cacoshf(x)  std::acosh(x)
    #define GB_cacosh(x)   std::acosh(x)
    #define GB_catanhf(x)  std::atanh(x)
    #define GB_catanh(x)   std::atanh(x)

#else

    //--------------------------------------------------------------------------
    // ANSI C11
    //--------------------------------------------------------------------------

    #define GB_crealf(x)   crealf(x)
    #define GB_creal(x)    creal(x)
    #define GB_cimagf(x)   cimagf(x)
    #define GB_cimag(x)    cimag(x)
    #define GB_cpowf(x,y)  cpowf(x,y)
    #define GB_cpow(x,y)   cpow(x,y)
    #define GB_cexpf(x)    cexpf(x)
    #define GB_cexp(x)     cexp(x)
    #define GB_clogf(x)    clogf(x)
    #define GB_clog(x)     clog(x)
    #define GB_cabsf(x)    cabsf(x)
    #define GB_cabs(x)     cabs(x)
    #define GB_csqrtf(x)   csqrtf(x)
    #define GB_csqrt(x)    csqrt(x)
    #define GB_conjf(x)    conjf(x)
    #define GB_conj(x)     conj(x)
    #define GB_cargf(x)    cargf(x)
    #define GB_carg(x)     carg(x)
    #define GB_csinf(x)    csinf(x)
    #define GB_csin(x)     csin(x)
    #define GB_ccosf(x)    ccosf(x)
    #define GB_ccos(x)     ccos(x)
    #define GB_ctanf(x)    ctanf(x)
    #define GB_ctan(x)     ctan(x)
    #define GB_casinf(x)   casinf(x)
    #define GB_casin(x)    casin(x)
    #define GB_cacosf(x)   cacosf(x)
    #define GB_cacos(x)    cacos(x)
    #define GB_catanf(x)   catanf(x)
    #define GB_catan(x)    catan(x)
    #define GB_csinhf(x)   csinhf(x)
    #define GB_csinh(x)    csinh(x)
    #define GB_ccoshf(x)   ccoshf(x)
    #define GB_ccosh(x)    ccosh(x)
    #define GB_ctanhf(x)   ctanhf(x)
    #define GB_ctanh(x)    ctanh(x)
    #define GB_casinhf(x)  casinhf(x)
    #define GB_casinh(x)   casinh(x)
    #define GB_cacoshf(x)  cacoshf(x)
    #define GB_cacosh(x)   cacosh(x)
    #define GB_catanhf(x)  catanhf(x)
    #define GB_catanh(x)   catanh(x)

#endif

//------------------------------------------------------------------------------
// complex macros for basic operations
//------------------------------------------------------------------------------

#if GB_COMPILER_MSC

    //--------------------------------------------------------------------------
    // Microsoft Visual Studio compiler with its own complex type
    //--------------------------------------------------------------------------

    // complex-complex multiply: z = x*y where both x and y are complex
    #define GB_FC32_mul(x,y) (_FCmulcc (x, y))
    #define GB_FC64_mul(x,y) ( _Cmulcc (x, y))

    // complex-complex addition: z = x+y where both x and y are complex
    #define GB_FC32_add(x,y) GxB_CMPLXF (GB_crealf (x) + GB_crealf (y), GB_cimagf (x) + GB_cimagf (y))
    #define GB_FC64_add(x,y) GxB_CMPLX  (GB_creal  (x) + GB_creal  (y), GB_cimag  (x) + GB_cimag  (y))

    // complex-complex subtraction: z = x-y where both x and y are complex
    #define GB_FC32_minus(x,y) GxB_CMPLXF (GB_crealf (x) - GB_crealf (y), GB_cimagf (x) - GB_cimagf (y))
    #define GB_FC64_minus(x,y) GxB_CMPLX  (GB_creal  (x) - GB_creal  (y), GB_cimag  (x) - GB_cimag  (y))

    // complex negation: z = -x
    #define GB_FC32_ainv(x) GxB_CMPLXF (-GB_crealf (x), -GB_cimagf (x))
    #define GB_FC64_ainv(x) GxB_CMPLX  (-GB_creal  (x), -GB_cimag  (x))

#else

    //--------------------------------------------------------------------------
    // native complex type support
    //--------------------------------------------------------------------------

    // complex-complex multiply: z = x*y where both x and y are complex
    #define GB_FC32_mul(x,y) ((x) * (y))
    #define GB_FC64_mul(x,y) ((x) * (y))

    // complex-complex addition: z = x+y where both x and y are complex
    #define GB_FC32_add(x,y) ((x) + (y))
    #define GB_FC64_add(x,y) ((x) + (y))

    // complex-complex subtraction: z = x-y where both x and y are complex
    #define GB_FC32_minus(x,y) ((x) - (y))
    #define GB_FC64_minus(x,y) ((x) - (y))

    // complex negation
    #define GB_FC32_ainv(x) (-(x))
    #define GB_FC64_ainv(x) (-(x))

#endif

// complex comparators
#define GB_FC32_eq(x,y) ((GB_crealf(x) == GB_crealf(y)) && (GB_cimagf(x) == GB_cimagf(y)))
#define GB_FC64_eq(x,y) ((GB_creal (x) == GB_creal (y)) && (GB_cimag (x) == GB_cimag (y)))

#define GB_FC32_ne(x,y) ((GB_crealf(x) != GB_crealf(y)) || (GB_cimagf(x) != GB_cimagf(y)))
#define GB_FC64_ne(x,y) ((GB_creal (x) != GB_creal (y)) || (GB_cimag (x) != GB_cimag (y)))

#define GB_FC32_iseq(x,y) GB_CMPLX32 ((float)  GB_FC32_eq (x,y), 0)
#define GB_FC64_iseq(x,y) GB_CMPLX64 ((double) GB_FC64_eq (x,y), 0)

#define GB_FC32_isne(x,y) GB_CMPLX32 ((float)  GB_FC32_ne (x,y), 0)
#define GB_FC64_isne(x,y) GB_CMPLX64 ((double) GB_FC64_ne (x,y), 0)

#define GB_FC32_eq0(x) ((GB_crealf (x) == 0) && (GB_cimagf (x) == 0))
#define GB_FC64_eq0(x) ((GB_creal  (x) == 0) && (GB_cimag  (x) == 0))

#define GB_FC32_ne0(x) ((GB_crealf (x) != 0) || (GB_cimagf (x) != 0))
#define GB_FC64_ne0(x) ((GB_creal  (x) != 0) || (GB_cimag  (x) != 0))

#endif

