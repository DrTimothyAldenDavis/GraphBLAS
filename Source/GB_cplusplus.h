//------------------------------------------------------------------------------
// GB_cplusplus.h: definitions for C++
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// These macros allow GraphBLAS to be compiled with a C++ compiler, and also
// define the strings required by GB_macrofy_binop and GB_macrofy_unop when
// using CUDA.  See:
// https://www.drdobbs.com/complex-arithmetic-in-the-intersection-o/184401628#

#ifndef GB_CPLUSPLUS_H
#define GB_CPLUSPLUS_H

// strings for GB_macrofy_binop and GB_macrofy_unop
#define GB_crealf_DEFN  "crealf(x)   std::real(x)"
#define GB_creal_DEFN   "creal(x)    std::real(x)"
#define GB_cimagf_DEFN  "cimagf(x)   std::imag(x)"
#define GB_cimag_DEFN   "cimag(x)    std::imag(x)"
#define GB_cpowf_DEFN   "cpowf(x,y)  std::pow(x,y)"
#define GB_cpow_DEFN    "cpow(x,y)   std::pow(x,y)"
#define GB_cexpf_DEFN   "cexpf(x)    std::exp(x)"
#define GB_cexp_DEFN    "cexp(x)     std::exp(x)"
#define GB_clogf_DEFN   "clogf(x)    std::log(x)"
#define GB_clog_DEFN    "clog(x)     std::log(x)"
#define GB_cabsf_DEFN   "cabsf(x)    std::abs(x)"
#define GB_cabs_DEFN    "cabs(x)     std::abs(x)"
#define GB_csqrtf_DEFN  "csqrtf(x)   std::sqrt(x)"
#define GB_csqrt_DEFN   "csqrt(x)    std::sqrt(x)"
#define GB_conjf_DEFN   "conjf(x)    std::conj(x)"
#define GB_conj_DEFN    "conj(x)     std::conj(x)"
#define GB_cargf_DEFN   "cargf(x)    std::arg(x)"
#define GB_carg_DEFN    "carg(x)     std::arg(x)"
#define GB_csinf_DEFN   "csinf(x)    std::sin(x)"
#define GB_csin_DEFN    "csin(x)     std::sin(x)"
#define GB_ccosf_DEFN   "ccosf(x)    std::cos(x)"
#define GB_ccos_DEFN    "ccos(x)     std::cos(x)"
#define GB_ctanf_DEFN   "ctanf(x)    std::tan(x)"
#define GB_ctan_DEFN    "ctan(x)     std::tan(x)"
#define GB_casinf_DEFN  "casinf(x)   std::asin(x)"
#define GB_casin_DEFN   "casin(x)    std::asin(x)"
#define GB_cacosf_DEFN  "cacosf(x)   std::acos(x)"
#define GB_cacos_DEFN   "cacos(x)    std::acos(x)"
#define GB_catanf_DEFN  "catanf(x)   std::atan(x)"
#define GB_catan_DEFN   "catan(x)    std::atan(x)"
#define GB_csinhf_DEFN  "csinhf(x)   std::sinh(x)"
#define GB_csinh_DEFN   "csinh(x)    std::sinh(x)"
#define GB_ccoshf_DEFN  "ccoshf(x)   std::cosh(x)"
#define GB_ccosh_DEFN   "ccosh(x)    std::cosh(x)"
#define GB_ctanhf_DEFN  "ctanhf(x)   std::tanh(x)"
#define GB_ctanh_DEFN   "ctanh(x)    std::tanh(x)"
#define GB_casinhf_DEFN "casinhf(x)  std::asinh(x)"
#define GB_casinh_DEFN  "casinh(x)   std::asinh(x)"
#define GB_cacoshf_DEFN "cacoshf(x)  std::acosh(x)"
#define GB_cacosh_DEFN  "cacosh(x)   std::acosh(x)"
#define GB_catanhf_DEFN "catanhf(x)  std::atanh(x)"
#define GB_catanh_DEFN  "catanh(x)   std::atanh(x)"

#if defined ( __cplusplus ) || defined ( __NVCC__ )

    #define GB_GUARD_crealf_DEFINED
    #define GB_GUARD_creal_DEFINED
    #define GB_GUARD_cimagf_DEFINED
    #define GB_GUARD_cimag_DEFINED
    #define GB_GUARD_cpowf_DEFINED
    #define GB_GUARD_cpow_DEFINED
    #define GB_GUARD_cexpf_DEFINED
    #define GB_GUARD_cexp_DEFINED
    #define GB_GUARD_clogf_DEFINED
    #define GB_GUARD_clog_DEFINED
    #define GB_GUARD_cabsf_DEFINED
    #define GB_GUARD_cabs_DEFINED
    #define GB_GUARD_csqrtf_DEFINED
    #define GB_GUARD_csqrt_DEFINED
    #define GB_GUARD_conjf_DEFINED
    #define GB_GUARD_conj_DEFINED
    #define GB_GUARD_cargf_DEFINED
    #define GB_GUARD_carg_DEFINED
    #define GB_GUARD_csinf_DEFINED
    #define GB_GUARD_csin_DEFINED
    #define GB_GUARD_ccosf_DEFINED
    #define GB_GUARD_ccos_DEFINED
    #define GB_GUARD_ctanf_DEFINED
    #define GB_GUARD_ctan_DEFINED
    #define GB_GUARD_casinf_DEFINED
    #define GB_GUARD_casin_DEFINED
    #define GB_GUARD_cacosf_DEFINED
    #define GB_GUARD_cacos_DEFINED
    #define GB_GUARD_casinf_DEFINED
    #define GB_GUARD_casin_DEFINED
    #define GB_GUARD_csinhf_DEFINED
    #define GB_GUARD_csinh_DEFINED
    #define GB_GUARD_ccoshf_DEFINED
    #define GB_GUARD_ccosh_DEFINED
    #define GB_GUARD_ctanhf_DEFINED
    #define GB_GUARD_ctanh_DEFINED
    #define GB_GUARD_casinhf_DEFINED
    #define GB_GUARD_casinh_DEFINED
    #define GB_GUARD_cacoshf_DEFINED
    #define GB_GUARD_cacosh_DEFINED
    #define GB_GUARD_catanhf_DEFINED
    #define GB_GUARD_catanh_DEFINED

    #define crealf(x)   std::real(x)
    #define creal(x)    std::real(x)
    #define cimagf(x)   std::imag(x)
    #define cimag(x)    std::imag(x)
    #define cpowf(x,y)  std::pow(x,y)
    #define cpow(x,y)   std::pow(x,y)
    #define cexpf(x)    std::exp(x)
    #define cexp(x)     std::exp(x)
    #define clogf(x)    std::log(x)
    #define clog(x)     std::log(x)
    #define cabsf(x)    std::abs(x)
    #define cabs(x)     std::abs(x)
    #define csqrtf(x)   std::sqrt(x)
    #define csqrt(x)    std::sqrt(x)
    #define conjf(x)    std::conj(x)
    #define conj(x)     std::conj(x)
    #define cargf(x)    std::arg(x)
    #define carg(x)     std::arg(x)
    #define csinf(x)    std::sin(x)
    #define csin(x)     std::sin(x)
    #define ccosf(x)    std::cos(x)
    #define ccos(x)     std::cos(x)
    #define ctanf(x)    std::tan(x)
    #define ctan(x)     std::tan(x)
    #define casinf(x)   std::asin(x)
    #define casin(x)    std::asin(x)
    #define cacosf(x)   std::acos(x)
    #define cacos(x)    std::acos(x)
    #define catanf(x)   std::atan(x)
    #define catan(x)    std::atan(x)
    #define csinhf(x)   std::sinh(x)
    #define csinh(x)    std::sinh(x)
    #define ccoshf(x)   std::cosh(x)
    #define ccosh(x)    std::cosh(x)
    #define ctanhf(x)   std::tanh(x)
    #define ctanh(x)    std::tanh(x)
    #define casinhf(x)  std::asinh(x)
    #define casinh(x)   std::asinh(x)
    #define cacoshf(x)  std::acosh(x)
    #define cacosh(x)   std::acosh(x)
    #define catanhf(x)  std::atanh(x)
    #define catanh(x)   std::atanh(x)

#endif
#endif

