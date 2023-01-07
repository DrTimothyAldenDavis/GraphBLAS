//------------------------------------------------------------------------------
// GB_cmplx.h: definitions for complex constructors
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CMPLX_H
#define GB_CMPLX_H

#if defined ( __cplusplus )                                             \
    || ( _MSC_VER && !(__INTEL_COMPILER || __INTEL_CLANG_COMPILER) )    \
    || !(defined (CMPLX) && defined (CMPLX))

    // The GxB_CMPLX* macros defined in GraphBLAS.h do no flops so they are
    // safe to use if the inputs are Inf or NaN.

    #define GB_CMPLX32(xreal,ximag) GxB_CMPLXF (xreal, ximag)
    #define GB_CMPLX64(xreal,ximag) GxB_CMPLX  (xreal, ximag)

#else

    // gcc on the Mac does not define the CMPLX and CMPLXF macros.  The macros
    // defined in GraphBLAS.h do arithmetic, so they are not safe with Inf or
    // NaN.

    #define GB_CMPLX32(xreal,ximag) GB_complexf (xreal, ximag)
    #define GB_CMPLX64(xreal,ximag) GB_complex  (xreal, ximag)

#endif

inline GxB_FC32_t GB_complexf (float xreal, float ximag)
{
    float z [2] ;
    z [0] = xreal ;
    z [1] = ximag ;
    return (* ((GxB_FC32_t *) z)) ;
}

inline GxB_FC64_t GB_complex (double xreal, double ximag)
{
    double z [2] ;
    z [0] = xreal ;
    z [1] = ximag ;
    return (* ((GxB_FC64_t *) z)) ;
}

#endif
