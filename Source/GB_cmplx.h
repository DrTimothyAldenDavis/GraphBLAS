//------------------------------------------------------------------------------
// GB_cmplx.h: definitions for complex constructors
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_CMPLX_H
#define GB_CMPLX_H

#if defined ( __cplusplus ) || GB_COMPILER_MSC || defined (CMPLX)

    // The GxB_CMPLX* macros defined in GraphBLAS.h do no flops so they are
    // safe to use if the inputs are Inf or NaN.

    #define GB_cmplxf(xreal,ximag) GxB_CMPLXF (xreal, ximag)
    #define GB_cmplx(xreal,ximag)  GxB_CMPLX  (xreal, ximag)

#else

    // gcc on the Mac does not define the CMPLX and CMPLXF macros.  The macros
    // defined in GraphBLAS.h do arithmetic, so they are not safe with Inf or
    // NaN.

    inline GxB_FC32_t GB_cmplxf (float xreal, float ximag)
    {
        float z [2] ;
        z [0] = xreal ;
        z [1] = ximag ;
        return (* ((GxB_FC32_t *) z)) ;
    }

    inline GxB_FC64_t GB_cmplx (double xreal, double ximag)
    {
        double z [2] ;
        z [0] = xreal ;
        z [1] = ximag ;
        return (* ((GxB_FC64_t *) z)) ;
    }

#endif
#endif

