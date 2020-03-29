//------------------------------------------------------------------------------
// GraphBLAS/Demo/Include/usercomplex.h:  complex numbers as a user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef USERCOMPLEX_H
#define USERCOMPLEX_H

#include "GraphBLAS.h"

// TODO HACK
#if 1
//#if ( _MSC_VER && !__INTEL_COMPILER )
// See the following link for C complex math support in Microsoft Visual Studio.
// https://docs.microsoft.com/en-us/cpp/c-runtime-library/complex-math-support?view=vs-2019
// The complex data type is not supported for this demo, in MS Visual Studio.
#define HAVE_COMPLEX 0

#else

#define HAVE_COMPLEX 1

#include <complex.h>

#ifndef CMPLX
#define CMPLX(real,imag) \
    ( \
    (double complex)((double)(real)) + \
    (double complex)((double)(imag) * _Complex_I) \
    )
#endif
#endif

// "I" is used in GraphBLAS to denote a list of row indices; remove it here
#undef I

//------------------------------------------------------------------------------
// 10 binary functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

GB_PUBLIC
GrB_BinaryOp Complex_first , Complex_second , Complex_min ,
             Complex_max   , Complex_plus   , Complex_minus ,
             Complex_times , Complex_div    , Complex_rdiv  ,
             Complex_rminus, Complex_pair ;

//------------------------------------------------------------------------------
// 6 binary comparison functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

GB_PUBLIC
GrB_BinaryOp Complex_iseq , Complex_isne ,
             Complex_isgt , Complex_islt ,
             Complex_isge , Complex_isle ;

//------------------------------------------------------------------------------
// 3 binary boolean functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

GB_PUBLIC
GrB_BinaryOp Complex_or , Complex_and , Complex_xor ;

//------------------------------------------------------------------------------
// 6 binary comparison functions, z=f(x,y), where CxC -> bool
//------------------------------------------------------------------------------

GB_PUBLIC
GrB_BinaryOp Complex_eq , Complex_ne ,
             Complex_gt , Complex_lt ,
             Complex_ge , Complex_le ;

//------------------------------------------------------------------------------
// 1 binary function, z=f(x,y), where double x double -> C
//------------------------------------------------------------------------------

GB_PUBLIC GrB_BinaryOp Complex_complex ;

//------------------------------------------------------------------------------
// 5 unary functions, z=f(x) where C -> C
//------------------------------------------------------------------------------

GB_PUBLIC
GrB_UnaryOp  Complex_identity , Complex_ainv , Complex_minv ,
             Complex_not ,      Complex_conj,
             Complex_one ,      Complex_abs  ;

//------------------------------------------------------------------------------
// 4 unary functions, z=f(x) where C -> double
//------------------------------------------------------------------------------

GB_PUBLIC 
GrB_UnaryOp Complex_real, Complex_imag,
            Complex_cabs, Complex_angle ;

//------------------------------------------------------------------------------
// 2 unary functions, z=f(x) where double -> C
//------------------------------------------------------------------------------

GB_PUBLIC GrB_UnaryOp Complex_complex_real, Complex_complex_imag ;

//------------------------------------------------------------------------------
// Complex type, scalars, monoids, and semiring
//------------------------------------------------------------------------------

GB_PUBLIC GrB_Type Complex ;

GB_PUBLIC GrB_Monoid   Complex_plus_monoid, Complex_times_monoid ;
GB_PUBLIC GrB_Semiring Complex_plus_times ;
#if HAVE_COMPLEX
GB_PUBLIC double complex Complex_1  ;
GB_PUBLIC double complex Complex_0 ;
#else
GB_PUBLIC double Complex_1  ;
GB_PUBLIC double Complex_0 ;
#endif
GrB_Info Complex_init ( ) ;
GrB_Info Complex_finalize ( ) ;

#endif

