//------------------------------------------------------------------------------
// GraphBLAS/Test/GB_mx_usercomplex.h:  complex numbers as a user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef USERCOMPLEX_H
#define USERCOMPLEX_H

//------------------------------------------------------------------------------
// 10 binary functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_first , Complex_second , Complex_min ,
             Complex_max   , Complex_plus   , Complex_minus ,
             Complex_times , Complex_div    , Complex_rdiv  ,
             Complex_rminus, Complex_pair ;

//------------------------------------------------------------------------------
// 6 binary comparators, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_iseq , Complex_isne ,
             Complex_isgt , Complex_islt ,
             Complex_isge , Complex_isle ;

//------------------------------------------------------------------------------
// 3 binary boolean functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_or , Complex_and , Complex_xor ;

//------------------------------------------------------------------------------
// 6 binary comparators, z=f(x,y), where CxC -> bool
//------------------------------------------------------------------------------

extern
GrB_BinaryOp Complex_eq , Complex_ne ,
             Complex_gt , Complex_lt ,
             Complex_ge , Complex_le ;

//------------------------------------------------------------------------------
// 1 binary function, z=f(x,y), where double x double -> C
//------------------------------------------------------------------------------

extern GrB_BinaryOp Complex_complex ;

//------------------------------------------------------------------------------
// 5 unary functions, z=f(x) where C -> C
//------------------------------------------------------------------------------

extern
GrB_UnaryOp  Complex_identity , Complex_ainv , Complex_minv ,
             Complex_not ,      Complex_conj,
             Complex_one ,      Complex_abs  ;

//------------------------------------------------------------------------------
// 4 unary functions, z=f(x) where C -> double
//------------------------------------------------------------------------------

extern 
GrB_UnaryOp Complex_real, Complex_imag,
            Complex_cabs, Complex_angle ;

//------------------------------------------------------------------------------
// 2 unary functions, z=f(x) where double -> C
//------------------------------------------------------------------------------

extern GrB_UnaryOp Complex_complex_real, Complex_complex_imag ;

//------------------------------------------------------------------------------
// Complex type, scalars, monoids, and semiring
//------------------------------------------------------------------------------

extern GrB_Type Complex ;
extern GrB_Monoid   Complex_plus_monoid, Complex_times_monoid ;
extern GrB_Semiring Complex_plus_times ;
GrB_Info Complex_init (bool builtin_complex) ;
GrB_Info Complex_finalize (void) ;

//------------------------------------------------------------------------------
// internal functions
//------------------------------------------------------------------------------

#define C GxB_FC64_t
#define X *x
#define Y *y
#define Z *z

void complex_first    (C Z, const C X, const C Y) ;
void complex_second   (C Z, const C X, const C Y) ;
void complex_pair     (C Z, const C X, const C Y) ;
void complex_plus     (C Z, const C X, const C Y) ;
void complex_minus    (C Z, const C X, const C Y) ;
void complex_rminus   (C Z, const C X, const C Y) ;
void complex_times    (C Z, const C X, const C Y) ;
void complex_div      (C Z, const C X, const C Y) ;
void complex_rdiv     (C Z, const C X, const C Y) ;
void complex_iseq     (C Z, const C X, const C Y) ;
void complex_isne     (C Z, const C X, const C Y) ;
void complex_eq       (bool Z, const C X, const C Y) ;
void complex_ne       (bool Z, const C X, const C Y) ;
void complex_complex  (C Z, const double X, const double Y) ;
void complex_one      (C Z, const C X) ;
void complex_identity (C Z, const C X) ;
void complex_ainv     (C Z, const C X) ;
void complex_minv     (C Z, const C X) ;
void complex_conj     (C Z, const C X) ;
void complex_real     (double Z, const C X) ;
void complex_imag     (double Z, const C X) ;
void complex_cabs     (double Z, const C X) ;
void complex_angle    (double Z, const C X) ;

void complex_min     (C Z, const C X, const C Y) ;
void complex_max     (C Z, const C X, const C Y) ;
void complex_isgt    (C Z, const C X, const C Y) ;
void complex_islt    (C Z, const C X, const C Y) ;
void complex_isge    (C Z, const C X, const C Y) ;
void complex_isle    (C Z, const C X, const C Y) ;
void complex_or      (C Z, const C X, const C Y) ;
void complex_and     (C Z, const C X, const C Y) ;
void complex_xor     (C Z, const C X, const C Y) ;
void complex_gt      (bool Z, const C X, const C Y) ;
void complex_lt      (bool Z, const C X, const C Y) ;
void complex_ge      (bool Z, const C X, const C Y) ;
void complex_le      (bool Z, const C X, const C Y) ;
void complex_abs     (C Z, const C X) ;
void complex_not     (C Z, const C X) ;

void complex_complex_real (C Z, const double X) ;
void complex_complex_imag (C Z, const double X) ;

#undef C
#undef X
#undef Y
#undef Z

#endif

