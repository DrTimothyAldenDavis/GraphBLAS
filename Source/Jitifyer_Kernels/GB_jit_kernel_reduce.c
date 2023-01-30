//------------------------------------------------------------------------------
// GB_jit_reduce.c:  JIT kernel for reduction to scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GB_jitifyer first constructs a header file with macro definitions
// specific to the problem instance, such as one of the following:
//
//      GB_jit_reduce__03feee0_complex_plus_mycomplex_mycomplex.h
//      GB_jit_reduce_2c1fbb2.h         <---- name of the constructed *.h file
//                                            (see example below)
//
// The first instance handles a user-defined type and/or operator, while the
// 2nd instance uses only built-in types and operators.  The file is #include'd
// into a second short file constructed by the GB_jitifyer_*:
//
//      // GB_jit_reduce_2c1fbb2.c:         // name of the constructed *.c file
//      #include "GB_jit_reduce.h"          // for all JIT reduce kernels
//      #define GB_JIT_KERNEL GB_jit_reduce_2c1fbb2
//      #include "GB_jit_reduce_2c1fbb2.h"  // example below
//      #include "GB_jit_reduce.c"          // this file
//
// GB_jit_reduce_2c1fbb2.h contains the following definitions for reduction of
// a non-iso bitmap matrix to a scalar, using the (plus, double) monoid:

#if comments
  
        // monoid type:
        #define GB_Z_TYPENAME double
  
        // reduction monoid:
        #define GB_ADD(z,x,y) z = (x) + (y)
        #define GB_UPDATE(z,y) z += (y)
        #define GB_DECLARE_MONOID_IDENTITY(z) double z = (double) (0) ;
        #define GB_IS_ANY_MONOID 0
        #define GB_MONOID_IS_TERMINAL 0
        #define GB_DECLARE_MONOID_TERMINAL(zterminal)
        #define GB_TERMINAL_CONDITION(z,zterminal) (false)
        #define GB_IF_TERMINAL_BREAK(z,zterminal)
        #define GB_GETA_AND_UPDATE(z,Ax,p) \
            GB_UPDATE(z, Ax [p]) ;    // z += Ax [p]
  
        // A matrix:
        #define GB_A_IS_PATTERN 0
        #define GB_A_ISO 0
        #define GB_A_HAS_ZOMBIES 0
        #define GB_A_IS_HYPER  0
        #define GB_A_IS_SPARSE 0
        #define GB_A_IS_BITMAP 1
        #define GB_A_IS_FULL   0
        #define GB_A_TYPENAME double
        #define GB_DECLAREA(a) double a
        #define GB_GETA(a,Ax,p,iso) a = (Ax [p])

        // not defined in GB_jit_reduce_2c1fbb2.h (see GB_reduce_panel.h):
        // #define GB_PANEL

#endif

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

// The two template files GB_reduce_to_scalar_template.c and GB_reduce_panel.c
// appear in GraphBLAS/Source/Template.  They are used by both the pre-compiled
// kernels in GraphBLAS/Source/Generated*, and by the JIT kernel here.

GrB_Info GB_JIT_KERNEL
(
    GB_Z_TYPENAME *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB_JIT_KERNEL
(
    GB_Z_TYPENAME *result,
    const GrB_Matrix A,
    GB_void *restrict W_space,
    bool *restrict F,
    int ntasks,
    int nthreads
)
{ 
    GB_Z_TYPENAME z = (*result) ;
    GB_Z_TYPENAME *restrict W = (GB_Z_TYPENAME *) W_space ;
    #if GB_A_HAS_ZOMBIES || GB_A_IS_BITMAP
    {
        #include "GB_reduce_to_scalar_template.c"
    }
    #else
    {
        #include "GB_reduce_panel.c"
    }
    #endif
    (*result) = z ;
    return (GrB_SUCCESS) ;
}

