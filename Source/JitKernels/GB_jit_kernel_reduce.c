//------------------------------------------------------------------------------
// GB_jit_kernel_reduce.c: JIT kernel for reduction to scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GB_jitifyer constructs a *.c file with macro definitions specific to the
// problem instance, such as the excerts for the GB_jit_reduce_2c1fbb2 kernel,
// below, which a kernel that computes the scalar reduce of a double matrix in
// bitmap form, using the GrB_PLUS_FP64_MONOID.  The code 2c1fbb2 is computed
// by GB_enumify_reduce.  The macros are followed by an #include with this
// file, to define the kernel routine itself.  The kernel is always called
// GB_jit_kernel, regardless of what it computes.

#if comments

        // example file: GB_jit_reduce_2c1fbb2.c

        #include "GB_jit_kernel_reduce.h"   // for all JIT reduce kernels

        // monoid: (plus, double)

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

        // panel size for reduction:
        #define GB_PANEL 16

        // reduction kernel
        #include "GB_jit_kernel_reduce.c"

#endif

//------------------------------------------------------------------------------
// reduce to a non-iso matrix to scalar, for monoids only
//------------------------------------------------------------------------------

// The two template files GB_reduce_to_scalar_template.c and GB_reduce_panel.c
// appear in GraphBLAS/Source/Template.  They are used by both the pre-compiled
// kernels in GraphBLAS/Source/Generated*, and by the JIT kernel here.

GrB_Info GB_jit_kernel
(
    GB_Z_TYPENAME *result,
    const GrB_Matrix A,
    GB_Z_TYPENAME *restrict W,
    bool *restrict F,
    int ntasks,
    int nthreads
) ;

GrB_Info GB_jit_kernel
(
    GB_Z_TYPENAME *result,
    const GrB_Matrix A,
    GB_Z_TYPENAME *restrict W,
    bool *restrict F,
    int ntasks,
    int nthreads
)
{ 
    GB_Z_TYPENAME z = (*result) ;
    #if GB_A_HAS_ZOMBIES || GB_A_IS_BITMAP || (GB_PANEL == 1)
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

