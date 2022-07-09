//------------------------------------------------------------------------------
// CUDA/GB_cuda_kernel.h: definitions for all GraphBLAS CUDA kernels
//------------------------------------------------------------------------------

// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is #include'd into all CUDA kernels for GraphBLAS.  It provides
// a

#pragma once
#undef  ASSERT
#define ASSERT(x)

//------------------------------------------------------------------------------
// TODO: this will be in the jit code:
#define chunksize 128 

//------------------------------------------------------------------------------
// GETA, GETB: get entries from input matrices A and B
//------------------------------------------------------------------------------

#if GB_FLIPXY

    #if GB_A_IS_PATTERN
        #define GB_DECLAREA(aval)
        #define GB_SHAREDA(aval)
        #define GB_GETA( aval, ax, p)
    #else
        #define GB_DECLAREA(aval) T_Y aval
        #define GB_SHAREDA(aval) __shared__ T_Y aval
        #if GB_A_ISO
            #define GB_GETA( aval, ax, p) aval = (T_Y) (ax [0]) ;
        #else
            #define GB_GETA( aval, ax, p) aval = (T_Y) (ax [p]) ;
        #endif
    #endif

    #if GB_B_IS_PATTERN
        #define GB_DECLAREB(bval)
        #define GB_SHAREDB(bval)
        #define GB_GETB( bval, bx, p)
    #else
        #define GB_DECLAREB(bval) T_X bval
        #define GB_SHAREDB(bval) __shared__ T_X bval
        #if GB_B_ISO
            #define GB_GETB( bval, bx, p) bval = (T_X) (bx [0]) ;
        #else
            #define GB_GETB( bval, bx, p) bval = (T_X) (bx [p]) ;
        #endif
    #endif

#else

    #if GB_A_IS_PATTERN
        #define GB_DECLAREA(aval)
        #define GB_SHAREDA(aval)
        #define GB_GETA( aval, ax, p)
    #else
        #define GB_DECLAREA(aval) T_X aval
        #define GB_SHAREDA(aval) __shared__ T_X aval
        #if GB_A_ISO
            #define GB_GETA( aval, ax, p) aval = (T_X) (ax [0]) ;
        #else
            #define GB_GETA( aval, ax, p) aval = (T_X) (ax [p]) ;
        #endif
    #endif

    #if GB_B_IS_PATTERN
        #define GB_DECLAREB(bval)
        #define GB_SHAREDB(bval)
        #define GB_GETB( bval, bx, p)
    #else
        #define GB_DECLAREB(bval) T_Y bval
        #define GB_SHAREDB(bval) __shared__ T_Y bval
        #if GB_B_ISO
            #define GB_GETB( bval, bx, p) bval = (T_Y) (bx [0]) ;
        #else
            #define GB_GETB( bval, bx, p) bval = (T_Y) (bx [p]) ;
        #endif
    #endif

#endif

//------------------------------------------------------------------------------
// operators
//------------------------------------------------------------------------------

#if GB_C_ISO

    #define GB_ADD_F( f , s)
    #define GB_C_MULT( c, a, b)
    #define GB_MULTADD( c, a ,b )
    #define GB_DOT_TERMINAL ( c )   
    #define GB_DOT_MERGE(pA,pB)                                         \
    {                                                                   \
        cij_exists = true ;                                             \
    }
    #define GB_CIJ_EXIST_POSTCHECK

#else

    #define GB_ADD_F( f , s)  f = GB_ADD ( f, s ) 
    #define GB_C_MULT( c, a, b)  c = GB_MULT( (a), (b) )
    #define GB_MULTADD( c, a ,b ) GB_ADD_F( (c), GB_MULT( (a),(b) ) )
    #define GB_DOT_TERMINAL ( c )
    //# if ( c == TERMINAL_VALUE) break;

    #if GB_IS_PLUS_PAIR_REAL_SEMIRING

        // cij += A(k,i) * B(k,j), for merge operation (plus_pair_real semiring)
        #if GB_ZTYPE_IGNORE_OVERFLOW
            // plus_pair for int64, uint64, float, or double
            #define GB_DOT_MERGE(pA,pB) cij++ ;
            #define GB_CIJ_EXIST_POSTCHECK cij_exists = (cij != 0) ;
        #else
            // plus_pair semiring for small integers
            #define GB_DOT_MERGE(pA,pB)                                     \
            {                                                               \
                cij_exists = true ;                                         \
                cij++ ;                                                     \
            }
            #define GB_CIJ_EXIST_POSTCHECK
        #endif

    #else

        // cij += A(k,i) * B(k,j), for merge operation (general case)
        #define GB_DOT_MERGE(pA,pB)                                         \
        {                                                                   \
            GB_GETA (aki, Ax, pA) ;         /* aki = A(k,i) */              \
            GB_GETB (bkj, Bx, pB) ;         /* bkj = B(k,j) */              \
            cij_exists = true ;                                             \
            GB_MULTADD (cij, aki, bkj) ;    /* cij += aki * bkj */          \
        }
        #define GB_CIJ_EXIST_POSTCHECK

    #endif

#endif

//------------------------------------------------------------------------------
// subset of GraphBLAS.h
//------------------------------------------------------------------------------

#ifndef GRAPHBLAS_H
#define GRAPHBLAS_H

#undef restrict
#undef GB_restrict
#if defined ( GB_CUDA_KERNEL ) || defined ( __NVCC__ )
    #define GB_restrict __restrict__
#else
    #define GB_restrict
#endif
#define restrict GB_restrict

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

// GB_STR: convert the content of x into a string "x"
#define GB_XSTR(x) GB_STR(x)
#define GB_STR(x) #x

#undef  GB_PUBLIC
#define GB_PUBLIC extern
#undef  GxB_MAX_NAME_LEN
#define GxB_MAX_NAME_LEN 128

typedef uint64_t GrB_Index ;
typedef struct GB_Descriptor_opaque *GrB_Descriptor ;
typedef struct GB_Type_opaque *GrB_Type ;
typedef struct GB_UnaryOp_opaque *GrB_UnaryOp ;
typedef struct GB_BinaryOp_opaque *GrB_BinaryOp ;
typedef struct GB_SelectOp_opaque *GxB_SelectOp ;
typedef struct GB_IndexUnaryOp_opaque *GrB_IndexUnaryOp ;
typedef struct GB_Monoid_opaque *GrB_Monoid ;
typedef struct GB_Semiring_opaque *GrB_Semiring ;
typedef struct GB_Scalar_opaque *GrB_Scalar ;
typedef struct GB_Vector_opaque *GrB_Vector ;
typedef struct GB_Matrix_opaque *GrB_Matrix ;

#define GxB_HYPERSPARSE 1   // store matrix in hypersparse form
#define GxB_SPARSE      2   // store matrix as sparse form (compressed vector)
#define GxB_BITMAP      4   // store matrix as a bitmap
#define GxB_FULL        8   // store matrix as full; all entries must be present

typedef void (*GxB_unary_function)  (void *, const void *) ;
typedef void (*GxB_binary_function) (void *, const void *, const void *) ;

typedef bool (*GxB_select_function)      // return true if A(i,j) is kept
(
    GrB_Index i,                // row index of A(i,j)
    GrB_Index j,                // column index of A(i,j)
    const void *x,              // value of A(i,j)
    const void *thunk           // optional input for select function
) ;

typedef void (*GxB_index_unary_function)
(
    void *z,            // output value z, of type ztype
    const void *x,      // input value x of type xtype; value of v(i) or A(i,j)
    GrB_Index i,        // row index of A(i,j)
    GrB_Index j,        // column index of A(i,j), or zero for v(i)
    const void *y       // input scalar y
) ;

typedef enum
{
    // for all GrB_Descriptor fields:
    GxB_DEFAULT = 0,    // default behavior of the method

    // for GrB_OUTP only:
    GrB_REPLACE = 1,    // clear the output before assigning new values to it

    // for GrB_MASK only:
    GrB_COMP = 2,       // use the structural complement of the input
    GrB_SCMP = 2,       // same as GrB_COMP (historical; use GrB_COMP instead)
    GrB_STRUCTURE = 4,  // use the only pattern of the mask, not its values

    // for GrB_INP0 and GrB_INP1 only:
    GrB_TRAN = 3,       // use the transpose of the input

    // for GxB_GPU_CONTROL only (DRAFT: in progress, do not use)
    GxB_GPU_ALWAYS  = 2001,
    GxB_GPU_NEVER   = 2002,

    // for GxB_AxB_METHOD only:
    GxB_AxB_GUSTAVSON = 1001,   // gather-scatter saxpy method
    GxB_AxB_DOT       = 1003,   // dot product
    GxB_AxB_HASH      = 1004,   // hash-based saxpy method
    GxB_AxB_SAXPY     = 1005    // saxpy method (any kind)
}
GrB_Desc_Value ;

#include "GB_opaque.h"
#endif

//------------------------------------------------------------------------------
// subset of GB.h
//------------------------------------------------------------------------------

#include "GB_imin.h"
#include "GB_zombie.h"
#include "GB_nnz.h"
#include "GB_partition.h"
#include "GB_binary_search.h"
#include "GB_search_for_vector_template.c"

