//------------------------------------------------------------------------------
// GB_nnz.h: number of entries in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_NNZ_H
#define GB_NNZ_H

//------------------------------------------------------------------------------
// macro for multiplying two integer values safely; result z is int64_t
//------------------------------------------------------------------------------

// This macro computes the same thing as GB_uint64_multiply, except that it
// does not return a boolean ok flag to indicate whether or not integer
// overflow is detected.  Instead, it just computes c as INT64_MAX if overflow
// occurs.  Both inputs x and y must be >= 0.

#define GB_INT64_MULT(z,x,y)                            \
{                                                       \
    uint64_t a = (uint64_t) (x) ;                       \
    uint64_t b = (uint64_t) (y) ;                       \
    if (a == 0 || b == 0)                               \
    {                                                   \
        (z) = 0 ;                                       \
    }                                                   \
    else                                                \
    {                                                   \
        uint64_t a1 = a >> 30 ;                         \
        uint64_t b1 = b >> 30 ;                         \
        if (a1 > 0 && b1 > 0)                           \
        {                                               \
            (z) = INT64_MAX ;                           \
        }                                               \
        else                                            \
        {                                               \
            uint64_t a0 = a & 0x3FFFFFFFL ;             \
            uint64_t b0 = b & 0x3FFFFFFFL ;             \
            uint64_t t0 = a1*b0 ;                       \
            uint64_t t1 = a0*b1 ;                       \
            if (t0 >= 0x40000000L || t1 >= 0x40000000L) \
            {                                           \
                (z) = INT64_MAX ;                       \
            }                                           \
            else                                        \
            {                                           \
                uint64_t t2 = t0 + t1 ;                 \
                uint64_t c = (t2 << 30) + a0*b0 ;       \
                (z) = (int64_t) c ;                     \
            }                                           \
        }                                               \
    }                                                   \
}

//------------------------------------------------------------------------------
// nnz functions
//------------------------------------------------------------------------------

#ifdef GB_CUDA_KERNEL

    //--------------------------------------------------------------------------
    // create static inline device functions for the GPU
    //--------------------------------------------------------------------------

    #include "GB_uint64_multiply.h"
    #include "GB_size_t_multiply.h"
    #include "GB_int64_multiply.h"
    #include "GB_nnz_full_template.c"
    #include "GB_nnz_held_template.c"
    #include "GB_nnz_max_template.c"
    #include "GB_nnz_template.c"

#else

    //--------------------------------------------------------------------------
    // declare the regular functions for the CPU
    //--------------------------------------------------------------------------

    // GB_nnz(A): # of entries in any matrix: includes zombies for hypersparse
    // and sparse, but excluding entries flagged as not present in a bitmap.
    int64_t GB_nnz (GrB_Matrix A) ;

    // GB_nnz_full(A): # of entries in A if A is full
    int64_t GB_nnz_full (GrB_Matrix A) ;

    // GB_nnz_held(A): # of entries held in the data structure, including
    // zombies and all entries in a bitmap.  For hypersparse, sparse, and full,
    // GB_nnz(A) and GB_nnz_held(A) are the same.  For bitmap, GB_nnz_held(A)
    // is the same as the # of entries in a full matrix (# rows times #
    // columns).
    int64_t GB_nnz_held (GrB_Matrix A) ;

    // GB_nnz_max(A): max number of entries that can be held in a matrix.
    // For iso full matrices, GB_nnz_max(A) can be less than GB_nnz_full(A),
    // and is typically 1.
    int64_t GB_nnz_max (GrB_Matrix A) ;

#endif

// Upper bound on nnz(A) when the matrix has zombies and pending tuples;
// does not need GB_MATRIX_WAIT(A) first.
#define GB_NNZ_UPPER_BOUND(A) \
    (GB_nnz ((GrB_Matrix) A) - (A)->nzombies + GB_Pending_n (A))

#endif
