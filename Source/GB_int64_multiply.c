//------------------------------------------------------------------------------
// GB_int64_multiply:  multiply two integers and guard against overflow
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// c = a*b where c is GrB_Index (uint64_t), and a and b are int64_t.
// Check for overflow.  Requires a >= 0 and b >= 0.

#ifdef GB_CUDA_KERNEL
__device__ static inline
#define restrict __restrict__

#include "../GB_index.h"

// "do not FIXME": This should probably be pulled out of graphblas.h into its
// own file.  comment by Tim: no, it has to be duplicated here since
// GraphBLAS.h need to be self-contained.
#define GrB_INDEX_MAX ((GrB_Index) (1ULL << 60) - 1)

#else
#include "GB.h"
#endif

bool GB_int64_multiply      // true if ok, false if overflow
(
    GrB_Index *restrict c,  // c = a*b, or zero if overflow occurs
    const int64_t a,
    const int64_t b
)
{

    ASSERT (c != NULL) ;

    (*c) = 0 ;
    if (a == 0 || b == 0)
    { 
        return (true) ;
    }

    if (a < 0 || a > GB_NMAX || b < 0 || b > GB_NMAX)
    { 
        // a or b are out of range
        return (false) ;
    }

#ifndef GB_CUDA_KERNEL // FIXME
    if (GB_CEIL_LOG2 (a) + GB_CEIL_LOG2 (b) > 60)
    { 
        // a * b may overflow
        return (false) ;
    }
#endif

    // a * b will not overflow
    (*c) = a * b ;
    return (true) ;
}

