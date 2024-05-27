//------------------------------------------------------------------------------
// GraphBLAS/CUDA/Template/GB_h_subset.cuh: subset of GB.h
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Note the header guard is the same as GB.h:
#ifndef GB_H
#define GB_H

#include "include/GB_iceil.h"
#if 0
// from GB_iceil.h:
#define GB_ICEIL(a,b) (((a) + (b) - 1) / (b))
#endif

#include "include/GB_math_macros.h"
#if 0
// from GB_imin.h:
#define GB_IMAX(x,y) (((x) > (y)) ? (x) : (y))
#define GB_IMIN(x,y) (((x) < (y)) ? (x) : (y))
#endif

#include "include/GB_zombie.h"
#if 0
// from GB_zombie.h:
#define GB_FLIP(i)             (-(i)-2)
#define GB_IS_FLIPPED(i)       ((i) < 0)
#define GB_IS_ZOMBIE(i)        ((i) < 0)
#define GB_IS_NOT_FLIPPED(i)   ((i) >= 0)
#define GB_UNFLIP(i)           (((i) < 0) ? GB_FLIP(i) : (i))
#define GBI_UNFLIP(Ai,p,avlen)      \
    ((Ai == NULL) ? ((p) % (avlen)) : GB_UNFLIP (Ai [p]))
#endif

#include "shared/GB_index.h"
#include "shared/GB_partition.h"
#include "shared/GB_pun.h"
#include "shared/GB_opaque.h"
#include "shared/GB_int64_mult.h"
#define GB_HAS_CMPLX_MACROS 1
#include "shared/GB_complex.h"
#include "include/GB_memory_macros.h"

// version for the GPU, with fewer branches
#define GB_TRIM_BINARY_SEARCH(i,X,pleft,pright)                             \
{                                                                           \
    /* binary search of X [pleft ... pright] for integer i */               \
    while (pleft < pright)                                                  \
    {                                                                       \
        int64_t pmiddle = (pleft + pright) >> 1 ;                           \
        bool less = (X [pmiddle] < i) ;                                     \
        pleft  = less ? (pmiddle+1) : pleft ;                               \
        pright = less ? pright : pmiddle ;                                  \
    }                                                                       \
    /* binary search is narrowed down to a single item */                   \
    /* or it has found the list is empty */                                 \
    ASSERT (pleft == pright || pleft == pright + 1) ;                       \
}

#define GB_BINARY_SEARCH(i,X,pleft,pright,found)                            \
{                                                                           \
    GB_TRIM_BINARY_SEARCH (i, X, pleft, pright) ;                           \
    found = (pleft == pright && X [pleft] == i) ;                           \
}

#define GB_SPLIT_BINARY_SEARCH(i,X,pleft,pright,found)                      \
{                                                                           \
    GB_BINARY_SEARCH (i, X, pleft, pright, found)                           \
    if (!found && (pleft == pright))                                        \
    {                                                                       \
        if (i > X [pleft])                                                  \
        {                                                                   \
            pleft++ ;                                                       \
        }                                                                   \
        else                                                                \
        {                                                                   \
            pright++ ;                                                      \
        }                                                                   \
    }                                                                       \
}


#endif

