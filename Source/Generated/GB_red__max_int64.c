
//------------------------------------------------------------------------------
// GB_red:  hard-coded functions for reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_reduce__include.h"

// The reduction is defined by the following types and operators:

// Reduce to scalar:  GB_red_scalar__max_int64

// C type:   int64_t
// A type:   int64_t

// Reduce:   s = GB_IMAX (s, aij)
// Identity: INT64_MIN
// Terminal: if (s == INT64_MAX) break ;

#define GB_ATYPE \
    int64_t

// monoid identity value
#define GB_IDENTITY \
    INT64_MIN

// scalar workspace for each thread
#define GB_REDUCE_WORKSPACE(w,nthreads) \
    int64_t w [nthreads] ;

// set t = identity
#define GB_REDUCE_INIT(t) \
    int64_t t = INT64_MIN ;

// t += Ax [p]
#define GB_REDUCE(t,Ax,p)   \
    t = GB_IMAX (t, Ax [p])

// w [tid] = t
#define GB_REDUCE_WRAPUP(w,tid,t) \
    w [tid] = t ;

// s += w [tid], sum up results of each thread
#define GB_REDUCE_W(s,w,tid)  \
    s = GB_IMAX (s, w [tid])

// break if terminal value of the monoid is reached
#define GB_REDUCE_TERMINAL(t) \
    if (s == INT64_MAX) break ;

//------------------------------------------------------------------------------
// reduce to a scalar
//------------------------------------------------------------------------------

void GB_red_scalar__max_int64
(
    int64_t *result,
    const GrB_Matrix A,
    int nthreads
)
{ 
    int64_t s = INT64_MIN ;
    #include "GB_reduce_to_scalar_template.c"
    (*result) = s ;
}

#endif

