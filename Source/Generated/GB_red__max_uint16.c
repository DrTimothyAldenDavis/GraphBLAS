
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

// Reduce to scalar:  GB_red_scalar__max_uint16

// C type:   uint16_t
// A type:   uint16_t

// Reduce:   s = GB_IMAX (s, aij)
// Identity: 0
// Terminal: if (s == UINT16_MAX) break ;

#define GB_ATYPE \
    uint16_t

// t += Ax [p]
#define GB_REDUCE(t,Ax,p)   \
    t = GB_IMAX (t, Ax [p])

// monoid identity value
#define GB_IDENTITY \
    0

// scalar workspace for each thread
#define GB_REDUCE_WORKSPACE(w,nthreads) \
    uint16_t w [nthreads] ;

// set t = identity
#define GB_REDUCE_INIT(t) \
    uint16_t t = 0 ;

// wrapup for each thread
#define GB_REDUCE_WRAPUP(w,tid,t) \
    w [tid] = t ;

// s += w [tid], sum up results of each thread
#define GB_REDUCE_W(s,w,tid)  \
    s = GB_IMAX (s, w [tid])

// break if terminal value of the monoid is reached
#define GB_REDUCE_TERMINAL(t) \
    if (s == UINT16_MAX) break ;

//------------------------------------------------------------------------------
// reduce to a scalar
//------------------------------------------------------------------------------

void GB_red_scalar__max_uint16
(
    uint16_t *result,
    const GrB_Matrix A,
    int nthreads
)
{ 
    uint16_t s = 0 ;
    #include "GB_reduce_to_scalar_template.c"
    (*result) = s ;
}

#endif

