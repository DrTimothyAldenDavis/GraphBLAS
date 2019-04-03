
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

// Reduce to scalar:  GB_red_scalar__times_int8

// C type:   int8_t
// A type:   int8_t

// Reduce:   s *= aij
// Identity: 1
// Terminal: if (s == 0) break ;

#define GB_ATYPE \
    int8_t

// t += Ax [p]
#define GB_REDUCE(t,Ax,p)   \
    t *= Ax [p]

// monoid identity value
#define GB_IDENTITY \
    1

// scalar workspace for each thread
#define GB_REDUCE_WORKSPACE(w,nthreads) \
    int8_t w [nthreads] ;

// set t = identity
#define GB_REDUCE_INIT(t) \
    int8_t t = 1 ;

// wrapup for each thread
#define GB_REDUCE_WRAPUP(w,tid,t) \
    w [tid] = t ;

// s += w [tid], sum up results of each thread
#define GB_REDUCE_W(s,w,tid)  \
    s *= w [tid]

// break if terminal value of the monoid is reached
#define GB_REDUCE_TERMINAL(t) \
    if (s == 0) break ;

//------------------------------------------------------------------------------
// reduce to a scalar
//------------------------------------------------------------------------------

void GB_red_scalar__times_int8
(
    int8_t *result,
    const GrB_Matrix A,
    int nthreads
)
{ 
    int8_t s = 1 ;
    #include "GB_reduce_to_scalar_template.c"
    (*result) = s ;
}

#endif

