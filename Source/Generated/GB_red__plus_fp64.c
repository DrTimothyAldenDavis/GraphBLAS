
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

// Reduce to scalar:  GB_red_scalar__plus_fp64

// C type:   double
// A type:   double

// Reduce:   s += aij
// Identity: 0
// Terminal: ;

#define GB_ATYPE \
    double

// t += Ax [p]
#define GB_REDUCE(t,Ax,p)   \
    t += Ax [p]

// monoid identity value
#define GB_IDENTITY \
    0

// scalar workspace for each thread
#define GB_REDUCE_WORKSPACE(w,nthreads) \
    double w [nthreads] ;

// set t = identity
#define GB_REDUCE_INIT(t) \
    double t = 0 ;

// wrapup for each thread
#define GB_REDUCE_WRAPUP(w,tid,t) \
    w [tid] = t ;

// s += w [tid], sum up results of each thread
#define GB_REDUCE_W(s,w,tid)  \
    s += w [tid]

// break if terminal value of the monoid is reached
#define GB_REDUCE_TERMINAL(t) \
    ;

//------------------------------------------------------------------------------
// reduce to a scalar
//------------------------------------------------------------------------------

void GB_red_scalar__plus_fp64
(
    double *result,
    const GrB_Matrix A,
    int nthreads
)
{ 
    double s = 0 ;
    #include "GB_reduce_to_scalar_template.c"
    (*result) = s ;
}

#endif

