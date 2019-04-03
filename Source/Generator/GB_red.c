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

// Reduce to scalar:  GB_red_scalar

// C type:   GB_ztype
// A type:   GB_atype

// Reduce:   GB_REDUCE_OP(s, aij)
// Identity: GB_identity
// Terminal: GB_terminal

#define GB_ATYPE \
    GB_atype

// monoid identity value
#define GB_IDENTITY \
    GB_identity

// scalar workspace for each thread
#define GB_REDUCE_WORKSPACE(w,nthreads) \
    GB_ztype w [nthreads] ;

// set t = identity
#define GB_REDUCE_INIT(t) \
    GB_ztype t = GB_identity ;

// t += Ax [p]
#define GB_REDUCE(t,Ax,p)   \
    GB_REDUCE_OP(t, Ax [p])

// w [tid] = t
#define GB_REDUCE_WRAPUP(w,tid,t) \
    w [tid] = t ;

// s += w [tid], sum up results of each thread
#define GB_REDUCE_W(s,w,tid)  \
    GB_REDUCE_OP(s, w [tid])

// break if terminal value of the monoid is reached
#define GB_REDUCE_TERMINAL(t) \
    GB_terminal

//------------------------------------------------------------------------------
// reduce to a scalar
//------------------------------------------------------------------------------

void GB_red_scalar
(
    GB_atype *result,
    const GrB_Matrix A,
    int nthreads
)
{ 
    GB_atype s = GB_identity ;
    #include "GB_reduce_to_scalar_template.c"
    (*result) = s ;
}

#endif

