//------------------------------------------------------------------------------
// GB_red:  hard-coded functions for reductions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_red__include.h"

// The reduction is defined by the following types and operators:

// Reduce to scalar:  GB_red_scalar
// Assemble tuples:   GB_bild

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
    GB_atype w [nthreads] ;

// t = identity
#define GB_REDUCE_INIT(t) \
    GB_atype t = GB_identity ;

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

// Tx [p] += S [k]
#define GB_BUILD_OP(Tx, p, S, k) \
    GB_REDUCE_OP(Tx [p], S [k])

// Tx [p] = S [k]
#define GB_BUILD_COPY(Tx, p, S, k) \
    Tx [p] = S [k] ;

//------------------------------------------------------------------------------
// reduce to a scalar, for monoids only
//------------------------------------------------------------------------------

if_is_monoid

void GB_red_scalar
(
    GB_atype *result,
    const GrB_Matrix A,
    int nthreads
)
{ 
    GB_REDUCE_INIT (s) ;
    #include "GB_reduce_to_scalar_template.c"
    (*result) = s ;
}

endif_is_monoid

//------------------------------------------------------------------------------
// build matrix
//------------------------------------------------------------------------------

void GB_bild
(
    GB_atype *restrict Tx,
    int64_t  *restrict Ti,
    const GB_atype *restrict S,
    int64_t nvals,
    int64_t ndupl,
    const int64_t *restrict I_work,
    const int64_t *restrict K_work,
    const int64_t *restrict tstart_slice,
    const int64_t *restrict tnz_slice,
    int nthreads
)
{
    #include "GB_build_template.c"
}

#endif

