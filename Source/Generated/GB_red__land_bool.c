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

// Reduce to scalar:  GB_red_scalar__land_bool
// Assemble tuples:   GB_bild__land_bool

// A type:   bool

// Reduce:   s = (s && aij)
// Identity: true
// Terminal: if (s == false) break ;

#define GB_ATYPE \
    bool

// monoid identity value
#define GB_IDENTITY \
    true

// scalar workspace for each thread
#define GB_REDUCE_WORKSPACE(w,nthreads) \
    bool w [nthreads] ;

// t = identity
#define GB_REDUCE_INIT(t) \
    bool t = true ;

// t += Ax [p]
#define GB_REDUCE(t,Ax,p)   \
    t = (t && Ax [p])

// w [tid] = t
#define GB_REDUCE_WRAPUP(w,tid,t) \
    w [tid] = t ;

// s += w [tid], sum up results of each thread
#define GB_REDUCE_W(s,w,tid)  \
    s = (s && w [tid])

// break if terminal value of the monoid is reached
#define GB_REDUCE_TERMINAL(t) \
    if (s == false) break ;

// Tx [p] += S [k]
#define GB_BUILD_OP(Tx, p, S, k) \
    Tx [p] = (Tx [p] && S [k])

// Tx [p] = S [k]
#define GB_BUILD_COPY(Tx, p, S, k) \
    Tx [p] = S [k] ;

//------------------------------------------------------------------------------
// reduce to a scalar, for monoids only
//------------------------------------------------------------------------------



void GB_red_scalar__land_bool
(
    bool *result,
    const GrB_Matrix A,
    int nthreads
)
{ 
    GB_REDUCE_INIT (s) ;
    #include "GB_reduce_to_scalar_template.c"
    (*result) = s ;
}



//------------------------------------------------------------------------------
// build matrix
//------------------------------------------------------------------------------

void GB_bild__land_bool
(
    bool *restrict Tx,
    int64_t  *restrict Ti,
    const bool *restrict S,
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

