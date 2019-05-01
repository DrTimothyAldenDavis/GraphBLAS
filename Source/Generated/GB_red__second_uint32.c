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

// Reduce to scalar:  GB_red_scalar__(none)
// Assemble tuples:   GB_bild__second_uint32

// A type:   uint32_t

// Reduce:   s = aij
// Identity: (none)
// Terminal: ;

#define GB_ATYPE \
    uint32_t

// monoid identity value
#define GB_IDENTITY \
    (none)

// scalar workspace for each thread
#define GB_REDUCE_WORKSPACE(w,nthreads) \
    uint32_t w [nthreads] ;

// t = identity
#define GB_REDUCE_INIT(t) \
    uint32_t t = (none) ;

// t += Ax [p]
#define GB_REDUCE(t,Ax,p)   \
    t = Ax [p]

// w [tid] = t
#define GB_REDUCE_WRAPUP(w,tid,t) \
    w [tid] = t ;

// s += w [tid], sum up results of each thread
#define GB_REDUCE_W(s,w,tid)  \
    s = w [tid]

// break if terminal value of the monoid is reached
#define GB_REDUCE_TERMINAL(t) \
    ;

// Tx [p] += S [k]
#define GB_BUILD_OP(Tx, p, S, k) \
    Tx [p] = S [k]

// Tx [p] = S [k]
#define GB_BUILD_COPY(Tx, p, S, k) \
    Tx [p] = S [k] ;

//------------------------------------------------------------------------------
// reduce to a scalar, for monoids only
//------------------------------------------------------------------------------

#if 0

    void GB_red_scalar__(none)
    (
        uint32_t *result,
        const GrB_Matrix A,
        int nthreads
    )
    { 
        GB_REDUCE_INIT (s) ;
        #include "GB_reduce_to_scalar_template.c"
        (*result) = s ;
    }

#endif

//------------------------------------------------------------------------------
// build matrix
//------------------------------------------------------------------------------

void GB_bild__second_uint32
(
    uint32_t *restrict Tx,
    int64_t  *restrict Ti,
    const uint32_t *restrict S,
    int64_t ntuples,
    int64_t ndupl,
    const int64_t *restrict iwork,
    const int64_t *restrict kwork,
    const int64_t *tstart_slice,
    const int64_t *tnz_slice,
    int nthreads
)
{
    #include "GB_build_template.c"
}

#endif

