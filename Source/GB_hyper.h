//------------------------------------------------------------------------------
// GB_hyper.h: definitions for hypersparse matrices and related methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_HYPER_H
#define GB_HYPER_H

GB_PUBLIC
int64_t GB_nvec_nonempty        // return # of non-empty vectors
(
    const GrB_Matrix A,         // input matrix to examine
    GB_Context Context
) ;

GrB_Info GB_hyper_realloc
(
    GrB_Matrix A,               // matrix with hyperlist to reallocate
    int64_t plen_new,           // new size of A->p and A->h
    GB_Context Context
) ;

GrB_Info GB_conform_hyper       // conform a matrix to sparse/hypersparse
(
    GrB_Matrix A,               // matrix to conform
    GB_Context Context
) ;

GrB_Info GB_hyper_prune
(
    // output, not allocated on input:
    int64_t *restrict *p_Ap, size_t *p_Ap_size,      // size plen+1
    int64_t *restrict *p_Ah, size_t *p_Ah_size,      // size plen
    int64_t *p_nvec,                // # of vectors, all nonempty
    int64_t *p_plen,                // size of Ap and Ah
    // input, not modified
    const int64_t *Ap_old,          // size nvec_old+1
    const int64_t *Ah_old,          // size nvec_old
    const int64_t nvec_old,         // original number of vectors
    GB_Context Context
) ;

GrB_Info GB_hypermatrix_prune
(
    GrB_Matrix A,               // matrix to prune
    GB_Context Context
) ;

GrB_Info GB_hyper_hash      // construct A->Y if not already constructed
(
    GrB_Matrix A,
    GB_Context Context
) ;

static inline void GB_hyper_hash_lookup
(
    // input, not modified
    const int64_t *restrict Ap,     // A->p [0..A->nvec  ]: pointers to vectors
    const int64_t *restrict Yp,     // A->Y->p
    const int64_t *restrict Yi,     // A->Y->i
    const int64_t *restrict Yx,     // A->Y->x
    const int64_t hash_bits,        // hash table size - 1
    const int64_t j,                // find j in Ah [0..anvec-1], using A->Y
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{
    const int64_t jhash = GB_HASHF2 (j, hash_bits) ;
    const int64_t ypstart = Yp [jhash] ;
    const int64_t ypend = Yp [jhash+1] ;
    for (int64_t p = ypstart ; p < ypend ; p++)
    {
        if (j == Yi [p])
        {
            // found: j = Ah [k] where k is given by k = Yx [p]
            const int64_t k = Yx [p] ;
            (*pstart) = Ap [k] ;
            (*pend  ) = Ap [k+1] ;
            return ;
//          return (k) ;
        }
    }
    // not found: j is not in the hyperlist Ah [0..anvec-1]
    (*pstart) = -1 ;
    (*pend  ) = -1 ;
//  return (-1) ;
}

#endif

