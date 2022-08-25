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

//------------------------------------------------------------------------------
// GB_hyper_hash_lookup: find k so that j == Ah [k], using the A->Y hyper_hash
//------------------------------------------------------------------------------

// Let j = Ah [k]
// k = A->Y (j, hash(j)), if present, or k=-1 if not found.

static 
#ifdef GB_CUDA_KERNEL
__device__ __inline__
#else
inline
#endif
int64_t GB_hyper_hash_lookup  // k if j==Ah[k], or -1 if not found
(
    // input, not modified
    const int64_t *restrict Ap,     // A->p [0..A->nvec]: pointers to vectors
    const int64_t *restrict Yp,     // A->Y->p
    const int64_t *restrict Yi,     // A->Y->i
    const int64_t *restrict Yx,     // A->Y->x
    const int64_t hash_bits,        // A->Y->vdim-1, which is hash table size-1
    const int64_t j,                // find j in Ah [0..anvec-1], using A->Y
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{
    const int64_t jhash = GB_HASHF2 (j, hash_bits) ;
    const int64_t ypstart = Yp [jhash] ;
    const int64_t ypend = Yp [jhash+1] ;
    int64_t k = -1 ;
    #if 0
    const int64_t yjnz = ypend - ypstart ;
    if (yjnz > 512)
    {
        // do a binary search as a fallback, if there are a crazy number
        // of hash collisions for this bucket jhash.
        bool found ;
        int64_t p = ypstart ;
        GB_BINARY_SEARCH (j, Yi, p, ypend, found) ;
        if (found)
        {
            k = Yx [p] ;
            break ;
        }
    }
    else
    #endif
    {
        for (int64_t p = ypstart ; p < ypend ; p++)
        {
            if (j == Yi [p])
            {
                // found: j = Ah [k] where k is given by k = Yx [p]
                k = Yx [p] ;
                break ;
            }
        }
    }
    if (k >= 0)
    {
        // found: j == Ah [k], get the vector A(:,j)
        (*pstart) = Ap [k] ;
        (*pend  ) = Ap [k+1] ;
    }
    else
    {
        // not found: j is not in the hyperlist Ah [0..anvec-1]
        (*pstart) = -1 ;
        (*pend  ) = -1 ;
    }
    return (k) ;
}

//------------------------------------------------------------------------------
// GB_lookup: find k so that j == Ah [k], without using the A->Y hyper_hash
//------------------------------------------------------------------------------

// For a sparse, bitmap, or full matrix j == k.
// For a hypersparse matrix, find k so that j == Ah [k], if it
// appears in the list.

// k is not needed by the caller, just pstart, pend, pleft, and found.

// Once k is found, find pstart and pend, the start and end of the vector.
// pstart and pend are defined for all sparsity structures: hypersparse,
// sparse, bitmap, or full.

static inline bool GB_lookup        // find j = Ah [k] in a hyperlist
(
    // input:
    const bool A_is_hyper,          // true if A is hypersparse
    const int64_t *restrict Ah,     // A->h [0..A->nvec-1]: list of vectors
    const int64_t *restrict Ap,     // A->p [0..A->nvec  ]: pointers to vectors
    const int64_t avlen,            // A->vlen
    // input/output:
    int64_t *restrict pleft,        // look only in A->h [pleft..pright]
    // input:
    int64_t pright,                 // normally A->nvec-1, but can be trimmed
    const int64_t j,                // vector to find, as j = Ah [k]
    // output:
    int64_t *restrict pstart,       // start of vector: Ap [k]
    int64_t *restrict pend          // end of vector: Ap [k+1]
)
{
    if (A_is_hyper)
    {
        // binary search of Ah [pleft...pright] for the value j
        bool found ;
        GB_BINARY_SEARCH (j, Ah, (*pleft), pright, found) ;
        if (found)
        { 
            // j appears in the hyperlist at Ah [pleft]
            // k = (*pleft)
            (*pstart) = Ap [(*pleft)] ;
            (*pend)   = Ap [(*pleft)+1] ;
        }
        else
        { 
            // j does not appear in the hyperlist Ah
            // k = -1
            (*pstart) = -1 ;
            (*pend)   = -1 ;
        }
        return (found) ;
    }
    else
    { 
        // A is sparse, bitmap, or full; j always appears
        // k = j
        (*pstart) = GBP (Ap, j, avlen) ;
        (*pend)   = GBP (Ap, j+1, avlen) ;
        return (true) ;
    }
}

#endif

