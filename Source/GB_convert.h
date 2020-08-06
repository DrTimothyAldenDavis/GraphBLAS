//------------------------------------------------------------------------------
// GB_convert.h: converting between sparsity structures
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#ifndef GB_CONVERT_H
#define GB_CONVERT_H

// these parameters define the hyper_switch needed to ensure matrix stays
// either always hypersparse, or never hypersparse.
#define GB_ALWAYS_HYPER (1.0)
#define GB_NEVER_HYPER  (-1.0)

#define GB_FORCE_HYPER 1
#define GB_HYPER GB_FORCE_HYPER
#define GB_FORCE_NONHYPER 0
#define GB_SPARSE GB_FORCE_NONHYPER
#define GB_AUTO_HYPER (-1)
#define GB_FULL 2
#define GB_BITMAP 3

#define GB_SAME_HYPER_AS(A_is_hyper) \
    ((A_is_hyper) ? GB_FORCE_HYPER : GB_FORCE_NONHYPER)

// true if A is bitmap
#define GB_IS_BITMAP(A) ((A) != NULL && ((A)->b != NULL))

// true if A is full (but not bitmap)
#define GB_IS_FULL(A) \
    ((A) != NULL && (A)->h == NULL && (A)->p == NULL && (A)->i == NULL \
        && (A)->b == NULL)

// true if A is hypersparse
#define GB_IS_HYPERSPARSE(A) ((A) != NULL && ((A)->h != NULL))

// true if A is sparse (but not hypersparse)
#define GB_IS_SPARSE(A) ((A) != NULL && ((A)->h == NULL) && (A)->p != NULL)

// if A is hypersparse but all vectors are present, then
// treat A as if it were non-hypersparse
#define GB_IS_HYPER(A) (GB_IS_HYPERSPARSE (A) && ((A)->nvec < (A)->vdim))

// true if A has any sparsit structure (only useful for commenting via
// assertions, since this is always true).
#define GB_IS_ANY_SPARSITY(A) \
    ((A) == NULL || GB_IS_FULL (A) || GB_IS_BITMAP (A) || \
     GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A))
    

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_convert_hyper_to_sparse // convert hypersparse to sparse
(
    GrB_Matrix A,           // matrix to convert from hypersparse to sparse
    GB_Context Context
) ;

GB_PUBLIC   // accessed by the MATLAB tests in GraphBLAS/Test only
GrB_Info GB_convert_sparse_to_hyper // convert from sparse to hypersparse
(
    GrB_Matrix A,           // matrix to convert to hypersparse
    GB_Context Context
) ;

bool GB_convert_hyper_to_sparse_test    // test for hypersparse to sparse
(
    float hyper_switch,     // A->hyper_switch
    int64_t k,              // # of non-empty vectors of A, an estimate is OK,
                            // but normally A->nvec_nonempty
    int64_t vdim            // A->vdim
) ;

bool GB_convert_sparse_to_hyper_test  // test sparse to hypersparse conversion
(
    float hyper_switch,     // A->hyper_switch
    int64_t k,              // # of non-empty vectors of A, an estimate is OK,
                            // but normally A->nvec_nonempty
    int64_t vdim            // A->vdim
) ;

GrB_Info GB_convert_full_to_sparse      // convert matrix from full to sparse
(
    GrB_Matrix A,               // matrix to convert from full to sparse
    GB_Context Context
) ;

GrB_Info GB_convert_full_to_bitmap      // convert matrix from full to bitmap
(
    GrB_Matrix A,               // matrix to convert from full to bitmap
    GB_Context Context
) ;

GrB_Info GB_convert_sparse_to_bitmap    // convert sparse/hypersparse to bitmap
(
    GrB_Matrix A,               // matrix to convert from sparse to bitmap
    GB_Context Context
) ;

GrB_Info GB_convert_bitmap_to_sparse    // convert matrix from bitmap to sparse
(
    GrB_Matrix A,               // matrix to convert from bitmap to sparse
    GB_Context Context
) ;

GrB_Info GB_convert_to_full     // convert matrix to full; delete prior values
(
    GrB_Matrix A                // matrix to convert to full
) ;

GrB_Info GB_convert_any_to_bitmap   // convert to bitmap
(
    GrB_Matrix A,           // matrix to convert to bitmap
    GB_Context Context
) ;

GB_PUBLIC                       // used by MATLAB interface
void GB_convert_any_to_full     // convert any matrix to full
(
    GrB_Matrix A                // matrix to convert to full
) ;

GrB_Info GB_convert_any_to_hyper // convert to hypersparse
(
    GrB_Matrix A,           // matrix to convert to hypersparse
    GB_Context Context
) ;

GrB_Info GB_convert_any_to_sparse // convert to sparse
(
    GrB_Matrix A,           // matrix to convert to sparse
    GB_Context Context
) ;

#define GB_ENSURE_SPARSE(C)                                 \
{                                                           \
    /* TODO: handle bitmap also */ \
    if (GB_IS_FULL (C))                                     \
    {                                                       \
        /* convert C from full to sparse */                 \
        GrB_Info info = GB_convert_full_to_sparse (C, Context) ;    \
        if (info != GrB_SUCCESS)                            \
        {                                                   \
            return (info) ;                                 \
        }                                                   \
    }                                                       \
}

#define GB_ENSURE_FULL(C)                                   \
{                                                           \
    ASSERT (GB_is_dense (C)) ;                              \
    /* convert C from any structure to full */              \
    GB_convert_any_to_full (C) ;                            \
}

static inline bool GB_is_dense
(
    const GrB_Matrix A
)
{
    // check if A is competely dense:  all entries present.
    // zombies, pending tuples, and jumbled status are not considered.
    // A can have any sparsity structure: hyper, sparse, bitmap, or full.
    if (A == NULL)
    {
        return (false) ;
    }
    if (GB_IS_FULL (A))
    { 
        // A is full; the pattern is not present
        return (true) ;
    }
    // A is sparse: check if all entries present
    GrB_Index anzmax ;
    bool ok = GB_Index_multiply (&anzmax, A->vlen, A->vdim) ;
    return (ok && (anzmax == GB_NNZ (A))) ;
}

static inline bool GB_is_packed
(
    const GrB_Matrix A
)
{
    // check if A is a packed matrix.  A is packed if it is bitmap or full.  If
    // A is hypersparse or sparse, it is packed if it is not jumbled, all
    // entries are present, and it has no zombies or pending tuples. 
    // If A is sparse or hypersparse, it can be converted to full via
    // GB_convert_any_to_full, by deleting A->p, A->h, and A->i.  If bitmap,
    // it cannot be converted to full unless GB_is_dense (A) is also true
    // (it must have all entries present).

    if (A == NULL)
    { 
        return (false) ;
    }
    if (GB_IS_FULL (A) || GB_IS_BITMAP (A))
    { 
        // A is full or bitmap
        return (true) ;
    }
    if (A->jumbled || GB_PENDING (A) || GB_ZOMBIES (A))
    { 
        // A is sparse or hypersparse with pending work
        return (false) ;
    }
    // A is sparse or hypersparse: check if all entries present
    GrB_Index anzmax ;
    bool ok = GB_Index_multiply (&anzmax, A->vlen, A->vdim) ;
    return (ok && (anzmax == GB_NNZ (A))) ;
}

GrB_Info GB_conform             // conform a matrix to its desired format
(
    GrB_Matrix A,               // matrix to conform
    GB_Context Context
) ;

#endif

