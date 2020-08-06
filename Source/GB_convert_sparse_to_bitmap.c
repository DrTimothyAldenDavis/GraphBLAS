//------------------------------------------------------------------------------
// GB_convert_sparse_to_bitmap: convert from sparse/hypersparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

#define GB_FREE_ALL     \
{                       \
    GB_FREE (Ab) ;      \
    GB_phbix_free (A) ; \
}

GrB_Info GB_convert_sparse_to_bitmap    // convert sparse/hypersparse to bitmap
(
    GrB_Matrix A,               // matrix to convert from sparse to bitmap
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting sparse/hypersparse to bitmap", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;        // A can be jumbled on input
    ASSERT (GB_ZOMBIES_OK (A)) ;        // A can have zombies on input
    GBURBLE ("(sparse to bitmap) ") ;

    int8_t *GB_RESTRICT Ab = NULL ;

    //--------------------------------------------------------------------------
    // allocate A->b (using calloc)
    //--------------------------------------------------------------------------

    // Entries not present in A have Ab [p] = 0, so calloc is used.

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;
    int64_t anzmax ;
    if (!GB_Index_multiply (&anzmax, avdim, avlen))
    { 
        // problem too large
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    anzmax = GB_IMAX (anzmax, 1) ;

    Ab = GB_CALLOC (anzmax, int8_t) ;
    if (Ab == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // allocate the new A->x (using calloc)
    //--------------------------------------------------------------------------

    const size_t asize = A->type->size ;
    GB_void *GB_RESTRICT Ax_new = NULL ;
    bool Ax_shallow ;

    // if in_place is true, then A->x does not change if A is dense and not
    // jumbled (zombies are OK).
    bool in_place = (GB_is_dense (A) && !(A->jumbled)) ;

    if (in_place)
    { 
        // keep the existing A->x, so remove it from the matrix for now so
        // that it is not freed by GB_phbix_free
        Ax_new = A->x ;
        Ax_shallow = A->x_shallow ;
        A->x = NULL ;
        A->x_shallow = false ;
    }
    else
    {
        // A->x must be modified to fit the bitmap structure.  calloc is used
        // so that all of A->x can be memcpy'd in bulk, with no complaints from
        // valgrind about uninitialized space, for GrB_Matrix_dup.
        Ax_new = GB_CALLOC (anzmax * asize, GB_void) ;
        Ax_shallow = false ;
        if (Ax_new == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // scatter the pattern and values into the new bitmap
    //--------------------------------------------------------------------------

    // TODO: do this in parallel

    // A retains its CSR/CSC format.

    const int64_t *GB_RESTRICT Ap = A->p ;
    const int64_t *GB_RESTRICT Ah = A->h ;
    const int64_t *GB_RESTRICT Ai = A->i ;
    const GB_void *GB_RESTRICT Ax = A->x ;
    const int64_t anvec = A->nvec ;
    int64_t anvals = 0 ;

    for (int64_t k = 0 ; k < anvec ; k++)
    { 
        // get A(:,j)
        int64_t j = GBH (Ah, k) ;
        int64_t pA_new   = j * avlen ;
        int64_t pA_start = GBP (Ap, k, avlen) ;
        int64_t pA_end   = GBP (Ap, k+1, avlen) ;
        for (int64_t p = pA_start ; p < pA_end ; p++)
        { 
            // A(i,j) has index i, value Ax [p]
            int64_t i = GBI (Ai, p, avlen) ;
            // TODO: make 2 versions: with and without zombies
            if (!GB_IS_ZOMBIE (i))
            { 
                // move A(i,j) to its new place in the bitmap
                int64_t pnew = i + pA_new ;  // for both CSC and CSR
                Ab [pnew] = 1 ;
                // TODO: use type-specific versions for built-in types,
                // and one for any type when A can be modified in-place.
                // Ax_new [pnew] = Ax [p]
                if (!in_place)
                { 
                    memcpy (Ax_new +(pnew)*asize, Ax +(p)*asize, asize) ;
                }
                anvals++ ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free prior content of A and transplant the new content
    //--------------------------------------------------------------------------

    // if done in place, A->x has been removed from A and is thus not freed
    GB_phbix_free (A) ;

    A->b = Ab ;
    A->b_shallow = false ;

    A->x = Ax_new ;
    A->x_shallow = Ax_shallow ;

    A->nzmax = anzmax ;
    A->nvals = anvals ;

    A->plen = -1 ;
    A->nvec = avdim ;
    A->nvec_nonempty = (avlen == 0) ? 0 : avdim ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from sparse to bitmap", GB0) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    return (GrB_SUCCESS) ;
}

