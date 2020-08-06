//------------------------------------------------------------------------------
// GB_convert_bitmap_to_sparse: convert a matrix from bitmap to sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// OK: BITMAP

#include "GB.h"

#define GB_FREE_ALL     \
{                       \
    GB_FREE (Ap) ;      \
    GB_FREE (Ai) ;      \
    GB_FREE (Ax_new) ;  \
    GB_phbix_free (A) ; \
}

GrB_Info GB_convert_bitmap_to_sparse    // convert matrix from bitmap to sparse
(
    GrB_Matrix A,               // matrix to convert from bitmap to sparse
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converting bitmap to sparse", GB0) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_PENDING (A)) ;      // bitmap never has pending tuples
    ASSERT (!GB_JUMBLED (A)) ;      // bitmap is never jumbled
    ASSERT (!GB_ZOMBIES (A)) ;      // bitmap never has zomies
    GBURBLE ("(bitmap to sparse) ") ;

    int64_t *GB_RESTRICT Ap = NULL ;
    int64_t *GB_RESTRICT Ai = NULL ;
    GB_void *GB_RESTRICT Ax_new = NULL ;

    //--------------------------------------------------------------------------
    // allocate A->p (using calloc)
    //--------------------------------------------------------------------------

    // calloc is used because A->p is used to count the entries in each vector.

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;
    int64_t anvec = avdim ;

    Ap = GB_CALLOC (avdim+1, int64_t) ;
    if (Ap == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector
    //--------------------------------------------------------------------------

    // TODO: do this in parallel using GB_reduce_to_vector template
    // (compare with GB_select_phase1)

    const int8_t *GB_RESTRICT Ab = A->b ;

    for (int64_t k = 0 ; k < avdim ; k++)
    { 
        // aknz = nnz (A (:,k))
        int64_t aknz = 0 ;
        int64_t pA_start = k * avlen ;
        for (int64_t i = 0 ; i < avlen ; i++)
        { 
            // see if A(i,j) is present in the bitmap
            int64_t p = i + pA_start ;
            aknz += (Ab [p] != 0) ;
            ASSERT (Ab [p] == 0 || Ab [p] == 1) ;
        }
        Ap [k] = aknz ;
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Ap 
    //--------------------------------------------------------------------------

    // TODO: do this in parallel

    int64_t anvec_nonempty ;
    GB_cumsum (Ap, anvec, &anvec_nonempty, 1) ;
    int64_t anzmax = Ap [anvec] ;
    anzmax = GB_IMAX (anzmax, 1) ;

    //--------------------------------------------------------------------------
    // allocate the new A->x and A->i
    //--------------------------------------------------------------------------

    const size_t asize = A->type->size ;
    Ai = GB_MALLOC (anzmax, int64_t) ;
    Ax_new = GB_MALLOC (anzmax * asize, GB_void) ;
    if (Ai == NULL || Ax_new == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // gather the pattern and values from the bitmap
    //--------------------------------------------------------------------------

    // TODO: do this in parallel

    // A retains its CSR/CSC format.

    const GB_void *GB_RESTRICT Ax = A->x ;

    for (int64_t k = 0 ; k < avdim ; k++)
    { 
        // gather from the bitmap into the new A (:,k)
        int64_t aknz = 0 ;
        int64_t pnew = Ap [k] ;
        int64_t pA_start = k * avlen ;
        for (int64_t i = 0 ; i < avlen ; i++)
        { 
            int64_t p = i + pA_start ;
            if (Ab [p])
            { 
                // A(i,j) is in the bitmap
                Ai [pnew] = i ;
                // Ax_new [pnew] = Ax [p]
                memcpy (Ax_new +(pnew)*asize, Ax +(p)*asize, asize) ;
                pnew++ ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free prior content of A and transplant the new content
    //--------------------------------------------------------------------------

    GB_phbix_free (A) ;

    A->p = Ap ;
    A->p_shallow = false ;

    A->i = Ai ;
    A->i_shallow = false ;

    A->x = Ax_new ;
    A->x_shallow = false ;

    A->nzmax = anzmax ;
    A->nvals = 0 ;              // only used when A is bitmap

    A->plen = avdim ;
    A->nvec = avdim ;
    A->nvec_nonempty = anvec_nonempty ;

    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A converted from to bitmap to sparse", GB0) ;
    ASSERT (A->h == NULL) ;     // A is not hypersparse
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    return (GrB_SUCCESS) ;
}

