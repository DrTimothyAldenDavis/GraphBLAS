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
    GB_FREE (W) ;       \
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

    int64_t *GB_RESTRICT W = NULL ;
    int64_t *GB_RESTRICT Ap = NULL ;
    int64_t *GB_RESTRICT Ai = NULL ;
    GB_void *GB_RESTRICT Ax_new = NULL ;

    //--------------------------------------------------------------------------
    // allocate A->p (using calloc)
    //--------------------------------------------------------------------------

    int64_t avdim = A->vdim ;
    int64_t avlen = A->vlen ;
    int64_t anvec = avdim ;

    Ap = GB_MALLOC (avdim+1, int64_t) ;
    if (Ap == NULL)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // count the entries in each vector
    //--------------------------------------------------------------------------

    const int8_t *GB_RESTRICT Ab = A->b ;

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;
    int nthreads = GB_nthreads (avlen*avdim, chunk, nthreads_max) ;
    bool by_vector = (nthreads <= avdim) ;

    if (by_vector)
    { 

        //----------------------------------------------------------------------
        // compute all vectors in parallel (no workspace)
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int64_t k = 0 ; k < avdim ; k++)
        { 
            // aknz = nnz (A (:,k))
            int64_t aknz = 0 ;
            int64_t pA_start = k * avlen ;
            for (int64_t i = 0 ; i < avlen ; i++)
            { 
                // see if A(i,j) is present in the bitmap
                int64_t p = i + pA_start ;
                aknz += Ab [p] ;
                ASSERT (Ab [p] == 0 || Ab [p] == 1) ;
            }
            Ap [k] = aknz ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // compute blocks of rows in parallel
        //----------------------------------------------------------------------

        // allocate one row of W per thread, each row of length avdim
        W = GB_MALLOC (nthreads * avdim, int64_t) ;
        if (W == NULL)
        {
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (taskid = 0 ; taskid < nthreads ; taskid++)
        { 
            int64_t *GB_RESTRICT Wtask = W + taskid * avdim ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, avlen, taskid, nthreads) ;
            for (int64_t k = 0 ; k < avdim ; k++)
            { 
                // aknz = nnz (A (istart:iend-1,k))
                int64_t aknz = 0 ;
                int64_t pA_start = k * avlen ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    // see if A(i,j) is present in the bitmap
                    int64_t p = i + pA_start ;
                    aknz += Ab [p] ;
                    ASSERT (Ab [p] == 0 || Ab [p] == 1) ;
                }
                Wtask [k] = aknz ;
            }
        }

        // cumulative sum to compute nnz(A(:,j)) for each vector j
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int64_t k = 0 ; k < avdim ; k++)
        {
            int64_t aknz = 0 ;
            for (int taskid = 0 ; taskid < nthreads ; taskid++)
            {
                int64_t *GB_RESTRICT Wtask = W + taskid * avdim ;
                int64_t c = Wtask [k] ;
                Wtask [k] = aknz ;
                aknz += c ;
            }
            Ap [k] = aknz ;
        }
    }

    //--------------------------------------------------------------------------
    // cumulative sum of Ap 
    //--------------------------------------------------------------------------

    int nth = GB_nthreads (avdim, chunk, nthreads_max) ;
    int64_t anvec_nonempty ;
    GB_cumsum (Ap, avdim, &anvec_nonempty, nth) ;
    int64_t anzmax = Ap [avdim] ;
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

    // A retains its CSR/CSC format.
    // TODO: add type-specific versions for built-in types

    const GB_void *GB_RESTRICT Ax = A->x ;

    if (by_vector)
    { 

        //----------------------------------------------------------------------
        // construct all vectors in parallel (no workspace)
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int64_t k = 0 ; k < avdim ; k++)
        { 
            // gather from the bitmap into the new A (:,k)
            int64_t pnew = Ap [k] ;
            int64_t pA_start = k * avlen ;
            for (int64_t i = 0 ; i < avlen ; i++)
            { 
                int64_t p = i + pA_start ;
                if (Ab [p])
                { 
                    // A(i,j) is in the bitmap
                    Ai [pnew] = i ;
                    // Ax_new [pnew] = Ax [p])
                    memcpy (Ax_new +(pnew)*asize, Ax +(p)*asize, asize) ;
                    pnew++ ;
                }
            }
            ASSERT (pnew == Ap [k+1]) ;
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // compute blocks of rows in parallel
        //----------------------------------------------------------------------

        int taskid ;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (taskid = 0 ; taskid < nthreads ; taskid++)
        { 
            int64_t *GB_RESTRICT Wtask = W + taskid * avdim ;
            int64_t istart, iend ;
            GB_PARTITION (istart, iend, avlen, taskid, nthreads) ;
            for (int64_t k = 0 ; k < avdim ; k++)
            { 
                // gather from the bitmap into the new A (:,k)
                int64_t pnew = Ap [k] + Wtask [k] ;
                int64_t pA_start = k * avlen ;
                for (int64_t i = istart ; i < iend ; i++)
                { 
                    // see if A(i,j) is present in the bitmap
                    int64_t p = i + pA_start ;
                    if (Ab [p])
                    { 
                        // A(i,j) is in the bitmap
                        Ai [pnew] = i ;
                        // Ax_new [pnew] = Ax [p] ;
                        memcpy (Ax_new +(pnew)*asize, Ax +(p)*asize, asize) ;
                        pnew++ ;
                    }
                }
            }
        }

        // free workspace
        GB_FREE (W) ;
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
    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    return (GrB_SUCCESS) ;
}

