//------------------------------------------------------------------------------
// GB_resize: change the size of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB_select.h"

#define GB_FREE_ALL GB_phix_free (A) ;

GrB_Info GB_resize              // change the size of a matrix
(
    GrB_Matrix A,               // matrix to modify
    const GrB_Index nrows_new,  // new number of rows in matrix
    const GrB_Index ncols_new,  // new number of columns in matrix
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A to resize", GB0) ;
    bool ok = true ;

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format
    //--------------------------------------------------------------------------

    int64_t vdim_old = A->vdim ;
    int64_t vlen_old = A->vlen ;
    int64_t vlen_new, vdim_new ;
    if (A->is_csc)
    {   GB_cov[2896]++ ;
// covered (2896): 42552
        vlen_new = nrows_new ;
        vdim_new = ncols_new ;
    }
    else
    {   GB_cov[2897]++ ;
// covered (2897): 42857
        vlen_new = ncols_new ;
        vdim_new = nrows_new ;
    }

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    // only do so if either dimension is shrinking, or if pending tuples exist
    // and vdim_old <= 1 and vdim_new > 1, since in that case, Pending->j has
    // not been allocated yet, but would be required in the resized matrix.

    if (vdim_new < vdim_old || vlen_new < vlen_old ||
        (GB_PENDING (A) && vdim_old <= 1 && vdim_new > 1))
    {   GB_cov[2898]++ ;
// covered (2898): 72429
        GB_MATRIX_WAIT (A) ;
        ASSERT_MATRIX_OK (A, "A to resize, wait", GB0) ;
    }

    //--------------------------------------------------------------------------
    // check for sparsity conversion
    //--------------------------------------------------------------------------

    if (vdim_new <= vdim_old && vlen_new <= vlen_old)
    {   GB_cov[2899]++ ;
// covered (2899): 43888
        // A is shrinking
        ASSERT (!GB_PENDING (A)) ;
        ASSERT (!GB_ZOMBIES (A)) ;
        if (GB_is_dense (A) && !GB_IS_FULL (A))
        {
            // A is dense but held in sparse format, and it is shrinking;
            // convert A to full
            GB_sparse_to_full (A) ;
        }
        if (vdim_new == vdim_old && vlen_new == vlen_old)
        {   GB_cov[2900]++ ;
// covered (2900): 832
            // nothing more to do
            return (GrB_SUCCESS) ;
        }
    }
    else
    {
        // at least one dimension of A is increasing
        GB_ENSURE_SPARSE (A) ;
    }

    //--------------------------------------------------------------------------
    // determine maximum number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS_MAX (nthreads_max, chunk, Context) ;

    //--------------------------------------------------------------------------
    // resize a full matrix
    //--------------------------------------------------------------------------

    if (GB_IS_FULL (A))
    {   GB_cov[2901]++ ;
// covered (2901): 12

        //----------------------------------------------------------------------
        // A must be shrinking; otherwise it will already be sparse, not full
        //----------------------------------------------------------------------

        ASSERT (vdim_new <= vdim_old && vlen_new <= vlen_old) ;

        //----------------------------------------------------------------------
        // get the old and new dimensions
        //----------------------------------------------------------------------

        int64_t anz_old = vlen_old * vdim_old ;
        int64_t anz_new = vlen_new * vdim_new ;
        size_t nzmax_new = GB_IMAX (anz_new, 1) ;
        size_t nzmax_old = A->nzmax ;
        bool in_place = (vlen_new == vlen_old || vdim_new <= 1) ;
        size_t asize = A->type->size ;
        GB_void *GB_RESTRICT Ax_new = NULL ;
        GB_void *GB_RESTRICT Ax_old = A->x ;

        //----------------------------------------------------------------------
        // allocate or reallocate A->x
        //----------------------------------------------------------------------

        if (in_place)
        {   GB_cov[2902]++ ;
// covered (2902): 12
            // reallocate A->x in place; no data movement needed
            A->x = GB_REALLOC (A->x, nzmax_new*asize, nzmax_old*asize,
                GB_void, &ok) ;
            Ax_new = A->x ;
        }
        else
        {   GB_cov[2903]++ ;
// NOT COVERED (2903):
            // allocate new space for A->x
            Ax_new = GB_MALLOC (nzmax_new*asize, GB_void) ;
            ok = (Ax_new != NULL) ;
        }

        if (!ok)
        {   GB_cov[2904]++ ;
// NOT COVERED (2904):
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // move data if necessary
        //----------------------------------------------------------------------

        if (!in_place)
        {   GB_cov[2905]++ ;
// NOT COVERED (2905):
            // determine number of threads to use
            int nthreads = GB_nthreads (anz_new, chunk, nthreads_max) ;

            // resize the matrix
            int64_t j ;
            if (vdim_new <= 4*nthreads)
            {   GB_cov[2906]++ ;
// NOT COVERED (2906):
                // use all threads for each vector
                for (j = 0 ; j < vdim_new ; j++)
                {   GB_cov[2907]++ ;
// NOT COVERED (2907):
                    GB_void *pdest = Ax_new + j * vlen_new * asize ;
                    GB_void *psrc  = Ax_old + j * vlen_old * asize ;
                    GB_memcpy (pdest, psrc, vlen_new * asize, nthreads) ;
                }
            }
            else
            {   GB_cov[2908]++ ;
// NOT COVERED (2908):
                // use a single thread for each vector
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (j = 0 ; j < vdim_new ; j++)
                {
                    GB_void *pdest = Ax_new + j * vlen_new * asize ;
                    GB_void *psrc  = Ax_old + j * vlen_old * asize ;
                    memcpy (pdest, psrc, vlen_new * asize) ;
                }
            }
        }

        //----------------------------------------------------------------------
        // adjust dimensions and return result
        //----------------------------------------------------------------------

        A->x = Ax_new ;
        A->vdim = vdim_new ;
        A->vlen = vlen_new ;
        A->nzmax = nzmax_new ;
        A->nvec = vdim_new ;
        A->nvec_nonempty = (vlen_new == 0) ? 0 : vdim_new ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // A is sparse or hypersparse; check for early conversion to hypersparse
    //--------------------------------------------------------------------------

    // If the # of vectors grows very large, it is costly to reallocate enough
    // space for the non-hypersparse A->p component.  So convert the matrix to
    // hypersparse if that happens.

    if (A->nvec_nonempty < 0)
    {   GB_cov[2909]++ ;
// covered (2909): 20086
        A->nvec_nonempty = GB_nvec_nonempty (A, Context) ;
    }

    if (A->h == NULL &&
        GB_to_hyper_test (A->hyper_ratio, A->nvec_nonempty, vdim_new))
    {   GB_cov[2910]++ ;
// covered (2910): 10660
        GB_OK (GB_to_hyper (A, Context)) ;
    }

    //--------------------------------------------------------------------------
    // resize the number of sparse vectors
    //--------------------------------------------------------------------------

    int64_t *GB_RESTRICT Ah = A->h ;
    int64_t *GB_RESTRICT Ap = A->p ;
    A->vdim = vdim_new ;

    if (Ah != NULL)
    {

        //----------------------------------------------------------------------
        // A is hypersparse: decrease size of A->p and A->h only if needed
        //----------------------------------------------------------------------

        if (vdim_new < A->plen)
        {   GB_cov[2911]++ ;
// covered (2911): 16842
            // reduce the size of A->p and A->h; this cannot fail
            info = GB_hyper_realloc (A, vdim_new, Context) ;
            ASSERT (info == GrB_SUCCESS) ;
            Ap = A->p ;
            Ah = A->h ;
        }
        if (vdim_new < vdim_old)
        {   GB_cov[2912]++ ;
// covered (2912): 17552
            // descrease A->nvec to delete the vectors outside the range
            // 0...vdim_new-1.
            int64_t pleft = 0 ;
            int64_t pright = GB_IMIN (A->nvec, vdim_new) - 1 ;
            bool found ;
            GB_SPLIT_BINARY_SEARCH (vdim_new, Ah, pleft, pright, found) ;
            A->nvec = pleft ;
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // A is not hypersparse: change size of A->p to match the new vdim
        //----------------------------------------------------------------------

        if (vdim_new != vdim_old)
        {
            // change the size of A->p
            A->p = GB_REALLOC (A->p, vdim_new+1, vdim_old+1, int64_t, &ok) ;
            if (!ok)
            {   GB_cov[2913]++ ;
// covered (2913): 4816
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
            Ap = A->p ;
            A->plen = vdim_new ;
        }

        if (vdim_new > vdim_old)
        {
            // number of vectors is increasing, extend the vector pointers
            int64_t anz = GB_NNZ (A) ;

            // determine number of threads to use
            int nthreads = GB_nthreads (vdim_new - vdim_old, chunk,
                nthreads_max) ;

            int64_t j ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (j = vdim_old + 1 ; j <= vdim_new ; j++)
            {   GB_cov[2914]++ ;
// covered (2914): 47543
                Ap [j] = anz ;          // ok: A is sparse
            }
            // A->nvec_nonempty does not change
        }
        A->nvec = vdim_new ;
    }

    if (vdim_new < vdim_old)
    {   GB_cov[2915]++ ;
// covered (2915): 31782
        // number of vectors is decreasing, need to count the new number of
        // non-empty vectors, unless it is done during pruning, just below.
        A->nvec_nonempty = -1 ;         // compute when needed
    }

    //--------------------------------------------------------------------------
    // resize the length of each vector
    //--------------------------------------------------------------------------

    // if vlen is shrinking, delete entries outside the new matrix
    if (vlen_new < vlen_old)
    {   GB_cov[2916]++ ;
// covered (2916): 62602
        GB_OK (GB_selector (NULL, GB_RESIZE_opcode, NULL, false, A, vlen_new-1,
            NULL, Context)) ;
    }

    //--------------------------------------------------------------------------
    // vlen has been resized
    //--------------------------------------------------------------------------

    A->vlen = vlen_new ;
    ASSERT_MATRIX_OK (A, "A vlen resized", GB0) ;

    //--------------------------------------------------------------------------
    // check for conversion to hypersparse or to non-hypersparse
    //--------------------------------------------------------------------------

    return (GB_to_hyper_conform (A, Context)) ;
}

