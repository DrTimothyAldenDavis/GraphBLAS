//------------------------------------------------------------------------------
// GB_slice: create hypersparse shallow slices of a matrix B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// For each thread t, create Bslice [t] as a purely hypersparse shallow slice
// of B.  The i and x arrays are the same as B.  The p array is an offset into
// Bp (that is, Bp + Slice [t]), which means that p [0] will not be zero
// (except for Bslice [0]).  If B is hypersparse, the h array is also an offset
// into B->h.  If B is standard, then Bslice [t] becomes an implicit
// hypersparse matrix.  Its h array is NULL, and the h list is implicit:
// h[0..nvec-1] is implicitly [hfirst, hfirst+1, ...  hfirst+nvec-1], where
// nvec = Slice [t+1] - Slice [t].

// The matrix dimensions of each slice are the same as B.  All slices have
// vector length B->vlen and vector dimension B->vdim.   The slices are subsets
// of the vectors of B, as defined by the Slice array.  The Bslice [t] consists
// of the vectors Slice [t] to Slice [t+1]-1.

// This function does only O(nthreads) work and allocates O(nthreads) space,
// so it does not need to be parallel.  If done in parallel, the out-of-memory
// condition needs to be handled with a reduction.

#include "GB.h"

GrB_Info GB_slice       // slice B into nthreads slices
(
    GrB_Matrix B,       // matrix to slice
    int nthreads,       // # of slices to create
    int64_t *Slice,     // array of size nthreads+1 that defines the slice
    GrB_Matrix *Bslice, // array of output slices, of size nthreads
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (B, "B to slice", GB0)) ;
    ASSERT (nthreads > 1) ;
    ASSERT (Bslice != NULL) ;
    ASSERT (Slice != NULL) ;
    ASSERT (Slice [0] == 0) ;
    ASSERT (Slice [nthreads] == B->nvec) ;
    for (int t = 0 ; t < nthreads ; t++)
    {
        // printf ("Slice [%d] = "GBd"\n", t, Slice [t]) ;
        ASSERT (Slice [t] <= Slice [t+1]) ;
    }
    // printf ("Slice [%d] = "GBd"\n", nthreads, Slice [nthreads]) ;

    GrB_Info info ;

    //--------------------------------------------------------------------------
    // create the slices or hyperslices
    //--------------------------------------------------------------------------

    for (int t = 0 ; t < nthreads ; t++)
    {
        // printf ("\n================== slice %d\n", t) ;

        // Bslice [t] = B (:, bcol_first:bcol_last)
        int64_t bvec_first  = Slice [t] ;
        int64_t bvec_last   = Slice [t+1] - 1 ;
        int64_t bslice_nvec = bvec_last - bvec_first + 1 ;

        // printf ("first "GBd" last "GBd" nvec "GBd"\n", 
        // bvec_first, bvec_last, bslice_nvec) ;

        // allocate just the header for Bslice [t]; all content will be shallow
        Bslice [t] = NULL ;
        GB_NEW (&(Bslice [t]), B->type, B->vlen, B->vdim, GB_Ap_null,
            B->is_csc, GB_SAME_HYPER_AS (B->is_hyper), B->hyper_ratio,
            bslice_nvec, NULL) ;
        if (info != GrB_SUCCESS)
        {
            // out of memory
            for (int i = 0 ; i < t ; i++)
            { 
                GB_MATRIX_FREE (&(Bslice [i])) ;
            }
            return (GB_OUT_OF_MEMORY) ;
        }

        // Bslice [t] is a slice or hyperslice
        (Bslice [t])->is_slice = true ;

        // Bslice has shallow pointers into B->i and B->x
        (Bslice [t])->i = B->i ; (Bslice [t])->i_shallow = true ;
        (Bslice [t])->x = B->x ; (Bslice [t])->x_shallow = true ;
        (Bslice [t])->h_shallow = true ;

        // Bslice->h hyperlist
        if (B->is_hyper)
        { 
            // the columns of Bslice [t] are B->h [bvec_first:bvec_last].
            // Bslice [t] is a hyperslice (with an explict h list).
            (Bslice [t])->h = B->h + bvec_first ;
            (Bslice [t])->hfirst = 0 ;      // unused
        }
        else
        { 
            // the columns of Bslice [t] are [bvec_first:bvec_last].
            // Bslice [t] is a slice (with an implicit h list)
            (Bslice [t])->h = NULL ;
            (Bslice [t])->hfirst = bvec_first ;
        }

        // Bslice->p pointers
        (Bslice [t])->p = B->p + bvec_first ;
        (Bslice [t])->p_shallow = true ;
        (Bslice [t])->plen = bslice_nvec ;

        (Bslice [t])->nvec = bslice_nvec ;

        if (B->nvec_nonempty == B->nvec)
        {
            // all vectors present in B, so all vectors present in the slice
            (Bslice [t])->nvec_nonempty = bslice_nvec ;
        }
        else
        {
            (Bslice [t])->nvec_nonempty = -1 ;
        }

        (Bslice [t])->nzmax = B->nzmax ;
        (Bslice [t])->magic = GB_MAGIC ;

        ASSERT_OK (GB_check (Bslice [t], "Bslice", GB0)) ;
    }

    //--------------------------------------------------------------------------
    // return the slices
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

