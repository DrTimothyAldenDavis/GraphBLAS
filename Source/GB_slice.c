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

// This function does only O(nthreads) work and allocates O(nthreads) space.

#include "GB.h"

GrB_Info GB_slice       // slice B into nthreads slices
(
    GrB_Matrix B,       // matrix to slice
    int nthreads,       // # of slices to create
    int64_t *Slice,     // array of size nthreads+1 that defines the slice
    GrB_Matrix **BsliceHandle,
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (B, "B to slice", GB3)) ;
    ASSERT (nthreads > 1) ;
    ASSERT (nthreads <= B->vdim) ;
    ASSERT (BsliceHandle != NULL) ;
    ASSERT (Slice != NULL) ;
    ASSERT (Slice [0] == 0) ;
    ASSERT (Slice [nthreads] == B->nvec) ;
    for (int t = 0 ; t < nthreads ; t++)
    {
        ASSERT (Slice [t] < Slice [t+1]) ;
    }

    GrB_Info info ;
    (*BsliceHandle) = NULL ;

    //--------------------------------------------------------------------------
    // create the slices
    //--------------------------------------------------------------------------

    GrB_Matrix *Bslice ;
    GB_CALLOC_MEMORY (Bslice, nthreads, sizeof (GrB_Matrix), NULL) ;
    if (Bslice == NULL)
    {
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }

    for (int t = 0 ; t < nthreads ; t++)
    {
        // Bslice [t] = B (:, bcol_first:bcol_last)
        int64_t bvec_first  = Slice [t] ;
        int64_t bvec_last   = Slice [t+1] - 1 ;
        int64_t bslice_nvec = bvec_last - bvec_first + 1 ;

        // allocate just the header for Bslice [t]; all content will be shallow
        GB_NEW (&(Bslice [t]), B->type, B->vlen, B->vdim, GB_Ap_null,
            B->is_csc, GB_FORCE_HYPER, GB_ALWAYS_HYPER, bslice_nvec, NULL) ;
        if (info != GrB_SUCCESS)
        {
            // out of memory
            GB_FREE_SLICE (*BsliceHandle, nthreads) ;
            return (GB_OUT_OF_MEMORY) ;
        }

        (Bslice [t])->i = B->i ; (Bslice [t])->i_shallow = true ;
        (Bslice [t])->x = B->x ; (Bslice [t])->x_shallow = true ;

        if (B->is_hyper)
        {
            // the columns of Bslice [t] are B->h [bvec_first:bvec_last].
            // Bslice [t] is a hypersparse matrix (with an explict h list).
            (Bslice [t])->h = B->h + bvec_first ;
            (Bslice [t])->hfirst = 0 ;      // unused
            (Bslice [t])->h_shallow = true ;
        }
        else
        {
            // the columns of Bslice [t] are [bvec_first:bvec_last].
            // Bslice [t] is an implicit hypersparse matrix>
            (Bslice [t])->h = NULL ;
            (Bslice [t])->hfirst = bvec_first ;
            (Bslice [t])->h_shallow = false ;
        }

        (Bslice [t])->p = B->p + bvec_first ;
        (Bslice [t])->p_shallow = true ;
        (Bslice [t])->nvec = bslice_nvec ;

        // TODO: change GB_matvec_check so that p[0] need not be zero.  If the
        // matrix is hypersparse, then h can be null, in which case the h list
        // is implicitly [hfirst..hfirst-nvec-1].  That is, h[k] == hfirst+k.
        // When h is NULL but is_hyper is true, the matrix is "implicitly
        // hypersparse" (since h is implicit).

        ASSERT_OK (GB_check (Bslice [t], "Bslice", GB3)) ;
    }

    //--------------------------------------------------------------------------
    // return the slices
    //--------------------------------------------------------------------------

    (*BsliceHandle) = Bslice ;
    return (GrB_SUCCESS) ;
}

