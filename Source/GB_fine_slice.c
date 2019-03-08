//------------------------------------------------------------------------------
// GB_fine_slice: create fine hyperslices of B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// For each thread t, create Bslice [t] as a fine hyperslice of B.  The i and x
// arrays are the same as B.  When this function returns, the rest of GraphBLAS
// will view Bslice [t] as a hyperslice, but with non-shallow Bslice [t]->p and
// either shallow Bslice [t]->h (if B is hypersparse) or non-shallow
// Bslice [t]->h (if B is sparse).

// For each fine hyperslice, Bslice [t]->p is allocated and created here; it is
// not shallow (unlike the coarse slices computed by GB_slice).

// Bslice [t]->i and Bslice [t]->x are offset pointers into B, so that
// Bslice [t]->p [0] == 0 for all slices t.

// if B is hypersparse, then Bslice [t]->h is a shallow pointer into B->h,
// where Bslice [t]->h [0] is the same as B->h [k] if the kth vector of B
// is the first vector of Bslice [t].

// The matrix dimensions of each slice are the same as B.  All slices have
// vector length B->vlen and vector dimension B->vdim.   The slices are subsets
// of the entries of B, as defined by the Slice array.  The Bslice [t] consists
// of the entries Slice [t] to Slice [t+1]-1 of B.

// This function does O(nthreads+B->nvec) work and allocates up to
// O(nthreads+B->nvec) space, so it could be parallel, but it will tend to be
// used when B->nvec is small (even 1, for GrB_mxv and GrB_vxm).  So it does
// not need to be parallel.

#include "GB.h"

GrB_Info GB_fine_slice  // slice B into nthreads fine hyperslices
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
    ASSERT (Slice [nthreads] == GB_NNZ (B)) ;
//  printf ("nthreads %d\n", nthreads) ;
    for (int t = 0 ; t < nthreads ; t++)
    {
//      printf ("Slice [%d] = "GBd"\n", t, Slice [t]) ;
        ASSERT (Slice [t] <= Slice [t+1]) ;
    }
//  printf ("Slice [%d] = "GBd"\n", nthreads, Slice [nthreads]) ;

    GrB_Info info ;

    //--------------------------------------------------------------------------
    // create the hyperslices
    //--------------------------------------------------------------------------

    for (int t = 0 ; t < nthreads ; t++)
    {
//      printf ("\n================== fine slice %d\n", t) ;

        // Bslice [t] will contain entries pfirst:plast-1 of B.
        int64_t pfirst = Slice [t] ;
        int64_t plast  = Slice [t+1] - 1 ;
        int64_t bslice_nz = plast - pfirst + 1 ;
        int64_t bvec_first = 0 ;
        int64_t bvec_last = 0 ;
        int64_t bslice_nvec = 0 ;

//      printf ("pfirst "GBd" plast "GBd" nz "GBd"\n", 
//          pfirst, plast, bslice_nz) ;

        if (bslice_nz > 0)
        {

            // find the first column of Bslice [t]: the column that contains
            // the entry at Bi [pfirst] and Bx [pfirst]
            int64_t pright = B->nvec ;
            bool found ;
            GB_BINARY_SPLIT_SEARCH (pfirst, B->p, bvec_first, pright, found) ;
            if (!found)
            { 
                bvec_first-- ;
            }
//          printf ("bvec_first: "GBd"\n", bvec_first) ;
//          printf ("pfirst: "GBd"\n", pfirst) ;
            ASSERT (B->p [bvec_first] <= pfirst) ;
            ASSERT (pfirst <= B->p [bvec_first+1]) ;

            // find the last column of Bslice [t]: the column that contains
            // the entry at Bi [plast] and Bx [plast]
            int64_t bvec_last = bvec_first ;
            pright = B->nvec ;
            GB_BINARY_SPLIT_SEARCH (plast, B->p, bvec_last, pright, found) ;
            if (!found)
            { 
                bvec_last-- ;
            }
//          printf ("bvec_last "GBd"\n", bvec_last) ;
            ASSERT (B->p [bvec_last] <= plast && plast < B->p [bvec_last+1]) ;

            // total number of vectors in B
            bslice_nvec = bvec_last - bvec_first + 1 ;

//          printf ("vec first "GBd" vec last "GBd" nvec "GBd"\n", 
//              bvec_first, bvec_last, bslice_nvec) ;
        }

        // allocate Bslice [t].  Bslice [t]->p is always allocated.  Bslice [t]
        // will always eventually be hypersparse.  However, Bslice[t]->h will
        // be a shallow offset into B->h if B is hypersparse, so GB_new should
        // not allocate h (initially creating a non-hypersparse Bslice [t]).
        // If B is not hypersparse, then Bslice[t]->h must be allocated.  As a
        // result, GB_new should create Bslice [t] as initially hypersparse if
        // B is not hypersparse.  Thus, in both cases, GB_new constructs
        // Bslice [t] with the opposite hypersparsity status of B.

        Bslice [t] = NULL ;
        GB_NEW (&(Bslice [t]), B->type, B->vlen, B->vdim, GB_Ap_malloc,
            B->is_csc, GB_SAME_HYPER_AS (!(B->is_hyper)), GB_ALWAYS_HYPER,
            bslice_nvec, NULL) ;
        if (info != GrB_SUCCESS)
        {
            // out of memory
            for (int i = 0 ; i < t ; i++)
            { 
                GB_MATRIX_FREE (&(Bslice [i])) ;
            }
            return (info) ;
        }

        // Bslice [t] is always a hyperslice
        (Bslice [t])->is_hyper = true ;
        (Bslice [t])->is_slice = true ;
        (Bslice [t])->hfirst = 0 ;      // unused
        (Bslice [t])->plen = bslice_nvec ;
        (Bslice [t])->nvec = bslice_nvec ;

        // Bslice has shallow pointers into B->i and B->x
        (Bslice [t])->i = B->i + pfirst ;
        (Bslice [t])->i_shallow = true ;
        (Bslice [t])->x = B->x + pfirst * B->type->size ;
        (Bslice [t])->x_shallow = true ;

        // Bslice->h hyperlist
        if (B->is_hyper)
        { 
            // the columns of Bslice [t] are B->h [bvec_first:bvec_last].
            // Bslice [t] is a hyperslice (with an explict h list, as a
            // shallow pointer into B->h).
            ASSERT ((Bslice [t])->h == NULL) ;
            (Bslice [t])->h = B->h + bvec_first ;
            (Bslice [t])->h_shallow = true ;
        }
        else
        { 
            // the columns of Bslice [t] are [bvec_first:bvec_last].
            // Bslice [t] is a hyperslice (with an explicit h list)
            GB_MALLOC_MEMORY ((Bslice [t])->h, bslice_nvec, sizeof (int64_t)) ;
            (Bslice [t])->h_shallow = false ;
            if ((Bslice [t])->h == NULL)
            {
                // out of memory
                for (int i = 0 ; i <= t ; i++)
                { 
                    GB_MATRIX_FREE (&(Bslice [i])) ;
                }
                return (GB_OUT_OF_MEMORY) ;
            }
            for (int64_t k = 0 ; k < bslice_nvec ; k++)
            {
                (Bslice [t])->h [k] = bvec_first + k ;
            }
        }

        // Bslice->p is always allocated fresh by GB_new.
        ASSERT (!(Bslice [t])->p_shallow) ;
        (Bslice [t])->p [0] = 0 ;
        for (int64_t k = 1 ; k < bslice_nvec ; k++)
        {
            (Bslice [t])->p [k] = B->p [bvec_first + k] - pfirst ;
        }
        (Bslice [t])->p [bslice_nvec] = bslice_nz ;
        (Bslice [t])->nvec_nonempty = -1 ;

        (Bslice [t])->nzmax = bslice_nz ;
        (Bslice [t])->magic = GB_MAGIC ;

        ASSERT_OK (GB_check (Bslice [t], "Bslice", GB0)) ;
    }

    //--------------------------------------------------------------------------
    // return the slices
    //--------------------------------------------------------------------------

    return (GrB_SUCCESS) ;
}

