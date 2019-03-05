//------------------------------------------------------------------------------
// GB_vcat_slice: vertical concatenation of the slices of C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Vertical concatenation of slices into the matrix C.

#include "GB.h"

GrB_Info GB_vcat_slice      // horizontal concatenation of the slices of C
(
    GrB_Matrix *Chandle,    // output matrix C to create
    int nthreads,           // # of slices to concatenate
    GrB_Matrix *Cslice,     // array of slices of size nthreads
    GB_Context Context
)
{
    abort ( ) ; // TODO

#if 0

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (nthreads > 1) ;
    ASSERT (Chandle != NULL) ;
    ASSERT (*Chandle == NULL) ;
    ASSERT (Cslice != NULL) ;
    for (int t = 0 ; t < nthreads ; t++)
    {
        ASSERT_OK (GB_check (Cslice [t], "a slice of C", GB0)) ;
        ASSERT (!GB_PENDING (Cslice [t])) ;
        ASSERT (!GB_ZOMBIES (Cslice [t])) ;
        ASSERT ((Cslice [t])->is_hyper) ;
        // each Cslice [t] is constructed as its own matrix, with Cslice [t] =
        // A * Bslice [t].  It is not a slice of an other matrix, so Cslice
        // [t]->is_slice is false.
        ASSERT (!(Cslice [t])->is_slice) ;
        ASSERT ((Cslice [t])->type == (Cslice [0])->type) ;
        ASSERT ((Cslice [t])->vlen == (Cslice [0])->vlen) ;
        ASSERT ((Cslice [t])->vdim == (Cslice [0])->vdim) ;
    }

    //--------------------------------------------------------------------------
    // find the size and type of C
    //--------------------------------------------------------------------------

    // all the slices have the same type and dimension
    GrB_Type ctype = (Cslice [0])->type ;
    int64_t  cvlen = (Cslice [0])->vlen ;
    int64_t  cvdim = (Cslice [0])->vdim ;
    size_t   csize = ctype->size ;

    int64_t cnz = 0 ;
    int64_t cnvec = 0 ;             // max # of vectors, not total
    int64_t cnvec_nonempty = 0 ;    // computed later

    for (int t = 0 ; t < nthreads ; t++)
    {
        // compute the cumulative sum of the # entries and # vectors
        cnz   += GB_NNZ (Cslice [t]) ;
        cnvec += (Cslice [t])->nvec ;
    }

    //--------------------------------------------------------------------------
    // merge the vectors of C
    //--------------------------------------------------------------------------

    GrB_Info info ;

    if (cnvec >= cvlen/16)
    {

        //----------------------------------------------------------------------
        // result C is not hypersparse
        //----------------------------------------------------------------------

        // allocate the result C
        GB_CREATE (Chandle, ctype, cvlen, cvdim, GB_Ap_calloc, true,
            GB_FORCE_NONHYPER, GB_Global.hyper_ratio, cvlen, cnz, true,
            Context) ;
        if (info != GrB_SUCCESS)
        {
            // out of memory
            return (GB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        int64_t *restrict Tp = NULL ;
        GB_MALLOC_MEMORY (Tp, n, sizeof (int64_t)) ;
        if (Tp == NULL)
        {
            // out of memory
            GrB_free (Chandle) ;
            return (GB_OUT_OF_MEMORY) ;
        }

        int64_t *restrict Cp = C->p ;
        int64_t *restrict Ci = C->i ;
        GB_void *restrict Cx = C->x ;

        //----------------------------------------------------------------------
        // count the entries in each column of C
        //----------------------------------------------------------------------

        for (int t = 0 ; t < nthreads ; t++)
        {
            int64_t *restrict Csliceh = (Cslice [t])->h ;
            int64_t *restrict Cslicep = (Cslice [t])->p ;
            int64_t cslice_nvec = (Cslice [t])->nvec ;

            #pragma omp parallel for
            for (int64_t k = 0 ; k < cslice_nvec ; k++)
            {
                // sum up the number of entries in C(:,j)
                int64_t j = Csliceh [k] ;
                int64_t csjnz = Cslicep [k+1] - Cslicep [k] ;
                Cp [j] += csjnz ;
            }
        }

        //----------------------------------------------------------------------
        // Tp = Cp = cumsum (Cp)
        //----------------------------------------------------------------------

        GB_cumsum (Cp, cvdim, &cnvec_nonempty, Context) ;
        C->nvec_nonempty = cnvec_nonempty ;
        memcpy (Tp, Cp, n * sizeof (int64_t)) ;

        //----------------------------------------------------------------------
        // copy each slice into C
        //----------------------------------------------------------------------

        for (int t = 0 ; t < nthreads ; t++)
        {
            int64_t *restrict Csliceh = (Cslice [t])->h ;
            int64_t *restrict Cslicep = (Cslice [t])->p ;
            int64_t *restrict Cslicei = (Cslice [t])->i ;
            GB_void *restrict Cslicex = (Cslice [t])->x ;
            int64_t cslice_nvec = (Cslice [t])->nvec ;

            #pragma omp parallel for
            for (int64_t k = 0 ; k < cslice_nvec ; k++)
            {
                // copy the entries into C(:,j)
                int64_t j = Csliceh [k] ;
                int64_t pslice = Cslicep [k] ;
                int64_t csjnz = Cslicep [k+1] - pslice ;
                int64_t p = Tp [j] ;
                Tp [j] += csjnz ;
                memcpy (Ci + p, Cslicei + pslice, csjnz * sizeof (int64_t)) ;
                memcpy (Cx + p*csize, Cslicex + pslice*csize, csjnz * csize) ;
            }
        }

        //----------------------------------------------------------------------
        // free workspace
        //----------------------------------------------------------------------

        GB_FREE_MEMORY (Tp, n, sizeof (int64_t)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C is hypersparse
        //----------------------------------------------------------------------

        // C will be hypersparse, so sort and merge the nonempty columns
        // of each slice.

        // TODO
        printf ("TODO\n") ;
        fprintf (stderr, "TODO\n") ;
        ASSERT (0) ;
        abort ( ) ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix, free workspace and return
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_OK (GB_check (C, "C from horizontal concatenation", GB0)) ;
#endif

    return (GrB_SUCCESS) ;
}

