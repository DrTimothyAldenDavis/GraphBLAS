//------------------------------------------------------------------------------
// GB_vcat_slice: vertical concatenation of the slices of C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Vertical concatenation of slices into the matrix C.

#include "GB.h"

GrB_Info GB_vcat_slice      // vertical concatenation of the slices of C
(
    GrB_Matrix *Chandle,    // output matrix C to create
    int nthreads,           // # of slices to concatenate
    GrB_Matrix *Cslice,     // array of slices of size nthreads
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (nthreads > 1) ;
    ASSERT (Chandle != NULL) ;
    ASSERT (*Chandle == NULL) ;
    ASSERT (Cslice != NULL) ;
    for (int tid = 0 ; tid < nthreads ; tid++)
    {
        ASSERT_OK (GB_check (Cslice [tid], "a slice of C", GB0)) ;
        ASSERT (!GB_PENDING (Cslice [tid])) ;
        ASSERT (!GB_ZOMBIES (Cslice [tid])) ;
        ASSERT ((Cslice [tid])->is_hyper) ;
        // each Cslice [tid] is constructed as its own matrix, with
        // Cslice [tid] = Aslice [tid] * B.  It is not a slice of an other
        // matrix, so Cslice [tid]->is_slice is false.
        ASSERT (!(Cslice [tid])->is_slice) ;
        ASSERT ((Cslice [tid])->type == (Cslice [0])->type) ;
        ASSERT ((Cslice [tid])->vlen == (Cslice [0])->vlen) ;
        ASSERT ((Cslice [tid])->vdim == (Cslice [0])->vdim) ;
    }

    //--------------------------------------------------------------------------
    // find the size and type of C
    //--------------------------------------------------------------------------

    // all the slices have the same type and dimension
    GrB_Type ctype = (Cslice [0])->type ;
    int64_t  cvlen = (Cslice [0])->vlen ;
    int64_t  cvdim = (Cslice [0])->vdim ;
    size_t   csize = ctype->size ;

    int64_t cnz = 0 ;               // sum of nnz (Cslice [0..nthreads-1])
    int64_t cnvec = 0 ;             // sum of # vectors in Cslices, not in C
    int64_t cnvec_nonempty = 0 ;    // computed later

    for (int tid = 0 ; tid < nthreads ; tid++)
    {
        // compute the cumulative sum of the # entries and # vectors
        cnz   += GB_NNZ (Cslice [tid]) ;
        cnvec += (Cslice [tid])->nvec ;
    }

    //--------------------------------------------------------------------------
    // merge the vectors of C
    //--------------------------------------------------------------------------

    // TODO: do hypersparse case.  The standard sparse case will work for all
    // cases, but it can take too much time if C and the Cslices are very
    // hypersparse.

    GrB_Info info ;
    GrB_Matrix C = NULL ;

    // if (cnvec >= cvlen/8)
    {

        //----------------------------------------------------------------------
        // result C is not hypersparse
        //----------------------------------------------------------------------

        // allocate the result C
        GB_CREATE (Chandle, ctype, cvlen, cvdim, GB_Ap_calloc, true,
            GB_FORCE_NONHYPER, GB_Global_hyper_ratio_get ( ), cvlen, cnz, true,
            Context) ;
        if (info != GrB_SUCCESS)
        {
            // out of memory
            return (GB_OUT_OF_MEMORY) ;
        }

        C = (*Chandle) ;

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        int64_t *restrict Tp = NULL ;
        GB_MALLOC_MEMORY (Tp, cvdim+1, sizeof (int64_t)) ;
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
        // 1st phase: count the entries in each column of C
        //----------------------------------------------------------------------

        // Note the parallelism is in the inner loop, not the outer one,
        // because of the reduction on Cp [j].  Also in the 2nd phase below.

        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            int64_t *restrict Cslice_h = (Cslice [tid])->h ;
            int64_t *restrict Cslice_p = (Cslice [tid])->p ;
            int64_t cslice_nvec = (Cslice [tid])->nvec ;

            #pragma omp parallel for
            for (int64_t k = 0 ; k < cslice_nvec ; k++)
            {
                // sum up the number of entries in C(:,j)
                int64_t j = Cslice_h [k] ;
                int64_t csjnz = Cslice_p [k+1] - Cslice_p [k] ;
                Cp [j] += csjnz ;
            }
        }

        //----------------------------------------------------------------------
        // Tp = Cp = cumsum (Cp), and also compute C->nvec_nonempty
        //----------------------------------------------------------------------

        GB_cumsum (Cp, cvdim, &cnvec_nonempty, Context) ;
        C->nvec_nonempty = cnvec_nonempty ;
        memcpy (Tp, Cp, (cvdim+1) * sizeof (int64_t)) ;

        //----------------------------------------------------------------------
        // 2nd phase: copy each slice into C
        //----------------------------------------------------------------------

        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            int64_t *restrict Cslice_h = (Cslice [tid])->h ;
            int64_t *restrict Cslice_p = (Cslice [tid])->p ;
            int64_t *restrict Cslice_i = (Cslice [tid])->i ;
            GB_void *restrict Cslice_x = (Cslice [tid])->x ;
            int64_t cslice_nvec = (Cslice [tid])->nvec ;

            #pragma omp parallel for
            for (int64_t k = 0 ; k < cslice_nvec ; k++)
            {
                // copy the entries into C(:,j)
                int64_t j = Cslice_h [k] ;
                int64_t pslice = Cslice_p [k] ;
                int64_t csjnz = Cslice_p [k+1] - pslice ;
                int64_t p = Tp [j] ;
                memcpy (Ci + p, Cslice_i + pslice, csjnz * sizeof (int64_t)) ;
                memcpy (Cx + p*csize, Cslice_x + pslice*csize, csjnz * csize) ;
                Tp [j] += csjnz ;
            }
        }

        //----------------------------------------------------------------------
        // free workspace
        //----------------------------------------------------------------------

        GB_FREE_MEMORY (Tp, cvdim+1, sizeof (int64_t)) ;
    }

    #if 0
    else
    {

        //----------------------------------------------------------------------
        // C is hypersparse
        //----------------------------------------------------------------------

        // C will be hypersparse, so sort and merge the nonempty columns
        // of each slice.

        // TODO
        // fprintf (stderr, "TODO\n") ; ASSERT (0) ; abort ( ) ;
    }
    #endif

    //--------------------------------------------------------------------------
    // finalize the matrix
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_OK (GB_check (C, "C from horizontal concatenation", GB0)) ;
    return (GrB_SUCCESS) ;
}

