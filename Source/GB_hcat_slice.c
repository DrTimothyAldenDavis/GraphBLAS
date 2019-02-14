//------------------------------------------------------------------------------
// GB_hcat_slice: horizontal concatenation of the slices of C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Horizontal concatenation of slices into the matrix C.

// PARALLEL: trivial

// TODO this is drafted but not tested

#include "GB.h"

GrB_Info GB_hcat_slice      // horizontal concatenation of the slices of C
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
    for (int t = 0 ; t < nthreads ; t++)
    {
        ASSERT_OK (GB_check (Cslice [t], "a slice of C", GB3)) ;
        ASSERT (!GB_PENDING (Cslice [t])) ;
        ASSERT (!GB_ZOMBIES (Cslice [t])) ;
        ASSERT ((Cslice [t])->is_hyper) ;
    }

    //--------------------------------------------------------------------------
    // find the size and type of C
    //--------------------------------------------------------------------------

    // Let cnz_slice [t] be the number of entries in Cslice [t], and let
    // cnvec_slice [t] be the number vectors in Cslice [t].  Then Cnzs and
    // Cnvecs are cumulative sums of cnz_slice and cnvec_slice, respectively:

    // Cnzs   [t] = sum of cnz_slice   [0:t-1]
    // Cnvecs [t] = sum of cnvec_slice [0:t-1]

    // both arrays are size nthreads+1.  Thus, both Cnzs [0] and Cnvecs [0]
    // are zero, and their last entries are the total # entries and vectors
    // in C, respectively.

    int64_t Cnzs   [nthreads+1] ;
    int64_t Cnvecs [nthreads+1] ;

    // all the slices have the same type and dimension
    GrB_Type ctype = (Cslice [0])->type ;
    int64_t  cvlen = (Cslice [0])->vlen ;
    int64_t  cvdim = (Cslice [0])->vdim ;

    int64_t cnz = 0 ;
    int64_t cnvec = 0 ;
    int64_t cnvec_nonempty = 0 ;
    for (int t = 0 ; t < nthreads ; t++)
    {
        // compute the cumulative sum of the # entries and # vectors
        Cnzs   [t] = cnz ;
        Cnvecs [t] = cnvec ;
        cnz   += GB_NNZ (Cslice [t]) ;
        cnvec += (Cslice [t])->nvec ;
        // also sum the total number of non-empty vectors in all the slices
        cnvec_nonempty += (Cslice [t])->nvec_nonempty ;
    }

    Cnzs   [nthreads] = cnz ;       // total # entries in C
    Cnvecs [nthreads] = cnvec ;     // total # vectors in C

    //--------------------------------------------------------------------------
    // create C and allocate all of its space
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CREATE (Chandle, ctype, cvlen, cvdim, GB_Ap_malloc, true,
        GB_FORCE_HYPER, GB_Global.hyper_ratio, cnvec, cnz, true, Context) ;
    if (info != GrB_SUCCESS)
    {
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }

    GrB_Matrix C = (*Chandle) ;

    int64_t *restrict Ch = C->h ;
    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ci = C->i ;
    GB_void *restrict Cx = C->x ;
    size_t  csize = ctype->size ;

    C->nvec_nonempty = cnvec_nonempty ;
    C->nvec = cnvec ;
    Cp [cnvec] = cnz ;

    //--------------------------------------------------------------------------
    // copy each slice into C
    //--------------------------------------------------------------------------

    // for all threads in parallel; no dependencies.  The work of each
    // thread easily vectorizes as well.
    for (int t = 0 ; t < nthreads ; t++)
    {
        // get the Cslice [t] and its position in C
        int64_t *restrict Csliceh = (Cslice [t])->h ;
        int64_t *restrict Cslicep = (Cslice [t])->p ;
        int64_t *restrict Cslicei = (Cslice [t])->i ;
        GB_void *restrict Cslicex = (Cslice [t])->x ;
        int64_t cnz         = Cnzs   [t] ;
        int64_t cnz_slice   = Cnzs   [t+1] - cnz ;
        int64_t cnvec       = Cnvecs [t] ;
        int64_t cnvec_slice = Cnvecs [t+1] - cnvec ;

        // copy the row indices and values of Cslice [t] into Ci and Cx
        memcpy (Ci + cnz        , Cslicei, cnz_slice * sizeof (int64_t)) ;
        memcpy (Cx + cnz * csize, Cslicex, cnz_slice * csize) ;

        // copy the column indices of Cslice into Ch
        memcpy (Ch + cnvec, Csliceh, cnvec_slice * sizeof (int64_t)) ;

        // construct the column pointers of C (shift upwards by cnz)
        for (int64_t k = 0 ; k < cnvec_slice ; k++)
        {
            Cp [k + cnvec] = Cslicep [k] + cnz ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix, free workspace and return
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_OK (GB_check (C, "C from horizontal concatenation", GB3)) ;
    return (GrB_SUCCESS) ;
}

