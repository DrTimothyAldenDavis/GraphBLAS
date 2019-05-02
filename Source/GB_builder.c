//------------------------------------------------------------------------------
// GB_builder: build a matrix from tuples
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// CALLED BY: GB_build, GB_wait, and GB_transpose

// This function is called by GB_build to build a matrix T for GrB_Matrix_build
// or GrB_Vector_build, by GB_wait to build a matrix T from the list of pending
// tuples, and by GB_transpose to transpose a matrix or vector.  Duplicates can
// appear if called by GB_build or GB_wait, but not GB_transpose.

// The indices are provided either as (I,J) or (iwork,jwork), not both.

// PARALLEL: done

#include "GB.h"
#ifndef GBCOMPACT
#include "GB_red__include.h"
#endif

#define GB_IWORK(t) (((t) < 0) ? -1 : iwork [t])
#define GB_JWORK(t) (((t) < 0) ? -1 : ((jwork == NULL) ? 0 : jwork [t]))
#define GB_KWORK(t) (((t) < 0) ? -1 : ((kwork == NULL) ? t : kwork [t]))

#define GB_FREE_WORK                                            \
{                                                               \
    GB_FREE_MEMORY (*iwork_handle, ijlen, sizeof (int64_t)) ;   \
    GB_FREE_MEMORY (*jwork_handle, ijlen, sizeof (int64_t)) ;   \
    GB_FREE_MEMORY (kwork,         nvals, sizeof (int64_t)) ;   \
}

GrB_Info GB_builder                 // build a matrix from tuples
(
    // matrix to build:
    GrB_Matrix *Thandle,            // matrix T to build
    const GrB_Type ttype,           // type of output matrix T
    const int64_t vlen,             // length of each vector of T
    const int64_t vdim,             // number of vectors in T
    const bool is_csc,              // true if T is CSC, false if CSR
    // if iwork is NULL then these are not yet allocated, or known:
    int64_t **iwork_handle,         // for (i,k) or (j,i,k) tuples
    int64_t **jwork_handle,         // for (j,i,k) tuples
    bool known_sorted,              // true if tuples known to be sorted
    bool known_no_duplicates,       // true if tuples known to not have dupl
    int64_t ijlen,                  // size of iwork and jwork arrays
    // only used if iwork is NULL:
    const bool is_matrix,           // true if T a GrB_Matrix, false if vector
    const bool ijcheck,             // true if I,J must be checked
    // original inputs:
    const int64_t *I,               // original indices, size nvals
    const int64_t *J,               // original indices, size nvals
    const GB_void *S,               // array of values of tuples, size nvals
    const int64_t nvals,            // number of tuples, and size of kwork
    const GrB_BinaryOp dup,         // binary function to assemble duplicates,
                                    // if NULL use the "SECOND" function to
                                    // keep the most recent duplicate.
    const GB_Type_code scode,       // GB_Type_code of S array
    GB_Context Context
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (Thandle != NULL) ;
    ASSERT (GB_IMPLIES (nvals > 0, S != NULL)) ;
    ASSERT (nvals >= 0) ;
    ASSERT (scode <= GB_UDT_code) ;
    ASSERT_OK (GB_check (ttype, "ttype for builder", GB0)) ;
    ASSERT_OK_OR_NULL (GB_check (dup, "dup for builder", GB0)) ;
    ASSERT (iwork_handle != NULL) ;
    ASSERT (jwork_handle != NULL) ;

    //==========================================================================
    // symbolic phase of the build =============================================
    //==========================================================================

    // The symbolic phase sorts the tuples and finds any duplicates.  The
    // output matrix T is constructed (not including T->i and T->x), and T->h
    // and T->p are computed.  Then iwork is transplanted into T->i, or T->i is
    // allocated.  T->x is then allocated.  It is not computed until the
    // numeric phase.

    // When this function returns, iwork is either freed or transplanted into
    // T->i.  jwork is freed, and the iwork and jwork pointers (in the caller)
    // are set to NULL by setting their handles to NULL.  Note that jwork may
    // already be NULL on input, if T has one or zero vectors (jwork_handle is
    // always non-NULL however).

    GrB_Info info ;
    GrB_Matrix T = NULL ;
    (*Thandle) = NULL ;
    int64_t *restrict iwork = *iwork_handle ;
    int64_t *restrict jwork = *jwork_handle ;
    int64_t *restrict kwork = NULL ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;

    //--------------------------------------------------------------------------
    // partition the tuples for the threads
    //--------------------------------------------------------------------------

    // Thread tid handles tuples tstart_slice [tid] to tstart_slice [tid+1]-1.
    // Each thread handles about the same number of tuples.  This partition
    // depends only on nvals.

    int64_t tstart_slice [nthreads+1] ; // first tuple in each slice
    tstart_slice [0] = 0 ;
    for (int tid = 1 ; tid < nthreads ; tid++)
    {
        tstart_slice [tid] = GB_PART (tid, nvals, nthreads) ;
    }
    tstart_slice [nthreads] = nvals ;

    // tnvec_slice [tid]: # of vectors that start in a slice.  If a vector
    //                    starts in one slice and ends in another, it is
    //                    counted as being in the first slice.
    // tnz_slice   [tid]: # of entries in a slice after removing duplicates

    int64_t tnvec_slice [nthreads+1] ;
    int64_t tnz_slice   [nthreads+1] ;

    // sentinel values for the final cumulative sum
    tnvec_slice [nthreads] = 0 ;
    tnz_slice   [nthreads] = 0 ;

    // this becomes true if the first pass computes tnvec_slice and tnz_slice,
    // and if the I,J tuples were found to be already sorted with no duplicates
    // present.
    bool tnvec_and_tnz_slice_computed = false ;

    //--------------------------------------------------------------------------
    // copy user input and check if valid
    //--------------------------------------------------------------------------

    if (iwork == NULL)
    {

double ttt = omp_get_wtime ( ) ;

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        // allocate workspace to load and sort the index tuples:

        // vdim <= 1: iwork and kwork for (i,k) tuples, where i = I(k)
        // vdim > 1: also jwork for (j,i,k) tuples where i = I(k) and j = J (k)

        // The k value in the tuple gives the position in the original set of
        // tuples: I[k] and S[k] when vdim <= 1, and also J[k] for matrices
        // with vdim > 1.

        // The workspace iwork and jwork are allocated here but freed (or
        // transplanted) inside GB_builder.  kwork is allocated, used, and
        // freed in GB_builder.

        GB_MALLOC_MEMORY (iwork, nvals, sizeof (int64_t)) ;
        bool ok = (iwork != NULL) ;
        ASSERT (jwork == NULL) ;
        if (vdim > 1)
        { 
            GB_MALLOC_MEMORY (jwork, nvals, sizeof (int64_t)) ;
            ok = ok && (jwork != NULL) ;
        }

        (*iwork_handle) = iwork ;
        (*jwork_handle) = jwork ;
        ijlen = nvals ;

        if (!ok)
        { 
            // out of memory
            GB_FREE_WORK ;
            return (GB_OUT_OF_MEMORY) ;
        }

        //----------------------------------------------------------------------
        // create the tuples to sort, and check for any invalid indices
        //----------------------------------------------------------------------

        known_sorted = true ;
        bool no_duplicates_found = true ;

        if (nvals == 0)
        { 

            //------------------------------------------------------------------
            // nothing to do
            //------------------------------------------------------------------

            ;

        }
        else if (is_matrix)
        {

            //------------------------------------------------------------------
            // C is a matrix; check both I and J
            //------------------------------------------------------------------

            // but if vdim <= 1, do not create jwork
            ASSERT (J != NULL) ;
            ASSERT (iwork != NULL) ;
            ASSERT (vdim >= 0) ;
            ASSERT ((vdim > 1) == (jwork != NULL)) ;
            ASSERT (I != NULL) ;

            int64_t kbad [nthreads] ;

            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(&&:known_sorted) reduction(&&:no_duplicates_found)
            for (int tid = 0 ; tid < nthreads ; tid++)
            {

                kbad [tid] = -1 ;
                int64_t my_tnvec = 0 ;
                int64_t kstart   = tstart_slice [tid] ;
                int64_t kend     = tstart_slice [tid+1] ;
                int64_t ilast = (kstart == 0) ? -1 : I [kstart-1] ;
                int64_t jlast = (kstart == 0) ? -1 : J [kstart-1] ;
                bool my_bad = false ;

                for (int64_t k = kstart ; k < kend ; k++)
                {
                    // get k-th index from user input: (i,j)
                    int64_t i = I [k] ;
                    int64_t j = J [k] ;

                    if (i < 0 || i >= vlen || j < 0 || j >= vdim)
                    { 
                        // halt if out of bounds
                        kbad [tid] = k ;
                        break ;
                    }

                    // check if the tuples are already sorted
                    known_sorted = known_sorted &&
                        ((jlast < j) || (jlast == j && ilast <= i)) ;

                    // check if this entry is a duplicate of the one before it
                    no_duplicates_found = no_duplicates_found &&
                        (!(jlast == j && ilast == i)) ;

                    // copy the tuple into the work arrays to be sorted
                    iwork [k] = i ;
                    if (jwork != NULL)
                    {
                        jwork [k] = j ;
                        if (j > jlast)
                        { 
                            // vector j starts in this slice (but this is 
                            // valid only if J is sorted on input)
                            my_tnvec++ ;
                        }
                    }
           
                    // log the last index seen
                    ilast = i ; jlast = j ;
                }

                // these are valid only if I and J are sorted on input,
                // with no duplicates present.
                tnvec_slice [tid] = my_tnvec ;
                tnz_slice   [tid] = kend - kstart ; 

            }

            // collect the report from each thread
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                if (kbad [tid] >= 0)
                { 
                    // invalid index
                    GB_FREE_WORK ;
                    int64_t i = I [kbad [tid]] ;
                    int64_t j = J [kbad [tid]] ;
                    int64_t row = is_csc ? i : j ;
                    int64_t col = is_csc ? j : i ;
                    int64_t nrows = is_csc ? vlen : vdim ;
                    int64_t ncols = is_csc ? vdim : vlen ;
                    return (GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS, (GB_LOG,
                        "index ("GBd","GBd") out of bounds,"
                        " must be < ("GBd", "GBd")", row, col, nrows, ncols))) ;
                }
            }

            // if the tuples were found to be already in sorted order, and if
            // no duplicates were found, then tnvec_slice and tnz_slice are now
            // valid, Otherwise, they can only be computed after sorting.
            tnvec_and_tnz_slice_computed =
                jwork != NULL && known_sorted && no_duplicates_found ;

        }
        else if (ijcheck)
        {

            //------------------------------------------------------------------
            // C is a typecasted GrB_Vector; check only I
            //------------------------------------------------------------------

            ASSERT (I != NULL) ;
            ASSERT (J == NULL) ;
            ASSERT (vdim == 1) ;
            int64_t kbad [nthreads] ;

            #pragma omp parallel for num_threads(nthreads) schedule(static) \
                reduction(&&:known_sorted) reduction(&&:no_duplicates_found)
            for (int tid = 0 ; tid < nthreads ; tid++)
            {

                kbad [tid] = -1 ;
                int64_t kstart   = tstart_slice [tid] ;
                int64_t kend     = tstart_slice [tid+1] ;
                int64_t ilast = (kstart == 0) ? -1 : I [kstart-1] ;

                for (int64_t k = kstart ; k < kend ; k++)
                {
                    // get k-th index from user input: (i)
                    int64_t i = I [k] ;

                    if (i < 0 || i >= vlen)
                    { 
                        // halt if out of bounds
                        kbad [tid] = k ;
                        break ;
                    }

                    // check if the tuples are already sorted
                    known_sorted = known_sorted && (ilast <= i) ;

                    // check if this entry is a duplicate of the one before it
                    no_duplicates_found = no_duplicates_found &&
                        (!(ilast == i)) ;

                    // copy the tuple into the work arrays to be sorted
                    iwork [k] = i ;

                    // log the last index seen
                    ilast = i ;
                }
            }

            // collect the report from each thread
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                if (kbad [tid] >= 0)
                { 
                    // invalid index
                    GB_FREE_WORK ;
                    int64_t i = I [kbad [tid]] ;
                    return (GB_ERROR (GrB_INDEX_OUT_OF_BOUNDS, (GB_LOG,
                        "index ("GBd") out of bounds, must be < ("GBd")",
                        i, vlen))) ;
                }
            }

        }
        else
        { 

            //------------------------------------------------------------------
            // GB_reduce_to_column: do not check I, assume not sorted
            //------------------------------------------------------------------

            // Many duplicates are possible, since the tuples are being used to
            // construct a single vector.  For a CSC format, each entry A(i,j)
            // becomes an (i,aij) tuple, with the column index j discarded.  All
            // entries in a single row i are reduced to a single entry in the
            // vector.  The input is unlikely to be sorted, so do not bother to
            // check.

            GB_memcpy (iwork, I, nvals * sizeof (int64_t), nthreads) ;
            known_sorted = false ;
        }

        //----------------------------------------------------------------------
        // determine if duplicates are possible
        //----------------------------------------------------------------------

        // The input is now known to be sorted, or not.  If it is sorted, and
        // if no duplicates were found, then it is known to have no duplicates.
        // Otherwise, duplicates might appear, but a sort is required first to
        // check for duplicates.

        known_no_duplicates = known_sorted && no_duplicates_found ;

ttt = omp_get_wtime ( ) - ttt ;
printf ("1st pass   %g\n", ttt) ;

    }


    //--------------------------------------------------------------------------

    ASSERT (iwork != NULL) ;
    ASSERT (GB_IMPLIES (vdim > 1, jwork != NULL)) ;

printf ("builder nthreads %d\n", nthreads) ;
double ttt ;

/*
{
    printf ("nvals: "GBd"\n", nvals) ;
    size_t tsize = ttype->size ;
    size_t ssize = GB_code_size (scode, tsize) ;
    for (int jj = 0 ; jj < nvals ; jj++)
    {
        printf ("%3d i "GBd" j "GBd", value: ", jj,
            GB_IWORK (jj), GB_JWORK (jj)) ;
        GB_code_check (scode, S +(jj)*ssize, stdout, Context) ;
        printf ("\n") ;
    }
}
*/


    //--------------------------------------------------------------------------
    // sort the tuples in ascending order (just the pattern, not the values)
    //--------------------------------------------------------------------------

    if (!known_sorted)
    {

ttt = omp_get_wtime ( ) ;

        // create the k part of each tuple
        GB_MALLOC_MEMORY (kwork, nvals, sizeof (int64_t)) ;
        if (kwork == NULL)
        { 
            // out of memory
            GB_FREE_WORK ;
            return (GB_OUT_OF_MEMORY) ;
        }

        // The k part of each tuple (i,k) or (j,i,k) records the original
        // position of the tuple in the input list.  This allows an unstable
        // sorting algorith to be used.  Since k is unique, it forces the
        // result of the sort to be stable regardless of whether or not the
        // sorting algorithm is stable.  It also keeps track of where the
        // numerical value of the tuple can be found; it is in S[k] for the
        // tuple (i,k) or (j,i,k), regardless of where the tuple appears in the
        // list after it is sorted.
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int64_t k = 0 ; k < nvals ; k++)
        { 
            kwork [k] = k ;
        }

        // sort all the tuples, in parallel
        if (jwork != NULL)
        { 
            // sort a set of (j,i,k) tuples
            GB_qsort_3 (jwork, iwork, kwork, nvals, Context) ;
        }
        else
        { 
            // sort a set of (i,k) tuples
            GB_qsort_2b (iwork, kwork, nvals, Context) ;
        }

ttt = omp_get_wtime ( ) - ttt ;
printf ("sort  time %g\n", ttt) ;

    }
    else
    { 

        // If the tuples were already known to be sorted on input, kwork is
        // NULL.  This implicitly means that kwork [k] = k for all k =
        // 0:nvals-1.  kwork is not allocated.
        ;
    }

/*
{
    printf ("\nsorted, nvals sliced: "GBd"\n", nvals) ;
    size_t tsize = ttype->size ;
    size_t ssize = GB_code_size (scode, tsize) ;
    for (int tid = 0 ; tid < nthreads ; tid++)
    {
        printf ("\n   tid: %d------\n", tid) ;
        for (int jj = tstart_slice [tid] ; jj < tstart_slice [tid+1]  ; jj++)
        {
            printf ("%3d i "GBd" j "GBd", k "GBd" value: ", jj,
            GB_IWORK (jj), GB_JWORK (jj), GB_KWORK (jj)) ;
            GB_code_check (scode, S +(GB_KWORK (jj))*ssize, stdout, Context) ;
            printf ("\n") ;
        }
    }
}
*/

    //--------------------------------------------------------------------------
    // count vectors and duplicates in each slice
    //--------------------------------------------------------------------------

    if (known_no_duplicates)
    {

        //----------------------------------------------------------------------
        // no duplicates: just count # vectors in each slice
        //----------------------------------------------------------------------

        #ifndef NDEBUG
        {
            // assert that there are no duplicates
            int64_t ilast = -1, jlast = -1 ;
            for (int64_t t = 0 ; t < nvals ; t++)
            {
                int64_t i= GB_IWORK (t), j= GB_JWORK (t) ;
                bool is_duplicate = (i == ilast && j == jlast) ;
                ASSERT (!is_duplicate) ;
                ilast = i ; jlast = j ;
            }
        }
        #endif

ttt = omp_get_wtime ( ) ;

        if (jwork == NULL)
        { 

            // all tuples appear in at most one vector, and there are no
            // duplicates, so there is no need to scan iwork or jwork.

            ASSERT (vdim == 0 || vdim == 1) ;
            for (int tid = 0 ; tid < nthreads ; tid++)
            { 
                int64_t tstart = tstart_slice [tid] ;
                int64_t tend   = tstart_slice [tid+1] ;
                tnvec_slice [tid] = 0 ;
                tnz_slice   [tid] = tend - tstart ;
            }
            tnvec_slice [0] = (nvals == 0) ? 0 : 1 ;

        }
        else
        {

            // count the # of unique vector indices in jwork.  No need to scan
            // iwork since there are no duplicates to be found.  Also no need
            // to compute them if already found in the first pass.

            if (!tnvec_and_tnz_slice_computed)
            {

                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (int tid = 0 ; tid < nthreads ; tid++)
                {
                    int64_t my_tnvec = 0 ;
                    int64_t tstart = tstart_slice [tid] ;
                    int64_t tend   = tstart_slice [tid+1] ;
                    int64_t jlast  = GB_JWORK (tstart-1) ;

                    for (int64_t t = tstart ; t < tend ; t++)
                    {
                        // get the t-th tuple
                        int64_t j = jwork [t] ;
                        if (j > jlast)
                        { 
                            // vector j starts in this slice
                            my_tnvec++ ;
                            jlast = j ;
                        }
                    }

                    tnvec_slice [tid] = my_tnvec ;
                    tnz_slice   [tid] = tend - tstart ;
                }
            }
        }

ttt = omp_get_wtime ( ) - ttt ;
printf ("nodup time %g\n", ttt) ;

    }
    else
    {

ttt = omp_get_wtime ( ) ;
        //----------------------------------------------------------------------
        // look for duplicates and count # vectors in each slice
        //----------------------------------------------------------------------

        int64_t ilast_slice [nthreads] ;
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            int64_t tstart = tstart_slice [tid] ;
            ilast_slice [tid] = GB_IWORK (tstart-1) ;
        }

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int tid = 0 ; tid < nthreads ; tid++)
        {

            int64_t my_tnvec = 0 ;
            int64_t my_ndupl = 0 ;
            int64_t tstart   = tstart_slice [tid] ;
            int64_t tend     = tstart_slice [tid+1] ;
            int64_t ilast    = ilast_slice [tid] ;
            int64_t jlast    = GB_JWORK (tstart-1) ;

            for (int64_t t = tstart ; t < tend ; t++)
            {
                // get the t-th tuple
                int64_t i = iwork [t] ;
                int64_t j = GB_JWORK (t) ;

                // tuples are now sorted but there may be duplicates
                ASSERT ((jlast < j) || (jlast == j && ilast <= i)) ;

                // check if (j,i,k) is a duplicate
                if (i == ilast && j == jlast)
                {
                    // flag the tuple as a duplicate
                    iwork [t] = -1 ;
//                  printf ("%d : %3d dupl ("GBd", "GBd")\n", tid, t, i, j) ;
                    my_ndupl++ ;
                    // the sort places earlier duplicate tuples (with smaller
                    // k) after later ones (with larger k).
                    ASSERT (GB_KWORK (t-1) < GB_KWORK (t)) ;
                }
                else
                {
                    // this is a new tuple
                    if (j > jlast)
                    { 
                        // vector j starts in this slice
                        my_tnvec++ ;
                        jlast = j ;
                    }
                    ilast = i ;
                }
            }
            tnvec_slice [tid] = my_tnvec ;
            tnz_slice   [tid] = (tend - tstart) - my_ndupl ;
        }
ttt = omp_get_wtime ( ) - ttt ;
printf ("dupl  time %g\n", ttt) ;
    }

    //--------------------------------------------------------------------------
    // find total # of vectors and duplicates in all tuples
    //--------------------------------------------------------------------------

    // Replace tnvec_slice with its cumulative sum, after which each slice tid
    // will be responsible for the # vectors in T that range from tnvec_slice
    // [tid] to tnvec_slice [tid+1]-1.
    GB_cumsum (tnvec_slice, nthreads, NULL, 1) ;
    int64_t tnvec = tnvec_slice [nthreads] ;

    // Replace tnz_slice with its cumulative sum
    GB_cumsum (tnz_slice, nthreads, NULL, 1) ;

    // find the total # of final entries, after assembling duplicates
    int64_t tnz = tnz_slice [nthreads] ;
    int64_t ndupl = nvals - tnz ;


/*
{
    printf ("\nnvals sliced: "GBd" dupl "GBd"\n", nvals, ndupl) ;
    size_t tsize = ttype->size ;
    size_t ssize = GB_code_size (scode, tsize) ;
    for (int tid = 0 ; tid < nthreads ; tid++)
    {
        printf ("\n   tid: %d------\n", tid) ;
        for (int jj = tstart_slice [tid] ; jj < tstart_slice [tid+1]  ; jj++)
        {
            printf ("%3d i "GBd" j "GBd", k "GBd" value: ", jj,
            GB_IWORK (jj), GB_JWORK (jj), GB_KWORK (jj)) ;
            GB_code_check (scode, S +(GB_KWORK (jj))*ssize, stdout, Context) ;
            printf ("\n") ;
        }
    }
}
*/


    //--------------------------------------------------------------------------
    // allocate T; always hypersparse
    //--------------------------------------------------------------------------

    // [ allocate T; allocate T->p and T->h but do not initialize them.
    // T is always hypersparse.
    GB_NEW (&T, ttype, vlen, vdim, GB_Ap_malloc, is_csc, GB_FORCE_HYPER,
        GB_ALWAYS_HYPER, tnvec, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_WORK ;
        return (info) ;
    }

    ASSERT (T->is_hyper) ;
    ASSERT (T->nzmax == 0) ;        // T->i and T->x not yet allocated

    //--------------------------------------------------------------------------
    // construct the vector pointers and hyperlist for T
    //--------------------------------------------------------------------------

    int64_t *restrict Th = T->h ;
    int64_t *restrict Tp = T->p ;

    if (ndupl == 0)
    {

        //----------------------------------------------------------------------
        // is it known that no duplicates appear
        //----------------------------------------------------------------------
ttt = omp_get_wtime ( ) ;

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int tid = 0 ; tid < nthreads ; tid++)
        {

            int64_t my_tnvec = tnvec_slice [tid] ;
            int64_t tstart   = tstart_slice [tid] ;
            int64_t tend     = tstart_slice [tid+1] ;
            int64_t jlast    = GB_JWORK (tstart-1) ;

            for (int64_t t = tstart ; t < tend ; t++)
            {
                // get the t-th tuple
                int64_t j = GB_JWORK (t) ;
                if (j > jlast)
                { 
                    // vector j starts in this slice
                    Th [my_tnvec] = j ;
                    Tp [my_tnvec] = t ;
                    my_tnvec++ ;
                    jlast = j ;
                }
            }
        }
ttt = omp_get_wtime ( ) - ttt ;
printf ("vec nodupl %g\n", ttt) ;

    }
    else
    {
ttt = omp_get_wtime ( ) ;

        //----------------------------------------------------------------------
        // it is known that at least one duplicate appears
        //----------------------------------------------------------------------

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int tid = 0 ; tid < nthreads ; tid++)
        {

            int64_t my_tnz   = tnz_slice [tid] ;
            int64_t my_tnvec = tnvec_slice [tid] ;
            int64_t tstart   = tstart_slice [tid] ;
            int64_t tend     = tstart_slice [tid+1] ;
            int64_t ilast    = GB_IWORK (tstart-1) ;
            int64_t jlast    = GB_JWORK (tstart-1) ;

            for (int64_t t = tstart ; t < tend ; t++)
            {
                // get the t-th tuple
                int64_t i = iwork [t] ;
                int64_t j = GB_JWORK (t) ;
                if (i >= 0)
                {
                    // this is a new tuple
                    if (j > jlast)
                    { 
                        // vector j starts in this slice 
                        Th [my_tnvec] = j ;
                        Tp [my_tnvec] = my_tnz ;
                        my_tnvec++ ;
                        jlast = j ;
                    }
                    my_tnz++ ;
                }
            }
        }
ttt = omp_get_wtime ( ) - ttt ;
printf ("vec dupl   %g\n", ttt) ;
    }

ttt = omp_get_wtime ( ) ;

    // log the end of the last vector
    T->nvec_nonempty = tnvec ;
    T->nvec = tnvec ;
    Tp [tnvec] = tnz ;
    ASSERT (T->nvec == T->plen) ;
    T->magic = GB_MAGIC ;                      // T->p and T->h are now valid ]

    //--------------------------------------------------------------------------
    // free jwork if it exists
    //--------------------------------------------------------------------------

    ASSERT (jwork_handle != NULL) ;
    GB_FREE_MEMORY (*jwork_handle, ijlen, sizeof (int64_t)) ;
    jwork = NULL ;

    //--------------------------------------------------------------------------
    // allocate T->i and T->x
    //--------------------------------------------------------------------------

    T->nzmax = GB_IMAX (tnz, 1) ;

    if (ndupl == 0)
    {
        // shrink iwork from size ijlen to size T->nzmax
        if (T->nzmax < ijlen)
        { 
            // this cannot fail since the size is shrinking.
            bool ok ;
            GB_REALLOC_MEMORY (iwork, T->nzmax, ijlen, sizeof (int64_t), &ok) ;
            ASSERT (ok) ;
        }
        // transplant iwork into T->i
        T->i = iwork ;
        iwork = NULL ;
        (*iwork_handle) = NULL ;
    }
    else
    { 
        // duplicates exist, so allocate a new T->i.  iwork must be freed later
        GB_MALLOC_MEMORY (T->i, tnz, sizeof (int64_t)) ;
    }

    GB_MALLOC_MEMORY (T->x, tnz, T->type->size) ;
    if (T->i == NULL || T->x == NULL)
    { 
        // out of memory
        GB_MATRIX_FREE (&T) ;
        GB_FREE_WORK ;
        return (GB_OUT_OF_MEMORY) ;
    }

    (*Thandle) = T ;

    GB_void *restrict Tx = T->x ;
    int64_t *restrict Ti = T->i ;

    //==========================================================================
    // numerical phase of the build: assemble any duplicates
    //==========================================================================

    // The tuples have been sorted.  Assemble any duplicates with a switch
    // factory of built-in workers, or four generic workers.  The vector
    // pointers T->p and hyperlist T->h (if hypersparse) have already been
    // computed.

    // If there are no duplicates, T->i holds the row indices of the tuple.
    // Otherwise, the row indices are still in iwork.  kwork holds the
    // positions of each tuple in the array S.  The tuples are sorted so that
    // duplicates are adjacent to each other and they appear in the order they
    // appeared in the original tuples.  This method assembles the duplicates
    // and computes T->i and T->x from iwork, kwork, and S.  into T, becoming
    // T->i.  If no duplicates appear, T->i is already computed, and S just
    // needs to be copied and permuted into T->x.

    // The (i,k,S[k]) tuples are held in two integer arrays: (1) iwork or T->i,
    // and (2) kwork, and an array S of numerical values.  S has not been
    // sorted, nor even accessed yet.  It is identical to the original unsorted
    // tuples.  The (i,k,S[k]) tuple holds the row index i, the position k, and
    // the value S [k].  This entry becomes T(i,j) = S [k] in the matrix T, and
    // duplicates (if any) are assembled via the dup operator.

    //--------------------------------------------------------------------------
    // get opcodes and check types
    //--------------------------------------------------------------------------

    // With GB_build, there can be 1 to 2 different types.
    //      T->type is identical to the types of x,y,z for z=dup(x,y).
    //      dup is never NULL and all its three types are the same
    //      The type of S (scode) can different but must be compatible
    //          with T->type

    // With GB_wait, there can be 1 to 5 different types:
    //      The pending tuples are in S, of type scode which must be
    //          compatible with dup->ytype and T->type
    //      z = dup (x,y): can be NULL or have 1 to 3 different types
    //      T->type: must be compatible with all above types.
    //      dup may be NULL, in which case it is assumed be the implicit SECOND
    //          operator, with all three types equal to T->type

    GrB_Type xtype, ytype, ztype ;
    GxB_binary_function fdup ;
    GB_Opcode opcode ;

    GB_Type_code tcode = ttype->code ;
    bool op_2nd ;

    ASSERT_OK (GB_check (ttype, "ttype for build_factorize", GB0)) ;

    if (dup == NULL)
    { 

        //----------------------------------------------------------------------
        // dup is the implicit SECOND operator
        //----------------------------------------------------------------------

        // z = SECOND (x,y) where all three types are the same as ttype
        // T(i,j) = (ttype) S(k) will be done for all tuples.

        opcode = GB_SECOND_opcode ;
        ASSERT (GB_op_is_second (dup, ttype)) ;
        xtype = ttype ;
        ytype = ttype ;
        ztype = ttype ;
        fdup = NULL ;
        op_2nd = true ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // dup is an explicit operator
        //----------------------------------------------------------------------

        // T(i,j) = (ttype) S[k] will be done for the first tuple.
        // for subsequent tuples: T(i,j) += S[k], via the dup operator and
        // typecasting:
        //
        //      y = (dup->ytype) S[k]
        //      x = (dup->xtype) T(i,j)
        //      z = (dup->ztype) dup (x,y)
        //      T(i,j) = (ttype) z

        ASSERT_OK (GB_check (dup, "dup for build_factory", GB0)) ;
        opcode = dup->opcode ;
        xtype = dup->xtype ;
        ytype = dup->ytype ;
        ztype = dup->ztype ;
        fdup = dup->function ;
        op_2nd = GB_op_is_second (dup, ttype) ;
    }

    //--------------------------------------------------------------------------
    // get the sizes and codes of each type
    //--------------------------------------------------------------------------

    GB_Type_code zcode = ztype->code ;
    GB_Type_code xcode = xtype->code ;
    GB_Type_code ycode = ytype->code ;

    ASSERT (GB_code_compatible (tcode, scode)) ;    // T(i,j) = (ttype) S
    ASSERT (GB_code_compatible (ycode, scode)) ;    // y = (ytype) S
    ASSERT (GB_Type_compatible (xtype, ttype)) ;    // x = (xtype) T(i,j)
    ASSERT (GB_Type_compatible (ttype, ztype)) ;    // T(i,j) = (ttype) z

    size_t tsize = ttype->size ;
    size_t zsize = ztype->size ;
    size_t xsize = xtype->size ;
    size_t ysize = ytype->size ;
    size_t ssize = GB_code_size (scode, tsize) ;

    //--------------------------------------------------------------------------
    // assemble the output
    //--------------------------------------------------------------------------

    // so that tcode can match scode
    GB_Type_code tcode2 = (tcode == GB_UCT_code) ? GB_UDT_code : tcode ;
    GB_Type_code scode2 = (scode == GB_UCT_code) ? GB_UDT_code : scode ;

    // no typecasting if all 5 types are the same
    bool nocasting = (tcode2 == scode2) &&
        (ttype == xtype) && (ttype == ytype) && (ttype == ztype) ;

    if (nocasting && known_sorted && ndupl == 0)
    { 

        //----------------------------------------------------------------------
        // copy S into T->x
        //----------------------------------------------------------------------

        // No typecasting is needed, the tuples were originally in sorted
        // order, and no duplicates appear.  All that is required is to copy S
        // into Tx.

        GB_memcpy (Tx, S, nvals * tsize, nthreads) ;

    }
    else if (nocasting)
    { 

        //----------------------------------------------------------------------
        // assemble the values, S, into T, no typecasting needed
        //----------------------------------------------------------------------

        // There are 44 common cases of this function for built-in types and
        // 8 associative operators: MIN, MAX, PLUS, TIMES for 10 types (all
        // but boolean; and OR, AND, XOR, and EQ for boolean.

        // In addition, the FIRST and SECOND operators are hard-coded, for
        // another 22 workers, since SECOND is used by GB_wait and since FIRST
        // is useful for keeping the first tuple seen.  It is controlled by the
        // GB_INCLUDE_SECOND_OPERATOR definition, so they do not appear in
        // GB_reduce_to_* where the FIRST and SECOND operators are not needed.

        // Early exit cannot be exploited, so the terminal value is ignored.

        #define GB_INCLUDE_SECOND_OPERATOR

        bool done = false ;

        #define GB_bild(opname,aname) GB_bild_ ## opname ## aname

        #define GB_ASSOC_WORKER(opname,aname,atype,terminal)                 \
        {                                                                    \
            GB_bild (opname, aname) ((atype *) Tx, Ti, (atype *) S, nvals,   \
                ndupl, iwork, kwork, tstart_slice, tnz_slice, nthreads) ;    \
            done = true ;                                                    \
        }                                                                    \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        #ifndef GBCOMPACT

            // controlled by opcode and typecode
            GB_Type_code typecode = tcode ;
            #include "GB_assoc_factory.c"

        #endif

        //----------------------------------------------------------------------
        // generic worker
        //----------------------------------------------------------------------

        if (!done)
        {

            //------------------------------------------------------------------
            // no typecasting, but use the fdup function pointer and memcpy
            //------------------------------------------------------------------

            // Tx [p] = S [k]
            #define GB_BUILD_COPY(Tx,p,S,k)                                 \
                memcpy (Tx +((p)*tsize), S +((k)*tsize), tsize) ;

            if (op_2nd)
            { 

                //--------------------------------------------------------------
                // dup is the SECOND operator, with no typecasting
                //--------------------------------------------------------------

                #define GB_BUILD_OP(Tx,p,S,k) GB_BUILD_COPY(Tx,p,S,k)
                #include "GB_build_template.c"

            }
            else
            { 

                //--------------------------------------------------------------
                // dup is another operator, with no typecasting needed
                //--------------------------------------------------------------

                // Tx [p] += S [k]
                #undef  GB_BUILD_OP
                #define GB_BUILD_OP(Tx,p,S,k)                               \
                    fdup (Tx +((p)*tsize), Tx +((p)*tsize), S +((k)*tsize)) ;
                #include "GB_build_template.c"
            }
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // assemble the values S into T, typecasting as needed
        //----------------------------------------------------------------------

        GB_cast_function cast_S_to_T = GB_cast_factory (tcode, scode) ;
        GB_cast_function cast_S_to_Y = GB_cast_factory (ycode, scode) ;
        GB_cast_function cast_T_to_X = GB_cast_factory (xcode, tcode) ;
        GB_cast_function cast_Z_to_T = GB_cast_factory (tcode, zcode) ;

        // The type of the S array differs from the type of T and dup, but both
        // types are built-in since user-defined types cannot be typecasted.
        ASSERT (scode <= GB_FP64_code) ;
        ASSERT (tcode <= GB_FP64_code) ;
        ASSERT (xcode <= GB_FP64_code) ;
        ASSERT (ycode <= GB_FP64_code) ;
        ASSERT (zcode <= GB_FP64_code) ;

        // Tx [p] = (ttype) (S [k])
        #undef  GB_BUILD_COPY
        #define GB_BUILD_COPY(Tx,p,S,k)                             \
            cast_S_to_T (Tx +((p)*tsize), S +((k)*ssize), ssize) ;

        if (op_2nd)
        { 

            //------------------------------------------------------------------
            // dup operator is the implicit SECOND operator, with typecasting
            //------------------------------------------------------------------

            #undef  GB_BUILD_OP
            #define GB_BUILD_OP(Tx,p,S,k) GB_BUILD_COPY(Tx,p,S,k)
            #include "GB_build_template.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // dup is another operator, with typecasting required
            //------------------------------------------------------------------

            // Tx [p] += S [k]
            #undef  GB_BUILD_OP
            #define GB_BUILD_OP(Tx,p,S,k)                               \
            {                                                           \
                /* ywork = (ytype) S [k] */                             \
                GB_void ywork [ysize] ;                                 \
                cast_S_to_Y (ywork, S +((k)*ssize), ssize) ;            \
                /* xwork = (xtype) Tx [p] */                            \
                GB_void xwork [xsize] ;                                 \
                cast_T_to_X (xwork, Tx +((p)*tsize), tsize) ;           \
                /* zwork = f (xwork, ywork) */                          \
                GB_void zwork [zsize] ;                                 \
                fdup (zwork, xwork, ywork) ;                            \
                /* Tx [tnz-1] = (ttype) zwork */                        \
                cast_Z_to_T (Tx +((p)*tsize), zwork, zsize) ;           \
            }
            #include "GB_build_template.c"
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORK ;
ttt = omp_get_wtime ( ) - ttt ;
printf ("numeric    %g\n", ttt) ;
    ASSERT_OK (GB_check (T, "T built", GB0)) ;
    return (GrB_SUCCESS) ;
}

