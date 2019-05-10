//------------------------------------------------------------------------------
// GB_delete_zombies: delete all zombies from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// TODO: put the bulk of this method in a template, and use for GB_select.

#include "GB.h"

#define GB_FREE_ALL                                     \
{                                                       \
    GB_FREE_MEMORY (Cp, anvec+1, sizeof (int64_t)) ;    \
    GB_FREE_MEMORY (Ci, cnz, sizeof (int64_t)) ;        \
    GB_FREE_MEMORY (Cx, cnz, asize) ;                   \
}

GrB_Info GB_delete_zombies
(
    GrB_Matrix A,               // matrix to delete zombies from
    GB_Context Context
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (A, "A input for delete zombies", GB_FLIP (GB0))) ;
    if (!GB_ZOMBIES (A)) return (GrB_SUCCESS) ;

    // There are zombies that will now be deleted.
    ASSERT (GB_ZOMBIES_OK (A)) ;

    // This step tolerates pending tuples
    // since pending tuples and zombies do not intersect
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    GB_GET_NTHREADS (nthreads, Context) ;
    int64_t anz = GB_NNZ (A) ;
    // TODO reduce nthreads for small problem (work: about O(anz))

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    int64_t *restrict Ah = A->h ;
    int64_t *restrict Ap = A->p ;
    int64_t *restrict Ai = A->i ;
    GB_void *restrict Ax = A->x ;
    int64_t asize = A->type->size ;
    int64_t anvec = A->nvec ;
    int64_t aplen = A->plen ;

    //--------------------------------------------------------------------------
    // allocate the new vector pointers
    //--------------------------------------------------------------------------

    int64_t *restrict Cp = NULL ;
    int64_t *restrict Ci = NULL ;
    GB_void *restrict Cx = NULL ;
    GB_CALLOC_MEMORY (Cp, aplen+1, sizeof (int64_t)) ;
    if (Cp == NULL)
    {
        // out of memory
        return (GB_OUT_OF_MEMORY) ;
    }
    Cp [anvec] = 0 ;

    //--------------------------------------------------------------------------
    // slice the entries for each thread
    //--------------------------------------------------------------------------

    // Thread tid does entries pstart_slice [tid] to pstart_slice [tid+1]-1 and
    // vectors kfirst_slice [tid] to klast_slice [tid].  The first and last
    // vectors may be shared with prior slices and subsequent slices.

    int64_t pstart_slice [nthreads+1] ;
    int64_t kfirst_slice [nthreads] ;
    int64_t klast_slice  [nthreads] ;

    GB_ek_slice (pstart_slice, kfirst_slice, klast_slice, A, nthreads) ;

    //--------------------------------------------------------------------------
    // count the live entries in each vector
    //--------------------------------------------------------------------------

    // Use the GB_reduce_each_vector template to count the number of live
    // entries in each vector of A.  The values Ax are not accessed.  Instead,
    // the 0/1 value GB_IS_NOT_ZOMBIE (Ai [p]) is summed up for each vector in
    // A.  The result is computed in Cp, where Cp [k] = # of live entries in
    // the kth vector of A.

    int64_t *restrict Tx = Cp ;     // for GB_reduce_each_vector

    #define GB_ATYPE int64_t
    #define GB_CTYPE int64_t

    // workspace for each thread
    int64_t Wfirst [nthreads] ;
    int64_t Wlast  [nthreads] ;
    for (int tid = 0 ; tid < nthreads ; tid++)
    {
        Wfirst [tid] = 0 ;
        Wlast  [tid] = 0 ;
    }
    #define GB_REDUCTION_WORKSPACE(W, nthreads) ;

    // ztype s = (ztype) Ax [p], with typecast
    #define GB_CAST_ARRAY_TO_SCALAR(s,Ax,p)                 \
        int64_t s = GB_IS_NOT_ZOMBIE (Ai [p]) ;

    // s += (ztype) Ax [p], with typecast
    #define GB_ADD_CAST_ARRAY_TO_SCALAR(s, Ax, p)           \
        s += GB_IS_NOT_ZOMBIE (Ai [p])

    // W [k] = s, no typecast
    #define GB_COPY_SCALAR_TO_ARRAY(W,k,s)                  \
        W [k] = s

    // W [k] = S [i], no typecast
    #define GB_COPY_ARRAY_TO_ARRAY(W,k,S,i)                 \
        W [k] = S [i]

    // W [k] += S [i], no typecast
    #define GB_ADD_ARRAY_TO_ARRAY(W,k,S,i)                  \
        W [k] += S [i]

    // no terminal value
    #define GB_BREAK_IF_TERMINAL(t) ;

    #include "GB_reduce_each_vector.c"

    //--------------------------------------------------------------------------
    // compute the new vector pointers
    //--------------------------------------------------------------------------

//  for (int64_t k = 0 ; k < anvec ; k++)
//  {
//      printf ("  k "GBd" j "GBd" Cp [k] = "GBd"\n",
//          k, (A->h == NULL) ? k : A->h [k], Cp [k]) ;
//  }

    // Cp = cumsum (Cp)
    int64_t C_nvec_nonempty ;
    GB_cumsum (Cp, anvec, &C_nvec_nonempty, nthreads) ;
    int64_t cnz = Cp [anvec] ;

    //--------------------------------------------------------------------------
    // determine the slice boundaries in the new C matrix
    //--------------------------------------------------------------------------

    int64_t C_pstart_slice [nthreads] ;
    GB_map_pslice (C_pstart_slice, Cp, kfirst_slice, klast_slice,
        Wfirst, Wlast, nthreads) ;

    //--------------------------------------------------------------------------
    // allocate new space for the compacted Ci and Cx
    //--------------------------------------------------------------------------
    
    GB_MALLOC_MEMORY (Ci, cnz, sizeof (int64_t)) ;
    GB_MALLOC_MEMORY (Cx, cnz, asize) ;
    if (Ci == NULL || Cx == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GB_OUT_OF_MEMORY) ;
    }

    // exactly A->nzombies will be deleted from A
//  printf ("znz "GBd"\n", GB_NNZ (A)) ;
//  printf ("nzombies cnz "GBd"\n", A->nzombies) ;
//  printf ("new cnz "GBd"\n", cnz) ;
    ASSERT (A->nzombies == (GB_NNZ (A) - cnz)) ;

    //--------------------------------------------------------------------------
    // delete the zombies
    //--------------------------------------------------------------------------

    // each thread does its own part in parallel
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int tid = 0 ; tid < nthreads ; tid++)
    {

        // if kfirst > klast then thread tid does no work at all
        int64_t kfirst = kfirst_slice [tid] ;
        int64_t klast  = klast_slice  [tid] ;

//      printf ("tid %d kfirst "GBd" klast "GBd"\n", tid, kfirst, klast) ;

        //----------------------------------------------------------------------
        // reduce vectors kfirst to klast
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) to be reduced by this thread
            //------------------------------------------------------------------

            int64_t pA_start, pA_end, pC ;
            if (k == kfirst)
            { 
                // First vector for thread tid; may only be partially owned.
                // It is reduced to Wfirst [tid].  Reduction always starts at
                // pstart_slice [tid], and ends either at the end of the vector
                // A(:,kfirst), or at the end of the entries this thread owns
                // in all of the A matrix, whichever comes first.
                pA_start = pstart_slice [tid] ;
                pA_end   = GB_IMIN (Ap [kfirst+1], pstart_slice [tid+1]) ;
                pC = C_pstart_slice [tid] ;
//              printf ("  at kfirst pA "GBd":"GBd" pC "GBd"\n",
//                  pA_start, pA_end, pC) ;
            }
            else if (k == klast)
            { 
                // Last vector for thread tid; may only be partially owned.
                // It is reduced to Wlast [tid].  If kfirst == klast then
                // this case is skipped.  If kfirst < klast, then thread tid
                // owns the first part of A(:,k), so it always starts its work
                // at Ap [klast].  It ends its work at the end of the entries
                // this thread owns in A.
                pA_start = Ap [k] ;
                pA_end   = pstart_slice [tid+1] ;
                pC = Cp [k] ;
//              printf ("  at klast  pA "GBd":"GBd" pC "GBd"\n",
//                  pA_start, pA_end, pC) ;
            }
            else
            { 
                // Thread tid fully owns this vector A(:,k), and reduces it
                // entirely to T(:,k).  No workspace is used.  The thread has
                // no such vectors if kfirst == klast.
                pA_start = Ap [k] ;
                pA_end   = Ap [k+1] ;
                pC = Cp [k] ;
//              printf ("  in middle pA "GBd":"GBd" pC "GBd"\n",
//                  pA_start, pA_end, pC) ;
            }

            //------------------------------------------------------------------
            // compact Ai and Ax [pA_start ... pA_end-1] into Ci and Cx
            //------------------------------------------------------------------

            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            {
                int64_t i = Ai [pA] ;

                if (GB_IS_NOT_ZOMBIE (i))
                {
//                  printf ("    move "GBd" to "GBd"\n", pC, pA) ;
                    Ci [pC] = i ;
                    // Cx [pC] = Ax [p] ;
                    memcpy (Cx +((pC)*asize), Ax +(pA)*asize, asize) ;
                    pC++ ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // prune empty vectors from A, if hypersparse, and transplant Cp into A->p
    //--------------------------------------------------------------------------

//    printf ("check for prune\n") ;
//    for (int64_t k = 0 ; k < anvec ; k++)
//    {
//        printf ("  k "GBd" j "GBd" Cp [k] = "GBd"\n",
//            k, (A->h == NULL) ? k : A->h [k], Cp [k]) ;
//    }
//    printf ("  Cp [anvec] "GBd"\n", Cp [anvec]) ;

    if (A->is_hyper && C_nvec_nonempty < anvec)
    {
        // prune empty vectors from Ah and Ap
//      printf ("prune empties\n") ;
        int64_t cnvec = 0 ;
        for (int64_t k = 0 ; k < anvec ; k++)
        {
            if (Cp [k] < Cp [k+1])
            {
//              printf ("keep k "GBd" j "GBd"\n", k, Ah [k]) ;
                Ah [cnvec] = Ah [k] ;
                Ap [cnvec] = Cp [k] ;
                cnvec++ ;
            }
        }
        Ap [cnvec] = Cp [anvec] ;
        A->nvec = cnvec ;
        ASSERT (A->nvec == C_nvec_nonempty) ;
        GB_FREE_MEMORY (Cp, aplen+1, sizeof (int64_t)) ;
    }
    else
    {
        GB_FREE_MEMORY (Ap, aplen+1, sizeof (int64_t)) ;
        A->p = Cp ; Cp = NULL ;
    }

    ASSERT (Cp == NULL) ;

    //--------------------------------------------------------------------------
    // transplant Ci and Cx back into A
    //--------------------------------------------------------------------------

    GB_FREE_MEMORY (Ai, A->nzmax, sizeof (int64_t)) ;
    GB_FREE_MEMORY (Ax, A->nzmax, asize) ;
    A->i = Ci ; Ci = NULL ;
    A->x = Cx ; Cx = NULL ;
    A->nzmax = cnz ;
    A->nzombies = 0 ;
    A->nvec_nonempty = C_nvec_nonempty ;

    if (A->nzmax == 0)
    {
        GB_FREE_MEMORY (A->i, A->nzmax, sizeof (int64_t)) ;
        GB_FREE_MEMORY (A->x, A->nzmax, asize) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    ASSERT_OK (GB_check (A, "A output for delete zombies", GB_FLIP (GB0))) ;
    ASSERT (A->nvec_nonempty == GB_nvec_nonempty (A, NULL)) ;
    return (GrB_SUCCESS) ;
}

