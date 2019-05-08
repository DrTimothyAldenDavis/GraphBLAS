//------------------------------------------------------------------------------
// GB_reduce_each_vector: T(j)=reduce(A(:,j)), reduce a matrix to a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Reduce a matrix to a vector.  The kth vector A(:,k) is reduced to the kth
// scalar T(k).  Each thread computes the reductions on roughly the same number
// of entries, which means that a vector A(:,k) may be reduced by more than one
// thread.  The first vector A(:,kfirst) reduced by thread tid may be partial,
// where the prior thread tid-1 (and other prior threads) may also do some of
// the reductions for this same vector A(:,kfirst).  The thread tid fully
// reduces all vectors A(:,k) for k in the range kfirst+1 to klast-1.  The last
// vector A(:,klast) reduced by thread tid may also be partial.  Thread tid+1,
// and following threads, may also do some of the reduces for A(:,klast).

// PARALLEL: done

{

    const GB_ATYPE *restrict Ax = A->x ;
    const int64_t  *restrict Ap = A->p ;

    //--------------------------------------------------------------------------
    // workspace for first and last vectors of each slice
    //--------------------------------------------------------------------------

    // ztype Wfirst [nthreads], Wlast [nthreads] ;
    GB_REDUCTION_WORKSPACE (Wfirst, nthreads) ;
    GB_REDUCTION_WORKSPACE (Wlast , nthreads) ;

    //--------------------------------------------------------------------------
    // reduce each slice
    //--------------------------------------------------------------------------

    // each thread reduces its own part in parallel
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int tid = 0 ; tid < nthreads ; tid++)
    {

        // if kfirst > klast then thread tid does no work at all
        int64_t kfirst = kfirst_slice [tid] ;
        int64_t klast  = klast_slice  [tid] ;

        //----------------------------------------------------------------------
        // reduce vectors kfirst to klast
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) to be reduced by this thread
            //------------------------------------------------------------------

            int64_t pA_start, pA_end ;
            if (k == kfirst)
            { 
                // First vector for thread tid; may only be partially owned.
                // It is reduced to Wfirst [tid].  Reduction always starts at
                // pstart_slice [tid], and ends either at the end of the vector
                // A(:,kfirst), or at the end of the entries this thread owns
                // in all of the A matrix, whichever comes first.
                pA_start = pstart_slice [tid] ;
                pA_end   = GB_IMIN (Ap [kfirst+1], pstart_slice [tid+1]) ;
            }
            else if (k == klast)
            { 
                // Last vector for thread tid; may only be partially owned.
                // It is reduced to Wlast [tid].  If kfirst == klast then
                // this case is skipped.  If kfirst < klast, then thread tid
                // owns the first part of A(:,k), so it always starts its work
                // at Ap [klast].  It ends its work at the end of the entries
                // this thread owns in A.
                pA_start = Ap [klast] ;
                pA_end   = pstart_slice [tid+1] ;
            }
            else
            { 
                // Thread tid fully owns this vector A(:,k), and reduces it
                // entirely to T(:,k).  No workspace is used.  The thread has
                // no such vectors if kfirst == klast.
                pA_start = Ap [k] ;
                pA_end   = Ap [k+1] ;
            }

            //------------------------------------------------------------------
            // reduce Ax [pA_start ... pA_end-1] to a scalar, if non-empty
            //------------------------------------------------------------------

            if (pA_start < pA_end)
            {

                //--------------------------------------------------------------
                // reduce the vector to the scalar s
                //--------------------------------------------------------------

                // ztype s = (ztype) Ax [pA_start], with typecast
                GB_CAST_ARRAY_TO_SCALAR (s, Ax, pA_start) ;
                for (int64_t p = pA_start+1 ; p < pA_end ; p++)
                {
                    // check for early exit
                    GB_BREAK_IF_TERMINAL (s) ;
                    // s += (ztype) Ax [p], with typecast
                    GB_ADD_CAST_ARRAY_TO_SCALAR (s, Ax, p) ;
                }

                //--------------------------------------------------------------
                // save the result s
                //--------------------------------------------------------------

                if (k == kfirst)
                {
                    // Wfirst [tid] = s ; no typecast
                    GB_COPY_SCALAR_TO_ARRAY (Wfirst, tid, s) ;
                }
                else if (k == klast)
                {
                    // Wlast [tid] = s ; no typecast
                    GB_COPY_SCALAR_TO_ARRAY (Wlast, tid, s) ;
                }
                else
                {
                    // Tx [k] = s ; no typecast
                    GB_COPY_SCALAR_TO_ARRAY (Tx, k, s) ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // reduce the first and last vector of each slice using a single thread
    //--------------------------------------------------------------------------

    // This step is sequential, but it takes only O(nthreads) time.  The only
    // case where this could be a problem is if a user-defined operator was
    // a very costly one.

    int64_t kprior = -1 ;

    for (int tid = 0 ; tid < nthreads ; tid++)
    {

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for kfirst
        //----------------------------------------------------------------------

        int64_t kfirst = kfirst_slice [tid] ;
        int64_t klast  = klast_slice  [tid] ;

        if (kfirst <= klast)
        {
            int64_t pA_start = pstart_slice [tid] ;
            int64_t pA_end   = GB_IMIN (Ap [kfirst+1], pstart_slice [tid+1]) ;
            if (pA_start < pA_end)
            {
                if (kprior < kfirst)
                { 
                    // This thread is the first one that did work on
                    // A(:,kfirst), so use it to start the reduction.
                    // Tx [kfirst] = Wfirst [tid], no typecast
                    GB_COPY_ARRAY_TO_ARRAY (Tx, kfirst, Wfirst, tid) ;
                }
                else
                { 
                    // Tx [kfirst] += Wfirst [tid], no typecast
                    GB_ADD_ARRAY_TO_ARRAY (Tx, kfirst, Wfirst, tid) ;
                }
                kprior = kfirst ;
            }
        }

        //----------------------------------------------------------------------
        // sum up the partial result that thread tid computed for klast
        //----------------------------------------------------------------------

        if (kfirst < klast)
        {
            int64_t pA_start = Ap [klast] ;
            int64_t pA_end   = pstart_slice [tid+1] ;
            if (pA_start < pA_end)
            {
                if (kprior < klast)
                { 
                    // This thread is the first one that did work on
                    // A(:,klast), so use it to start the reduction.
                    // Tx [klast] = Wlast [tid], no typecase
                    GB_COPY_ARRAY_TO_ARRAY (Tx, klast, Wlast, tid) ;
                }
                else
                { 
                    // Tx [klast] += Wlast [tid], no typecase
                    GB_ADD_ARRAY_TO_ARRAY (Tx, klast, Wlast, tid) ;
                }
                kprior = klast ;
            }
        }
    }
}

