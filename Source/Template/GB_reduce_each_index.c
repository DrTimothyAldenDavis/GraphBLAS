//------------------------------------------------------------------------------
// GB_reduce_each_index: T(i)=reduce(A(i,:)), reduce a matrix to a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Reduce a matrix to a vector.  All entries in A(i,:) are reduced to T(i).
// First, all threads reduce their slice to their own Sauna workspace,
// operating on roughly the same number of entries each.  The vectors in A are
// ignored; the reduction only depends on the indices.  Next, the threads
// cooperate to reduce all Sauna workspaces to the Sauna of thread 0.  Finally,
// this last Sauna workspace is collected into T.

// PARALLEL: done

{

    const GB_ATYPE *restrict Ax = A->x ;
    const int64_t  *restrict Ai = A->i ;
    const int64_t n = A->vlen ;

    //--------------------------------------------------------------------------
    // reduce each slice it its own Sauna
    //--------------------------------------------------------------------------

    GB_CTYPE **Sauna_Works [nth] ;
    int64_t  **Sauna_Marks [nth] ;
    int64_t  Tnz [nth] ;

    // each thread reduces its own slice in parallel
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (int tid = 0 ; tid < nth ; tid++)
    {

        // get the Sauna for this thread
        GB_Sauna Sauna = Saunas [tid] ;
        GB_CTYPE *restrict Sauna_Work = Sauna->Sauna_Work ;
        int64_t  *restrict Sauna_Mark = Sauna->Sauna_Mark ;
        Sauna_Works [tid] = Sauna_Work ;
        Sauna_Marks [tid] = Sauna_Mark ;
        int64_t my_tnz = 0 ;

        // reduce the entries
        for (int64_t p = pstart_slice [tid] ; p < pstart_slice [tid+1] ; p++)
        {
            int64_t i = Ai [p] ;
            // ztype aij = (ztype) Ax [p], with typecast
            GB_CAST_ARRAY_TO_SCALAR (aij, Ax, p) ;
            if (Sauna_Mark [i] < hiwater)
            {
                // first time index i has been seen
                // Sauna_Work [i] = aij ; no typecast
                GB_COPY_SCALAR_TO_ARRAY (Sauna_Work, i, aij) ;
                Sauna_Mark [i] = hiwater ;
                my_tnz++ ;
            }
            else
            {
                // Sauna_Work [i] += aij ; no typecast
                GB_ADD_SCALAR_TO_ARRAY (Sauna_Work, i, aij) ;
            }
        }
        Tnz [tid] = my_tnz ;
    }

    //--------------------------------------------------------------------------
    // reduce all Saunas to Sauna [0] and count # entries in T
    //--------------------------------------------------------------------------

    GB_CTYPE *restrict Sauna_Work0 = Sauna_Works [0] ;
    int64_t  *restrict Sauna_Mark0 = Sauna_Marks [0] ;
    int64_t tnz = Tnz [0] ;

    if (nth > 1)
    {
        #pragma omp parallel for num_threads(nthreads) schedule(static) \
            reduction(+:tnz)
        for (int64_t i = 0 ; i < n ; i++)
        {
            for (int tid = 1 ; tid < nth ; tid++)
            {
                const GB_CTYPE *restrict Sauna_Work = Sauna_Works [tid] ;
                const int64_t  *restrict Sauna_Mark = Sauna_Marks [tid] ;
                if (Sauna_Mark [i] == hiwater)
                {
                    // Sauna for thread tid has a contribution to index i
                    if (Sauna_Mark0 [i] < hiwater)
                    {
                        // first time index i has been seen
                        // Sauna_Work0 [i] = Sauna_Work [i] ; no typecast
                        GB_COPY_ARRAY_TO_ARRAY (Sauna_Work0, i,
                            Sauna_Work, i) ;
                        Sauna_Mark0 [i] = hiwater ;
                        tnz++ ;
                    }
                    else
                    {
                        // Sauna_Work0 [i] += Sauna_Work [i] ; no typecast
                        GB_ADD_ARRAY_TO_ARRAY (Sauna_Work0, i,
                            Sauna_Work, i) ;
                    }
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // allocate T
    //--------------------------------------------------------------------------

    // since T is a GrB_Vector, it is CSC and not hypersparse
    GB_CREATE (&T, ttype, n, 1, GB_Ap_calloc, true,
        GB_FORCE_NONHYPER, GB_HYPER_DEFAULT, 1, tnz, true, Context) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GB_OUT_OF_MEMORY) ;
    }

    T->p [0] = 0 ;
    T->p [1] = tnz ;
    int64_t  *restrict Ti = T->i ;
    GB_CTYPE *restrict Tx = T->x ;
    T->nvec_nonempty = (tnz > 0) ? 1 : 0 ;

    //--------------------------------------------------------------------------
    // gather the results into T
    //--------------------------------------------------------------------------

    if (tnz == n)
    {
        // T is dense: copy from Sauna_Work0
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int64_t i = 0 ; i < n ; i++)
        { 
            Ti [i] = i ;
        }
        size_t zsize = ttype->size ;
        GB_memcpy (Tx, Sauna_Work0, n * zsize, nthreads) ;
    }
    else
    {
        // T is sparse: gather from Sauna_Work0 and Sauna_Mark0
        // FUTURE: this is not yet parallel
        GB_CTYPE *restrict Tx = T->x ;
        int64_t p = 0 ;
        for (int64_t i = 0 ; i < n ; i++)
        {
            if (Sauna_Mark0 [i] == hiwater)
            { 
                Ti [p] = i ;
                // Tx [p] = Sauna_Work0 [i], no typecast
                GB_COPY_ARRAY_TO_ARRAY (Tx, p, Sauna_Work0, i) ;
                p++ ;
            }
        }
        ASSERT (p == tnz) ;
    }
}

