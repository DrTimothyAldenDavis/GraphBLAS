//------------------------------------------------------------------------------
// GB_hcat_fine_slice: horizontal concatenation and summation of slices of C
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// Horizontal concatenation and summation of fine slices into the matrix C.

#include "GB.h"

GrB_Info GB_hcat_fine_slice // horizontal concatenation and sum of slices of C
(
    GrB_Matrix *Chandle,    // output matrix C to create
    int nthreads,           // # of slices to concatenate
    GrB_Matrix *Cslice,     // array of slices of size nthreads
    GrB_Monoid add,         // monoid to use to sum up the entries
    bool any_Gustavson,     // true if any thread used Gustavson's method
    int *Sauna_ids,         // size nthreads, Sauna id's of each thread
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
        ASSERT_OK (GB_check (Cslice [tid], "a fine slice of C", GB0)) ;
        ASSERT (!GB_PENDING (Cslice [tid])) ;
        ASSERT (!GB_ZOMBIES (Cslice [tid])) ;
        ASSERT ((Cslice [tid])->is_hyper) ;
        // each Cslice [tid] is constructed as its own matrix, with
        // Cslice [tid] = A * Bslice [tid].  It is not a slice of an other
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

    // cnz and cnvec are upper bounds; exact values computed later
    int64_t cnz = 0 ;               // upper bound on nnz(C)
    int64_t cnvec = 0 ;             // upper bound on # vectors in C

    for (int tid = 0 ; tid < nthreads ; tid++)
    {
        // compute the cumulative sum of the # entries and # vectors
        cnz   += GB_NNZ (Cslice [tid]) ;
        cnvec += (Cslice [tid])->nvec ;
    }

    // printf ("bound: cnz "GBd" cnvec "GBd"\n", cnz, cnvec) ;

    //--------------------------------------------------------------------------
    // create C and allocate all of its space
    //--------------------------------------------------------------------------

    #define GB_FREE_ALL 

    GrB_Info info ;
    GB_CREATE (Chandle, ctype, cvlen, cvdim, GB_Ap_malloc, true,
        GB_FORCE_HYPER, GB_Global_hyper_ratio_get ( ), cnvec, cnz, true,
        Context) ;
    if (info != GrB_SUCCESS)
    {
        // out of memory
        if (any_Gustavson)
        {
            // at least one thread used a Sauna; free and release all
            // Sauna workspaces
            for (int tid = 0 ; tid < nthreads ; tid++)
            {
                int Sauna_id = Sauna_ids [tid] ;
                if (Sauna_id >= 0)
                { 
                    GB_Sauna_free (Sauna_id) ;
                }
            }
            GB_OK (GB_Sauna_release (nthreads, Sauna_ids)) ;
        }
        return (GB_OUT_OF_MEMORY) ;
    }

    #undef  GB_FREE_ALL 
    #define GB_FREE_ALL                     \
    {                                       \
        GB_MATRIX_FREE (Chandle) ;          \
    }

    GrB_Matrix C = (*Chandle) ;

    int64_t *restrict Ch = C->h ;
    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ci = C->i ;
    GB_void *restrict Cx = C->x ;
    size_t csize = ctype->size ;

    C->nvec_nonempty = -1 ;

    //--------------------------------------------------------------------------
    // acquire a single Sauna
    //--------------------------------------------------------------------------

    GB_Sauna Sauna = NULL ;
    int Sauna_id = -2 ;

    if (any_Gustavson)
    {
        // at least one thread already has a Sauna.  Use it.
        for (int tid = 0 ; tid < nthreads ; tid++)
        {
            Sauna_id = Sauna_ids [tid] ;
            // printf ("%d : Sauna id %d\n", tid, Sauna_id) ;
            if (Sauna_id >= 0)
            { 
                break ;
            }
        }
        // printf ("got sauna %d\n", Sauna_id) ;
        ASSERT (Sauna_id >= 0) ;
        Sauna = GB_Global_Saunas_get (Sauna_id) ;
    }
    else
    {

        // TODO: use a hypersparse-friendly method instead of the Sauna

        // no thread used a Sauna; acquire a single one
        int method = GxB_AxB_GUSTAVSON ;
        GB_OK (GB_Sauna_acquire (1, &Sauna_id, &method, Context)) ;

        // TODO: put this in a function:
        if (Sauna == NULL || Sauna->Sauna_n < cvlen || Sauna->Sauna_size <csize)
        {
            // get a new Sauna: the Sauna either does not exist, or is too small
            GB_Sauna_free (Sauna_id) ;
            info = GB_Sauna_alloc (Sauna_id, cvlen, csize) ;
            if (info != GrB_SUCCESS)
            { 
                GB_FREE_ALL ;
                GB_OK (GB_Sauna_release (1, &Sauna_id)) ;
                return (GB_OUT_OF_MEMORY) ;
            }
        }
        Sauna = GB_Global_Saunas_get (Sauna_id) ;
        // printf ("allocated new sauna %d\n", Sauna_id) ;
    }

    // Sauna has been acquired
    ASSERT (Sauna != NULL) ;
    ASSERT (Sauna->Sauna_n >= cvlen) ;
    ASSERT (Sauna->Sauna_size >= csize) ;

    // hiwater++
    int64_t hiwater = GB_Sauna_reset (Sauna, 1, 0) ;
    int64_t *restrict Sauna_Mark = Sauna->Sauna_Mark ;

    for (int64_t i = 0 ; i < C->vlen ; i++)
    {
        // printf ("Sauna_Mark ["GBd"] = "GBd"  hi "GBd"\n", i, Sauna_Mark [i],
        //     hiwater) ;
        ASSERT (Sauna_Mark [i] < hiwater) ;
    }

    // Sauna_Mark [0..cvlen-1] < hiwater holds
    ASSERT_SAUNA_IS_RESET ;

    // Sauna_Work has size cvlen, each entry of size csize.  Not initialized.
    GB_void *restrict Sauna_Work = Sauna->Sauna_Work ;

    //--------------------------------------------------------------------------
    // copy and sum each slice into C
    //--------------------------------------------------------------------------

    cnz = 0 ;           // now compute the exact cnz and cnvec of C
    cnvec = 0 ;

    // last_vector of prior slice
    int64_t last_vector = -1 ;

    GxB_binary_function fadd = add->op->function ;
    ASSERT (C->type == add->op->ztype) ;

    for (int tid = 0 ; tid < nthreads ; tid++)
    {

        //----------------------------------------------------------------------
        // get the Cslice [tid] and its position in C
        //----------------------------------------------------------------------

//      printf ("\n============================= do fine slice tid=%d\n", tid) ;
        ASSERT_OK (GB_check (Cslice [tid], "Cslice [tid]", GB0)) ;
//      printf ("last_vector "GBd"\n", last_vector) ;

        int64_t *restrict Csliceh = (Cslice [tid])->h ;
        int64_t *restrict Cslicep = (Cslice [tid])->p ;
        int64_t *restrict Cslicei = (Cslice [tid])->i ;
        GB_void *restrict Cslicex = (Cslice [tid])->x ;
        int64_t cnz_slice   = GB_NNZ (Cslice [tid]) ;
        int64_t cnvec_slice = (Cslice [tid])->nvec ;

        // skip if Cslice [tid] is empty
        if (cnvec_slice == 0) continue ;

        //----------------------------------------------------------------------
        // discard the first vector in Cslice [tid], if already summed
        //----------------------------------------------------------------------

        int64_t kfirst = 0 ;
        if (Csliceh [0] == last_vector)
        { 
            kfirst = 1 ;
//          printf ("discard first vector "GBd"\n", Csliceh [0]) ;
        }

//          int64_t first_vector_nz = Cslicep [1] - Cslicep [0] ;
//          Csliceh++ ;
//          Cslicep++ ;
//          Cslicei += first_vector_nz ;
//          Cslicex += first_vector_nz * csize ;
//          cnz_slice -= first_vector_nz ;
//          cnvec_slice-- ;
//          printf ("\nafter discard:\n") ;
//          for (int64_t k = 0 ; k < cnvec_slice ; k++)
//          {
//              printf ("Csliceh ["GBd"]="GBd"  ", k, Csliceh [k]) ;
//              printf ("Cslicep ["GBd"]="GBd"\n", k, Cslicep [k]) ;
//              for (int64_t p = Cslicep [k] ; p < Cslicep [k+1] ; p++)
//              {
//                  printf ("   row "GBd"\n", Cslicei [p]) ;
//              }
//          }
//          printf ("\n") ;

        // skip if Cslice [tid] is now empty
        if (cnvec_slice - kfirst == 0) continue ;

        //----------------------------------------------------------------------
        // search for last vector in subsequent slices
        //----------------------------------------------------------------------

        // search for the last vector of Cslice [tid] in all slices tid+1:tfine
        // that contain it as their first vector
        last_vector = Csliceh [cnvec_slice-1] ;
//      printf ("new last vector "GBd"\n", last_vector) ;
        int tfine = tid ;
        while (tfine+1 < nthreads)
        {
            if ((Cslice [tfine+1])->nvec == 0)
            {
                // slice tfine+1 is empty; include it and keep looking
                tfine++ ;
                continue ;
            }
            int64_t first_vector = (Cslice [tfine+1])->h [0] ;
            if (last_vector == first_vector)
            {
                // the last vector of Cslice [tid] is the same as the first
                // vector of Cslice [tfine+1].  Add tfine+1 to the list, and
                // continue looking for last_vector in subsequent slices.
                tfine++ ;
                continue ;
            }
            else
            {
                // Cslice [tfine+1] starts with a different vector than the
                // last vector of Cslice [tid], so it is not part of the list.
                break ;
            }
        }

        #ifdef GB_DEBUG
        // check the list
        for (int tid2 = tid + 1 ; tid2 <= tfine ; tid2++)
        {
            // slice tid2 is in the set tid+1:tfine.  If it has any vectors, its
            // first vector is last_vector
            ASSERT ((Cslice [tid2])->nvec >= 0) ;
            if ((Cslice [tid2])->nvec > 0) 
            {
                ASSERT ((Cslice [tid2])->h [0] == last_vector) ;
            }
        }
        if (tfine+1 < nthreads)
        {
            // slice tfine+1, if it exists, contains at least one vector, and
            // it differs from last_vector
            ASSERT ((Cslice [tfine+1])->nvec > 0) ;
            ASSERT ((Cslice [tfine+1])->h [0] != last_vector) ;
        }
//      printf ("tid %d tfine %d\n", tid, tfine) ;
        #endif

        //----------------------------------------------------------------------
        // copy the bulk of Cslice [tid] into C
        //----------------------------------------------------------------------

        // exclude the first vector from Cslice [tid] if it's been discarded
        int64_t pfirst = Cslicep [kfirst] ;

        // copy the row indices of Cslice [tid] into Ci and Cx
        memcpy (Ci +(cnz), Cslicei +(pfirst),
            (cnz_slice - pfirst) * sizeof (int64_t)) ;

        // copy the values of Cslice [tid] into Ci and Cx
        memcpy (Cx +(cnz * csize), Cslicex +(pfirst * csize),
            (cnz_slice - pfirst) * csize) ;

        // copy the column indices of Cslice into Ch
        memcpy (Ch +(cnvec), Csliceh +(kfirst),
            (cnvec_slice - kfirst) * sizeof (int64_t)) ;

        // construct the column pointers of C (shift upwards by cnz)
//      printf ("\n construct Cp\n") ;
        for (int64_t k = kfirst ; k < cnvec_slice ; k++)
        {
            Cp [cnvec++] = Cslicep [k] + cnz - pfirst ;
        }


        // entries and vectors have been appended to C
        cnz   += (cnz_slice - pfirst) ;
//      printf ("now cnz "GBd"\n", cnz) ;

        // the last_vector in C starts at Ci [cnz_last] and Cx [cnz_last]
        int64_t cnz_last = Cp [cnvec-1] ;

//      for (int64_t k = 0 ; k < cnvec ; k++)
//      {
//          printf ("Cp ["GBd"] = "GBd"\n", k, Cp [k]) ;
//      }
//
//      printf ("cnz_last "GBd" cnvec now "GBd" Cp[cnvec-1] "GBd"\n",
//          cnz_last, cnvec, Cp [cnvec-1]) ;

        //----------------------------------------------------------------------
        // handle the last vector, if it needs summation
        //----------------------------------------------------------------------

        if (tfine > tid)
        {

            //------------------------------------------------------------------
            // scatter the last_vector from Cslice [tid] into the Sauna
            //------------------------------------------------------------------

            int64_t pstart = Cslicep [cnvec_slice-1] ;
            int64_t pend   = Cslicep [cnvec_slice] ;
            for (int64_t p = pstart ; p < pend ; p++)
            {
                int64_t i = Cslicei [p] ;
//              printf ("last vector, row "GBd"\n", i) ;
                Sauna_Mark [i] = hiwater ;
                // Sauna_Work [i] = Cslicex [p]
                memcpy (Sauna_Work +(i*csize), Cslicex +(p*csize), csize) ;
            }

            bool unsorted = false ;

            //------------------------------------------------------------------
            // scatter and add each subsequent vector
            //------------------------------------------------------------------

            for (int tid2 = tid + 1 ; tid2 <= tfine ; tid2++)
            {
                // skip if Cslice [tid2] is empty
                if ((Cslice [tid2])->nvec == 0) continue ;

                int64_t *restrict Cslice2p = (Cslice [tid2])->p ;
                int64_t *restrict Cslice2i = (Cslice [tid2])->i ;
                GB_void *restrict Cslice2x = (Cslice [tid2])->x ;

                // scatter/add first vector from Cslice [tid2] into the Sauna
                int64_t pstart = Cslice2p [0] ;
                int64_t pend   = Cslice2p [1] ;
                for (int64_t p = pstart ; p < pend ; p++)
                {
                    int64_t i = Cslice2i [p] ;
//                  printf ("tid %d subsequent row "GBd"\n", tid, i) ;
                    if (Sauna_Mark [i] < hiwater)
                    {
                        // first time row index i has been seen
                        Sauna_Mark [i] = hiwater ;
                        // Sauna_Work [i] = Cslice2x [p]
                        memcpy (Sauna_Work +(i*csize), Cslice2x +(p*csize),
                                csize) ;
                        // append row index i to C
                        Ci [cnz++] = i ;
                        unsorted = true ;
                    }
                    else
                    {
                        // C(i,last_vector) += Cslice2 (i,last_vector)
                        fadd (Sauna_Work +(i*csize), Sauna_Work +(i*csize),
                              Cslice2x +(p*csize)) ;
                    }
                }
            }

            //------------------------------------------------------------------
            // sort the pattern of C(:,j)
            //------------------------------------------------------------------

            if (unsorted)
            {
                // sort the pattern of C(:,j)
                int64_t len = cnz - cnz_last ;
                if (len == cvlen)
                {
                    // no need to sort C(:,j) if dense; just recreate it
                    for (int64_t pC = cnz_last, i = 0 ; pC < cnz ; pC++, i++)
                    { 
                        Ci [pC] = i ;
                    }
                }
                else
                { 
                    // sort the nonzero indices in C(:,j)
                    GB_qsort_1 (Ci + cnz_last, len, NULL) ;
                }
            }

            //------------------------------------------------------------------
            // gather the values into C(:,j)
            //------------------------------------------------------------------

            for (int64_t pC = cnz_last ; pC < cnz ; pC++)
            {
                int64_t i = Ci [pC] ;
//              printf ("gathered row "GBd"\n", i) ;
                // Cx [pC] = Sauna_Work [i]
                memcpy (Cx +(pC*csize), Sauna_Work +(i*csize), csize) ;
            }
//          printf ("wrapup cnz "GBd"\n", cnz) ;

            //------------------------------------------------------------------
            // hiwater++
            //------------------------------------------------------------------

            hiwater = GB_Sauna_reset (Sauna, 1, 0) ;
        }
    }

    //--------------------------------------------------------------------------
    // release Sauna workspaces
    //--------------------------------------------------------------------------

    if (any_Gustavson)
    { 
        // release all Sauna workspaces, if any
        GB_OK (GB_Sauna_release (nthreads, Sauna_ids)) ;
    }
    else
    {
        // release the newly acquired Sauna
        GB_OK (GB_Sauna_release (1, &Sauna_id)) ;
    }

    //--------------------------------------------------------------------------
    // finalize the matrix
    //--------------------------------------------------------------------------

    C->nvec = cnvec ;
    Cp [cnvec] = cnz ;
    C->magic = GB_MAGIC ;
    ASSERT_OK (GB_check (C, "C from fine concatenation", GB0)) ;
    return (GrB_SUCCESS) ;
}

