//------------------------------------------------------------------------------
// GB_dense_subassign_23_template: C += A where C is dense and A is sparse
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

// All entries in C+=A are computed fully in parallel, using the same kind of
// parallelism as Template/GB_AxB_colscale.c.

{

    //--------------------------------------------------------------------------
    // get C and A
    //--------------------------------------------------------------------------

    const int64_t  *GB_RESTRICT Ap = A->p ;
    const int64_t  *GB_RESTRICT Ah = A->h ;
    const int64_t  *GB_RESTRICT Ai = A->i ;
    const GB_ATYPE *GB_RESTRICT Ax = A->x ;

    GB_CTYPE *GB_RESTRICT Cx = C->x ;
    const int64_t cvlen = C->vlen ;

    //--------------------------------------------------------------------------
    // C += A
    //--------------------------------------------------------------------------

    int taskid ;
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic,1)
    for (taskid = 0 ; taskid < ntasks ; taskid++)
    {

        // if kfirst > klast then taskid does no work at all
        int64_t kfirst = kfirst_slice [taskid] ;
        int64_t klast  = klast_slice  [taskid] ;

        //----------------------------------------------------------------------
        // C(:,kfirst:klast) += A(:,kfirst:klast)
        //----------------------------------------------------------------------

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) and C(:,k) to be operated on by this task
            //------------------------------------------------------------------

            int64_t j = (Ah == NULL) ? k : Ah [k] ;
            int64_t pA_start, pA_end ;
            GB_get_pA_and_pC (&pA_start, &pA_end, NULL,
                taskid, k, kfirst, klast, pstart_slice, NULL, NULL, Ap) ;

            // pC points to the start of C(:,j) if C is dense
            int64_t pC = j * cvlen ;

            // TODO handle the case when A is dense, so that the pattern
            // of A need not be accessed.

            //------------------------------------------------------------------
            // C(:,j) += A(:,j)
            //------------------------------------------------------------------

            GB_PRAGMA_VECTORIZE
            for (int64_t pA = pA_start ; pA < pA_end ; pA++)
            { 
                int64_t p = pC + Ai [pA] ;
                GB_GETB (aij, Ax, pA) ;                     // aij = A(i,j)
                GB_BINOP (GB_CX (p), GB_CX (p), aij) ;      // C(i,j) += aij
            }
        }
    }
}

