//------------------------------------------------------------------------------
// GB_dense_ewise3_noaccum_template: C = A+B where all 3 matrices are dense
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    // any matrix may be aliased to any other (C==A, C==B, and/or A==B)
    GB_ATYPE *Ax = A->x ;
    GB_BTYPE *Bx = B->x ;
    GB_CTYPE *Cx = C->x ;
    const int64_t cnz = GB_NNZ (C) ;
    ASSERT (GB_is_dense (A)) ;
    ASSERT (GB_is_dense (B)) ;
    ASSERT (GB_is_dense (C)) ;
    int64_t p ;

    //--------------------------------------------------------------------------
    // C = A+B where all 3 matrices are dense
    //--------------------------------------------------------------------------

    if (C == B)
    {

        //----------------------------------------------------------------------
        // C = A+C
        //----------------------------------------------------------------------

        #if GB_HAS_CBLAS & GB_OP_IS_PLUS_REAL

            GB_CBLAS_AXPY (cnz, (GB_CTYPE) 1, Ax, Cx, nthreads) ;   // C += A

        #else

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GETA (aij, Ax, p) ;                  // aij = Ax [p]
                GB_BINOP (GB_CX (p), aij, GB_CX (p)) ;  // Cx [p] = aij + Cx [p]
            }

        #endif

    }
    else if (C == A)
    {

        //----------------------------------------------------------------------
        // C = C+B
        //----------------------------------------------------------------------

        #if GB_HAS_CBLAS & GB_OP_IS_PLUS_REAL

            GB_CBLAS_AXPY (cnz, (GB_CTYPE) 1, Bx, Cx, nthreads) ;   // C += B

        #else

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GETB (bij, Bx, p) ;                  // bij = Bx [p]
                GB_BINOP (GB_CX (p), GB_CX (p), bij) ;  // Cx [p] += bij
            }

        #endif

    }
    else
    {

        //----------------------------------------------------------------------
        // C = A+B
        //----------------------------------------------------------------------

        #if GB_HAS_CBLAS & GB_OP_IS_PLUS_REAL

            GB_memcpy (Cx, Ax, cnz * sizeof (GB_CTYPE), nthreads) ; // C = A
            GB_CBLAS_AXPY (cnz, (GB_CTYPE) 1, Bx, Cx, nthreads) ;   // C += B

        #else

            #if 0
            int taskid ;
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (taskid = 0 ; taskid < nthreads ; taskid++)
            {
                int64_t p1, p2 ;
                GB_PARTITION (p1, p2, cnz, taskid, nthreads) ;
                GB_PRAGMA_VECTORIZE
                for (int64_t p = p1 ; p < p2 ; p++)
                {
                    GB_GETA (aij, Ax, p) ;              // aij = Ax [p]
                    GB_GETB (bij, Bx, p) ;              // bij = Bx [p]
                    GB_BINOP (GB_CX (p), aij, bij) ;    // Cx [p] = aij + bij
                }
            }
            #else

            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for (p = 0 ; p < cnz ; p++)
            {
                GB_GETA (aij, Ax, p) ;              // aij = Ax [p]
                GB_GETB (bij, Bx, p) ;              // bij = Bx [p]
                GB_BINOP (GB_CX (p), aij, bij) ;    // Cx [p] = aij + bij
            }

            #endif

        #endif
    }
}

